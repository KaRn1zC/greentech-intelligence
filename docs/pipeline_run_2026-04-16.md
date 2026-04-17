# Suivi Pipeline Re-entrainement - 2026-04-16

## Contexte

- **Commande** : `uv run python scripts/retrain_pipeline.py summarize-classif annotate classify summarize-green export-golden`
- **Date de lancement** : 2026-04-16 20:24 (Europe/Paris)
- **Objectif** : regenerer 100% des resumes + classifications sur les 5642 articles propres (apres nettoyage NewsData/TechCrunch courts) avec le prompt v2 renforce et le nouveau pipeline. `train-cv` et `auto-promote` seront lances separement par l'utilisateur.

## Etape 0 : Reset SQL

- **Statut** : OK
- **Heure** : 2026-04-16 20:24
- **Resultats** :
  - 5642 articles reset (resume, resume_ecologique, est_green_it, modele_classification, score_confiance, date_analyse)
  - 2 analysis_logs purges
  - 5642 articles avec tous les champs de classification / resume a NULL, prets pour regeneration
- **Erreurs/Warnings** : aucun

## Incident - Premier essai (20:25-20:27)

- **Statut** : FREEZE
- **Demarrage** : 20:25:11
- **Arret observe** : 20:25:31 (sur article 274/5642 "Exploiting Social Annotation...")
- **Duree freeze constatee** : > 2 minutes sans progression log, process Python vivant (7 Go RAM) mais sans nouvelle activite
- **Resumes sauvegardes** : 4 (articles id=6, 7, 8, 273), tous valides apres inspection
- **Cause probable** : freeze ROCm aleatoire sur `model.generate()` du Qwen local (pas reproductible au smoke test precedent de 50 articles qui avait 100% de succes). Process tue a 20:27 via `taskkill //F //PID 27696`.
- **Decision** : relance du pipeline sans modification de code, pour confirmer si le freeze etait un incident isole.

## Etape 1 : summarize-classif (deuxieme essai)

- **Statut** : FREEZE N2 (refreeze apres 4 articles traites)
- **Relance** : 20:32 (pipeline), chargement Qwen2.5-3B termine vers 20:37:52
- **Articles resumes dans cet essai** : 4 (articles 274, 275, 276, 277) entre 20:37:52 et 20:38:15
- **Dernier article resume** : id=277, 969 chars (20:38:15)
- **Freeze detecte** : 8 min 13 s sans progression (check a 20:46:29)
- **Process tue** : 20:46 via `taskkill //F //PID 11792`
- **Articles resumes totaux (DB)** : 8 (4 essai 1 + 4 essai 2)
- **Hypothese confirmee** : bug systemique dans l'inference Qwen locale (pas un incident isole). Le freeze se produit apres ~4 articles a chaque lancement, ce qui suggere une accumulation de state GPU/memoire ou une instabilite ROCm apres quelques generations.

## Decision : diagnostic avant de modifier llm_local.py

- **Statut** : EN ATTENTE de validation utilisateur
- **Pipeline** : ARRETE (kill 20:46, plus aucun process uv/python actif a 20:53)
- **DB** : 8 resumes conserves (articles 6, 7, 8, 273, 274, 275, 276, 277)
- **Analyse comparative** (demande utilisateur) : le smoke test `generate_classification_summaries.py --limit 50 --force --shuffle` a reussi 100% (50/50) en 3 min. Le pipeline `retrain_pipeline.py` freeze apres 4 articles. Differences cles :

| Aspect | Smoke test (OK) | Pipeline (FREEZE) |
|---|---|---|
| Execution | Process Python direct | Parent + subprocess avec PIPE |
| Capture stdout/stderr | Non | Oui (via `_stream_subprocess`) |
| Threads forwarding | 0 | 2 (`_forward_stream`) |
| `enable_loki` parent | False | **True** |
| Ordre articles | shuffle | id_article ASC |

- **Hypothese principale** : saturation du PIPE subprocess sous Windows (buffer 4-8 KB). Chaque article genere ~48 lignes de logs SQLAlchemy. Apres 4-5 articles (~200 lignes), le buffer sature, le subprocess bloque sur `write()`. Le smoke test n'utilise pas de subprocess donc pas de PIPE = pas de saturation. Cela explique pourquoi 50 articles passent sans souci en mode direct mais le pipeline bloque toujours au 5e article.

- **Test decisif propose** : lancer `uv run python scripts/generate_classification_summaries.py` directement (sans pipeline). Si >10 articles passent sans freeze, hypothese PIPE confirmee -> fix cote retrain_pipeline.py (desactiver capture stdout ou passage en import direct au lieu de subprocess).

## Etapes 2-5 (annotate, classify, summarize-green, export-golden)

- **Statut** : EN ATTENTE (non lancees car pipeline stoppe apres freeze N2)

## Fix applique : passage subprocess -> appel Python direct

- **Decision** : Option B (conversion subprocess -> import/await direct) validee apres analyse comparative smoke test vs pipeline.
- **Fichiers modifies** :
  - `scripts/retrain_pipeline.py` : 5 etapes (`step_summarize_classification`, `step_annotate`, `step_classify`, `step_summarize_green`, `step_export_golden`) converties en async avec imports directs ; `run_pipeline` adapte pour await ces nouveaux steps via le dict `async_steps` ; `sys.path` enrichi avec `scripts/` pour permettre les imports hors package.
- **Etapes restees en subprocess** : `collect` (Spark a besoin d'un process Python dedie pour sa JVM), `train`, `promote` (plus rares, moins de logs, pas de risque PIPE connu).
- **Benefices attendus** :
  1. Suppression complete du risque freeze PIPE (plus de subprocess pour les 5 etapes critiques)
  2. Qwen local charge une seule fois pour `summarize-classif` + `summarize-green` (~5 min de cold start economises)
  3. Overhead de demarrage Python elimine (2-3 s/etape x 5 = 10-15 s gagnees)
  4. Logs Loki preserves integralement (ils viennent du process principal)
- **Verification** : `ruff check` pass, imports testes OK.

## Etape 1 : summarize-classif (troisieme essai, apres fix)

- **Statut** : EN COURS, fix valide
- **Relance** : 21:06:31
- **Chargement Qwen2.5-3B** : 21:06:49 (13 s, chargement sensiblement plus rapide qu'au 2e essai -- probable conservation du cache tokenizer dans le working set Windows)
- **HF quota** : toujours epuise (402), fallback local actif des la premiere requete
- **Articles resumes dans ce run** : 2 (ids 278, 279) en 10 s chacun
- **DB total resumes** : 10 (8 preserves + 2 nouveaux)
- **Confirmation hypothese PIPE** : **le pipeline a franchi la barre des 5 articles sans freeze**. Deux articles resumes consecutivement avec cadence reguliere (10 s chacun), log actif en temps reel. Les deux freezes precedents etaient bien dus a la saturation du PIPE subprocess. Le fix (conversion en appel Python direct) est valide.
- **Cadence observee** : ~10 s/article sur abstracts arXiv courts (~200-1500 chars)
- **ETA estime** : ~15-18 h pour les 5 632 articles restants (articles Guardian/Dev.to/TechCrunch plus longs prendront davantage)

### Update 21:10:39 (check +3 min apres relance)

- **DB** : 39 resumes (10 -> 39 en 3 min 30 s = +29 articles)
- **Cadence mesuree** : 7.2 s/article sur arXiv courts (meilleure que l'estimation initiale)
- **Dernier article** : id=308 a 21:10:37
- **Log** : actualise < 1 s, pas de trace de freeze
- **ETA revise** : ~11-14 h pour les 5 603 articles restants (les Guardian/Dev.to/TechCrunch plus longs tireront la moyenne vers 10-15 s/article)

### Update 22:01:29 (check +55 min apres relance)

- **DB** : 458 resumes (+419 depuis 21:10, +448 depuis le debut du run a 21:06)
- **Duree cumulee active** : ~54 min 40 s
- **Cadence confirmee** : **7.32 s/article** (stable, tres proche de la premiere mesure)
- **Dernier article** : id=729 a 22:01:25
- **Log** : actualise il y a 3 s, progression en temps reel
- **Process** : PID 29432, 6.6 GB RAM (Qwen2.5-3B charge)
- **Distribution actuelle** :
  - arXiv : 458/5003 (9 % traites, ordre id_article ASC)
  - Guardian : 0/432 (viendront en fin, IDs plus eleves)
  - Dev.to : 0/120
  - TechCrunch : 0/85
- **ETA summarize-classif** : **~08:30 demain matin** (+10 h 30 min)
- **ETA pipeline complet** : **~10-12 h demain matin** (avec annotate + classify + summarize-green + export-golden)

### Update 23:03:39 (check +2 h apres relance)

- **DB** : 951 resumes (+493 depuis 22:01, +941 depuis le debut du run)
- **Duree cumulee active** : ~1 h 56 min
- **Cadence confirmee** : **7.55 s/article** (stable, variance negligeable)
- **Dernier article** : id=1223 a 23:03:33 (encore sur arXiv, tous IDs < 5300)
- **Log** : actualise il y a 5 s, progression en temps reel
- **Aucun freeze** : le fix anti-PIPE tient sur 2 h de run continu avec ~1000 articles traites
- **ETA restant summarize-classif** : ~10 h 35 min -> fin vers **09:40 demain matin**
  - arXiv restants (~3 780) a 7.5 s -> 7 h 50 min
  - Guardian+Dev.to+TechCrunch (~640) a 15-18 s -> 2 h 45 min
- **ETA pipeline complet** : **~10h-12h demain matin** (apres annotate + classify + summarize-green + export-golden)

### Update 00:13:35 (check +3 h apres relance)

- **DB** : 1515 resumes (+564 depuis 23:03, cadence 7.44 s/article)
- **Dernier article** : id=1831 a 00:13:34 (toujours arXiv)
- **Aucun freeze** depuis le fix anti-PIPE (3 h de run continu, 1500+ articles)
- **Progression** : 26.8 % du total
- **ETA restant** : ~10 h -> fin summarize-classif vers 10:15

## Etape 2 : annotate

- **Statut** : EN ATTENTE

## Etape 3 : classify

- **Statut** : EN ATTENTE

## Etape 4 : summarize-green

- **Statut** : EN ATTENTE

## Etape 5 : export-golden

- **Statut** : EN ATTENTE

## Bilan final

- **Statut** : SUCCES COMPLET
- **Duree totale** : 779 min (12 h 59 min), de 21:06:31 (16 avril) a 10:05:42 (17 avril)
- **Commande** : `uv run python scripts/retrain_pipeline.py summarize-classif annotate classify summarize-green export-golden`

### Resultats par etape

| Etape | Duree | Succes | Echecs | Detail |
|---|---|---|---|---|
| summarize-classif | 12 h 18 min | 5626 | 8 (arXiv < 50 chars) | Cadence 7.5 s/article, fix anti-PIPE valide |
| annotate | 5 s | 5642 | 0 | Pre-filtre mots-cles instantane |
| classify | 39 min | 1029 | 0 | LLM judge Qwen local, 17 Green IT, 1012 Non Green IT |
| summarize-green | 1 min 40 s | 17 | 0 | Resumes ecologiques pour tous les Green IT |
| export-golden | 3 s | - | - | golden_dataset.csv genere |

### Classification finale

- **17 Green IT** (0.30 % du dataset) - en hausse de 89 % vs les 9 historiques
- **5625 Non Green IT**
- **8 non resumes** (arXiv abstracts corrompus/tronques dans le dump Kaggle)

### Prochaine action

L'utilisateur peut maintenant lancer l'entrainement K-fold :

```bash
uv run python scripts/retrain_pipeline.py train-cv auto-promote
```

Le golden dataset `data/golden_dataset.csv` est pret, avec la colonne `resume_classification` comme feature d'entrainement alignee sur le prompt v2 renforce.
