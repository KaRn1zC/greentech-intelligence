# Prompts Perplexity pour Veille Hebdomadaire

> **Projet** : GreenTech Intelligence
> **Rédigé par** : KaRn1zC - 2026-02-09

---

## 🚨 Option 1 : Tâche Planifiée Perplexity (VERSION COURTE)

**Limite** : ~1500-2000 caractères maximum

**⚠️ ATTENTION** : Cette version est **fortement réduite** en raison des limitations de Perplexity. La qualité sera inférieure à la version manuelle.

```
Effectue une veille technique hebdomadaire pour le projet GreenTech Intelligence (plateforme de classification automatique d'articles selon leur pertinence Green IT).

Stack : Python 3.12, FastAPI, PostgreSQL, PyTorch (AMD ROCm), Transformers (Hugging Face), Spark, MLflow, DVC, CodeCarbon, Docker, Prometheus, Grafana.

Thématiques à analyser (semaine du [DATE] au [DATE]) :

1. Green IT : réglementations environnementales, métriques carbone IT, éco-conception, hébergement vert
2. Sustainable AI : optimisation modèles (quantization, pruning), NLP efficient, benchmarks énergétiques, fine-tuning (LoRA, QLoRA)
3. Cloud & Infrastructure : optimisations PyTorch, support AMD ROCm, Spark, DevOps durables
4. APIs IA : services Hugging Face, alternatives éco-responsables, comparaisons énergétiques

Format de sortie :
- 5-7 actualités majeures (titre, source, date, résumé, pertinence projet)
- 2-3 analyses approfondies (contexte technique, implications projet, ressources officielles)
- Opportunités d'implémentation (bibliothèques Python, techniques d'optimisation, métriques)
- Ressources complémentaires (articles, repos GitHub, documentations)

Prioriser sources fiables (papers, blogs officiels, conférences), données quantitatives, compatibilité Windows + AMD ROCm.
```

**Nombre de caractères** : ~1150 caractères ✅

---

## ✅ Option 2 : MANUELLE (VERSION COMPLÈTE - RECOMMANDÉE)

**Avantages** :
- Contrôle total sur le prompt
- Qualité supérieure des résultats
- Possibilité d'ajuster selon les besoins de la semaine
- Pas de limite de caractères

**Workflow** : Tous les lundis à 11:00 AM

### Étape 1 : Copier le prompt complet

```
Tu es un expert en veille technologique pour un projet de recherche académique nommé "GreenTech Intelligence". Ta mission est de réaliser une analyse approfondie des avancées de la semaine dernière (du [DATE DÉBUT] au [DATE FIN]).

## CONTEXTE DU PROJET

GreenTech Intelligence est une plateforme d'analyse et de classification automatique d'articles technologiques selon leur pertinence "Green IT" (informatique durable et éco-responsable).

**Stack technique** :
- Backend : Python 3.12, FastAPI, PostgreSQL, Apache Spark
- IA : PyTorch (AMD ROCm), Transformers (Hugging Face), DeBERTa fine-tuné
- MLOps : MLflow, DVC, CodeCarbon
- DevOps : Docker, Prometheus, Grafana, GitHub Actions

## THÉMATIQUES À ANALYSER

### 1. Green IT & Numérique Responsable
- Nouvelles réglementations environnementales (Europe, France)
- Métriques et outils de mesure d'empreinte carbone IT
- Bonnes pratiques éco-conception web et logicielle
- Initiatives d'hébergement vert et cloud durable

### 2. Sustainable AI & Model Efficiency
- Techniques d'optimisation de modèles (quantization, pruning, distillation)
- Nouveaux modèles NLP légers ou efficaces énergétiquement
- Benchmarks d'efficacité énergétique des modèles IA (LLMs, transformers)
- Outils de tracking carbone pour ML (CodeCarbon, alternatives)
- Fine-tuning efficient (LoRA, QLoRA, adaptateurs)

### 3. Cloud & Infrastructure Efficiency
- Optimisations PyTorch et frameworks deep learning
- Support GPU AMD (ROCm, DirectML) et optimisations
- Avancées Apache Spark pour Big Data
- Pratiques DevOps durables (green CI/CD)
- Monitoring et observabilité (Prometheus, Grafana)

### 4. APIs & Services IA Durables
- Nouveaux services d'inférence Hugging Face
- Alternatives éco-responsables aux APIs IA commerciales
- Comparaisons énergétiques entre providers (AWS, Azure, etc.)

## FORMAT DE SORTIE ATTENDU

Structure ta réponse en 4 sections :

### 📰 Faits Marquants de la Semaine
Liste 5-7 actualités majeures avec :
- Titre et source
- Date de publication
- Résumé (2-3 phrases)
- Pertinence pour GreenTech Intelligence (impact direct/indirect)

### 🔬 Analyses Approfondies
Sélectionne 2-3 sujets et fournis pour chacun :
- Contexte technique détaillé
- Implications pour le projet (applicabilité concrète)
- Liens vers ressources officielles (documentation, papers, GitHub)

### 💡 Opportunités d'Implémentation
Identifie des éléments actionnables :
- Nouvelles bibliothèques Python à tester
- Techniques d'optimisation à intégrer
- Métriques supplémentaires à monitorer
- Améliorations possibles du pipeline MLOps

### 📚 Ressources Complémentaires
Liste :
- Articles de référence (blogs techniques, papers)
- Repositories GitHub pertinents
- Documentations officielles mises à jour
- Webinars ou conférences à venir

## CRITÈRES DE QUALITÉ

- Prioriser les sources fiables : papers académiques, blogs officiels (Hugging Face, PyTorch, etc.), conférences reconnues
- Inclure des données quantitatives (benchmarks, métriques énergétiques)
- Vérifier la compatibilité avec l'écosystème Windows + AMD ROCm
- Mentionner les implications RGPD/légales si pertinent
- Éviter les buzzwords marketing, privilégier l'analyse technique

## PÉRIODE D'ANALYSE

Semaine du [LUNDI DATE] au [DIMANCHE DATE]

---

Effectue une recherche approfondie en mode "Deep Research" et fournis une synthèse structurée et exploitable.
```

### Étape 2 : Remplacer les dates

**Exemple pour la semaine du 10 au 16 février 2026** :
- `[DATE DÉBUT]` → `lundi 10 février 2026`
- `[DATE FIN]` → `dimanche 16 février 2026`
- `[LUNDI DATE]` → `lundi 10 février 2026`
- `[DIMANCHE DATE]` → `dimanche 16 février 2026`

### Étape 3 : Utilisation dans Perplexity

1. Ouvrir https://www.perplexity.ai/
2. **Cliquer sur le bouton "New Thread"** (nouvelle conversation)
3. Coller le prompt complet avec les dates remplacées
4. **Sélectionner le modèle "Deep Research"** (menu déroulant en haut)
5. Appuyer sur Entrée
6. Attendre 2-5 minutes (Deep Research prend plus de temps)
7. Copier le résultat généré

### Étape 4 : Sauvegarder la synthèse

```bash
# Copier le template
cp docs/veille/TEMPLATE_synthese_hebdomadaire.md docs/veille/2026-02-10_synthese.md

# Ouvrir avec VSCode
code docs/veille/2026-02-10_synthese.md
```

Coller le résultat de Perplexity dans le fichier, compléter les sections manquantes (statistiques, actions), puis sauvegarder.

---

## 📊 Comparaison des Options

| Critère | Tâche Planifiée | Manuelle |
|---------|----------------|----------|
| **Automatisation** | ✅ Automatique | ❌ Manuelle (5 min setup) |
| **Qualité résultats** | ⚠️ Moyenne (prompt court) | ✅ Excellente (prompt détaillé) |
| **Flexibilité** | ❌ Fixe | ✅ Adaptable chaque semaine |
| **Limite caractères** | ❌ ~2000 caractères | ✅ Illimitée |
| **Validation C6** | ⚠️ Acceptable | ✅ Optimale |

---

## 🎯 Recommandation Finale

**Utilise la méthode MANUELLE (Option 2)** pour ces raisons :

1. **Qualité** : Le prompt complet donne des résultats beaucoup plus précis et exploitables
2. **Flexibilité** : Tu peux ajuster le prompt selon les besoins de la semaine (exemple : "cette semaine, focus sur PyTorch ROCm")
3. **Validation C6** : Le jury appréciera la profondeur des analyses
4. **Temps** : 5 minutes de setup le lundi matin, c'est acceptable

**Si tu veux vraiment automatiser**, utilise la version courte (Option 1), mais sache que tu devras probablement **compléter manuellement** les résultats pour avoir la qualité requise.

---

## 🔄 Workflow Recommandé (Méthode Manuelle)

### Lundi matin 11:00 AM (30-45 min total)

1. **☕ Café + Inoreader** (10 min)
   - Parcourir les nouveaux articles de la semaine
   - Marquer 3-5 articles intéressants avec une étoile ⭐

2. **🔍 Perplexity Deep Research** (15 min)
   - Ouvrir `docs/veille/PROMPT_PERPLEXITY.md`
   - Copier le prompt complet (Option 2)
   - Remplacer les dates
   - Coller dans Perplexity + Deep Research
   - Attendre résultat (2-5 min)

3. **📝 Sauvegarde synthèse** (10 min)
   - Copier template → nouveau fichier daté
   - Coller résultat Perplexity
   - Compléter section "📊 Statistiques de Veille"
   - Ajouter notes personnelles depuis Inoreader

4. **🎯 Actions** (10 min)
   - Lire section "💡 Opportunités d'Implémentation"
   - Créer 1-2 issues GitHub si pertinent (label `veille`)
   - Ajouter au backlog si nécessaire

---

**Prochain test** : Lundi 10 février 2026 à 11:00 AM ! 🚀
