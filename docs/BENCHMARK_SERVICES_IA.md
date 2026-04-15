# Benchmark des Services IA de Resume Automatique

> **Projet** : GreenTech Intelligence
> **Competence validee** : C7 - Identifier des services IA (Benchmark)

> **Note de lecture** : Les sections 1 a 7 documentent le benchmark initial
> (phase de selection) qui a conduit au choix de **BART-large-CNN** comme
> premiere solution retenue. Apres les crash tests en conditions reelles
> (cold-starts >120s, resume en anglais pour un public francophone,
> qualite extractive limitee), le choix a evolue vers **Qwen2.5-7B-Instruct**.
> L'historique complet de la migration est detaille dans la section 8.
> L'architecture finale unifiee (un seul modele pour les deux resumes) est
> decrite dans la section 9.

---

## 1. Reformulation du Besoin

### Problematique fonctionnelle

La plateforme GreenTech Intelligence collecte quotidiennement des articles technologiques depuis 3 sources (API NewsData.io, scraping TechCrunch, fichiers arXiv). Les articles stockes en base PostgreSQL necessitent un **resume automatique** pour :

- Permettre aux utilisateurs de parcourir rapidement les articles sans lire le contenu integral
- Alimenter le tableau de bord avec des aperçus synthetiques
- Ameliorer la pertinence de la classification Green IT en fournissant un texte condense

### Problematique technique

Le service de resume doit :

- Traiter des textes en **anglais** (articles techniques)
- Generer des resumes de **30 a 150 mots**
- Etre appelable de maniere **asynchrone** depuis Python (FastAPI)
- Respecter les contraintes **RGPD** (pas de retention de donnees par le fournisseur)
- Avoir un **impact carbone mesurable et minimal**

---

## 2. Contraintes du Projet

### Contraintes budgetaires
| Critere | Contrainte |
|---------|-----------|
| Budget | Projet etudiant — **gratuit ou quasi-gratuit** |
| Volume | ~50-200 articles/jour maximum |
| Duree | Projet sur 6 mois (fevrier - juillet 2026) |

### Contraintes techniques
| Critere | Contrainte |
|---------|-----------|
| Langage | Python 3.12 (UV) |
| Framework | FastAPI (async obligatoire) |
| GPU | AMD Radeon RX 7900 XTX (ROCm) — pas de NVIDIA |
| Hebergement | Render (free tier pour la pre-prod) |
| Stack existante | PostgreSQL, MinIO, SQLAlchemy async, httpx |

### Contraintes operationnelles
| Critere | Contrainte |
|---------|-----------|
| Disponibilite | Service accessible 24/7 (pas d'auto-hebergement complexe) |
| Latence | < 5 secondes par article acceptable |
| RGPD | Pas de retention des donnees envoyees |
| Green IT | Impact carbone minimal (coherent avec la philosophie du projet) |

---

## 3. Services Etudies

### 3.1 OpenAI API (GPT-4o-mini)

**Description** : API commerciale d'OpenAI, modele GPT-4o-mini optimise pour les taches rapides.

| Critere | Evaluation | Score |
|---------|-----------|-------|
| Cout | $0.15/1M tokens input, $0.60/1M output. Free tier : $5 credits initiaux | ★★★☆☆ |
| Integration Python | SDK `openai` officiel, tres bien documente | ★★★★★ |
| Qualite du resume | Excellente comprehension du contexte technique | ★★★★★ |
| Impact carbone | Non mesure officiellement, datacenters energivores, PUE non publie | ★★☆☆☆ |
| RGPD | Donnees potentiellement traitees aux USA, DPA disponible mais complexe | ★★☆☆☆ |
| Latence | ~1-3 secondes par requete | ★★★★☆ |
| Vendor lock-in | Eleve — modeles proprietaires, pas de self-hosting | ★☆☆☆☆ |
| Open source | Non | ★☆☆☆☆ |

**Raison d'exclusion** : Impact carbone non transparent, donnees potentiellement traitees hors UE (RGPD), vendor lock-in eleve, cout qui pourrait depasser le budget etudiant en cas de montee en charge.

### 3.2 Mistral AI (Mistral Small)

**Description** : API francaise, modele Mistral Small optimise pour les taches de texte.

| Critere | Evaluation | Score |
|---------|-----------|-------|
| Cout | $0.1/1M tokens input, $0.3/1M output. Free tier : credits limites | ★★★☆☆ |
| Integration Python | SDK `mistralai`, bonne documentation | ★★★★☆ |
| Qualite du resume | Bonne sur l'anglais technique, excellente sur le francais | ★★★★☆ |
| Impact carbone | Hebergement en France (Scaleway), PUE < 1.2 annonce | ★★★★☆ |
| RGPD | Donnees traitees en France/UE, DPA clair | ★★★★★ |
| Latence | ~1-4 secondes | ★★★★☆ |
| Vendor lock-in | Modere — modeles partiellement open-weight | ★★★☆☆ |
| Open source | Partiellement (poids disponibles, licence restrictive) | ★★★☆☆ |

**Raison d'exclusion** : Bonne option RGPD mais le free tier est trop limite pour un projet etudiant sur 6 mois. Le modele est surdimensionne pour de la simple summarization (7B params pour un resume de 150 mots).

### 3.3 Google Cloud Vertex AI (Gemini 2.0 Flash)

**Description** : API Google Cloud avec le modele Gemini Flash optimise pour la vitesse.

| Critere | Evaluation | Score |
|---------|-----------|-------|
| Cout | $0.075/1M tokens. Free tier : 15 req/min pendant 1 an | ★★★★☆ |
| Integration Python | SDK `google-cloud-aiplatform`, complexe a configurer | ★★★☆☆ |
| Qualite du resume | Tres bonne, multimodale | ★★★★★ |
| Impact carbone | Google compense 100% carbone, mais datacenters massifs | ★★★☆☆ |
| RGPD | Region UE disponible, DPA Google Cloud robuste | ★★★★☆ |
| Latence | ~0.5-2 secondes (Flash) | ★★★★★ |
| Vendor lock-in | Eleve — ecosysteme GCP complet necessaire | ★★☆☆☆ |
| Open source | Non | ★☆☆☆☆ |

**Raison d'exclusion** : Configuration GCP complexe pour un projet etudiant, vendor lock-in Google, et le free tier necessite un compte billing actif.

### 3.4 Hugging Face Serverless Inference API (BART-large-CNN)

**Description** : API d'inference gratuite pour les modeles open source heberges sur Hugging Face Hub.

| Critere | Evaluation | Score |
|---------|-----------|-------|
| Cout | **Gratuit** (rate-limited) pour les modeles publics | ★★★★★ |
| Integration Python | SDK `huggingface_hub` officiel, `InferenceClient` simple | ★★★★★ |
| Qualite du resume | BART-large-CNN : reference pour la summarization, 406M params | ★★★★☆ |
| Impact carbone | Modele leger (406M vs 7B+), inference mutualisee, neutralite carbone HF | ★★★★★ |
| RGPD | Pas de retention des donnees, serveurs UE disponibles | ★★★★★ |
| Latence | ~2-5 secondes (serverless cold start possible) | ★★★☆☆ |
| Vendor lock-in | **Aucun** — modele telechargeble, self-hostable a tout moment | ★★★★★ |
| Open source | Oui — modele MIT/Apache, poids publics, reproductible | ★★★★★ |

**Selection initiale** : ✅ Retenue comme solution principale a l'issue du
benchmark de phase 1.

> **Mise a jour post-integration** : BART-large-CNN a finalement ete remplace
> par **Qwen2.5-7B-Instruct** (voir section 8 pour le detail). Le fournisseur
> retenu reste identique (Hugging Face Serverless Inference API), seul le
> modele appele a change, pour les raisons de qualite abstractive et de
> coherence linguistique francaise.

### 3.5 Cohere (Command-R)

**Description** : API specialisee en NLP, modele Command-R pour la generation.

| Critere | Evaluation | Score |
|---------|-----------|-------|
| Cout | Free tier : 1000 req/mois (Trial), puis $1/1M tokens | ★★★☆☆ |
| Integration Python | SDK `cohere`, bien documente | ★★★★☆ |
| Qualite du resume | Bonne qualite, RAG integre | ★★★★☆ |
| Impact carbone | Non publie | ★★☆☆☆ |
| RGPD | Traitement aux USA principalement | ★★☆☆☆ |
| Latence | ~1-3 secondes | ★★★★☆ |
| Vendor lock-in | Modere — API proprietaire mais modeles partiellement ouverts | ★★★☆☆ |
| Open source | Partiellement | ★★★☆☆ |

**Raison d'exclusion** : Free tier trop limite (1000 req/mois), impact carbone non publie (incoherent avec la philosophie Green IT du projet), traitement hors UE.

---

## 4. Service Non Etudie

### AWS Bedrock (Amazon)

**Raison de non-etude** :
- **Complexite** : Necessite un compte AWS avec configuration IAM, VPC, et billing
- **Cout** : Pas de free tier pour l'inference sur Bedrock (minimum $0.0008/1K tokens)
- **Surdimensionne** : Infrastructure enterprise inadaptee a un projet etudiant
- **Vendor lock-in** : Ecosysteme AWS complet
- L'ecosysteme AWS est trop complexe et couteux pour les besoins du projet

---

## 5. Adequation Fonctionnelle

### Tableau Comparatif Final

| Critere (poids) | OpenAI | Mistral | Google | **HF** | Cohere |
|-----------------|--------|---------|--------|--------|--------|
| Cout (25%) | 3/5 | 3/5 | 4/5 | **5/5** | 3/5 |
| Integration (15%) | 5/5 | 4/5 | 3/5 | **5/5** | 4/5 |
| Qualite (20%) | 5/5 | 4/5 | 5/5 | **4/5** | 4/5 |
| Impact carbone (15%) | 2/5 | 4/5 | 3/5 | **5/5** | 2/5 |
| RGPD (10%) | 2/5 | 5/5 | 4/5 | **5/5** | 2/5 |
| Vendor lock-in (10%) | 1/5 | 3/5 | 2/5 | **5/5** | 3/5 |
| Open source (5%) | 1/5 | 3/5 | 1/5 | **5/5** | 3/5 |
| **Score pondere** | **3.15** | **3.70** | **3.50** | **4.80** | **3.10** |

### Detail des calculs

```
OpenAI  : (3×0.25) + (5×0.15) + (5×0.20) + (2×0.15) + (2×0.10) + (1×0.10) + (1×0.05) = 3.15
Mistral : (3×0.25) + (4×0.15) + (4×0.20) + (4×0.15) + (5×0.10) + (3×0.10) + (3×0.05) = 3.70
Google  : (4×0.25) + (3×0.15) + (5×0.20) + (3×0.15) + (4×0.10) + (2×0.10) + (1×0.05) = 3.50
HF      : (5×0.25) + (5×0.15) + (4×0.20) + (5×0.15) + (5×0.10) + (5×0.10) + (5×0.05) = 4.80
Cohere  : (3×0.25) + (4×0.15) + (4×0.20) + (2×0.15) + (2×0.10) + (3×0.10) + (3×0.05) = 3.10
```

---

## 6. Demarche Eco-Responsable

### Comparaison de l'empreinte carbone estimee

| Service | Taille modele | Infra | Compensation | Score Green IT |
|---------|--------------|-------|-------------|---------------|
| OpenAI GPT-4o-mini | ~8B params | Datacenters USA | Non publie | ★★☆☆☆ |
| Mistral Small | ~7B params | Scaleway FR (PUE<1.2) | Partielle | ★★★★☆ |
| Google Gemini Flash | Non publie | Google (100% compense) | Oui | ★★★☆☆ |
| **HF BART-large-CNN** | **406M params** | **Mutualize, UE** | **Neutralite HF** | **★★★★★** |
| Cohere Command-R | ~35B params | AWS USA | Non publie | ★★☆☆☆ |

### Justification Green IT (phase initiale)

Le choix de **BART-large-CNN via Hugging Face** etait le plus coherent avec la philosophie Green IT du projet lors de la phase de benchmark initiale :

1. **Modele 18x plus leger** que les alternatives generatives (406M vs 7-35B parametres)
2. **Concu specifiquement pour la summarization** — pas besoin d'un LLM generique surdimensionne
3. **Inference mutualisee** — le serveur Hugging Face partage les ressources entre utilisateurs
4. **Possibilite de self-hosting** — deploiement local ulterieur sur le GPU AMD 7900 XTX pour eliminer totalement le transfert reseau
5. **Mesurable** — CodeCarbon peut mesurer l'impact en self-hosting

> **Arbitrage final** : BART a ete remplace par Qwen2.5-7B-Instruct apres les
> crash tests (section 8). Le surcout carbone lie a un modele de 7B parametres
> est compense par (1) la mutualisation HF, (2) la suppression d'une etape
> de traduction necessaire avec BART pour un public francophone, et (3) la
> consolidation sur un **seul modele pour les deux resumes** (general + Green IT)
> au lieu de deux modeles distincts, ce qui reduit la duplication d'appels.

---

## 7. Contraintes Techniques et Pre-requis

### Hugging Face Serverless Inference API

| Pre-requis | Detail |
|-----------|--------|
| Compte | Gratuit sur huggingface.co |
| Token | API token (Read) dans le fichier .env |
| SDK Python | `huggingface-hub >= 0.20.0` (deja dans pyproject.toml) |
| Reseau | Acces HTTPS sortant vers api-inference.huggingface.co |
| Rate limit | ~30 req/min (free tier), suffisant pour 200 articles/jour |
| Cold start | 5-20 sec si le modele n'est pas en cache (rare) |
| Input max | ~1024 tokens (BART) / ~32k tokens (Qwen2.5-7B-Instruct), troncature cote client |
| Output | JSON avec `summary_text` (BART) ou `chat_completion.choices[0].message.content` (Qwen) |

### Integration dans la stack existante

> **Version actuelle (post-migration)** : le SDK reste `huggingface_hub` mais
> l'appel passe desormais par `chat_completion` (format OpenAI-compatible) avec
> un prompt systeme francais. Voir le module `src/greentech/ai/services/summarizer.py`
> pour l'implementation de reference.

```python
from huggingface_hub import AsyncInferenceClient

client = AsyncInferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    token=settings.huggingface_token,
)
response = await client.chat_completion(
    messages=[
        {"role": "system", "content": "Tu es un redacteur technique..."},
        {"role": "user", "content": f"Resume cet article en francais :\n{texte}"},
    ],
    max_tokens=250,
    temperature=0.3,
)
resume = response.choices[0].message.content
```

- Compatible avec FastAPI (async natif)
- Resultat stocke dans la colonne `resume` de la table `articles`
  (et `resume_ecologique` pour les articles classifies Green IT)
- Module developpe : `src/greentech/ai/services/summarizer.py`

---

## 8. Conclusions et Preconisations

### Solution retenue

**Hugging Face Serverless Inference API** avec le modele **`Qwen/Qwen2.5-7B-Instruct`**
(LLM instructif d'Alibaba, licence Apache-2.0), appele via `chat_completion`
avec un **prompt systeme specialise** en redaction francaise de resumes
technologiques.

**Historique du choix** :

1. **Phase 1 (abandonne)** : `facebook/bart-large-cnn` evalue en premier mais
   souffrant de cold-starts >120s sur les HF Inference Providers du plan gratuit.
2. **Phase 2 (abandonne)** : `sshleifer/distilbart-cnn-12-6` (version distillee
   de BART) evaluee comme solution de repli stable. Fonctionnelle, mais produit
   un resume **extractif** (copie des phrases existantes) dans la langue source
   de l'article (anglais pour la majorite de notre corpus). Resultat : un resume
   moins lisible pour l'utilisateur francophone.
3. **Phase 3 (retenue)** : **`Qwen/Qwen2.5-7B-Instruct`** produit un resume
   **abstractif** directement en francais via un prompt system approprie, avec
   une latence equivalente (~2s) et une qualite nettement superieure.

### Justification du choix final

1. **Resume directement en francais** — evite la dependance a un service de
   traduction supplementaire et offre une UX fluide pour les utilisateurs FR.
2. **Qualite abstractive** — le LLM reformule et synthetise, contrairement a
   l'approche extractive de BART qui se limite a selectionner des phrases.
3. **Architecture mono-modele** — le meme modele genere le resume general
   **et** le resume oriente Green IT (voir section 9). Un seul service SaaS,
   une seule dependance, une seule facture d'inference a surveiller.
4. **Licence Apache-2.0** — plus permissive que la Llama Community License,
   pas de formalite d'acces.
5. **Disponibilite immediate** sur les HF Inference Providers du compte gratuit,
   contrairement a Llama-3.2-3B-Instruct (absent des providers) ou Llama-3.1-8B
   (provider payant requis).
6. **Gratuit via le fair use HF** — aucune limite rencontree sur le volume
   anticipe du projet (50-200 articles/jour).
7. **RGPD-compliant** — HF pratique la non-retention de donnees, serveurs UE.

### Plan d'evolution

| Phase | Service | Justification |
|-------|---------|---------------|
| Dev & Pre-prod | HF Serverless API (gratuit) | Prototypage rapide, zero config |
| Production v1 | HF Serverless API (gratuit) | Suffisant pour 200 articles/jour |
| Production v2 (optionnel) | Self-hosting sur GPU AMD | Latence reduite, zero dependance externe |

### Alternatives de secours

En cas de limitation du free tier Hugging Face ou d'indisponibilite Qwen :
1. **Mistral AI API** (Mistral-Small) — alternative LLM instructif, compatible
   chat_completion, heberge en France, quota gratuit limite.
2. **Meta Llama-3.1-8B-Instruct** via Together AI ou Fireworks AI (paliers
   gratuits limites, providers non actives par defaut).
3. **Self-hosting Qwen2.5-3B-Instruct** sur le GPU AMD 7900 XTX via ROCm
   (version plus legere, elimine toute dependance externe).

---

## 9. Architecture mono-modele : deux resumes via un seul LLM instructif

### Besoin fonctionnel

Deux types de resumes sont attendus sur chaque article analyse :

1. **Resume general** (toujours genere) : synthese neutre, informative, en
   francais, pour donner une vue d'ensemble rapide de l'article.
2. **Resume oriente "aspects ecologiques"** (uniquement si l'article est
   classifie Green IT) : extraction ciblee des elements de durabilite,
   d'efficacite energetique, de sobriete numerique, etc.

### Pourquoi un seul modele pour les deux usages

L'architecture initiale prevoyait **deux modeles distincts** : un modele de
summarization extractif (BART) pour le resume general, et un LLM instructif
(Qwen) pour le resume ecologique. Les crash tests ont revele deux problemes :

1. **Incoherence linguistique** : le resume general etait en anglais (langue
   source), le resume ecologique en francais. Cela creait un rendu visuel
   deroutant pour l'utilisateur francophone.
2. **Incoherence qualitative** : le resume ecologique "specialise" etait
   paradoxalement **mieux redige** que le resume "general", car le LLM
   instructif genere du texte abstractif de qualite superieure au simple
   extractif de BART.

**Decision architecturale** : unifier sur **`Qwen/Qwen2.5-7B-Instruct`** pour
les deux resumes, avec deux prompts systeme distincts. Cette approche apporte
la coherence linguistique (tout en francais), la coherence qualitative (meme
niveau de redaction), et simplifie l'infrastructure (un seul service SaaS,
un seul token, un seul quota a surveiller).

### Services evalues pour le LLM instructif

Plusieurs modeles instructifs ont ete testes via l'API HF Serverless avant
de retenir celui disponible sur les providers actifs de notre compte :

| Service | Type | Cout | RGPD | Disponible via HF Providers | Retenu |
|---------|------|------|------|-----------------------------|--------|
| HF Serverless API + **Qwen/Qwen2.5-7B-Instruct** | LLM instructif | Gratuit (fair use) | Conforme | ✅ Oui | ✅ |
| HF Serverless API + Llama-3.2-3B-Instruct | LLM instructif | Gratuit | Conforme | ❌ `not deployed by any Inference Provider` | Non |
| HF Serverless API + Mistral-7B-Instruct-v0.3 | LLM instructif | Gratuit | Conforme | ❌ Provider non active | Non |
| HF Serverless API + zephyr-7b-beta | LLM instructif | Gratuit | Conforme | ❌ Provider non active | Non |
| HF Serverless API + Llama-3.1-8B-Instruct | LLM instructif | Payant selon provider | Conforme | ✅ (Together AI, payant) | Non |
| OpenAI GPT-4o-mini | LLM proprietaire | Payant | Serveurs US | N/A | Non |
| Mistral AI API (Mistral Small) | LLM FR | Gratuit limite | Conforme FR | N/A | Non (eviter multi-fournisseurs) |

### Justification du choix Qwen2.5-7B-Instruct

1. **Disponibilite immediate** : seul LLM instructif de la shortlist reellement
   utilisable via les HF Inference Providers actifs sur un compte gratuit
   standard (Llama-3.2-Instruct, Mistral-Instruct, Zephyr, Gemma → tous
   renvoient `model_not_supported` ou ne sont deployes par aucun provider).
2. **Licence Apache-2.0** (Alibaba) — aucune formalite d'acces, contrairement
   a Llama (Community License qui exige l'approbation Meta pour chaque depot).
3. **Qualite multilingue superieure** : Qwen2.5-7B-Instruct obtient des scores
   superieurs a Mistral-7B et Llama-3-8B sur les benchmarks multilingues,
   notamment en francais. Ideal pour resumer des articles anglais en sortie
   francaise.
4. **Support officiel du `chat_completion`** sur HF Serverless, format
   standard OpenAI-compatible.
5. **Gratuit via le fair use HF** — aucune limite rencontree sur le volume
   anticipe (50-200 articles/jour, 2 resumes par article max).
6. **Coherence architecturale** : un seul fournisseur SaaS IA (HF), un seul
   modele, une seule integration a maintenir.

### Architecture mise en place

```
Article analyse
    |
    +-- Classification Green IT (Qwen3-4B + LoRA local, bloc E3)
    |       |
    |       +-- est_green_it = True ?
    |               |
    |               +-- OUI -- Appel parallele (asyncio.gather) :
    |               |           - summarize_text()          -> Qwen (prompt general)
    |               |           - summarize_green_aspects() -> Qwen (prompt Green IT)
    |               |
    |               +-- NON -- Appel unique :
    |                           - summarize_text()          -> Qwen (prompt general)
```

- Les deux resumes sont persistes dans `articles.resume` et `articles.resume_ecologique`.
- Le resume ecologique est affiche dans un bloc distinct (vert) sur le
  Dashboard et la page detail d'article.
- Cout supplementaire negligeable (un appel LLM court par article Green IT,
  soit ~0.4% du volume total vu la rarete des Green IT dans le corpus).

### Prompt systeme

Defini dans `src/greentech/ai/services/summarizer.py` :

> "Tu es un analyste specialise en Green IT et en informatique eco-responsable.
> Ta mission est de lire un article technologique identifie comme Green IT
> et d'en extraire les aspects ecologiques les plus saillants, en restant
> concis, factuel et fidele au texte source."

Le prompt utilisateur demande un resume de 3-5 phrases (80-120 mots) centre
exclusivement sur les aspects ecologiques, avec instruction explicite de ne
pas inventer ce qui n'est pas dans l'article source.

### Parametres d'inference

- **max_tokens** : 250 (equivalent ~150-180 mots)
- **temperature** : 0.3 (faible creativite, extraction factuelle)
- **Modele configurable** via `HUGGINGFACE_MODEL_GREEN_SUMMARIZER` dans `.env`

### Impact sur la competence C7

Ce second service conforte la validation de C7 :
- Demarche de veille et de benchmark appliquee deux fois (BART vs alternatifs,
  puis Llama-Instruct vs alternatifs).
- Double usage du SaaS HF dans le cadre du meme fournisseur (reduction de
  la complexite d'integration et du risque fournisseur).
- Preuve de la capacite a enrichir une architecture existante avec un nouveau
  service IA tout en respectant les contraintes initiales (cout zero, RGPD,
  Green IT).

---

## 10. Resilience : fallback local Qwen (GPU AMD ROCm) apres epuisement du quota HF

### 10.1 Probleme rencontre

Lors de la premiere execution du pipeline de classification hybride sur
l'ensemble du corpus (892 candidats), la limite mensuelle du plan gratuit
Hugging Face Inference Providers a ete atteinte apres environ 300 requetes.
Les appels suivants ont tous echoue avec un code HTTP 402
("Payment Required", "You have depleted your monthly included credits").

Le pipeline de classification et de resume, qui ne reposait jusqu'alors que
sur le SaaS HF, se retrouvait incapable de terminer le traitement sans :

- attendre le debut du mois suivant (reset automatique du quota),
- ou souscrire a l'offre PRO (9 USD/mois, 20x plus de credits inclus).

### 10.2 Solution retenue : fallback local automatique

L'architecture a ete etendue pour supporter une bascule **transparente**
vers une execution locale du **meme modele** (`Qwen/Qwen2.5-7B-Instruct`)
sur le GPU AMD Radeon RX 7900 XTX via ROCm 7.2. Le modele etant identique
entre HF et local, aucune divergence de qualite ne peut etre introduite
par la bascule.

### 10.3 Mecanisme de detection et de bascule

Un module central `llm_dispatcher` encapsule la strategie :

1. Etat de session `_hf_quota_exhausted` initialement `False`.
2. Chaque appel passe d'abord par `AsyncInferenceClient` contre l'API HF.
3. Sur exception HTTP 402 (detectee par le code status ou par le pattern
   `"Payment Required"`), l'etat passe a `True` et la requete en cours
   est re-tentee immediatement sur le backend local.
4. Tous les appels suivants basculent directement sur le local, sans
   solliciter HF inutilement.
5. A chaque nouveau processus Python, l'etat est reset : on retente HF
   (utile quand le quota est recharge entre-temps).

### 10.4 Architecture comparee

| Critere | HF Serverless API | Local ROCm (fallback) |
|---------|-------------------|----------------------|
| Modele | `Qwen/Qwen2.5-7B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` (identique) |
| Cout | Gratuit (fair use mensuel) | Nul (GPU deja possede) |
| Quota | Mensuel, reset debut de mois | Illimite |
| Latence | ~2 s / article | ~2-5 s / article (selon contexte) |
| Disponibilite | Soumise a la dispo de l'API | Depend uniquement du PC local |
| RGPD | Serveurs HF (UE disponibles) | 100% local, aucune donnee sortante |
| Empreinte carbone | Mutualisee HF, neutralite revendiquee | Mesurable par CodeCarbon, ~15 Wh / article |
| Materiel requis | Aucun | GPU 16+ Go VRAM |

### 10.5 Impact sur la competence C7 et sur la resilience

Cette evolution renforce la validation de C7 :

- **Demarche de benchmark defensif** : anticipation de la limite d'un
  service SaaS et choix documente d'une alternative self-hosted equivalente.
- **Continuite de service** : absence d'interruption pour l'utilisateur
  final lorsque le quota est epuise ; les articles continuent d'etre
  classifies et resumes sans alerte manuelle.
- **Maitrise de plusieurs modes de deploiement** : meme modele execute en
  SaaS ou en self-hosting, choix dynamique selon le contexte.
- **Economie green IT en bonus** : quand HF est disponible, on profite de
  la mutualisation ; quand HF est indisponible, on evite le cout carbone
  d'un autre fournisseur en utilisant du materiel deja possede.

### 10.6 Variables d'environnement

- `HUGGINGFACE_MODEL_LOCAL_FALLBACK` : identifiant du modele local
  (defaut : `Qwen/Qwen2.5-7B-Instruct`, meme que HF pour continuite).

### 10.7 Traces en base

Le champ `articles.modele_classification` distingue les decisions prises
par le LLM judge mais ne distingue pas le backend (HF vs local) : la
valeur reste `keyword_filter+qwen_llm_judge` dans les deux cas, puisque
le modele rendu est identique. Les logs Loguru du dispatcher conservent
l'historique du basculement pour audit.

