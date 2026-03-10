# Benchmark des Services IA de Resume Automatique

> **Projet** : GreenTech Intelligence
> **Redige par** : KaRn1zC - 2026-03-10
> **Competence validee** : C7 - Identifier des services IA (Benchmark)

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

**Selection** : ✅ Retenue comme solution principale.

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

### Justification Green IT

Le choix de **BART-large-CNN via Hugging Face** est le plus coherent avec la philosophie Green IT du projet :

1. **Modele 18x plus leger** que les alternatives generatives (406M vs 7-35B parametres)
2. **Concu specifiquement pour la summarization** — pas besoin d'un LLM generique surdimensionne
3. **Inference mutualisee** — le serveur Hugging Face partage les ressources entre utilisateurs
4. **Possibilite de self-hosting** — deploiement local ulterieur sur le GPU AMD 7900 XTX pour eliminer totalement le transfert reseau
5. **Mesurable** — CodeCarbon peut mesurer l'impact en self-hosting

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
| Input max | ~1024 tokens (BART), texte tronque automatiquement |
| Output | JSON avec `summary_text` (30-150 mots) |

### Integration dans la stack existante

```python
from huggingface_hub import AsyncInferenceClient

client = AsyncInferenceClient(
    model="facebook/bart-large-cnn",
    token=settings.huggingface_token,
)
result = await client.summarization(text)
```

- Compatible avec FastAPI (async natif)
- Resultat stocke dans la colonne `resume` de la table `articles`
- Module developpe : `src/greentech/ai/services/summarizer.py`

---

## 8. Conclusions et Preconisations

### Solution retenue

**Hugging Face Serverless Inference API** avec le modele **facebook/bart-large-cnn**.

### Justification

1. **Score le plus eleve** (4.80/5) sur l'ensemble des criteres ponderes
2. **Gratuit** — aucun cout pour un projet etudiant
3. **RGPD-compliant** — pas de retention de donnees, serveurs UE
4. **Green IT exemplaire** — modele leger, specifiquement concu pour la tache
5. **Zero vendor lock-in** — modele open source, self-hostable
6. **Integration simple** — SDK Python officiel, compatible async

### Plan d'evolution

| Phase | Service | Justification |
|-------|---------|---------------|
| Dev & Pre-prod | HF Serverless API (gratuit) | Prototypage rapide, zero config |
| Production v1 | HF Serverless API (gratuit) | Suffisant pour 200 articles/jour |
| Production v2 (optionnel) | Self-hosting sur GPU AMD | Latence reduite, zero dependance externe |

### Alternatives de secours

En cas de limitation du free tier Hugging Face :
1. **Mistral AI** (meilleur score apres HF, RGPD-compliant, heberge en France)
2. **Self-hosting BART** sur le GPU AMD 7900 XTX via ROCm (elimine toute dependance)

---

**Document redige par KaRn1zC - 2026-03-10**
