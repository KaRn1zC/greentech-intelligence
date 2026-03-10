# État d'avancement du projet - Session du 2026-03-10

> **Dernière mise à jour** : 2026-03-10
> **Rédigé par** : KaRn1zC

---

## 📍 Où nous en sommes

### ✅ ÉTAPE 1 : Installation & Configuration - **TERMINÉE**
Toutes les dépendances et outils sont installés et configurés.

### 🔄 ÉTAPE 2 : Data Factory & Gestion de Données - **EN COURS**

#### Section 2.1 : Conception & Conformité - **✅ TERMINÉE**
- ✅ Spécifications techniques rédigées (`docs/SPECIFICATIONS_TECHNIQUES.md`)
- ✅ MCD/MLD créé et documenté
- ✅ Script SQL `init.sql` corrigé en nomenclature française (conforme au MCD/MLD)
- ✅ Registre RGPD complet (`docs/REGISTRE_RGPD.md`)
- ✅ Procédures d'anonymisation documentées
- ✅ Architecture stockage documentée (section 9 des specs)

#### Section 2.2 : Infrastructure de Stockage - **✅ TERMINÉE**
- ✅ Déploiement PostgreSQL 15 (Docker) - conteneur healthy, init.sql exécuté
- ✅ Déploiement MinIO (Docker) - conteneur healthy, API S3 opérationnelle
- ✅ Création des 4 buckets (raw-data, clean-data, models, mlflow)
- ✅ Création utilisateur applicatif `greentech_app` (droits restreints)
- ✅ Script de vérification `scripts/verify_infrastructure.py` (3/3 PASS)
- ✅ Package Python `minio` ajouté aux dépendances

#### Section 2.3 : Programmation de la Collecte - **✅ TERMINÉE**
- ✅ Configuration dynamique SQL (table search_config + get_config_from_db)
- ✅ Module API (httpx → NewsData.io → MinIO raw-data)
- ✅ Module Scraping (Scrapy + Playwright → TechCrunch → MinIO raw-data)
- ✅ Module Fichier (arXiv JSON Lines → MinIO raw-data)

#### Section 2.4 : Traitement Big Data & Nettoyage - **✅ TERMINÉE**
- ✅ Session PySpark 4.1.1 configurée avec connecteur S3A/MinIO (Hadoop 3.4.2)
- ✅ Lecture récursive des 3 sources depuis MinIO raw-data via Spark
- ✅ Agrégation des 3 DataFrames en jeu de données unifié
- ✅ Pipeline de nettoyage complet (fonctions Spark SQL natives) :
  - Suppression balises HTML (regexp_replace)
  - Anonymisation auteurs RGPD ("Jean Dupont" → "J.D.")
  - Normalisation dates ISO 8601
  - Suppression entrées corrompues + déduplication par URL
- ✅ Sauvegarde en Parquet dans MinIO clean-data
- ✅ Test end-to-end validé (6 articles → 4 après nettoyage)

#### Section 2.5 : Mise à disposition structurée (SQL) - **✅ TERMINÉE**
- ✅ Script d'ingestion SQL async (`sql_ingester.py`) avec SQLAlchemy 2.0
- ✅ Lecture des Parquet depuis MinIO clean-data via PyArrow
- ✅ Mapping dynamique des sources (source_nom → id_source)
- ✅ Upsert avec ON CONFLICT (url) DO NOTHING (idempotent)
- ✅ Vérification post-ingestion (requêtes SQL de contrôle)
- ✅ Test validé : 8 articles insérés (3 API + 2 Scraping + 3 arXiv)
- ✅ Idempotence confirmée : relance = 0 doublons

---

## 📝 Travail effectué lors de cette session

### 1. Vérification et correction du schéma SQL

**Problème identifié** : Le fichier `scripts/sql/init.sql` utilisait des noms de colonnes en **anglais**, alors que les spécifications MCD/MLD définissaient une nomenclature **française**.

**Action réalisée** : Correction complète du fichier `init.sql` pour aligner tous les noms de tables et colonnes avec les spécifications.

**Fichier modifié** :
- `scripts/sql/init.sql` (réécriture complète avec nomenclature française)

**Exemples de changements** :
- `id` → `id_article`, `id_source`, `id_config`, etc.
- `title` → `titre`
- `content` → `contenu`
- `is_green_it` → `est_green_it`
- `created_at` → `date_creation`
- `updated_at` → `date_modification`
- Tous les index, triggers, vues mis à jour

### 2. Documentation de l'architecture stockage

**Action réalisée** : Ajout d'une section complète (Section 9) dans `docs/SPECIFICATIONS_TECHNIQUES.md` pour expliquer la distinction entre `init.sql` (SQL pur) et `models.py` (SQLAlchemy ORM).

**Fichier modifié** :
- `docs/SPECIFICATIONS_TECHNIQUES.md` (ajout section 9)

**Contenu ajouté** :
- Distinction init.sql vs SQLAlchemy ORM 2.0
- Workflow complet illustré
- Exemples de code SQLAlchemy
- Tableau "Quand utiliser l'un ou l'autre ?"
- Mention d'Alembic pour les migrations

### 3. Création du registre RGPD

**Action réalisée** : Rédaction complète du registre des traitements de données personnelles conforme au RGPD (Article 30).

**Fichier créé** :
- `docs/REGISTRE_RGPD.md` (nouveau fichier, 10 sections)

**Contenu du registre** :
1. Informations générales sur le traitement
2. Inventaire des traitements (2 traitements identifiés)
3. Mesures de sécurité (techniques + organisationnelles)
4. Procédures d'anonymisation (algorithme Python documenté)
5. Droits des personnes (accès, rectification, effacement)
6. Sous-traitants et tiers (NewsData.io, Hugging Face, Render)
7. AIPD (non nécessaire - justification)
8. Violations de données (procédure CNIL)
9. Conformité et audits (checklist RGPD complète)
10. Mentions légales et contacts

**Points clés** :
- Anonymisation systématique : `"John Doe"` → `"J.D."`
- Architecture en 3 couches (raw → clean → API)
- Durées de conservation : 90 jours (brut) / 2 ans (nettoyé)
- Procédure de droit à l'oubli documentée

### 4. Mise à jour des checklists

**Fichiers modifiés** :
- `docs/PLAN_ETAPES.md` (section 2.1 cochée)
- `docs/CHECKLIST_SUIVI.md` (Bloc E1 - C4 : cases RGPD cochées)

---

## 🎯 Prochaine étape à réaliser

### ÉTAPE 2.2 : Infrastructure de Stockage

**Objectif** : Déployer PostgreSQL et MinIO via Docker pour préparer l'environnement de stockage.

**Actions à effectuer** :

#### 1. Créer/mettre à jour `docker-compose.yml`
- Définir le service PostgreSQL 15
- Définir le service MinIO
- Configurer les variables d'environnement
- Monter le script `init.sql` dans PostgreSQL
- Configurer les volumes persistants

#### 2. Déployer PostgreSQL
```bash
docker-compose up -d postgres
```
- Vérifier que le conteneur démarre
- Vérifier que `init.sql` s'exécute correctement
- Se connecter à la base pour vérifier les tables créées

#### 3. Déployer MinIO
```bash
docker-compose up -d minio
```
- Vérifier que le conteneur démarre
- Accéder à l'interface web (localhost:9001)
- Créer les buckets `raw-data` et `clean-data`

#### 4. Vérifications
- Tester la connexion PostgreSQL depuis Python (SQLAlchemy)
- Tester la connexion MinIO (boto3 ou minio-py)
- Documenter les commandes de vérification

---

## 📂 Fichiers créés/modifiés aujourd'hui

| Fichier | Action | Description |
|---------|--------|-------------|
| `scripts/sql/init.sql` | ✏️ Modifié | Réécriture complète avec nomenclature française |
| `docs/SPECIFICATIONS_TECHNIQUES.md` | ✏️ Modifié | Ajout section 9 (Architecture stockage) |
| `docs/REGISTRE_RGPD.md` | ✨ Créé | Registre RGPD complet (10 sections) |
| `docs/PLAN_ETAPES.md` | ✏️ Modifié | Section 2.1 cochée |
| `docs/CHECKLIST_SUIVI.md` | ✏️ Modifié | Bloc E1 - C4 : cases RGPD cochées |
| `docs/ETAT_AVANCEMENT.md` | ✨ Créé | Ce fichier (mémorisation état) |

---

## 💡 Points importants à retenir

### 1. Nomenclature française confirmée
Le projet utilise des noms de colonnes en **français** dans toute la stack :
- SQL : `id_article`, `titre`, `contenu`, `est_green_it`, etc.
- SQLAlchemy : Les modèles devront refléter exactement ces noms
- API : Les schémas Pydantic utiliseront ces noms

### 2. Architecture en 2 couches
- `init.sql` : Initialisation de la structure (une fois)
- `models.py` : Interaction quotidienne depuis Python (SQLAlchemy ORM 2.0 async)

### 3. Conformité RGPD assurée
- Registre complet et professionnel
- Anonymisation automatique (noms → initiales)
- Procédures documentées
- Checklist de conformité OK

### 4. Prêt pour le développement
- Toute la conception est terminée
- Le schéma SQL est conforme
- Les spécifications sont complètes
- On peut maintenant passer à l'infrastructure

---

## 🔧 Commandes utiles pour la reprise

### Vérifier l'état du projet
```bash
cd C:\Users\aboys\Documents\Simplon\Projet_Chef_Oeuvre\greentech-intelligence
git status
```

### Lancer l'environnement
```bash
# Activer l'environnement uv (si nécessaire)
uv sync

# Lancer les services Docker (quand docker-compose.yml sera créé)
docker-compose up -d
```

### Consulter les documents de référence
- `docs/PLAN_ETAPES.md` : Ordre précis de développement
- `docs/CHECKLIST_SUIVI.md` : Suivi des compétences validées
- `docs/SPECIFICATIONS_TECHNIQUES.md` : Référence technique complète
- `docs/REGISTRE_RGPD.md` : Conformité RGPD

---

## 📞 Pour reprendre la session

**Phrase à dire** : "Reprends où on s'est arrêté"

ou

"Je veux continuer l'ÉTAPE 2.2 : Infrastructure de Stockage"

---

**Date de sauvegarde** : 2026-03-10
**Prochaine action** : ÉTAPE 2.3 - Configuration dynamique SQL + Collecte de données
