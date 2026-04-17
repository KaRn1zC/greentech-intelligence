# Registre des Traitements de Données Personnelles

>
> Document de conformité RGPD pour le projet GreenTech Intelligence.
> Conformément à l'article 30 du RGPD, ce registre documente l'ensemble
> des traitements de données personnelles effectués dans le cadre du projet.

---

## 1. Informations générales sur le traitement

### 1.1 Identification du responsable de traitement

| Champ | Valeur |
|-------|--------|
| **Nom du projet** | GreenTech Intelligence |
| **Nature** | Projet académique - Chef d'œuvre Simplon |
| **Responsable de traitement** | KaRn1zC (Étudiant) |
| **Finalité globale** | Plateforme d'analyse et classification automatique d'articles technologiques selon leur pertinence "Green IT" |
| **Type de structure** | Projet éducatif (non commercial) |
| **Date de création** | 2026-02-09 |
| **Dernière mise à jour** | 2026-02-11 |

---

## 2. Inventaire des traitements de données

### 2.1 TRAITEMENT N°1 : Collecte et analyse d'articles technologiques

#### 2.1.1 Description du traitement

**Finalité** : Collecter, nettoyer, stocker et analyser des articles technologiques publics pour classifier leur pertinence selon les critères du Green IT (informatique éco-responsable).

**Base légale** : Article 6.1.f du RGPD - Intérêt légitime
- Intérêt légitime : Projet éducatif et de recherche sur les pratiques Green IT
- Traitement de données publiques accessibles librement sur Internet

**Catégories de personnes concernées** :
- Auteurs d'articles technologiques
- Chercheurs scientifiques (publications arXiv)
- Journalistes technologiques

**Nature des données collectées** :

| Donnée | Source | Type | Sensible | Collectée | Stockée | Exposée API |
|--------|--------|------|----------|-----------|---------|-------------|
| Nom complet de l'auteur | The Guardian, Dev.to, TechCrunch, arXiv | Identité | ⚠️ Oui | Oui | **Non** (anonymisée) | **Non** |
| Initiales de l'auteur | Transformation automatique | Pseudonyme | Non | Non | Oui | Oui |
| Adresse e-mail | (non collectée) | Contact | ⚠️ Oui | **Non** | **Non** | **Non** |
| Titre de l'article | Toutes sources | Contenu éditorial | Non | Oui | Oui | Oui |
| Contenu de l'article | Toutes sources | Contenu éditorial | Non | Oui | Oui | Oui |
| URL de publication | Toutes sources | Référence publique | Non | Oui | Oui | Oui |
| Date de publication | Toutes sources | Métadonnée | Non | Oui | Oui | Oui |
| Nom de la source | Toutes sources | Métadonnée | Non | Oui | Oui | Oui |

**Durée de conservation** :
- **Données brutes (MinIO `raw-data`)** : 90 jours puis suppression automatique
- **Données nettoyées (PostgreSQL)** : 2 ans maximum, puis archivage ou suppression
- **Données anonymisées (API)** : Conservation indéfinie (plus de données personnelles)

**Destinataires des données** :
- Base de données locale : Accès restreint au responsable de traitement uniquement
- API publique : Données anonymisées uniquement (initiales, pas de noms complets)
- Aucune transmission à des tiers

**Transferts hors UE** : Aucun transfert hors Union Européenne

---

### 2.2 TRAITEMENT N°2 : Gestion des comptes utilisateurs (API)

#### 2.2.1 Description du traitement

**Finalité** : Permettre aux utilisateurs de créer un compte pour accéder aux fonctionnalités avancées de l'API (déclenchement d'analyses IA, historique personnel).

**Base légale** : Article 6.1.b du RGPD - Exécution d'un contrat
- Création de compte nécessaire pour l'utilisation des services

**Catégories de personnes concernées** :
- Utilisateurs de la plateforme web

**Nature des données collectées** :

| Donnée | Type | Obligatoire | Stockée | Durée de conservation |
|--------|------|-------------|---------|------------------------|
| Adresse e-mail | Identité | Oui | Oui (table `users`) | Tant que le compte est actif |
| Mot de passe hashé | Authentification | Oui | Oui (bcrypt/Argon2) | Tant que le compte est actif |
| Date de création du compte | Métadonnée | Oui | Oui | Tant que le compte est actif |
| Statut du compte (actif/inactif) | Métadonnée | Oui | Oui | Tant que le compte est actif |

**Durée de conservation** :
- Tant que le compte est actif
- Suppression définitive à la demande de l'utilisateur
- Suppression automatique après 2 ans d'inactivité (notification préalable par e-mail)

**Destinataires des données** :
- Base de données PostgreSQL : Accès restreint au responsable de traitement
- Aucune transmission à des tiers

**Transferts hors UE** : Aucun transfert hors Union Européenne

---

## 3. Mesures de sécurité mises en œuvre

### 3.1 Sécurité technique

| Mesure | Description | Niveau |
|--------|-------------|--------|
| **Chiffrement des mots de passe** | Hashage bcrypt avec salt (via FastAPI Users) | ✅ Haute |
| **Connexion BDD sécurisée** | PostgreSQL avec authentification par mot de passe | ✅ Moyenne |
| **Variables d'environnement** | Secrets stockés dans `.env` (non versionné sur Git) | ✅ Haute |
| **Accès restreint MinIO** | Buckets privés avec authentification obligatoire | ✅ Haute |
| **API sécurisée** | JWT pour l'authentification, HTTPS en production | ✅ Haute |
| **Validation des entrées** | Pydantic pour prévenir les injections SQL/XSS | ✅ Haute |
| **Logs sécurisés** | Loguru configuré pour ne pas logger les données sensibles | ✅ Moyenne |

### 3.2 Sécurité organisationnelle

| Mesure | Description |
|--------|-------------|
| **Accès restreint** | Un seul administrateur (responsable de traitement) |
| **Environnement de dev** | Données de test uniquement en développement |
| **Backup réguliers** | Sauvegarde automatique de la base de données |
| **Journalisation** | Logs des accès et modifications (audit trail) |
| **Mise à jour régulière** | Dépendances Python et Docker maintenues à jour |

---

## 4. Procédures d'anonymisation et de pseudonymisation

### 4.1 Identification des données personnelles

Les données personnelles identifiées dans les sources de collecte sont :

| Source | Donnée | Champ brut | Nature | Traitement |
|--------|--------|------------|--------|------------|
| The Guardian | Nom de l'auteur | `fields.byline` (string) | Identité | Anonymisation |
| Dev.to | Nom de l'auteur | `user.name` (string) | Identité | Anonymisation |
| TechCrunch | Nom de l'auteur | Extrait du HTML (`a[href*="/author/"]`) | Identité | Anonymisation |
| arXiv | Noms des chercheurs | `authors` (string) | Identité | Anonymisation |

### 4.2 Règles d'anonymisation automatique

#### 4.2.1 Transformation des noms en initiales

**Algorithme appliqué lors du nettoyage Spark** :

```
Entrée : "John Doe"
Sortie : "J.D."

Entrée : "Marie-Claire Dubois"
Sortie : "M.D."

Entrée : "Li Wei Zhang"
Sortie : "L.W.Z."

Entrée : null ou chaîne vide
Sortie : "Auteur anonyme"
```

**Implémentation** : Fonction Python dans le pipeline PySpark (`processors/spark_cleaner.py`)

```python
def anonymiser_auteur(nom_complet: str | None) -> str:
    """Transforme un nom complet en initiales.

    Args:
        nom_complet: Nom complet de l'auteur (ex: "John Doe")

    Returns:
        Initiales formatées (ex: "J.D.")
    """
    if not nom_complet or nom_complet.strip() == "":
        return "Auteur anonyme"

    # Séparation des mots (gestion des tirets et espaces)
    mots = nom_complet.replace("-", " ").split()

    # Extraction de la première lettre de chaque mot
    initiales = [mot[0].upper() for mot in mots if mot]

    # Formatage avec points
    return ".".join(initiales) + "."
```

#### 4.2.2 Suppression des adresses e-mail

**Règle** : Aucune adresse e-mail n'est collectée ni stockée.

Si une adresse e-mail est détectée dans le contenu d'un article :
- Elle est **conservée** dans le contenu (information publique de l'article)
- Elle n'est **jamais extraite** comme champ séparé

### 4.3 Stockage en couches avec accès restreint

Le projet utilise une **architecture en couches** pour limiter l'exposition des données personnelles :

```
┌─────────────────────────────────────────────────────────────────┐
│  COUCHE 1 : Données brutes (MinIO raw-data)                     │
│  ─────────────────────────────────────────────────────────────  │
│  Contenu : JSON/HTML brut avec noms complets possibles          │
│  Accès : Restreint (connexion MinIO requise)                    │
│  Exposition API : NON                                            │
│  Durée : 90 jours maximum                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓ Nettoyage Spark
┌─────────────────────────────────────────────────────────────────┐
│  COUCHE 2 : Données nettoyées (MinIO clean-data + PostgreSQL)   │
│  ─────────────────────────────────────────────────────────────  │
│  Contenu : Données structurées avec initiales uniquement         │
│  Accès : Restreint (connexion BDD requise)                      │
│  Exposition API : OUI (via authentification JWT)                 │
│  Durée : 2 ans maximum                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓ API REST
┌─────────────────────────────────────────────────────────────────┐
│  COUCHE 3 : Données exposées (API publique)                     │
│  ─────────────────────────────────────────────────────────────  │
│  Contenu : Articles avec initiales uniquement                    │
│  Accès : Public (lecture) ou authentifié (écriture)             │
│  Format : JSON via endpoints REST                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Droits des personnes concernées

### 5.1 Information préalable

**Aucune collecte directe** : Les données sont collectées depuis des sources publiques (API, sites web publics). Les personnes concernées sont les auteurs d'articles déjà publiés publiquement.

**Transparence** : Une page "Politique de confidentialité" sera accessible sur l'application web, expliquant :
- La nature des données collectées
- La finalité du traitement
- Les procédures d'anonymisation
- Les droits des personnes

### 5.2 Droits applicables

Les personnes concernées disposent des droits suivants :

| Droit | Description | Procédure |
|-------|-------------|-----------|
| **Droit d'accès** (Art. 15) | Obtenir une copie des données personnelles stockées | Demande par e-mail au responsable de traitement |
| **Droit de rectification** (Art. 16) | Corriger des données inexactes | Demande par e-mail avec justificatif |
| **Droit à l'effacement** (Art. 17) | Supprimer les données (« droit à l'oubli ») | Suppression immédiate sur demande |
| **Droit d'opposition** (Art. 21) | S'opposer au traitement | Suppression des données de la personne |
| **Droit à la limitation** (Art. 18) | Limiter temporairement le traitement | Mise en "quarantaine" des données |

**Contact pour l'exercice des droits** : À définir (e-mail de contact du responsable de traitement)

**Délai de réponse** : Maximum 1 mois à compter de la réception de la demande

### 5.3 Procédure de suppression des données (« Droit à l'oubli »)

Si un auteur demande la suppression de ses données :

1. **Vérification de l'identité** : Demande de justificatif (preuve de l'identité de l'auteur)
2. **Recherche dans la base** : Requête SQL pour identifier tous les articles de cet auteur
3. **Suppression en cascade** :
   - Suppression de l'article dans PostgreSQL (table `articles`)
   - Suppression des logs d'analyse associés (table `analysis_logs`)
   - Suppression des fichiers bruts dans MinIO (si encore présents)
4. **Confirmation** : E-mail de confirmation envoyé à la personne
5. **Délai** : Suppression effective sous 48h maximum

**Requête SQL de suppression** :

```sql
-- Recherche des articles d'un auteur (avant anonymisation)
SELECT id_article, titre, url, auteur
FROM articles
WHERE auteur ILIKE '%Nom Complet%';

-- Suppression (les logs seront supprimés en cascade via ON DELETE CASCADE)
DELETE FROM articles WHERE id_article IN (...);
```

---

## 6. Sous-traitants et tiers

### 6.1 Services tiers utilisés

| Service | Fournisseur | Localisation | Données transmises | Base légale |
|---------|-------------|--------------|-------------------|-------------|
| **The Guardian API** | Guardian News & Media | Royaume-Uni | Mots-clés de recherche uniquement | Clause contractuelle type |
| **Dev.to API** | Forem Inc. | États-Unis | Tags de recherche uniquement | Clause contractuelle type |
| **Hugging Face API** | Hugging Face Inc. | États-Unis | Texte des articles (anonymisé) | Clause contractuelle type |
| **Hébergement** | Render | États-Unis | Données applicatives | Clause contractuelle type |

**Note importante** : Aucune donnée personnelle brute (noms complets) n'est transmise aux services tiers. Seules les données anonymisées sont envoyées.

### 6.2 Garanties de conformité

- Tous les services tiers sont conformes RGPD (Privacy Shield ou clauses contractuelles types)
- Aucune transmission de données vers des pays sans niveau de protection adéquat
- Revue régulière des politiques de confidentialité des services tiers

---

## 7. Analyse d'impact relative à la protection des données (AIPD)

### 7.1 AIPD nécessaire ?

**Non, l'AIPD n'est pas obligatoire** pour ce projet selon l'article 35 du RGPD, car :
- ❌ Pas de traitement à grande échelle de données sensibles
- ❌ Pas de profilage systématique
- ❌ Pas de surveillance systématique de zones accessibles au public
- ❌ Pas de traitement de données de santé, biométriques ou génétiques
- ✅ Traitement de données publiques déjà accessibles en ligne
- ✅ Anonymisation systématique des données personnelles

### 7.2 Risques identifiés et mesures

| Risque | Gravité | Probabilité | Mesure de mitigation |
|--------|---------|-------------|----------------------|
| Accès non autorisé à la BDD | Haute | Faible | Authentification forte, accès restreint |
| Ré-identification des auteurs | Moyenne | Très faible | Transformation en initiales (non réversible) |
| Fuite de données via l'API | Moyenne | Faible | Authentification JWT, validation des entrées |
| Perte de données | Moyenne | Faible | Backups automatiques quotidiens |

---

## 8. Violations de données (Data Breach)

### 8.1 Procédure en cas de violation

En cas de violation de données personnelles (accès non autorisé, perte, destruction) :

1. **Détection** : Monitoring via Grafana/Prometheus (alertes automatiques)
2. **Confinement** : Isolation immédiate du système compromis
3. **Évaluation** : Détermination de la gravité et de l'étendue de la violation
4. **Notification** : Si risque pour les droits et libertés des personnes :
   - Notification à la CNIL sous 72h
   - Information des personnes concernées
5. **Documentation** : Consignation de la violation dans un registre
6. **Correction** : Mise en place de mesures correctives

### 8.2 Contact en cas de violation

**CNIL** : https://www.cnil.fr/fr/notifier-une-violation-de-donnees-personnelles

---

## 9. Conformité et audits

### 9.1 Revue périodique

Ce registre doit être revu et mis à jour :
- ✅ À chaque modification du traitement
- ✅ Tous les 6 mois (revue de routine)
- ✅ En cas de nouvelle réglementation

### 9.2 Checklist de conformité RGPD

| Exigence RGPD | Statut | Commentaire |
|---------------|--------|-------------|
| Registre des traitements (Art. 30) | ✅ | Ce document |
| Base légale identifiée (Art. 6) | ✅ | Intérêt légitime + Exécution contrat |
| Information des personnes (Art. 13) | ✅ | Page "Politique de confidentialité" |
| Droits des personnes (Art. 15-22) | ✅ | Procédures définies |
| Sécurité des données (Art. 32) | ✅ | Mesures techniques et organisationnelles |
| Notification violations (Art. 33) | ✅ | Procédure définie |
| Durées de conservation définies | ✅ | 90 jours (brut) / 2 ans (nettoyé) |
| Minimisation des données (Art. 5) | ✅ | Anonymisation systématique |
| Privacy by Design (Art. 25) | ✅ | Architecture en couches, anonymisation dès la collecte |

---

## 10. Mentions légales et contacts

### 10.1 Responsable de traitement

**Nom** : KaRn1zC
**Qualité** : Étudiant développeur (Projet Chef d'œuvre Simplon)
**Contact** : *(à définir : e-mail de contact)*

### 10.2 Délégué à la protection des données (DPO)

**Non applicable** : Projet académique de taille réduite, pas d'obligation de désigner un DPO.

### 10.3 Autorité de contrôle

**CNIL** (Commission Nationale de l'Informatique et des Libertés)
**Adresse** : 3 Place de Fontenoy, TSA 80715, 75334 Paris Cedex 07
**Site web** : https://www.cnil.fr
**Téléphone** : +33 1 53 73 22 22

---

**Date de dernière mise à jour** : 2026-02-11
**Version** : 1.0
