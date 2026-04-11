# Dossier de Veille Technique - GreenTech Intelligence

> **Objectif** : Centraliser la veille technologique hebdomadaire et mensuelle pour valider la compétence **C6 : Veille technique et réglementaire** (Bloc E2 - A3)

---

## 📂 Organisation des Fichiers

### Structure Recommandée

```
docs/veille/
├── README.md                          # Ce fichier
├── TEMPLATE_synthese_hebdomadaire.md  # Template à copier pour chaque semaine
├── 2026-02-10_synthese.md             # Semaine du 3 au 9 février 2026
├── 2026-02-17_synthese.md             # Semaine du 10 au 16 février 2026
├── 2026-02-24_synthese.md             # Semaine du 17 au 23 février 2026
├── 2026-03-03_synthese.md             # Semaine du 24 février au 2 mars 2026
├── 2026-02_rapport_mensuel.md         # Rapport mensuel de février 2026
├── 2026-03_rapport_mensuel.md         # Rapport mensuel de mars 2026
└── ...
```

### Convention de Nommage

- **Synthèses hebdomadaires** : `YYYY-MM-DD_synthese.md` (date du lundi)
- **Rapports mensuels** : `YYYY-MM_rapport_mensuel.md`

---

## 🔄 Routine Hebdomadaire

### Quand : Tous les lundis à 11:00 AM

1. **Préparer la recherche** (5 min)
   - Ouvrir Perplexity Pro
   - Copier le prompt depuis `docs/ACTIONS_MANUELLES_ETAPE1.md` section 2.4
   - Remplacer les dates par celles de la semaine précédente

2. **Lancer la recherche** (2-5 min)
   - Coller le prompt dans Perplexity Pro
   - Activer le mode "Deep Research"
   - Attendre la génération complète

3. **Sauvegarder la synthèse** (5 min)
   - Copier `TEMPLATE_synthese_hebdomadaire.md` → `YYYY-MM-DD_synthese.md`
   - Coller le contenu généré par Perplexity
   - Ajuster la mise en forme si nécessaire
   - Compléter la section "📊 Statistiques de Veille"

4. **Exploiter les résultats** (15-30 min)
   - Lire attentivement la section "💡 Opportunités d'Implémentation"
   - Identifier les actions à court terme
   - Créer des issues GitHub si pertinent
   - Marquer les articles Inoreader correspondants avec une étoile

**Durée totale** : 30-45 minutes

---

## 📅 Routine Mensuelle

### Quand : Dernier lundi du mois (après la synthèse hebdomadaire)

1. **Compiler les 4 synthèses hebdomadaires**
   - Relire les 4 synthèses du mois
   - Identifier les tendances récurrentes
   - Extraire les faits marquants les plus impactants

2. **Rédiger le rapport mensuel**
   - Structure (voir template ci-dessous) :
     - Vue d'ensemble du mois
     - Top 10 des actualités
     - Analyses approfondies (3-5 sujets)
     - Bilan des actions entreprises
     - Axes d'amélioration pour le mois suivant

3. **Mettre à jour le backlog**
   - Transformer les opportunités identifiées en issues GitHub
   - Prioriser les actions pour le mois suivant

**Durée totale** : 1-2 heures

---

## 📝 Template Rapport Mensuel

```markdown
# Rapport Mensuel de Veille - [MOIS ANNÉE]

> **Période** : Du [DATE DÉBUT] au [DATE FIN]
> **Projet** : GreenTech Intelligence

---

## 📊 Vue d'Ensemble du Mois

[Résumé global des tendances du mois : quels ont été les sujets dominants ? Quelles évolutions majeures ?]

### Statistiques
- **Synthèses hebdomadaires réalisées** : 4
- **Articles Inoreader consultés** : [Total]
- **Recherches Perplexity** : 4
- **Issues GitHub créées** : [Nombre]
- **Bibliothèques ajoutées au projet** : [Nombre]

---

## 🏆 Top 10 des Actualités du Mois

1. **[Titre]** - [Date] - [Source]
   - [Résumé 1 phrase]
   - Impact : 🔴 Majeur / 🟡 Modéré / 🟢 Mineur

2. **[Titre]** - [Date] - [Source]
   - [Résumé 1 phrase]
   - Impact : 🔴 Majeur / 🟡 Modéré / 🟢 Mineur

[...]

10. **[Titre]** - [Date] - [Source]
    - [Résumé 1 phrase]
    - Impact : 🔴 Majeur / 🟡 Modéré / 🟢 Mineur

---

## 🔬 Analyses Approfondies

### Sujet 1 : [Thématique principale du mois]
[Analyse détaillée avec implications pour le projet]

### Sujet 2 : [Thématique secondaire]
[Analyse détaillée avec implications pour le projet]

### Sujet 3 : [Thématique tertiaire]
[Analyse détaillée avec implications pour le projet]

---

## ✅ Bilan des Actions Entreprises

### Actions Réalisées
- [Action 1 issue de la veille] - Statut : ✅ Terminé / 🚧 En cours / ⏳ En attente
- [Action 2 issue de la veille] - Statut : ✅ Terminé / 🚧 En cours / ⏳ En attente

### Intégrations Concrètes au Projet
- **[Bibliothèque/Technique intégrée]** : [Module où elle a été utilisée]
- **[Bibliothèque/Technique intégrée]** : [Module où elle a été utilisée]

---

## 🎯 Axes d'Amélioration pour le Mois Suivant

### Thématiques à Approfondir
- [Thématique 1]
- [Thématique 2]
- [Thématique 3]

### Actions Prioritaires
- [ ] [Action prioritaire 1]
- [ ] [Action prioritaire 2]
- [ ] [Action prioritaire 3]

---

## 📚 Ressources Clés du Mois

### Articles de Référence
- [Article 1](URL)
- [Article 2](URL)

### Repositories GitHub
- [Repo 1](URL)
- [Repo 2](URL)

### Documentations
- [Doc 1](URL)
- [Doc 2](URL)

---

**Prochain rapport mensuel** : [DATE]
```

---

## ✅ Validation Compétence C6

### Critères de Validation

Pour valider la compétence **C6 : Veille technique et réglementaire**, le jury attend :

1. **📅 Régularité** : Synthèses hebdomadaires datées sur minimum 3 mois
2. **📊 Structuration** : Organisation claire (thématiques, sources, analyses)
3. **🔍 Qualité** : Sources fiables, analyses pertinentes, applicabilité au projet
4. **💡 Exploitation** : Preuves d'intégration des découvertes dans le projet
5. **📝 Communication** : Capacité à synthétiser et présenter les résultats

### Livrables pour la Soutenance

- **Dossier `docs/veille/`** avec toutes les synthèses hebdomadaires et mensuelles
- **Section dédiée** dans le rapport de projet expliquant la démarche de veille
- **Exemples concrets** d'éléments découverts en veille et intégrés au projet
- **Dashboard GitHub Projects** montrant les issues créées suite à la veille (label `veille`)

---

## 🛠️ Outils Utilisés

| Outil | Rôle | Fréquence |
|-------|------|-----------|
| **Inoreader** | Agrégation flux RSS (9-10 sources) | Quotidien (10 min) |
| **Perplexity Pro** | Deep research hebdomadaire | Hebdomadaire (30-45 min) |
| **GitHub Projects** | Gestion backlog issu de la veille | Hebdomadaire |
| **Markdown** | Rédaction synthèses | Hebdomadaire |

---

## 🔗 Liens Utiles

- [Prompt Perplexity Pro](PROMPT_PERPLEXITY.md)
- [GitHub Projects Kanban](../GESTION_KANBAN_GITHUB.md)

