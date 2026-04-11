# Checklist de Fin d'Étape - Référence Rapide

> **À exécuter SYSTÉMATIQUEMENT à la fin de chaque ÉTAPE**
> **Durée estimée** : 5-10 minutes

---

## ✅ Checklist Complète (9 Actions)

### Action 1 : Ouvrir le Guide
- [ ] Ouvrir `docs/GESTION_KANBAN_GITHUB.md`
- [ ] Aller à la section de l'ÉTAPE terminée (ex: "ISSUE #2 : ÉTAPE 2")

### Action 2 : Mettre à Jour l'Issue GitHub
- [ ] Ouvrir l'issue sur GitHub Projects
- [ ] Cocher ✅ TOUTES les sous-tâches de l'issue

### Action 3 : Modifier le Statut de l'Issue
- [ ] Cliquer sur "Edit" dans l'issue
- [ ] Modifier la section **"📊 Statut"** :
  ```
  Remplacer : 🚧 **EN COURS** - Démarré le YYYY-MM-DD
  Par       : ✅ **TERMINÉ** - Complété le YYYY-MM-DD
  ```
- [ ] Sauvegarder

### Action 4 : Mettre à Jour les Labels
- [ ] Retirer le label : `en-cours`
- [ ] Ajouter le label : `terminé`

### Action 5 : Déplacer et Fermer l'Issue
- [ ] Glisser-déposer l'issue vers la colonne **"Done"**
- [ ] Cliquer sur **"Close issue"** (bouton en bas de l'issue)

### Action 6 : Mettre à Jour PLAN_ETAPES.md
- [ ] Ouvrir `docs/PLAN_ETAPES.md`
- [ ] Trouver la section de l'ÉTAPE terminée
- [ ] Cocher **toutes** les cases `- [ ]` → `- [x]` de cette étape

### Action 7 : Mettre à Jour CHECKLIST_SUIVI.md
- [ ] Ouvrir `docs/CHECKLIST_SUIVI.md`
- [ ] Identifier les compétences validées par cette étape (voir liste ci-dessous)
- [ ] Cocher les cases correspondantes

### Action 8 : Préparer l'Étape Suivante
- [ ] Dans GitHub Projects, déplacer l'issue de l'ÉTAPE suivante de **"Backlog"** vers **"Ready"**
- [ ] Ouvrir l'issue et modifier le statut :
  ```
  Remplacer : ⏳ **EN ATTENTE** (après ÉTAPE N)
  Par       : 🚧 **PRÊTE À DÉMARRER**
  ```

### Action 9 : Commit (Optionnel mais recommandé)
- [ ] Si modifications de fichiers documentation : faire un commit
  ```bash
  git add docs/
  git commit -m "docs: mise à jour étape N terminée"
  git push
  ```

---

## 📚 Correspondance Étapes ↔ Compétences

### ÉTAPE 1 : Installation & Configuration
**Compétences** : Infrastructure (pas de compétences spécifiques C1-C21)
**Cases à cocher** : Toute la section 1.1 à 1.10 dans PLAN_ETAPES.md

### ÉTAPE 2 : Data Factory (Bloc E1)
**Compétences validées** :
- **C1** : Automatiser l'extraction de données
- **C2** : Développer des requêtes SQL d'extraction
- **C3** : Agrégation et nettoyage des données
- **C4** : Création de la base de données (et RGPD)
- **C5** : Développer une API de mise à disposition (REST)

**Cases à cocher** :
- `CHECKLIST_SUIVI.md` : Toutes les cases de A1 (C1, C2, C3) et A2 (C4, C5)
- `PLAN_ETAPES.md` : Toute la section 2.1 à 2.5

### ÉTAPE 3 : Intelligence Artificielle (Blocs E2 & E3)
**Compétences validées** :
- **C6** : Veille technique et réglementaire
- **C7** : Identifier des services IA (Benchmark)
- **C8** : Paramétrer un service IA
- **C11** : Monitorer le modèle IA
- **C12** : Tests automatisés du modèle
- **C13** : Chaîne de livraison continue (CI/CD pour IA)

**Cases à cocher** :
- `CHECKLIST_SUIVI.md` : A3 (C6, C7, C8) et A5 (C11, C12, C13)
- `PLAN_ETAPES.md` : Toute la section 3.1 à 3.5

### ÉTAPE 4 : Backend & API (Blocs E1 & E4)
**Compétences validées** :
- **C5** : Développer une API de mise à disposition (REST) - Complément ÉTAPE 2
- **C9** : Développer une API exposant un modèle IA
- **C10** : Intégrer l'API IA dans une application
- **C14** : Analyser le besoin
- **C15** : Concevoir le cadre technique
- **C17** : Développer composants et interfaces
- **C18** : Automatiser les tests (CI)

**Cases à cocher** :
- `CHECKLIST_SUIVI.md` : A4 (C9, C10), A6 (C14, C15), A7 (C17), A8 (C18)
- `PLAN_ETAPES.md` : Toute la section 4.1 à 4.6

### ÉTAPE 5 : Frontend (Bloc E4)
**Compétences validées** :
- **C10** : Intégrer l'API IA dans une application - Complément
- **C14** : Analyser le besoin - Complément (User Stories + Wireframes)
- **C15** : Concevoir le cadre technique - Complément (POC)
- **C17** : Développer composants et interfaces - Complément (UI)
- **C18** : Automatiser les tests (CI) - Complément (accessibilité)

**Cases à cocher** :
- `CHECKLIST_SUIVI.md` : Compléments de A4, A6, A7, A8 si pas déjà cochés
- `PLAN_ETAPES.md` : Toute la section 5.1 à 5.5

### ÉTAPE 6 : DevOps & Maintenance (Blocs E4 & E5)
**Compétences validées** :
- **C18** : Automatiser les tests (CI) - Finalisation
- **C19** : Livraison continue (CD)
- **C20** : Surveiller l'application (Monitoring)
- **C21** : Résoudre les incidents techniques

**Cases à cocher** :
- `CHECKLIST_SUIVI.md` : A8 (C18, C19), A9 (C20, C21)
- `PLAN_ETAPES.md` : Toute la section 6.1 à 6.5

---

## 🔄 Exemple Complet : Fin de l'ÉTAPE 2

### Dans GitHub Projects
1. Ouvrir issue #2 "ÉTAPE 2 : Data Factory"
2. Cocher toutes les sous-tâches (2.1 à 2.9)
3. Éditer description : `✅ **TERMINÉ** - Complété le 2026-02-15`
4. Labels : `-en-cours` `+terminé`
5. Déplacer vers "Done" + Fermer l'issue

### Dans docs/PLAN_ETAPES.md
Cocher toutes les cases de la section "ÉTAPE 2" :
```markdown
- [x] Rédiger les spécifications techniques de collecte
- [x] Réaliser le MCD/MLD avec Looping
- [x] Créer la base greentech_db
...
```

### Dans docs/CHECKLIST_SUIVI.md
Cocher les compétences C1 à C5 :
```markdown
#### C1. Automatiser l'extraction de données
- [x] Identifier les contraintes techniques
- [x] Rédiger les spécifications techniques
...

#### C5. Développer une API de mise à disposition
- [x] Rédiger les spécifications techniques de l'API
- [x] Configurer les accès aux données
...
```

### Préparer ÉTAPE 3
1. Déplacer issue #3 de "Backlog" vers "Ready"
2. Modifier statut : `🚧 **PRÊTE À DÉMARRER**`

---

## ⏱️ Fréquence de Mise à Jour

| Moment | Action | Fichier |
|--------|--------|---------|
| **Fin de session quotidienne** | Cocher sous-tâches terminées | Issue GitHub |
| **Fin de sous-section** (ex: 2.1, 2.2) | Cocher cases correspondantes | PLAN_ETAPES.md |
| **FIN D'ÉTAPE COMPLÈTE** | **EXÉCUTER CETTE CHECKLIST** | Tous |

---

## 🎯 Rappel Visuel

```
ÉTAPE N terminée
     ↓
📋 Ouvrir GESTION_KANBAN_GITHUB.md
     ↓
✅ Cocher issue GitHub (toutes sous-tâches)
     ↓
✏️ Modifier statut → "TERMINÉ + Date"
     ↓
🏷️ Labels : -en-cours +terminé
     ↓
📦 Déplacer "Done" + Fermer issue
     ↓
📄 Cocher PLAN_ETAPES.md (ÉTAPE N)
     ↓
📋 Cocher CHECKLIST_SUIVI.md (C1, C2...)
     ↓
🚀 Préparer ÉTAPE N+1 (Ready)
     ↓
✅ C'EST BON !
```

---

**À utiliser à la fin de chaque ÉTAPE !**
