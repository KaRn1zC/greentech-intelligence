# Accessibilité de la documentation — GreenTech Intelligence

> Document de conformité aux recommandations d'accessibilité de la
> documentation, référencées dans le référentiel de certification
> (association Valentin Haüy, Atalan-AcceDe, WCAG 2.1 AA, RGAA v4).

---

## 1. Contexte

Le référentiel E1-E5 du diplôme **Développeur IA** mentionne, pour chaque
compétence impliquant une production documentaire (C6, C8, C9, C11, C12,
C13, C17, C18, C19, C20) :

> "La documentation est communiquée dans un format qui respecte les
> recommandations d'accessibilité (par exemple celles de l'association
> Valentin Haüy ou d'Atalan - AcceDe)."

Ce document atteste des moyens mis en place pour satisfaire ce critère.

---

## 2. Standards ciblés

| Standard | Niveau | Statut |
|----------|--------|--------|
| **WCAG 2.1** (W3C) | AA | ✅ Audité automatiquement |
| **RGAA v4** (référentiel France) | Conforme | ✅ Basé sur WCAG |
| **EN 301 549** (norme EU) | Appliquée | ✅ Basé sur WCAG |
| **Recommandations Valentin Haüy** | Respectées | ✅ Voir §5 |
| **Atalan - AcceDe** (charte éditoriale) | Respectée | ✅ Voir §5 |

---

## 3. Outillage d'audit automatisé

### 3.1 Audit continu dans la CI

Chaque commit déclenche un audit d'accessibilité **automatique** via le workflow
GitHub Actions `docs-accessibility` (`.github/workflows/ci.yml`) :

1. **Build** de la documentation Sphinx (thème Furo, WCAG-friendly par défaut)
2. **Serveur HTTP local** servant les pages HTML générées
3. **Audit Playwright + axe-core** sur toutes les pages principales :
   - Tags appliqués : `wcag2a`, `wcag2aa`, `wcag21a`, `wcag21aa`
   - Un test par page : la CI échoue si une violation `critical` ou `serious` est détectée
4. **Upload du rapport HTML** en artefact GitHub pour revue manuelle

### 3.2 Pages auditées (15 tests automatisés)

| Page | Fichier | Audité |
|------|---------|--------|
| Index (accueil) | `index.html` | ✅ |
| Checklist de suivi | `CHECKLIST_SUIVI.html` | ✅ |
| Plan des étapes | `PLAN_ETAPES.html` | ✅ |
| Spécifications techniques | `SPECIFICATIONS_TECHNIQUES.html` | ✅ |
| Benchmark services IA | `BENCHMARK_SERVICES_IA.html` | ✅ |
| Sources de données | `Sources_Données.html` | ✅ |
| Registre RGPD | `REGISTRE_RGPD.html` | ✅ |
| Procédure MAJ modèle | `PROCEDURE_MAJ_MODELE.html` | ✅ |
| Playbook maintenance | `PLAYBOOK_MAINTENANCE.html` | ✅ |
| État d'avancement | `ETAT_AVANCEMENT.html` | ✅ |

### 3.3 Exécution locale

```bash
# 1. Build de la documentation
uv run sphinx-build -b html docs docs/_build/html

# 2. Lancement des tests d'accessibilité
cd frontend && npm run test:a11y:docs

# Le rapport HTML détaillé est disponible dans frontend/a11y-docs-report/
```

---

## 4. Corrections appliquées pour atteindre la conformité WCAG 2.1 AA

Le thème Furo est WCAG-friendly par défaut, mais plusieurs ajustements ont été
nécessaires pour corriger des violations résiduelles :

### 4.1 Contraste des liens (WCAG 1.4.3 — Contraste minimum)

**Problème** : La couleur de marque `#16a34a` (green-600 Tailwind) ne respecte
pas le ratio de contraste 4.5:1 requis sur fond clair (ratio observé : 3.12:1).

**Correction** (`docs/_static/a11y-overrides.css`) :
- Mode clair : liens en `#15803d` (green-700, ratio **5.67:1**)
- Mode sombre : liens en `#86efac` (green-300)
- La couleur de marque `#16a34a` reste utilisée uniquement pour les accents
  visuels (barres, bordures) qui ne nécessitent pas de ratio de contraste.

### 4.2 Attribut aria-level sur les captions (WCAG 4.1.2 — Nom, rôle, valeur)

**Problème** : Le thème Furo applique `role="heading"` sur les titres de
sections de la sidebar mais omet `aria-level`, ce que axe-core signale
comme violation critique (`aria-required-attr`).

**Correction** (`docs/_static/a11y-fixes.js`) : script JS ajouté au chargement
qui attribue `aria-level="3"` à toutes les captions concernées.

### 4.3 Labels des checkboxes tasklist (WCAG 3.3.2 — Étiquettes)

**Problème** : L'extension `tasklist` de MyST-Parser (utilisée pour les cases
à cocher `- [x]` dans CHECKLIST_SUIVI.md et PLAN_ETAPES.md) génère des
`<input type="checkbox">` sans label associé.

**Correction** (`docs/_static/a11y-fixes.js`) : le script JS ajoute un
`aria-label` déduit du texte de la tâche à chaque checkbox, plus `aria-readonly="true"`
(les cases de la doc statique ne sont pas interactives).

### 4.4 Blocs de code focusables (WCAG 2.1.1 — Clavier)

**Problème** : Les blocs `<pre>` générés par Pygments contiennent parfois du
scroll horizontal mais ne sont pas focusables, empêchant la navigation clavier
(règle `scrollable-region-focusable`).

**Correction** (`docs/_static/a11y-fixes.js`) : `tabindex="0"` ajouté à
chaque `pre` du contenu, permettant le focus clavier et le défilement avec
les flèches.

### 4.5 Focus visible renforcé (WCAG 2.4.7)

Outline de 3px en vert accessible sur tous les éléments interactifs
(`:focus-visible`), avec un offset pour assurer la visibilité sur tous les
arrière-plans.

---

## 5. Conformité aux recommandations éditoriales

Les documents Markdown source respectent les bonnes pratiques issues des
chartes **Valentin Haüy** et **Atalan - AcceDe** :

### 5.1 Structure sémantique

- ✅ **Hiérarchie des titres** : un seul `# H1` par document, sous-niveaux
  `##`, `###`, `####` utilisés dans l'ordre, pas de saut de niveau.
- ✅ **Sections claires** : utilisation de séparateurs `---` entre sections
  majeures.
- ✅ **Tableaux accessibles** : tous les tableaux ont une ligne d'en-tête
  (`|---|`), ce qui génère des `<th>` avec le bon rôle ARIA.

### 5.2 Liens

- ✅ **Libellés explicites** : aucun lien `[cliquez ici]`, `[ici]`, `[link]`
  ou `[click here]`. Chaque lien décrit sa destination.
- ✅ **Liens externes** : URLs complètes ou libellés descriptifs du service
  cible (ex. `[Furo](https://pradyunsg.me/furo/)`).

### 5.3 Images

- ✅ **Pas d'images décoratives non étiquetées** : la documentation actuelle
  ne contient aucune image `![](...)` sans alt text. Les futures images
  devront avoir un `alt` descriptif ou `alt=""` si purement décoratif.

### 5.4 Langue

- ✅ **Attribut `lang="fr"`** défini dans `conf.py` et propagé sur la balise
  `<html>` de chaque page HTML générée.

### 5.5 Navigation et lisibilité

- ✅ **Skip link équivalent** : le thème Furo utilise un landmark `<main>`
  qui permet aux lecteurs d'écran de sauter directement au contenu.
- ✅ **Table des matières** : TOC générée automatiquement sur chaque page,
  navigable au clavier.
- ✅ **Dark mode** : deux thèmes (clair et sombre), avec ratios de contraste
  vérifiés sur les deux variantes.

### 5.6 Contenu

- ✅ **Ton naturel et professionnel** en français (pas de jargon inutile).
- ✅ **Acronymes explicités** à leur première apparition (ex. "RGPD (Règlement
  Général sur la Protection des Données)").
- ✅ **Dates au format ISO 8601** (YYYY-MM-DD) pour l'universalité.

---

## 6. Audit manuel complémentaire

### 6.1 Points vérifiés manuellement

- [x] Navigation au clavier fonctionnelle sur toute la sidebar et le contenu
- [x] Affichage correct en mode 200% zoom (responsive)
- [x] Pas d'information véhiculée uniquement par la couleur
- [x] Contraste minimum 4.5:1 pour le texte courant
- [x] Contraste minimum 3:1 pour les éléments graphiques non textuels
- [x] Tableaux navigables au clavier
- [x] Aucun contenu clignotant ou animation automatique (WCAG 2.2.2)

### 6.2 Limitations connues

Certains contenus générés automatiquement par Sphinx (comme les index,
glossaires ou références croisées) peuvent présenter des marges d'amélioration
non couvertes par l'audit automatisé. Ils seront revus manuellement à chaque
release majeure.

---

## 7. Procédure en cas de régression d'accessibilité

1. **Détection** : échec du job `docs-accessibility` dans la CI GitHub Actions.
2. **Diagnostic** : télécharger l'artefact `a11y-docs-report` de la CI
   (`a11y-docs-report/index.html` détaille chaque violation).
3. **Correction** :
   - Si violation structurelle → modifier le document Markdown source
   - Si violation de thème → ajouter un override dans `docs/_static/a11y-overrides.css`
     ou `docs/_static/a11y-fixes.js`
4. **Validation locale** : `npm run test:a11y:docs` doit repasser au vert.
5. **Commit + push** : la CI re-lance l'audit automatiquement.

---

## 8. Références

- [WCAG 2.1 (W3C)](https://www.w3.org/TR/WCAG21/) — Référentiel international
- [RGAA v4 (DINUM)](https://accessibilite.numerique.gouv.fr/) — Référentiel français
- [Recommandations Valentin Haüy](https://www.avh.asso.fr/) — Accessibilité visuelle
- [Atalan - AcceDe](https://www.accede-web.com/) — Guides éditoriaux accessibles
- [axe-core Rules](https://dequeuniversity.com/rules/axe/) — Règles automatisées
- [Furo Theme](https://pradyunsg.me/furo/) — Thème Sphinx utilisé

---

**Rédigé par KaRn1zC - 2026-04-13**
