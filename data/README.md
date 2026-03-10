# Données du projet

Ce dossier contient les datasets versionnés via **DVC** (Data Version Control).

## Fichiers

| Fichier | Description | Versionné par |
|---------|-------------|---------------|
| `articles_a_annoter.csv` | Export brut des articles pour annotation manuelle | DVC |
| `golden_dataset.csv` | Dataset annoté (labels Green IT 0/1) | DVC |

## Workflow

```bash
# Exporter les articles à annoter
uv run python scripts/create_annotation_dataset.py

# Annoter manuellement le CSV (colonne label_green_it : 0 ou 1)
# Renommer en golden_dataset.csv

# Importer les labels en base
uv run python scripts/create_annotation_dataset.py import

# Versionner avec DVC
uv run dvc add data/golden_dataset.csv
git add data/golden_dataset.csv.dvc data/.gitignore
```

Rédigé par KaRn1zC - 2026-03-10
