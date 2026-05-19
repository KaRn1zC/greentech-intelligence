"""Configuration Sphinx pour la documentation GreenTech Intelligence."""

project = "GreenTech Intelligence"
author = "Arnaud KaRn1zC BOY"
release = "1.0.0"

# Extensions
extensions = [
    "myst_parser",              # Support Markdown (.md) en plus de RST
    "sphinx.ext.autodoc",       # Docstrings Python (utilisé en interne par autoapi)
    "sphinx.ext.napoleon",      # Support Google-style docstrings (FR)
    "sphinx.ext.viewcode",      # Liens vers le code source dans la doc
    "sphinx.ext.intersphinx",   # Liens vers docs externes (Python, etc.)
    "autoapi.extension",        # Génération auto de la doc API à partir de src/
]

# Formats de fichiers sources
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# MyST-Parser config (Markdown avance)
myst_enable_extensions = [
    "colon_fence",      # ::: directives
    "deflist",          # Listes de definitions
    "fieldlist",        # Listes de champs
    "tasklist",         # Cases a cocher [x]
]
myst_heading_anchors = 3

# Theme Furo (moderne, responsive, dark mode)
html_theme = "furo"
html_title = "GreenTech Intelligence"
html_theme_options = {
    "light_css_variables": {
        # Couleur de marque utilisee pour headings et accents visuels uniquement.
        # Le ratio de contraste n'a pas besoin de respecter 4.5:1 car non utilisee
        # pour du texte courant.
        "color-brand-primary": "#16a34a",
        # Couleur utilisee pour les liens du contenu : green-700 Tailwind
        # (ratio 5.67:1 sur fond #f8f9fb) pour respecter WCAG 2.1 AA.
        "color-brand-content": "#15803d",
    },
    "dark_css_variables": {
        # En mode sombre, le vert clair passe le contraste sur fond tres fonce.
        "color-brand-primary": "#4ade80",
        "color-brand-content": "#86efac",
    },
}

# Fichiers statiques (CSS/JS custom pour corrections d'accessibilite)
html_static_path = ["_static"]
html_css_files = ["a11y-overrides.css"]
html_js_files = ["a11y-fixes.js"]

# Langue
language = "fr"

# Exclure certains patterns
# - Mémoire/ : dossier local des rapports RNCP (gitignored, ne doit pas être publié)
# - superpowers/ : specs internes de tooling local
# - **/CLAUDE.md : fichiers d'instructions persistantes locales
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "Mémoire",
    "Mémoire/**",
    "superpowers",
    "superpowers/**",
    "**/CLAUDE.md",
]

# Autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Sphinx-AutoAPI : génération automatique de la doc API à partir des docstrings
# du code source. Parse via AST (n'importe pas le code) pour éviter les
# dépendances lourdes (PyTorch, Transformers, ROCm) au moment du build.
autoapi_type = "python"
autoapi_dirs = ["../src/greentech"]
autoapi_root = "api"
autoapi_template_dir = None
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    # "imported-members" volontairement omis : déclenche des duplicate object
    # description sur les dataclasses dont les attributs sont déjà listés dans
    # la section "Attributes:" de la docstring Google-style (Napoleon).
]
autoapi_python_class_content = "both"   # docstring de la classe + de __init__
autoapi_member_order = "bysource"
autoapi_keep_files = False               # rebuild propre à chaque fois
autoapi_add_toctree_entry = False        # on insère l'entrée manuellement dans index.md

# Napoleon : générer les Attributes: avec :ivar: (non indexé) pour éviter
# les doublons object-description sur les dataclasses.
napoleon_use_ivar = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Le champ `type` apparaît à la fois sur `Source` (colonne SQLAlchemy) et
# `SourceStatsItem` (schéma Pydantic). Les deux noms sont légitimes dans
# leur contexte métier respectif ; Sphinx ne peut pas désambigüer la
# cross-référence et émet un avertissement pour chaque occurrence.
# On supprime cette catégorie spécifique pour conserver un build clean.
suppress_warnings = ["ref.python"]

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
