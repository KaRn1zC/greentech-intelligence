"""Configuration Sphinx pour la documentation GreenTech Intelligence."""

project = "GreenTech Intelligence"
author = "Arnaud KaRn1zC BOY"
release = "1.0.0"

# Extensions
extensions = [
    "myst_parser",              # Support Markdown (.md) en plus de RST
    "sphinx.ext.autodoc",       # Docstrings Python automatiques
    "sphinx.ext.napoleon",      # Support Google-style docstrings
    "sphinx.ext.viewcode",      # Liens vers le code source
    "sphinx.ext.intersphinx",   # Liens vers docs externes (Python, etc.)
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
        "color-brand-primary": "#16a34a",
        "color-brand-content": "#16a34a",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4ade80",
        "color-brand-content": "#4ade80",
    },
}

# Fichiers statiques
html_static_path = []

# Langue
language = "fr"

# Exclure certains patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
