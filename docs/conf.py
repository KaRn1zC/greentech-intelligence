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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
