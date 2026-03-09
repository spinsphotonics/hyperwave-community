# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "Hyperwave Community"
copyright = "2025-2026, SPINS Photonics"
author = "SPINS Photonics"
release = "0.1.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
master_doc = "index"

# Prevent autodoc entries from appearing in sidebar/toctree
toc_object_entries = False

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "Hyperwave Community"
html_favicon = "_static/spins_logo.png"
html_css_files = ["custom.css"]

html_sidebars = {
    "**": ["sidebar-nav-bs"],
}

html_theme_options = {
    "logo": {
        "image_light": "_static/spins_logo.png",
        "image_dark": "_static/spins_logo.png",
        "text": "Hyperwave Community",
    },
    "github_url": "https://github.com/spinsphotonics/hyperwave-community",
    "collapse_navigation": False,
    "show_nav_level": 2,
    "navigation_depth": 3,
    "show_toc_level": 2,
    "navbar_center": [],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "SPINS Photonics",
            "url": "https://spinsphotonics.com",
            "icon": "fa-solid fa-globe",
        },
    ],
    "secondary_sidebar_items": ["page-toc"],
    "show_version_warning_banner": False,
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": False,
    "exclude-members": "__weakref__",
}

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

suppress_warnings = [
    "autodoc.import_object",
]
