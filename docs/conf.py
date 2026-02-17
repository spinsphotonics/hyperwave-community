# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'Hyperwave Community'
copyright = '2025, SPINS Photonics'
author = 'SPINS Photonics'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Master document
master_doc = 'index'

# Global TOC depth
html_theme_options_globaltoc_depth = 3

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# PyData theme options
html_theme_options = {
    "logo": {
        "image_light": "_static/spins_logo.png",
        "image_dark": "_static/spins_logo.png",
        "text": "SPINS Photonics",
    },
    "github_url": "https://github.com/spinsphotonics/hyperwave-community",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "SPINS Photonics",
            "url": "https://spinsphotonics.com",
            "icon": "fa-solid fa-globe",
        }
    ],
    "show_nav_level": 2,
    "show_toc_level": 3,
    "navigation_depth": 4,
    "secondary_sidebar_items": ["page-toc"],
    "navigation_with_keys": False,
    "show_version_warning_banner": False,
    "primary_sidebar_end": [],
}

# Don't show version in title
html_title = "Hyperwave Community"

# Favicon (browser tab icon)
html_favicon = '_static/spins_logo.png'

# -- Extension configuration -------------------------------------------------

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Autosummary settings
autosummary_generate = True

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
}


# Suppress specific warnings that come from docstring formatting
# and removed/renamed functions
suppress_warnings = [
    'autodoc.import_object',
    'duplicate.object.description',
]
# -- Custom configuration ----------------------------------------------------

# Custom CSS and JS files
html_css_files = [
    'custom.css',
]
html_js_files = [
    'sortable.js',
    'gpu_calculator.js',
]
