import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# Project information
project = 'Spacylize'
copyright = '2025, Stephan Baier'
author = 'Stephan Baier'
release = '0.1.0'

# Sphinx extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
]

# Autodoc settings
autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# HTML theme settings
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'titles_only': False,
}

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
