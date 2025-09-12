# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FDTDX'
copyright = '2025, Yannik Mahlau'
author = 'Yannik Mahlau'
release = '0.4.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    "myst_nb",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
html_favicon = '_static/fdtdx_icon_64.ico'
html_logo = '_static/fdtdx_icon_200.png'

html_theme_options = {
    "repository_url": "https://github.com/ymahlau/fdtdx",
    "repository_branch": "main",  # or "master" if that's your default branch
    "use_repository_button": True,  # This enables the repository button
}

napoleon_google_docstring = True
autosummary_generate = True
# Set to 'separated' to display signature as a method instead of in the class header
# autodoc_class_signature = 'separated'
# autodoc_typehints = 'signature'

# autodoc_default_options = {
#     'exclude-members': '__init__, __new__, __post_init__, __repr__, __eq__, __hash__, __weakref__',
#     'undoc-members': False,  # Don't document members without docstrings
# }

autodoc_default_options = {
    'undoc-members': False,  # Don't document members without docstrings
}


nb_execution_mode = "off"

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

# MathJax configuration (optional, for customization)
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    }
}

