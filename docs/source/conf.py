# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

project = 'kval'
copyright = '2025, NPI Oceanography'
author = 'NPI Oceanography'
release = '0.3'



sys.path.insert(0, os.path.abspath('../src/kval/'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.coverage',
              'sphinx_copybutton',
              'sphinx.ext.napoleon',
              'sphinx.ext.mathjax',
              'myst_nb']  # For LaTeX support]


myst_enable_extensions = [
    "dollarmath",   # Enables LaTeX math using $...$
]

templates_path = ['_templates']
exclude_patterns = []


# Set the logo
html_logo = "https://raw.githubusercontent.com/npiocean/kval/master/graphics/kval_logo.png"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_title = "kval docs"