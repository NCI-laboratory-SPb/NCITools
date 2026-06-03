# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NCITools'
copyright = '2026, Mark Kaplanskiy'
author = 'Mark Kaplanskiy'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))   # чтобы Sphinx видел ваш пакет

extensions = [
    'sphinx.ext.autodoc',    # автодокументация из docstrings
    'sphinx.ext.napoleon',   # поддержка Google/NumPy стиля docstrings
    'sphinx.ext.viewcode',   # ссылки на исходный код
    'sphinx_rtd_theme',
]

html_theme = 'sphinx_rtd_theme'