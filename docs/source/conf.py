# -*- coding: utf-8 -*-

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import sphinx_materialdesign_theme
import csharpy


# -- Project information -----------------------------------------------------

project = 'Custom Extensions to ML.net'
copyright = '2018'
author = 'Xavier Dupré'
version = '0.4.0'
release = version


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    #,
    'matplotlib.sphinxext.plot_directive',
    'jupyter_sphinx.embed_widgets',
    "nbsphinx",
    #
    'pyquickhelper.sphinxext.sphinx_runpython_extension',
    'pyquickhelper.sphinxext.sphinx_faqref_extension',
    'pyquickhelper.sphinxext.sphinx_epkg_extension',
    'pyquickhelper.sphinxext.sphinx_exref_extension',
    'pyquickhelper.sphinxext.sphinx_collapse_extension',
    #,
    'sphinx_mlext',
]

exclude_patterns = []
source_suffix = '.rst'
source_encoding = 'utf-8'
language = None
master_doc = 'index'
pygments_style = 'sphinx'
templates_path = ['_templates']

# -- Shortcuts ---------------------------------------------------------------

owner = "xadupre"

epkg_dictionary = {
    'DataFrame': 'https://github.com/%s/machinelearningext/blob/master/machinelearningext/DataManipulation/DataFrame.cs' % owner,
    'C#': 'https://en.wikipedia.org/wiki/C_Sharp_(programming_language)',
    'Iris': 'http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html',
    'Microsoft': 'https://www.microsoft.com/',
    'ML.net': 'https://github.com/dotnet/machinelearning',
    'Python': 'https://www.python.org/',
    'R': 'https://www.r-project.org/',
}

# -- Options for HTML output -------------------------------------------------

html_output_encoding = 'utf-8'
html_theme = 'sphinx_materialdesign_theme'
html_theme_options = {}
html_static_path = ['_static']
html_theme_path = [sphinx_materialdesign_theme.get_path()]

# html_sidebars = {}
