# -*- coding: utf-8 -*-

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import csharpy


# -- Project information -----------------------------------------------------

project = 'Custom Extensions to ML.net'
copyright = '2020'
author = 'Xavier Dupré'
version = '0.8.0'
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
    'pyquickhelper.sphinxext.sphinx_md_builder',
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
html_logo = "project_ico.png"

# -- Shortcuts ---------------------------------------------------------------

owner = "xadupre"

epkg_dictionary = {
    'AVX': 'https://fr.wikipedia.org/wiki/Advanced_Vector_Extensions',
    'C#': 'https://en.wikipedia.org/wiki/C_Sharp_(programming_language)',
    'C++': 'https://en.wikipedia.org/wiki/C%2B%2B',
    'cffi': 'https://cffi.readthedocs.io/en/latest/',
    'DataFrame': 'https://github.com/%s/machinelearningext/blob/master/machinelearningext/DataManipulation/DataFrame.cs' % owner,
    'DBSCAN': 'https://fr.wikipedia.org/wiki/DBSCAN',
    'dotnet/machinelearning': 'https://github.com/dotnet/machinelearning',
    'Google Protobuf': 'https://developers.google.com/protocol-buffers/',
    'IDataView': 'https://github.com/dotnet/machinelearning/blob/master/docs/code/IDataViewDesignPrinciples.md',
    'Iris': 'http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html',
    'LightGBM': 'http://lightgbm.apachecn.org/en/latest/index.html',
    'lightgbm': 'http://lightgbm.apachecn.org/en/latest/index.html',
    'linear regression': 'https://en.wikipedia.org/wiki/Linear_regression',
    'logistic regression': 'https://en.wikipedia.org/wiki/Logistic_regression',
    'Microsoft': 'https://www.microsoft.com/',
    'MIT License': 'https://github.com/dotnet/machinelearning/blob/master/LICENSE',
    'ML.net': 'https://github.com/dotnet/machinelearning',
    'NimbusML': 'https://github.com/Microsoft/NimbusML',
    'nimbusml': 'https://github.com/Microsoft/NimbusML',
    'numba': 'http://numba.pydata.org/',
    'ONNX': 'https://onnx.ai/',
    'onnx': 'https://github.com/onnx/onnx',
    'onnxmltools': 'https://github.com/onnx/onnxmltools',
    'onnxruntime': 'https://docs.microsoft.com/en-us/python/api/overview/azure/onnx/intro?view=azure-onnx-py',
    'onnx ml functions': 'https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md',
    'onnx operators': 'https://github.com/onnx/onnx/blob/master/docs/Operators.md',
    'OPTICS': 'https://fr.wikipedia.org/wiki/OPTICS',
    'PCA': 'https://en.wikipedia.org/wiki/Principal_component_analysis',
    'Python': 'https://www.python.org/',
    'python': 'https://www.python.org/',
    'R': 'https://www.r-project.org/',
    'scikit-learn': 'http://scikit-learn.org/',
    'xadupre/machinelearningext': 'https://github.com/xadupre/machinelearningext',
}

# -- Options for HTML output -------------------------------------------------

# import sphinx_materialdesign_theme as theme_ext
# html_theme = 'sphinx_materialdesign_theme'
# html_theme_path = [theme_ext.get_path()]

import alabaster
html_theme_path = [alabaster.get_path()]
html_theme = 'alabaster'

html_output_encoding = 'utf-8'
html_theme_options = {}
html_static_path = ['_static']

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
    ]
}

from recommonmark.parser import CommonMarkParser
source_parsers = {'.md': CommonMarkParser}
source_suffix = ['.rst', '.md']

