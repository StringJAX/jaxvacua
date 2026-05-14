# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

    
# -- Project information -----------------------------------------------------
from jaxvacua import __version__

project = "JAXVacua"
copyright = "2024, Andreas Schachner"

release = __version__
version = ".".join(release.split(".")[:2])


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_autodoc_typehints",
    "sphinx_togglebutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "myst_nb",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

source_suffix = [".rst", ".ipynb", ".md"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

pygments_style = None

autodoc_default_flags = ["members"]
autosummary_generate = True
napolean_use_rtype = False

napoleon_custom_sections = [('Returns', 'params_style')]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/AndreasSchachner/JAXVacua_public",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "logo_only": True,
}

# ADD LOGO HERE!!!
#html_logo = "_static/jaxvacua.svg"
#html_favicon = "_static/jaxvacua.ico"
#html_logo = "_static/jaxvacua_new.svg"
#html_favicon = "_static/jaxvacua_new.ico"
#html_logo = "_static/jaxvacua_release.svg"
#html_favicon = "_static/jaxvacua_release.ico"
#html_logo = "_static/jaxvacua_release_v22.svg"
#html_favicon = "_static/jaxvacua_release_v2.ico"
html_logo = "_static/jaxvacua_release_v2.png"
html_favicon = "_static/jaxvacua_release_v2.png"
#html_logo = "_static/jaxvacua_release_v2_Kopie.svg"
#html_favicon = "_static/jaxvacua_release_v2_Kopie.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options myst -------------------------------------------------
#nb_execution_mode = "force"
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath"]
myst_dmath_double_inline = True
nb_execution_allow_errors = False
nb_merge_streams = True
nb_execution_timeout = 120

# -- Custom CSS / MathJax for raw-HTML figures --------------------
# Shared stylesheet for the workflow / module-graph figures embedded
# via raw HTML in `intro/index.md` and the API rst pages.
html_css_files = ["css/jaxvacua-figures.css"]

# Enable MathJax 3 to scan for inline $...$ in raw HTML blocks. Sphinx's
# default mathjax setup only renders math wrapped in :math: nodes; raw
# HTML carrying $...$ (as the workflow figure does) needs this hook.
mathjax3_config = {
    "tex": {"inlineMath": [["$", "$"], ["\\(", "\\)"]]},
    "svg": {"fontCache": "global"},
}

# Removes module and file name in title
add_module_names = False

toc_object_entries_show_parents = "hide"

