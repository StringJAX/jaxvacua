"""Sphinx configuration for the JAXVacua documentation.

Purpose
-------
Configure paths, project metadata, Sphinx extensions, notebook handling and
HTML output options for the documentation build.

Main public API
---------------
- Module-level Sphinx configuration variables such as ``project``,
  ``extensions``, ``html_theme`` and ``myst_enable_extensions``.

Design notes
------------
This file is executed by Sphinx, so imports should stay minimal and
documentation-specific.  Package imports are limited to metadata needed for
the rendered documentation version.
"""

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
import re
import sys
from pathlib import Path
from sphinx.ext import autosummary as _sphinx_autosummary
sys.path.insert(0, os.path.abspath('../../'))

    
# -- Project information -----------------------------------------------------
def _read_package_version() -> str:
    """Read the package version without importing the full JAXVacua stack."""
    init_py = Path(__file__).resolve().parents[2] / "jaxvacua" / "__init__.py"
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", init_py.read_text(), re.M)
    if match is None:
        return "0.0.0"
    return match.group(1)


project = "JAXVacua"
copyright = "2024, Andreas Schachner"

release = _read_package_version()
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
exclude_patterns = ["notebooks/_template.md"]
suppress_warnings = [
    # ``cytools`` annotations are optional external types.  Keep the rendered
    # annotations, but do not fail docs builds when Sphinx cannot import the
    # package namespace while resolving forward references.
    "sphinx_autodoc_typehints.forward_reference",
]

pygments_style = None

# Module pages are curated indexes.  Class pages opt into ``:members:``
# explicitly through ``custom-class-template.rst`` so each object has one
# canonical indexed definition.
autodoc_default_options = {}
autosummary_generate = True
napoleon_use_rtype = False

napoleon_custom_sections = [('Returns', 'params_style')]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/AndreasSchachner/JAXVacua_public",
    "repository_branch": "main",
    "path_to_docs": "documentation/source",
    "use_repository_button": True,
    "use_edit_page_button": True,
}

html_logo = "_static/jaxvacua_release_v2.png"
html_favicon = "_static/jaxvacua_release_v2.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options myst -------------------------------------------------
#nb_execution_mode = "force"
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 4
myst_dmath_double_inline = True
nb_execution_allow_errors = False
nb_merge_streams = True
nb_execution_timeout = 120

# Number every displayed equation in the rendered documentation.  Explicit
# labels should still be used for equations that are referenced from prose.
math_number_all = True

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


def _strip_leading_description_marker(lines):
    """Remove the project's leading Markdown-style description marker."""
    lines = list(lines)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        marker = "**Description:**"
        if stripped == marker:
            del lines[idx]
        elif stripped.startswith(marker):
            indent = line[: len(line) - len(line.lstrip())]
            lines[idx] = indent + stripped[len(marker):].lstrip()
        break
    return lines


def _extract_project_summary(doc, settings):
    """Extract autosummary text without parsing project docstring markers."""
    first_stanza = []
    content_started = False
    for line in _strip_leading_description_marker(doc):
        stripped = line.strip()
        if not content_started:
            if not stripped:
                continue
            content_started = True

        if (
            not stripped
            or stripped.startswith("```")
            or stripped in {"Args:", "Arguments:", "Returns:", "Raises:", "Example:", "Examples:"}
            or stripped.startswith(".. ")
        ):
            break
        first_stanza.append(stripped)

    if not first_stanza:
        return ""

    summary = " ".join(first_stanza).strip()
    return re.sub(r"::$", ".", summary)


_sphinx_autosummary.extract_summary = _extract_project_summary


def _normalise_project_docstring(app, what, name, obj, options, lines):
    """Adapt the project's Markdown-flavoured docstrings for Sphinx/ReST."""
    lines[:] = _strip_leading_description_marker(lines)

    normalised = []
    in_fence = False
    fence_indent = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence:
                in_fence = False
                normalised.append("")
            else:
                language = stripped[3:].strip() or "text"
                fence_indent = line[: len(line) - len(line.lstrip())]
                normalised.append(f"{fence_indent}.. code-block:: {language}")
                normalised.append("")
                in_fence = True
            continue

        if in_fence:
            normalised.append(f"{fence_indent}    {line}")
        else:
            normalised.append(line)

    lines[:] = normalised


def setup(app):
    app.connect("autodoc-process-docstring", _normalise_project_docstring)
