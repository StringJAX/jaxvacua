jaxvacua.cy_io
==================

.. currentmodule:: jaxvacua.cy_io

.. automodule:: jaxvacua.cy_io


Database classes
-----------------------------------

Pure-I/O classes for reading the HuggingFace-hosted ``cy-database``.  These
classes have **no jaxvacua dependencies** and are intended to be extractable
into a standalone PyPI package later.  For model construction (``lcs_tree`` /
``FluxVacuaFinder``) and vacua persistence, see
:doc:`jaxvacua.lcs_database` and :doc:`jaxvacua.vacua_writer`.

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    CYDatabase
    TDFDatabase
    CICYDatabase


Discovery
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    CYDatabase.info
    CYDatabase.query
    CYDatabase.query_conifolds


Module-level convenience functions
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    load_catalog
    query_models