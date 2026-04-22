jaxvacua.database
==================

.. currentmodule:: jaxvacua.database

.. automodule:: jaxvacua.database


Database classes
-----------------------------------

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


Loading models
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    CYDatabase.load
    CYDatabase.load_model
    CYDatabase.load_batch
    CYDatabase.sample


Module-level convenience functions
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    load_tdf_model
    load_cicy_model
    query_models