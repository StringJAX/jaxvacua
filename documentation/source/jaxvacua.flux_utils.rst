jaxvacua.flux_utils
=====================

.. currentmodule:: jaxvacua.flux_utils

.. automodule:: jaxvacua.flux_utils

Role in solver workflows
-----------------------------------

This module collects stateless helpers used by flux EFTs and vacuum finders.
The functions convert between integral flux vectors and PFV variables, derive
PFV-level moduli, and post-process candidate solutions after a numerical
search.

PFV and flux conversion workflow
-----------------------------------

``flux_to_pfv`` extracts the PFV variables from a full flux vector,
``pfv_to_flux`` reconstructs compatible flux data, and ``pfv_to_moduli``
computes the associated moduli at PFV level.  These helpers are useful when a
search alternates between a full flux description and the reduced PFV
description used by specialised conifold or ISD-inspired pipelines.

Solution post-processing
-----------------------------------

``dedup_key`` builds stable keys for identifying repeated candidates,
``classify_solution`` labels the outcome of a solver run, ``is_physical``
applies the physicality checks used by the finder, and ``map_to_fd`` maps
solutions to a chosen fundamental domain before comparison or storage.

PFV algebra
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    flux_to_pfv
    pfv_to_flux
    pfv_to_moduli


Solution processing
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    dedup_key
    classify_solution
    is_physical
    map_to_fd
