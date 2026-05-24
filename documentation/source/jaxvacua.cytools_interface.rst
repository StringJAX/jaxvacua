jaxvacua.cytools_interface
============================

.. currentmodule:: jaxvacua.cytools_interface

.. automodule:: jaxvacua.cytools_interface

When to use this bridge
-----------------------------------

Use this module when the starting point is a CYTools polytope,
triangulation, Calabi-Yau object, cone, or facet and the next step is a
JAXVacua model-data object.  If the workflow already has a saved
``lcs_tree`` or bundled model data, this bridge is usually not needed.

Data extracted
-----------------------------------

The initialisation helpers collect the geometry needed by the LCS and
coniLCS pipelines: Hodge numbers, intersection numbers, second Chern data,
Mori/Kähler cone information, triangulation and polytope identifiers, and
optionally instanton data or conifold candidates when those are available
from the CYTools object.

Typical route
-----------------------------------

1. Start from a CYTools geometry object.
2. Use ``cytools_model_data_init`` to build the core model-data dictionary.
3. Use ``cytools_instanton_data_init`` when instanton data should be attached
   to that dictionary.
4. Pass the result to ``lcs_tree.from_dict`` or to a higher-level model
   constructor.

.. raw:: html
   :file: _static/figures/f10_cytools_interface.html

CYTools interface
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    cytools_model_data_init
    cytools_instanton_data_init
    compute_intersection_numbers_coo
    remove_zeros


Geometry adapters
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    eft_to_poly
    eft_to_cy
    eft_to_coninop
    eft_to_cone
    eft_to_facet
