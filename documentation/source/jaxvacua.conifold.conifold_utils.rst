jaxvacua.conifold.conifold_utils
================================

.. currentmodule:: jaxvacua.conifold.conifold_utils

.. automodule:: jaxvacua.conifold.conifold_utils

Structural helpers used across the conifold subsystem: lattice / basis
algebra and flux / index manipulation.

The general-purpose number-theoretic helpers ``extended_euclidean`` and
``orthogonal_lattice`` live in :doc:`jaxvacua.util` (they have non-conifold
callers); they are re-exported here for convenience.


Lattice / basis algebra
-----------------------

.. autosummary::
    :toctree: _autosummary

    get_basis_change
    compute_a_matrix
    get_projection


Flux & index helpers (attached to ``periods`` / ``FluxEFT``)
------------------------------------------------------------

.. autosummary::
    :toctree: _autosummary

    conifold_fluxes
    delete_coni_index


Re-exports from :doc:`jaxvacua.util`
------------------------------------

.. autosummary::
    :toctree: _autosummary

    extended_euclidean
    orthogonal_lattice