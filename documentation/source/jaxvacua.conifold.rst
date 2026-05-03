jaxvacua.conifold
=================

.. currentmodule:: jaxvacua.conifold

.. automodule:: jaxvacua.conifold

Subpackage layout
-----------------

The conifold subsystem is organised into four focused modules.  This
umbrella page collects the public surface; each submodule has its own
dedicated page.

.. toctree::
    :maxdepth: 1

    jaxvacua.conifold.conifold_utils
    jaxvacua.conifold.coni
    jaxvacua.conifold.coniLCS_prepotential
    jaxvacua.conifold.zcf_solver


``_ConifoldGated`` — gating descriptor
---------------------------------------

The ``_ConifoldGated`` descriptor surfaces a method only when the
instance's ``limit`` lies in the coniLCS family
(``coniLCS`` / ``coniLCS_series`` / ``coniLCS_bulk``); otherwise it raises
``AttributeError`` so ``hasattr()`` returns ``False``.  Used by
``periods.py`` / ``css.py`` / ``flux_eft.py`` to attach the conifold
methods only when meaningful.

.. autoclass:: _ConifoldGated
    :members:


Method-attachment lists
-----------------------

Single source of truth for which conifold methods are attached to which
class; consumed by the ``setattr`` blocks at the bottom of
``periods.py`` / ``css.py`` / ``flux_eft.py``.

* ``_PERIODS_METHODS`` — methods attached to :class:`jaxvacua.periods.periods`
  (per-period prepotential family + ``delete_coni_index``).
* ``_CSS_METHODS`` — methods attached to :class:`jaxvacua.css.css`
  (per-modulus prepotential + ``dK_cf_bulk``).
* ``_FLUXEFT_METHODS`` — methods attached to
  :class:`jaxvacua.flux_eft.FluxEFT` (z_cf solver + bulk-EFT building blocks
  + ``conifold_fluxes``).

Adding a new attached method is a one-place edit: define the function in
the appropriate submodule, then append its name to the relevant list in
``conifold/__init__.py``.  No edit to the consumer module is required.


Public re-exports
-----------------

Every public symbol from the four submodules is re-exported here, so user
code can write ``from jaxvacua.conifold import find_conifolds, Conifold,
compute_zcf, …`` without reaching into the submodule namespaces.


See also
--------

* General-purpose lattice helpers — ``extended_euclidean``,
  ``orthogonal_lattice`` — live in :doc:`jaxvacua.util` (they have non-
  conifold callers and are re-exported here only for convenience).
* :doc:`jaxvacua.freezer` — :class:`ConifoldFreezer` consumes ``compute_zcf``
  and ``zcf_handling`` from this subpackage to provide the light-field EFT
  interface (``DW_x_light``, ``dDW_x_light``).