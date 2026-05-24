jaxvacua.freezer
=================

.. currentmodule:: jaxvacua.freezer

.. automodule:: jaxvacua.freezer

When to use this module
-----------------------------------

Use a freezer after constructing a full flux effective theory, when one or
more complex-structure fields are treated as heavy and should be solved away
before scanning the remaining light directions.  ``Freezer`` provides the
base reduced-EFT interface; ``ConifoldFreezer`` is the specialised
implementation for integrating out a conifold modulus with the coniLCS
``z_cf`` equation of motion.

Reduced-EFT workflow
-----------------------------------

1. Wrap the full model with ``ConifoldFreezer`` and specify the conifold
   modulus through ``conifold_index``.
2. Call ``solve_heavy`` to determine the heavy modulus for fixed light
   moduli, axio-dilaton, and fluxes.
3. Use ``reconstruct_full_moduli`` when a full moduli vector is needed again,
   for example before evaluating quantities defined on the original model.
4. Evaluate reduced quantities with ``superpotential``, ``DW_light``,
   ``DW_x_light``, ``dDW_x_light``, ``V_x_light``, ``dV_x_light``, and
   ``ddV_x_light``.

Index conventions
-----------------------------------

``heavy_indices`` names the frozen complex-structure moduli and
``light_indices`` is its complement.  The counters ``n_heavy`` and
``n_light`` refer to complex moduli, while real light-field vectors used by
the ``*_x_light`` methods contain real and imaginary parts of the light
moduli plus the real and imaginary parts of ``tau``.

.. raw:: html
   :file: _static/figures/f9_freezer.html


Freezer base class
-----------------------------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    Freezer


Light-field EFT interface
-----------------------------------

.. autosummary::

    Freezer.solve_heavy
    Freezer.reconstruct_full_moduli
    Freezer.superpotential
    Freezer.DW_light
    Freezer.DW_x_light
    Freezer.dDW_x_light
    Freezer.V_x_light
    Freezer.dV_x_light
    Freezer.ddV_x_light


Conifold freezer
-----------------------------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    ConifoldFreezer


Conifold EOM
-----------------------------------

.. autosummary::

    ConifoldFreezer.solve_heavy
    ConifoldFreezer.reconstruct_full_moduli
