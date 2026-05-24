jaxvacua.sampling
===================

.. currentmodule:: jaxvacua.sampling

.. automodule:: jaxvacua.sampling

When to use this module
-----------------------------------

Use ``data_sampler`` when a workflow needs reproducible batches of fluxes,
axions, dilaton values, complex-structure moduli, or solver initial guesses.
It is usually the preparation layer before calling
``jaxvacua.flux_vacua_finder.FluxVacuaFinder``, but it is also useful on its
own for probing cone geometry and checking how filtering cuts affect a sample.

Common workflow
-----------------------------------

1. Prepare cone-compatible real moduli with ``find_interior_points``,
   ``sample_ray``, ``sample_rays``, ``sample_interior_point``, and
   ``rescale_points``.
2. Convert these real samples into moduli data with ``get_moduli`` and apply
   geometric or instanton-control cuts with ``filter_by_instantons``,
   ``filter_by_km``, ``filter_points``, or ``filter_moduli``.
3. Combine moduli with axion, dilaton, and flux batches from ``get_axions``,
   ``get_axion``, ``get_dilaton``, ``get_complex_tau``, and ``get_fluxes``.
4. Build solver seeds with ``initial_guesses``.  Use
   ``ISD_sampling`` and ``initial_guesses_ISD`` when the search should start
   from ISD- or PFV-informed data.

Choosing an entry point
-----------------------------------

* ``get_moduli`` samples geometric moduli only.
* ``get_fluxes`` samples integral flux batches only.
* ``initial_guesses`` assembles general flux/moduli/tau starting points for
  numerical vacuum searches.
* ``initial_guesses_ISD`` assembles starting points from the specialised ISD
  sampling pipeline.
* ``filter_moduli`` post-processes existing moduli samples without drawing a
  new batch.

.. raw:: html
   :file: _static/figures/f8_sampling.html

Sampling class
-----------------------------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    data_sampler


Moduli sampling
-----------------------------------

.. autosummary::

    data_sampler.get_axion
    data_sampler.get_axions
    data_sampler.get_complex_moduli
    data_sampler.get_complex_tau
    data_sampler.get_dilaton
    data_sampler.get_moduli
    data_sampler.sample_sphere
    data_sampler.sample_ray
    data_sampler.sample_rays
    data_sampler.sample_interior_point
    data_sampler.find_interior_points
    data_sampler.filter_points
    data_sampler.filter_moduli
    data_sampler.rescale_points


Flux sampling
-----------------------------------

.. autosummary::

    data_sampler.get_fluxes
    data_sampler.ISD_sampling


Initial guesses
-----------------------------------

.. autosummary::

    data_sampler.initial_guesses
    data_sampler.initial_guesses_ISD
