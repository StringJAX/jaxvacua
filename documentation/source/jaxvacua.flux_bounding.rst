jaxvacua.flux_bounding
========================

.. currentmodule:: jaxvacua.flux_bounding

.. automodule:: jaxvacua.flux_bounding

Bounded fluxes class
-----------------------------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    bounded_fluxes


Bounding box computation
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    bounded_fluxes.compute_bounding_box
    bounded_fluxes.get_h_box
    bounded_fluxes.compute_evs
    bounded_fluxes.compute_evs_vmap
    bounded_fluxes.update_global
    bounded_fluxes.update_evs
    bounded_fluxes.update_local


Flux enumeration
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    bounded_fluxes.enumerate_fluxes
    bounded_fluxes.sample_bounded_fluxes
    bounded_fluxes.get_h_candidates


Batch flux checking
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    bounded_fluxes.check_bounds_batch
    bounded_fluxes.compute_tadpole_batch
    bounded_fluxes.newton_refine_batch
    bounded_fluxes.in_patch_batch


Flux bounds
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    bounded_fluxes.bound_h1_local
    bounded_fluxes.bound_h1_global
    bounded_fluxes.bound_h2_local
    bounded_fluxes.bound_h2_global
    bounded_fluxes.bound_f1_local
    bounded_fluxes.bound_f1_global
    bounded_fluxes.bound_f2_local
    bounded_fluxes.bound_f2_global
    bounded_fluxes.bound_s_local
    bounded_fluxes.bound_s_global
    bounded_fluxes.bound_h_local
    bounded_fluxes.bound_h_global
    bounded_fluxes.bound_f_local
    bounded_fluxes.bound_f_global


Flux utilities
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    bounded_fluxes.get_nflux
    bounded_fluxes.get_fh
    bounded_fluxes.get_flux_split
    bounded_fluxes.get_subvector
    bounded_fluxes.compute_norm
    bounded_fluxes.check_bounds
    bounded_fluxes.check_bounds_flat
