jaxvacua.flux_eft
==================

.. currentmodule:: jaxvacua.flux_eft

.. automodule:: jaxvacua.flux_eft

EFT flux class
-----------------------------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    FluxEFT


Superpotential and F-terms
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    FluxEFT.superpotential
    FluxEFT.superpotential_gauge_invariant
    FluxEFT.DW
    FluxEFT.DW_z
    FluxEFT.DW_tau
    FluxEFT.DW_x
    FluxEFT.DcDW
    FluxEFT.DDW
    FluxEFT.canonical_fterms


Scalar potential
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    FluxEFT.scalar_potential
    FluxEFT.V
    FluxEFT.V_x
    FluxEFT.ddV
    FluxEFT.ddV_x
    FluxEFT.mass_matrix
    FluxEFT.hessian


ISD condition
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    FluxEFT.ISD_condition
    FluxEFT.ISD_matrix
    FluxEFT.projection_fluxes


Flux utilities
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    FluxEFT.map_to_FD_tau
    FluxEFT.tadpole
    FluxEFT.flux_to_pfv
    FluxEFT.pfv_to_flux
    FluxEFT.pfv_to_moduli
