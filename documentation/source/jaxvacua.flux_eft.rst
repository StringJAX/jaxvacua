jaxvacua.flux_eft
==================

.. currentmodule:: jaxvacua.flux_eft

.. automodule:: jaxvacua.flux_eft

Computational graph
-------------------

This layer assembles the integer flux background into the GVW
superpotential, its covariant F-term derivatives, the no-scale
scalar potential, and the D3-tadpole contribution. The defining
relations are

.. math::
   :label: eq:jaxvacua-flux-eft-01

   \begin{aligned}
       W_{\text{GVW}}(z, \tau;\, \text{flux})
           &= \int_X G_3 \wedge \Omega
            = \bigl(F - \tau H\bigr) \cdot \Pi(z), \\[2pt]
       D_I W &= \partial_I W + (\partial_I K)\, W,
           \quad I = (z^i,\, \tau), \\[2pt]
       V_{\rm ns} &= e^{K}\!\left(
              g^{i\bar\jmath}\, D_i W\, \overline{D_j W}
              + (\operatorname{Im}\tau)^{-2} |D_\tau W|^2
           \right), \\[2pt]
       N_{\text{flux}} &= f^T \Sigma\, h,
   \end{aligned}

with :math:`G_3 = F_3 - \tau H_3` the complex three-form flux and
:math:`\Sigma` the symplectic intersection form on
:math:`H^3(X, \mathbb{Z})`.  The full supergravity
:math:`-3|W|^2` contribution can be included by setting
``noscale=False``.

The diagram below splits the inputs into two visually distinct
groups: those *inherited* from the upstream period and complex-structure
layers (light grey, solid border) and the *external* user-supplied
flux integers and axio-dilaton (white, dashed border). Public
outputs of the layer are highlighted in orange.

.. raw:: html
   :file: _static/figures/f5_flux_eft.html


EFT flux class
-----------------------------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    FluxEFT


Superpotential and F-terms
-----------------------------------

.. autosummary::

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

    FluxEFT.ISD_condition
    FluxEFT.ISD_matrix
    FluxEFT.projection_fluxes


Flux utilities
-----------------------------------

.. autosummary::

    FluxEFT.map_to_fd_tau
    FluxEFT.tadpole
    FluxEFT.flux_to_pfv
    FluxEFT.pfv_to_flux
    FluxEFT.pfv_to_moduli
