jaxvacua.flux_vacua_finder
============================

.. currentmodule:: jaxvacua.flux_vacua_finder

.. automodule:: jaxvacua.flux_vacua_finder

Computational graph
-------------------

The vacuum-search loop wraps a :class:`jaxvacua.flux_eft.FluxEFT`
instance and runs a Newton iteration on the F-term residual

.. math::
   :label: eq:jaxvacua-flux-vacua-finder-01

   \begin{aligned}
       r_I(z, \bar z, \tau, \bar\tau;\, \mathrm{flux})
           &= D_I W = \partial_I W + (\partial_I K)\, W,
              \quad I = (z^i,\, \tau), \\[2pt]
       \text{converged when } \;
           \lVert r \rVert_\infty &< \texttt{tol}.
   \end{aligned}

For each converged point the layer also computes the bosonic
mass-squared spectrum,

.. math::
   :label: eq:jaxvacua-flux-vacua-finder-02

   m^2 = \operatorname{eig}\!\left(M^2_{IJ}\right),
   \qquad
   M^2_{IJ} = \frac{\partial^2 V}{\partial \phi^I \partial \phi^J}
   \Big|_{D_I W = 0},

via :meth:`jaxvacua.flux_eft.FluxEFT.hessian` and
:meth:`jaxvacua.flux_eft.FluxEFT.mass_matrix`. Inherited input (the
upstream :class:`jaxvacua.flux_eft.FluxEFT` instance) is shown light
grey; the diamond is the convergence filter; orange callouts are the
public outputs of the layer.

.. raw:: html
   :file: _static/figures/f6_flux_vacua_finder.html


Flux vacua finder
-----------------------------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-no-inherited-template.rst

    FluxVacuaFinder


Newton minimisation
-----------------------------------

.. autosummary::

    FluxVacuaFinder.newton_method_flux_vacua
    FluxVacuaFinder.compute_residual


Linearised shifts
-----------------------------------

.. autosummary::

    FluxVacuaFinder.linearised_shifts
    FluxVacuaFinder.linearised_shifts_ISD
    FluxVacuaFinder.linearised_shifts_H
    FluxVacuaFinder.linearised_shifts_F


Flux vacuum sampling
-----------------------------------

.. autosummary::

    FluxVacuaFinder.fterm_solver
    FluxVacuaFinder.sample_SUSY_flux_vacua
    FluxVacuaFinder.sample_SUSY_vacua_from_fluxes
    FluxVacuaFinder.deduplicate_vacua


Critical-point sampling (non-SUSY)
-----------------------------------

The non-SUSY workflow — sampling Gaussian-M-prior fluxes, ISD-completing
them, refining via Newton / optax / scipy, and filtering — lives directly
on :class:`FluxVacuaFinder`.

.. autosummary::

    FluxVacuaFinder.sample_critical_points
    FluxVacuaFinder.run_calibration
    FluxVacuaFinder.calibrate_priors
    FluxVacuaFinder.save_calibration
    FluxVacuaFinder.load_calibration
    FluxVacuaFinder.from_model


Shared finder utilities
-----------------------------------

Thin delegators over the stateless helpers in
:mod:`jaxvacua.flux_utils`.  Useful for post-hoc analysis of any
converged candidate, regardless of which solver produced it.

.. autosummary::

    FluxVacuaFinder.dedup_key
    FluxVacuaFinder.classify_solution
    FluxVacuaFinder.is_physical
    FluxVacuaFinder.to_fd
