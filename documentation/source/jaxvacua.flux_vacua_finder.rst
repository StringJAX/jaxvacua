jaxvacua.flux_vacua_finder
============================

.. currentmodule:: jaxvacua.flux_vacua_finder

.. automodule:: jaxvacua.flux_vacua_finder

Computational graph
-------------------

The vacuum-search loop wraps a :class:`jaxvacua.flux_eft.FluxEFT`
instance and runs a Newton iteration on the F-term residual

.. math::
   :nowrap:

   \begin{align*}
       r_I(z, \bar z, \tau, \bar\tau;\, \mathrm{flux})
           &= D_I W = \partial_I W + (\partial_I K)\, W,
              \quad I = (z^i,\, \tau), \\[2pt]
       \text{converged when } \;
           \lVert r \rVert_\infty &< \texttt{tol}.
   \end{align*}

For each converged point the layer also computes the bosonic
mass-squared spectrum,

.. math::

   m^2 = \operatorname{eig}\!\left(M^2_{IJ}\right),
   \qquad
   M^2_{IJ} = \frac{\partial^2 V}{\partial \phi^I \partial \phi^J}
   \Big|_{D_I W = 0},

via ``jax.hessian`` on the scalar potential :math:`V`. Inherited
input (the upstream :class:`FluxEFT` instance) is shown light grey;
the diamond is the convergence filter; orange callouts are the
public outputs of the layer.

.. raw:: html
   :file: _static/figures/f6_flux_vacua_finder.html


Flux vacua finder
-----------------------------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    FluxVacuaFinder


Newton minimization
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    FluxVacuaFinder.newton_method_flux_vacua
    FluxVacuaFinder.compute_residual


Linearised shifts
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    FluxVacuaFinder.linearised_shifts
    FluxVacuaFinder.linearised_shifts_ISD
    FluxVacuaFinder.linearised_shifts_H
    FluxVacuaFinder.linearised_shifts_F


Flux vacuum sampling
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    FluxVacuaFinder.fterm_solver
    FluxVacuaFinder.sample_SUSY_flux_vacua
    FluxVacuaFinder.sample_SUSY_vacua_from_fluxes
    FluxVacuaFinder.CheckConstraints
