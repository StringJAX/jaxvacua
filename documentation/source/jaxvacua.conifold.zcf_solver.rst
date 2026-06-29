jaxvacua.conifold.zcf_solver
============================

.. currentmodule:: jaxvacua.conifold.zcf_solver

.. automodule:: jaxvacua.conifold.zcf_solver

The bulk effective theory after integrating out the conifold modulus
:math:`z_{\rm cf}`.  All public functions take ``flux``; private helpers
are tightly coupled to the flux pathway.

Mathematical structure (notes/main.tex eq:Wtilde1Explicit,
eq:zcf_corrected):

.. math::
   :label: eq:jaxvacua-conifold-zcf-solver-01

    \partial_{z_{\rm cf}} W_{\rm coni} =
        \texttt{log\_prefactor} \cdot \ln(-2\pi i\, z_{\rm cf})
        + \texttt{W\_log\_coeff} + \mathcal{O}(z_{\rm cf})

with

.. math::
   :label: eq:jaxvacua-conifold-zcf-solver-02

    \texttt{log\_prefactor} = (M^1 - \tau H^1) \cdot n_{\rm cf} / (2\pi i)\,,

so that

.. math::
   :label: eq:jaxvacua-conifold-zcf-solver-03

    z_{\rm cf} = -\frac{1}{2\pi i}\,
        \exp\!\Bigl(-\bigl(\texttt{W\_log\_coeff}
            \,[+\,\texttt{log\_coeff\_K\_corr}]\bigr)
            /\,\texttt{log\_prefactor}\Bigr).


Building blocks
---------------

.. autosummary::
    :toctree: _autosummary

    W_bulk
    dK_cf_bulk
    log_prefactor


W̃₁ assembly (three independent routes)
---------------------------------------

The closed-form ``manual`` and the autodiff route ``autodiff`` agree
numerically; ``pfv`` is a deliberate racetrack approximation that becomes
exact at the racetrack-stationary point.

.. autosummary::
    :toctree: _autosummary

    W_log_coeff


Kähler correction + final exponentiation
----------------------------------------

.. autosummary::
    :toctree: _autosummary

    log_coeff_K_corr


Top-level dispatchers (attached to ``FluxEFT``)
-----------------------------------------------

.. autosummary::
    :toctree: _autosummary

    compute_zcf
    compute_zcf_x
    zcf_handling