jaxvacua.css
===============

.. currentmodule:: jaxvacua.css

.. automodule:: jaxvacua.css

Computational graph
-------------------

The complex-structure sector consumes the period vector :math:`\Pi(z)`
and the Kähler potential :math:`K(z, \bar z)` from
:mod:`jaxvacua.periods` and produces the moduli-space metric, the ISD
matrix and the gauge-kinetic matrix:

.. math::
   :nowrap:

   \begin{align*}
       g_{i \bar\jmath}(z, \bar z)
           &= \partial_i \partial_{\bar\jmath}\, K(z, \bar z), \\[2pt]
       M_{AB}(z, \bar z)
           &= e^{K(z, \bar z)}\, \Pi_A(z)\, \overline{\Pi_B(z)}, \\[2pt]
       \mathcal{N}_{IJ}(z, \bar z)
           &= \bar{F}_{IJ}
              + 2 i\, \frac{\bigl(\operatorname{Im} F\bigr)_{IK} X^K\,
                              \bigl(\operatorname{Im} F\bigr)_{JL} X^L}
                              {\bigl(\operatorname{Im} F\bigr)_{KL} X^K X^L}.
   \end{align*}

In the diagram, inherited inputs (light grey, from the upstream
period layer) flow into the layer's three computed objects; the
public outputs (orange) feed downstream into
:class:`jaxvacua.flux_eft.FluxEFT` (via :math:`\mathcal{N}_{IJ}`) and
:mod:`jaxvacua.sampling` (via :math:`M_{AB}`).

.. raw:: html
   :file: _static/figures/f4_css.html


Complex structure sector class
-----------------------------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    css


Coordinate transformations
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    css.moduli_to_periods
    css.periods_to_moduli


Prepotential
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    css.prepot
    css.dF
    css.period_vector
    css.F_LCS_poly
    css.F_inst
    css.F_LCS
    css.F_coniLCS_bulk
    css.F_coniLCS_series


Kähler potential
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    css.mirror_volume
    css.kahler_potential
    css.dK
    css.dK_c
    css.dK_z
    css.dK_cz
    css.dK_tau
    css.dK_ctau


Kähler metric
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    css.kahler_metric
    css.inverse_kahler_metric
    css.inverse_kahler_metric_grad
    css._kahler_metric_general
    css._kahler_metric_block_diagonal
    css.ddK_z_cz
    css.ddK_cz_z
    css.ddK_z_tau
    css.ddK_cz_ctau
    css.ddK_z_ctau
    css.ddK_cz_tau
    css.ddK_tau_ctau
    css.ddK_z_z
    css.ddK_cz_cz
    css.ddK_tau_tau
    css.ddK_ctau_ctau


Gauge kinetic matrix and ISD matrix
--------------------------------------

.. autosummary::
    :toctree: _autosummary

    css.gauge_kinetic_matrix
    css.ISD_matrix
