jaxvacua.periods
=================

.. currentmodule:: jaxvacua.periods

.. automodule:: jaxvacua.periods

Computational graph
-------------------

This layer assembles the prepotential, the period vector and the
Kähler potential from the topological data carried on ``lcs_tree``.
At large complex structure (LCS) the prepotential decomposes into a
cubic polynomial part (driven by the intersection numbers
:math:`\kappa_{ijk}`, the linear coefficients :math:`a_{ij}`, the
constant :math:`b_i` and the Euler characteristic :math:`\chi`) and a
non-perturbative instanton sum (driven by the genus-zero
Gopakumar–Vafa invariants :math:`n_\beta^0`):

.. math::
   :label: eq:jaxvacua-periods-01

   \begin{aligned}
       F_{\text{poly}}(z)
           &= -\tfrac{1}{6}\, \kappa_{ijk}\, z^i z^j z^k
              + \tfrac{1}{2}\, a_{ij}\, z^i z^j
              + b_i\, z^i + K_0, \\[2pt]
       F_{\text{inst}}(z)
           &= -\,\frac{1}{(2\pi i)^3}
              \sum_{\beta \in H_2(X, \mathbb{Z})_{>0}}
              n_\beta^0\, \mathrm{Li}_3\!\bigl(q^{\beta}\bigr),
              \quad q^{\beta} = e^{\,2\pi i\, \beta \cdot z}, \\[2pt]
       F(z) &= F_{\text{poly}}(z) + F_{\text{inst}}(z).
   \end{aligned}

The period vector and the Kähler potential then read

.. math::
   :label: eq:jaxvacua-periods-02

   \begin{aligned}
       \Pi(z)
           &= \bigl( X^0,\ X^i,\ F_i,\ F_0 \bigr),
              \quad X^0 = 1,\ X^i = z^i,\
              F_i = \partial_i F,\
              F_0 = 2 F - z^i \partial_i F, \\[2pt]
       K(z, \bar z)
           &= -\log\!\bigl(\, i\, \Pi^\dagger \cdot \Sigma \cdot \Pi \,\bigr),
   \end{aligned}

with :math:`\Sigma` the symplectic intersection form on
:math:`H^3(X, \mathbb{Z})`. In the diagram, inherited inputs (light
grey) come straight from ``lcs_tree``; the public outputs of the
layer (orange) are :math:`F`, :math:`\Pi` and :math:`K`.

.. raw:: html
   :file: _static/figures/f3_periods.html


Period class
-----------------------------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    periods

Prepotential
-----------------------------------

.. autosummary::

    periods.prepot_per
    periods.prepot_grad_per
    periods.prepot_grad_grad_per



Period vector and derivatives
-----------------------------------

.. autosummary::

    periods.period_vector_per
    periods.grad_period_vector_per
    periods.D_period_vector_per
    periods.PQ_per
    periods.P_per
    periods.Q_inv_per
    periods.Q_per


Mirror dual volume and Kähler potential
----------------------------------------

.. autosummary::

    periods.A_per
    periods.kahler_potential_per
    periods.grad_kahler_potential_per
    periods.sigma
    periods.compute_a_shift_monodromy


Gauge kinetic matrix
-----------------------------------

.. autosummary::

    periods.gauge_kinetic_matrix
    periods.gauge_kinetic_matrix_periods
    periods.gauge_kinetic_matrix_prepotential


ISD matrix
-----------------------------------

.. autosummary::

    periods.ISD_matrix
