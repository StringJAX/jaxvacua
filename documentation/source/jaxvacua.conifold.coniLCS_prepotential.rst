jaxvacua.conifold.coniLCS_prepotential
======================================

.. currentmodule:: jaxvacua.conifold.coniLCS_prepotential

.. automodule:: jaxvacua.conifold.coniLCS_prepotential

The coniLCS prepotential family.  Two layers, both pure-math (no flux
dependence):

1. **Per-period (X-coordinate) family** — attached to
   :class:`jaxvacua.periods.periods`.
2. **Per-modulus (z-coordinate) family** — attached to
   :class:`jaxvacua.css.css`.

These helpers underpin the dilogarithmic conifold contribution to the LCS
prepotential and its Taylor expansion in the conifold modulus
:math:`X^{\rm cf}` / :math:`z_{\rm cf}`.


Per-period family (attached to ``periods``)
-------------------------------------------

.. autosummary::
    :toctree: _autosummary

    F_coniLCS_bulk_per
    F_coniLCS_poly_split_per
    dF_coniLCS_poly_per
    ddF_coniLCS_poly_per
    dddF_coniLCS_poly_per
    ddddF_coniLCS_poly_per
    F_inst_per_coni
    F_coni_per
    F_coniLCS_exp_per
    F_coniLCS_series_per


Per-modulus family (attached to ``css``)
----------------------------------------

.. autosummary::
    :toctree: _autosummary

    F_coniLCS_bulk
    F_coniLCS_exp
    dF_coniLCS_exp
    F_coniLCS_series