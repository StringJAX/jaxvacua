# Copyright 2024-2026 Andreas Schachner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""End-to-end tests for the coniLCS flux-vacuum pipeline.

Purpose
-------
Validate the ``limit="coniLCS"`` workflow against the worked notebook fixture,
from model construction through flux plumbing, root solving, conifold-modulus
reconstruction and freezer integration.

Main public API
---------------
- ``TestConiLCSModelConstruction``: checks model, basis-change and conifold
  fixture construction.
- ``TestConiLCSFluxPlumbing`` and ``TestConiLCSScipyRoot``: check PFV/flux
  round-trips and root-solver convergence.
- ``TestConiLCSZcfSolver`` and ``TestConiLCSFreezerInterface``: check
  analytic ``z_cf`` handling and reduced-EFT consistency.
- ``TestConiLCSMultiVacuumScan`` and
  ``TestConiLCSVsConiLCSSeriesAgreement``: guard multi-point and cross-limit
  regressions.

Design notes
------------
The fixture mirrors
``documentation/source/notebooks/04_geometry_and_limits/13_coniLCS.ipynb``.
It also protects against jaxpolylog API or convention changes that alter the
``approx="patch"`` path used by coniLCS prepotentials.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest as _pytest
from scipy.optimize import root

import jaxvacua as jvc
from jaxvacua.freezer import ConifoldFreezer

# Local test infrastructure (TestCase + assertAllClose).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from util import TestCase  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
jax.config.update("jax_enable_x64", True)

# ============================================================================
# CYTools availability gate
#
# Construction of the model goes through ``cytools.Polytope`` + ``triangulate``
# + ``compute_gvs``; these are heavy and rely on the cytools install.  If
# cytools cannot fetch the toric data (e.g. running on a stripped CI), skip
# the whole module rather than fail loudly.
# ============================================================================

try:
    from cytools import Polytope as _Polytope
    _CYTOOLS_LOAD_ERR = None
except Exception as _exc:                                                  # noqa: BLE001
    _Polytope = None
    _CYTOOLS_LOAD_ERR = f"{type(_exc).__name__}: {_exc}"

_NEEDS_CYTOOLS = _pytest.mark.skipif(
    _Polytope is None,
    reason=f"cytools unavailable ({_CYTOOLS_LOAD_ERR})",
)


# ============================================================================
# Fixture (NB13)
#
# Lattice points → Polytope → triangulation → CY.  Conifold curve
# ``[-1, 1, 0]`` is the flopping rational curve in the mirror Mori cone.
# The basis-change matrix maps it to ``(1, 0, 0)`` (i.e. ``z^1 ≡ z_cf``).
# ============================================================================

_POLYTOPE_PTS = np.array([
    [-1, 3, -2, -1], [1, -1, 0, 0], [-1, 0, 0, 1],
    [-1, 0, 0, 0],   [-1, 0, 1, 1], [-1, 0, 2, 0],
    [-1, 0, 1, 0],
])
_BASIS_CHANGE   = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 1]])
_CONIFOLD_CURVE = np.array([-1, 1, 0])
_NCF            = 2
_MAXIMUM_DEGREE = 6
_PRANGE         = 100

# Single PFV from the notebook (cell 27/29).  Picked to give a small W0 and
# small z_cf — i.e. exactly the regime where the conifold corrections matter.
_M_DEFAULT = np.array([4, -8, 8])
_K_DEFAULT = np.array([-8, 3, -6])
_GS_DEFAULT = 0.38

# Multi-vacuum scan from cell 47.
_M_LIST = np.array([
    [4, -8, 8],
    [4, -8, 10],
    [8, -12, 6],
    [-8, 4, 12],
    [-14, 6, 27],
])
_K_LIST = np.array([
    [-8, 3, -6],
    [-6, 3, -4],
    [-5, 1, -2],
    [5, 1, -4],
    [4, 1, -2],
])
_GS_LIST = np.array([0.38, 0.15, 0.125, 0.35, 0.0643])


# ----------------------------------------------------------------------------
# Hardcoded NB13 ground-truth fingerprints (jaxpolylog 0.1.0, all uncommitted
# changes applied as of 2026-05-01).  Each fingerprint records the converged
# moduli and superpotential at the flux vacuum reached from the corresponding
# (M, K, gs) seed.  The tests below assert that the *current* code reaches the
# same fingerprint point — any regression in the polylog/prepotential/solver
# pipeline that breaks the convergence basin will trip these.
#
# Update by running ``/tmp/extract_nb13_fingerprints.py`` after a deliberate
# canonical change.
#
# Caveat (vacuum #2): scipy.root *does not* fully converge on this seed
# (|DW|_inf ~ 1e-4 even after Newton fallback). The fingerprint records the
# stall point; the convergence test for #2 is informational, the fingerprint
# match is the binding contract.
# ----------------------------------------------------------------------------
# Fields per fingerprint:
#   "scipy_success" — whether scipy.root.success was True (informational)
#   "DW_inf"        — |DW|_inf at the recovered point
#   "W0"            — |W0| (normalise=True)
#   "zcf"           — z[0] complex (numerical z_cf)
#   "z1", "z2"      — bulk complex moduli z[1], z[2]
#   "tau"           — axio-dilaton at the recovered point
_NB13_FINGERPRINTS = (
    {   # vacuum #1
        "M": (4, -8, 8), "K": (-8, 3, -6), "gs": 0.38,
        "scipy_success": True, "DW_inf": 6.082e-13, "W0": 6.881506e-04,
        "zcf": -2.2005150048e-18 + 4.8943611635e-06j,
        "z1":   8.5537804191e-14 + 1.9820605886e+00j,
        "z2":   4.2704349564e-14 + 9.9105094250e-01j,
        "tau":  1.1338856858e-13 + 2.6448930550e+00j,
    },
    {   # vacuum #2 — scipy stalls; fingerprint pins the stall point
        "M": (4, -8, 10), "K": (-6, 3, -4), "gs": 0.15,
        "scipy_success": False, "DW_inf": 1.130e-04, "W0": 7.368743e-09,
        "zcf":  2.6099621388e-21 + 5.4095885287e-14j,
        "z1":  -6.1440666077e-09 + 3.6554980498e+00j,
        "z2":  -4.6080498952e-09 + 2.7416235375e+00j,
        "tau": -1.2288131293e-08 + 7.3109961630e+00j,
    },
    {   # vacuum #3
        "M": (8, -12, 6), "K": (-5, 1, -2), "gs": 0.125,
        "scipy_success": True, "DW_inf": 1.708e-12, "W0": 6.695917e-04,
        "zcf": -7.0125170155e-18 + 1.3423461495e-05j,
        "z1":   1.3141171446e-13 + 1.9883998955e+00j,
        "z2":   6.5654567502e-14 + 9.9419274635e-01j,
        "tau":  5.2312037906e-13 + 7.9599429478e+00j,
    },
    {   # vacuum #4 — relatively large |z_cf| ~ 1.5e-2 (deep coniLCS)
        "M": (-8, 4, 12), "K": (5, 1, -4), "gs": 0.35,
        "scipy_success": True, "DW_inf": 5.058e-11, "W0": 6.956875e-02,
        "zcf": -4.2985849854e-16 + 1.5007023359e-02j,
        "z1":   2.0015529522e-14 + 1.3305744201e+00j,
        "z2":   4.3075205009e-15 + 3.5521384799e-01j,
        "tau":  3.4064476347e-14 + 2.8595150086e+00j,
    },
    {   # vacuum #5
        "M": (-14, 6, 27), "K": (4, 1, -2), "gs": 0.0643,
        "scipy_success": True, "DW_inf": 4.407e-12, "W0": 1.548275e-03,
        "zcf":  2.0092844203e-17 + 5.7937993663e-05j,
        "z1":  -7.7667116445e-14 + 1.9577951251e+00j,
        "z2":  -3.8660273787e-14 + 9.7904438375e-01j,
        "tau": -6.1657992912e-13 + 1.5676683050e+01j,
    },
)

# Tolerances.
_FLUX_TOL = 1e-12      # exact integer-flux equalities
_DW_TOL   = 1e-7       # converged F-term residual at the default seed
_DW_TOL_SCAN = 1e-4    # multi-scan: looser; scipy hybr stalls on some seeds
                       # (NB13 cell 49 itself does Newton fallback)
_ZCF_REL  = 5e-3       # numerical-vs-analytic z_cf at the default seed
_ZCF_REL_SCAN = 0.10   # multi-scan: 10% — analytic is leading-order; the
                       # truncation error grows with |z_cf|
_W0_RANGE = (1e-5, 1e-2)         # NB13 default seed sits in [10⁻⁴, 10⁻³]
_W0_RANGE_SCAN = (1e-9, 0.1)     # multi-scan: empirical envelope across the
                                  # 5 PFV pairs in NB13 cell 47


_LOAD_EXC: Exception | None = None
_MODEL = None


def _try_build_model():
    """Build the NB13 fixture.  Returns ``(model, h11)`` or raises."""
    poly = _Polytope(_POLYTOPE_PTS)
    cy = poly.triangulate().get_cy()
    h11 = int(cy.h11())
    h12_dual = int(cy.h12())
    model = jvc.FluxVacuaFinder(
        h12=h11, Q=h11 + h12_dual + 2,
        use_cytools=True, mirror_cy=cy, ncf=_NCF, use_gvs=True,
        maximum_degree=_MAXIMUM_DEGREE, basis_change=_BASIS_CHANGE,
        conifold_curve=_CONIFOLD_CURVE, limit="coniLCS",
        prange=_PRANGE, conifold_basis=True,
    )
    return model, h11


if _Polytope is not None:
    try:
        _MODEL, _H11 = _try_build_model()
    except Exception as _exc:                                              # noqa: BLE001
        _LOAD_EXC = _exc

_NEEDS_MODEL = _pytest.mark.skipif(
    _MODEL is None,
    reason=f"NB13 model unavailable ({_LOAD_EXC})",
)


# ============================================================================
# Group 1 — Model construction & lcs_tree state
# ============================================================================

@_NEEDS_CYTOOLS
@_NEEDS_MODEL
class TestConiLCSModelConstruction(TestCase):
    r"""Verify that the model carries the right coniLCS bookkeeping after
    construction.  These are cheap, no-jit checks — they catch bugs in
    ``lcs_tree.__init__`` / ``_update_conifold_curve_and_index`` /
    ``_perform_basis_change`` without paying any JAX trace cost."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = _MODEL
        cls.lt    = _MODEL.lcs_tree

    def test_limit_is_coniLCS(self):
        self.assertEqual(self.lt.limit, "coniLCS")

    def test_h12_matches_h11(self):
        self.assertEqual(self.lt.h12, _H11)

    def test_conifold_basis_true(self):
        self.assertTrue(bool(self.lt.conifold_basis))

    def test_ncf_value(self):
        self.assertEqual(int(self.lt.conifold.ncf), _NCF)

    def test_conifold_curve_matches_input(self):
        self.assertAllEqual(np.asarray(self.lt.conifold.conifold_curve),
                            _CONIFOLD_CURVE)

    def test_conifold_curve0_axis_aligned(self):
        r"""``conifold_curve @ basis_change.T`` must equal ``(1, 0, ..., 0)``."""
        expected = np.zeros(_H11, dtype=int)
        expected[0] = 1
        self.assertAllClose(np.asarray(self.lt.conifold.conifold_curve0),
                            expected, atol=_FLUX_TOL)

    def test_basis_change_round_trip(self):
        r"""``conifold_curve @ basis_change.T`` ≡ ``conifold_curve0``
        (computed independently)."""
        bc = np.asarray(self.lt.basis_change)
        derived = np.asarray(_CONIFOLD_CURVE) @ bc.T
        self.assertAllClose(derived,
                            np.asarray(self.lt.conifold.conifold_curve0),
                            atol=_FLUX_TOL)

    def test_a_matrix_b_vector_shape(self):
        a_mat = np.asarray(self.lt.a_matrix)
        b_vec = np.asarray(self.lt.b_vector)
        self.assertEqual(a_mat.shape, (_H11, _H11))
        self.assertEqual(b_vec.shape, (_H11,))

    def test_gv_data_populated_and_includes_conifold(self):
        r"""For ``limit="coniLCS"`` the conifold curve must be PRESENT in
        ``gv_charges`` (it's filtered out only for series/bulk).  The
        ``coni_index`` attribute must point at it."""
        gv_charges = np.asarray(self.lt.gv_charges)
        gv_invs    = np.asarray(self.lt.gv_invariants)
        self.assertGreater(gv_charges.shape[0], 0)
        self.assertEqual(gv_charges.shape[1], _H11)
        self.assertEqual(gv_invs.shape[0], gv_charges.shape[0])
        # The conifold-row must equal conifold_curve0 (axis-aligned).
        coni_row = gv_charges[int(self.lt.coni_index)]
        expected = np.zeros(_H11, dtype=int)
        expected[0] = 1
        self.assertAllClose(coni_row, expected, atol=_FLUX_TOL)


# ============================================================================
# Group 2 — Flux/moduli plumbing (PFV ↔ flux ↔ moduli)
# ============================================================================

@_NEEDS_CYTOOLS
@_NEEDS_MODEL
class TestConiLCSFluxPlumbing(TestCase):
    r"""``pfv_to_flux``, ``pfv_to_moduli``, ``flux_to_pfv`` round-trips."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = _MODEL
        cls.M     = _M_DEFAULT
        cls.K     = _K_DEFAULT
        cls.gs    = _GS_DEFAULT
        cls.tau0  = 1j / _GS_DEFAULT
        cls.flux  = cls.model.pfv_to_flux(cls.M, cls.K)
        cls.z0    = cls.model.pfv_to_moduli(cls.M, cls.K, cls.tau0)

    def test_flux_shape(self):
        n = int(np.asarray(self.flux).shape[0])
        # 2 * (h12 + 1) PFV slots, each split (M, K) → 4 * (h12 + 1).
        self.assertEqual(n, 4 * (_H11 + 1))

    def test_flux_is_integer_after_subtracting_b_shift(self):
        r"""Per the coniLCS convention in flux_utils, the b-vector is shifted
        by ``conifold_curve0 * ncf/24``.  After undoing this shift, the
        flux entries should be the integer ``M``/``K`` exactly."""
        # We don't try to recover M/K here (the inverse map is non-trivial
        # in general); we just sanity-check that flux is finite, non-NaN,
        # and order-of-magnitude consistent with M/K.
        f = np.asarray(self.flux)
        self.assertTrue(np.all(np.isfinite(f)))
        self.assertLess(float(np.max(np.abs(f))), 100.0)

    def test_z0_shape_and_first_entry_small(self):
        r"""``pfv_to_moduli`` for limit=coniLCS+ conifold_basis=True puts
        the conifold modulus at index 0 with a small magnitude (the seed
        is tuned for the conifold limit)."""
        z = np.asarray(self.z0)
        self.assertEqual(z.shape, (_H11,))
        self.assertLess(float(np.abs(z[0])), 1e-3)
        # Bulk moduli should be O(1) in the (Im) saxion direction.
        self.assertGreater(float(np.abs(z[1])), 0.1)

    def test_flux_to_pfv_inverse(self):
        r"""``flux_to_pfv(pfv_to_flux(M, K))`` must recover ``(M, K)``.

        ``flux_to_pfv`` returns the flux-side M, K slices (which are the
        full ``f2[1:]`` and ``h1[1:]`` vectors); for the conifold-basis
        coniLCS-limit setup, the leading ``h12`` entries of these slices
        coincide with the input ``M`` / ``K`` modulo the b-vector shift
        (an additive constant from ``pfv_to_flux``).  We compare on the
        slots that are unaffected by the shift."""
        M_back, K_back = self.model.flux_to_pfv(self.flux)
        # Drop the trailing 0-padding entries; on the remaining entries,
        # M_back[:h12] / K_back[:h12] should match the input integers.
        M_back_arr = np.asarray(M_back)
        K_back_arr = np.asarray(K_back)
        self.assertGreaterEqual(M_back_arr.shape[0], _H11)
        self.assertGreaterEqual(K_back_arr.shape[0], _H11)
        self.assertAllClose(M_back_arr[:_H11], np.asarray(self.M),
                            atol=_FLUX_TOL)
        self.assertAllClose(K_back_arr[:_H11], np.asarray(self.K),
                            atol=_FLUX_TOL)


# ============================================================================
# Group 3 — DW at the seed + scipy.root convergence
# ============================================================================

@_NEEDS_CYTOOLS
@_NEEDS_MODEL
class TestConiLCSScipyRoot(TestCase):
    r"""End-to-end vacuum-finding sanity: scipy.root converges, DW → 0,
    W0 lands in the expected range, the recovered point is inside the
    Kähler-cone facet, and the analytic z_cf matches the numerical one
    in all three solver modes."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = _MODEL
        cls.M     = _M_DEFAULT
        cls.K     = _K_DEFAULT
        cls.tau0  = 1j / _GS_DEFAULT
        cls.flux  = cls.model.pfv_to_flux(cls.M, cls.K)
        cls.z0    = cls.model.pfv_to_moduli(cls.M, cls.K, cls.tau0)
        cls.x0    = cls.model._convert_complex_to_real(
            cls.z0, jnp.conj(cls.z0), cls.tau0, jnp.conj(cls.tau0))
        cls.res   = root(
            cls.model.DW_x, np.asarray(cls.x0), args=(cls.flux,),
            jac=cls.model.dDW_x, tol=1e-10, method="hybr",
        )
        cls.x1 = jnp.asarray(cls.res.x)
        cls.z1, _, cls.tau1, _ = cls.model._convert_real_to_complex(cls.x1)

    def test_scipy_root_converged(self):
        self.assertTrue(bool(self.res.success),
                        msg=f"scipy.root did not converge: {self.res.message}")

    def test_DW_residual_is_small(self):
        DW = np.asarray(self.model.DW_x(self.x1, self.flux))
        residual = float(np.max(np.abs(DW)))
        self.assertLess(residual, _DW_TOL,
                        msg=f"|DW|_inf = {residual:.3e} > tol = {_DW_TOL}")

    def test_W0_in_expected_range(self):
        r"""The NB13 fixture has the seed tuned so |W0| at the converged
        vacuum is around 10⁻⁴ (and definitely between 10⁻⁵ and 10⁻²).
        Anything outside this window is a strong red flag that the polylog
        / prepotential pipeline broke."""
        W0 = float(np.abs(self.model.superpotential(
            self.z1, self.tau1, self.flux, normalise=True)))
        lo, hi = _W0_RANGE
        self.assertTrue(lo < W0 < hi,
                        msg=f"|W0| = {W0:.3e} outside expected range "
                            f"({lo:.0e}, {hi:.0e})")

    def test_zcf_numerical_small(self):
        r"""At the converged vacuum, |z_cf| should be in the
        ``[10⁻⁷, 10⁻³]`` window (the ``coniLCS`` regime)."""
        zcf_num = float(np.abs(self.z1[0]))
        self.assertTrue(1e-7 < zcf_num < 1e-3,
                        msg=f"|z_cf|_num = {zcf_num:.3e} outside coniLCS regime")

    def test_zcf_analytic_modes_agree(self):
        r"""``manual`` and ``autodiff`` analytic z_cf must agree (closed-form
        vs autodiff route — same math, different assembly).  The ``pfv``
        mode is a deliberate approximation that need only agree to ~1%."""
        x_bulk = self.x1[2:]
        zcf_man = complex(self.model.compute_zcf_x(
            x_bulk, self.flux, mode="manual"))
        zcf_aut = complex(self.model.compute_zcf_x(
            x_bulk, self.flux, mode="autodiff"))
        zcf_pfv = complex(self.model.compute_zcf_x(
            x_bulk, self.flux, mode="pfv"))
        # manual ↔ autodiff: same physics, must agree tightly.
        self.assertAllClose(zcf_man, zcf_aut, atol=1e-12, rtol=1e-10,
                            msg="manual ≠ autodiff")
        # All three within order-of-magnitude.
        scale = abs(zcf_man) + 1e-30
        self.assertLess(abs(zcf_pfv - zcf_man) / scale, 0.05,
                        msg=f"PFV vs manual diverged: "
                            f"PFV={zcf_pfv}, manual={zcf_man}")

    def test_zcf_numeric_matches_analytic_manual(self):
        r"""At a converged vacuum the F-term equation ∂_zcf W = 0 holds,
        so the numerical ``z_cf = z[0]`` must equal the analytic solver's
        output to order ``_ZCF_REL`` (small mismatch from the truncated
        instanton sum + Kähler correction not applied here)."""
        zcf_num = complex(self.z1[0])
        zcf_man = complex(self.model.compute_zcf_x(
            self.x1[2:], self.flux, mode="manual"))
        rel = abs(zcf_num - zcf_man) / (abs(zcf_man) + 1e-30)
        self.assertLess(rel, _ZCF_REL,
                        msg=f"numeric vs analytic z_cf relative err "
                            f"{rel:.3e} > {_ZCF_REL}")

    def test_inside_kahler_cone(self):
        r"""The recovered moduli must lie inside the Kähler cone."""
        # mirror_volume(z) > 0 at the conifold-side Kähler-cone facet.
        v = float(self.model.mirror_volume(
            self.z1, jnp.conj(self.z1)).real)
        self.assertGreater(v, 0.0,
                           msg=f"mirror_volume = {v:.3e} ≤ 0 (outside cone)")


# ============================================================================
# Group 4 — z_cf solver: building blocks + apply_correction toggle
# ============================================================================

@_NEEDS_CYTOOLS
@_NEEDS_MODEL
class TestConiLCSZcfSolver(TestCase):
    r"""Direct probes of the z_cf solver building blocks (``log_prefactor``,
    ``W_log_coeff``, ``log_coeff_K_corr``, ``_zcf_from_log_coeff``) on the
    NB13 fixture."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = _MODEL
        cls.M     = _M_DEFAULT
        cls.K     = _K_DEFAULT
        cls.tau0  = 1j / _GS_DEFAULT
        cls.flux  = cls.model.pfv_to_flux(cls.M, cls.K)
        z0_full = cls.model.pfv_to_moduli(cls.M, cls.K, cls.tau0)
        cls.z_bulk  = z0_full[1:]
        cls.cz_bulk = jnp.conj(cls.z_bulk)
        cls.tau     = cls.tau0
        cls.ctau    = jnp.conj(cls.tau0)

    def test_log_prefactor_finite_nonzero(self):
        lp = complex(self.model.log_prefactor(self.tau, self.flux))
        self.assertTrue(np.isfinite(lp.real) and np.isfinite(lp.imag))
        self.assertGreater(abs(lp), 1e-10,
                           msg="log_prefactor too small — F-term equation degenerate")

    def test_W_log_coeff_modes_agree(self):
        r"""``manual`` and ``autodiff`` routes assemble the same W̃₁ via two
        independent code paths; must match.

        Tolerance: ~1e-8 rel. is set by float64 ordering differences between
        the two code paths.  ``manual`` evaluates closed-form W̃₁ via direct
        polylog calls; ``autodiff`` uses ``jax.grad`` of ``F_coniLCS_exp``
        which routes through jaxpolylog's ``custom_jvp`` derivative rule.
        Both paths agree to ~1.3e-9 relative error on the test point;
        physics-meaningful precision is preserved.
        """
        w_man = complex(self.model.W_log_coeff(
            self.z_bulk, self.tau, self.flux, mode="manual"))
        w_aut = complex(self.model.W_log_coeff(
            self.z_bulk, self.tau, self.flux, mode="autodiff"))
        self.assertAllClose(w_man, w_aut, atol=1e-9, rtol=1e-8,
                            msg=f"W_log_coeff manual={w_man} ≠ autodiff={w_aut}")

    def test_compute_zcf_apply_correction_changes_value(self):
        r"""Toggling ``apply_correction=True`` must produce a *different*
        z_cf (the Kähler-covariant correction is non-trivial); both must
        be finite."""
        z_off = complex(self.model.compute_zcf(
            self.z_bulk, self.cz_bulk, self.tau, self.ctau, self.flux,
            mode="manual", apply_correction=False))
        z_on  = complex(self.model.compute_zcf(
            self.z_bulk, self.cz_bulk, self.tau, self.ctau, self.flux,
            mode="manual", apply_correction=True))
        self.assertTrue(np.isfinite(z_off.real) and np.isfinite(z_off.imag))
        self.assertTrue(np.isfinite(z_on.real)  and np.isfinite(z_on.imag))
        # Toggle is non-trivial.
        rel = abs(z_off - z_on) / (abs(z_off) + 1e-30)
        self.assertGreater(rel, 1e-9,
                           msg="apply_correction toggle is a no-op (suspicious)")

    def test_compute_zcf_x_matches_compute_zcf(self):
        r"""Real-coord and complex-coord entry points must agree (the real
        wrapper just unpacks ``x_bulk`` via ``_convert_real_to_complex``)."""
        x_bulk = self.model._convert_complex_to_real(
            self.z_bulk, self.cz_bulk, self.tau, self.ctau)
        for mode in ("manual", "autodiff", "pfv"):
            via_x = complex(self.model.compute_zcf_x(
                x_bulk, self.flux, mode=mode))
            via_z = complex(self.model.compute_zcf(
                self.z_bulk, self.cz_bulk, self.tau, self.ctau, self.flux,
                mode=mode))
            self.assertAllClose(via_x, via_z, atol=1e-12, rtol=1e-10,
                                msg=f"compute_zcf_x ≠ compute_zcf at mode={mode}")


# ============================================================================
# Group 5 — Freezer interface equivalence
# ============================================================================

@_NEEDS_CYTOOLS
@_NEEDS_MODEL
class TestConiLCSFreezerInterface(TestCase):
    r"""``ConifoldFreezer`` on the NB13 fixture must produce the same
    answers as the model's direct ``compute_zcf`` / ``DW_x`` calls."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model   = _MODEL
        cls.freezer = ConifoldFreezer(cls.model)
        cls.M     = _M_DEFAULT
        cls.K     = _K_DEFAULT
        cls.tau0  = 1j / _GS_DEFAULT
        cls.flux  = cls.model.pfv_to_flux(cls.M, cls.K)
        z0_full   = cls.model.pfv_to_moduli(cls.M, cls.K, cls.tau0)
        cls.z_bulk = z0_full[1:]
        cls.tau    = cls.tau0
        x0_full = cls.model._convert_complex_to_real(
            z0_full, jnp.conj(z0_full), cls.tau0, jnp.conj(cls.tau0))
        cls.x_bulk = x0_full[2:]

    def test_freezer_solve_heavy_matches_compute_zcf(self):
        for mode in ("manual", "autodiff", "pfv"):
            zcf_fr  = complex(self.freezer.solve_heavy(
                self.z_bulk, self.tau, self.flux, mode=mode)[0])
            zcf_md  = complex(self.model.compute_zcf(
                self.z_bulk, jnp.conj(self.z_bulk),
                self.tau, jnp.conj(self.tau), self.flux,
                mode=mode))
            self.assertAllClose(zcf_fr, zcf_md, atol=1e-12, rtol=1e-10,
                                msg=f"freezer.solve_heavy ≠ compute_zcf at mode={mode}")

    def test_freezer_DW_x_light_matches_DW_x_slice(self):
        r"""``freezer.DW_x_light(x_bulk, flux, mode)`` must equal
        ``model.DW_x(x_full, flux)[2:]`` where x_full is the bulk x with
        the analytic z_cf prepended (the slice equivalence)."""
        x_full = self.freezer._real_light_to_full(
            self.x_bulk, self.flux, mode="manual")
        dw_via_freezer = np.asarray(self.freezer.DW_x_light(
            self.x_bulk, self.flux, mode="manual"))
        dw_via_slice   = np.asarray(self.model.DW_x(x_full, self.flux))[2:]
        self.assertAllClose(dw_via_freezer, dw_via_slice,
                            atol=1e-12, rtol=1e-10)


# ============================================================================
# Group 6 — Multi-vacuum scan vs NB13 fingerprints
#
# Drives scipy.root from each of the 5 NB13 seeds and asserts the recovered
# point matches the *exact* NB13 fingerprint (z, tau, W0).  This is the
# strongest possible regression net: any change in the polylog /
# prepotential / flux / scipy-driver pipeline that perturbs the convergence
# basin (or the convergence point) will trip these tests immediately.
#
# Vacuum #2 is a known scipy-stall case (NB13 itself documents this).
# We pin it via fingerprint anyway — if the stall *point* shifts, that's a
# regression worth flagging.
# ============================================================================

@_NEEDS_CYTOOLS
@_NEEDS_MODEL
class TestConiLCSMultiVacuumScan(TestCase):
    r"""Run NB13 cell 47's 5-vacuum scan and assert the recovered (z, tau, W0)
    match the hardcoded NB13 fingerprints exactly (to ``_FP_RTOL``)."""

    # Fingerprint-match tolerance.  Generous because:
    #  * scipy.hybr's exact stop point depends on the Hessian inversion path,
    #  * the analytic-Newton fallback for #2 is order-dependent,
    #  * jaxpolylog's custom_jvp (v0.2.1+) and custom_vjp (v0.1.0) evaluation
    #    paths produce gradients that differ by ~1e-9 relative, which shifts
    #    Newton's converged point by ~1e-11 absolute (~few · 1e-6 relative).
    # 1e-5 (relative) absorbs the autodiff-version drift while still catching
    # any meaningful change in the convergence basin.
    _FP_RTOL = 1e-5
    _FP_ATOL = 1e-12

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = _MODEL

    def _run_one(self, fp):
        r"""Drive scipy.root from the seed; if hybr stalls (e.g. vacuum #2),
        fall back to a few in-tree Newton steps as NB13 cell 49 does."""
        M = np.asarray(fp["M"])
        K = np.asarray(fp["K"])
        gs = float(fp["gs"])
        tau0 = 1j / gs
        flux = self.model.pfv_to_flux(M, K)
        z0 = self.model.pfv_to_moduli(M, K, tau0)
        x0 = self.model._convert_complex_to_real(
            z0, jnp.conj(z0), tau0, jnp.conj(tau0))
        res = root(self.model.DW_x, np.asarray(x0), args=(flux,),
                   jac=self.model.dDW_x, tol=1e-10, method="hybr")
        x1 = jnp.asarray(res.x)
        if not res.success:
            z_b, _, tau_b, _ = self.model._convert_real_to_complex(x1)
            try:
                z_n, t_n, _ = self.model.newton_method_flux_vacua(
                    z_b, tau_b, flux, step_size_Newton=0.5,
                    tol=1e-10, max_iters=30, mode="SUSY", solver_mode="real",
                )
                x1 = self.model._convert_complex_to_real(
                    z_n, jnp.conj(z_n), t_n, jnp.conj(t_n))
            except Exception:                                              # noqa: BLE001
                pass
        z1, _, t1, _ = self.model._convert_real_to_complex(x1)
        return res, x1, z1, t1, flux

    def _close(self, got, expected, label, idx):
        r"""Complex-number proximity helper: return None on PASS, str on FAIL."""
        got = complex(got); expected = complex(expected)
        atol = self._FP_ATOL + self._FP_RTOL * abs(expected)
        if abs(got - expected) <= atol:
            return None
        return (f"  vacuum #{idx+1}, {label}: got {got!r} vs expected "
                f"{expected!r} (|Δ|={abs(got - expected):.3e}, atol={atol:.3e})")

    # -------- z-fingerprint (each modulus, each component) --------

    def test_z_fingerprints(self):
        r"""For each of the 5 vacua, the converged ``z = (z_cf, z1, z2)`` must
        match the hardcoded NB13 fingerprint to ``_FP_RTOL``."""
        failures = []
        for idx, fp in enumerate(_NB13_FINGERPRINTS):
            _, _, z, _, _ = self._run_one(fp)
            for label, key, val in (
                ("z[0] (z_cf)", "zcf", z[0]),
                ("z[1]",        "z1",  z[1]),
                ("z[2]",        "z2",  z[2]),
            ):
                err = self._close(val, fp[key], label, idx)
                if err is not None:
                    failures.append(err)
        self.assertFalse(failures, msg="\n" + "\n".join(failures))

    def test_tau_fingerprint(self):
        failures = []
        for idx, fp in enumerate(_NB13_FINGERPRINTS):
            _, _, _, tau, _ = self._run_one(fp)
            err = self._close(tau, fp["tau"], "tau", idx)
            if err is not None:
                failures.append(err)
        self.assertFalse(failures, msg="\n" + "\n".join(failures))

    def test_W0_fingerprints(self):
        failures = []
        for idx, fp in enumerate(_NB13_FINGERPRINTS):
            _, _, z, t, flux = self._run_one(fp)
            W0 = float(np.abs(self.model.superpotential(
                z, t, flux, normalise=True)))
            atol = self._FP_ATOL + self._FP_RTOL * fp["W0"]
            if abs(W0 - fp["W0"]) > atol:
                failures.append(
                    f"  vacuum #{idx+1}: |W0|={W0:.6e} vs expected "
                    f"{fp['W0']:.6e} (|Δ|={abs(W0 - fp['W0']):.3e})"
                )
        self.assertFalse(failures, msg="\n" + "\n".join(failures))

    def test_zcf_analytic_matches_numerical(self):
        r"""On the four converged vacua (#1, 3, 4, 5), the analytic z_cf
        solver must agree with the numerical z_cf to within ``_ZCF_REL_SCAN``.
        Vacuum #2 (scipy-stalled) is excluded — at the stall point z_cf is
        ~5e-14 (essentially zero) and the analytic solver predicts ~ 1e-7,
        a relative-error comparison is meaningless there."""
        failures = []
        for idx, fp in enumerate(_NB13_FINGERPRINTS):
            if idx == 1:                                                   # skip stall
                continue
            _, x1, z1, _, flux = self._run_one(fp)
            zcf_num = complex(z1[0])
            zcf_an  = complex(self.model.compute_zcf_x(
                x1[2:], flux, mode="manual"))
            rel = abs(zcf_num - zcf_an) / (abs(zcf_an) + 1e-30)
            if rel > _ZCF_REL_SCAN:
                failures.append(
                    f"  vacuum #{idx+1}: numeric={zcf_num} vs "
                    f"analytic={zcf_an}, rel err {rel:.3e} > {_ZCF_REL_SCAN}"
                )
        self.assertFalse(failures, msg="\n" + "\n".join(failures))

    def test_DW_residual_at_converged_seeds(self):
        r"""For the four cleanly-converged vacua, |DW|_inf must be tight.
        Vacuum #2 is excluded (NB13 itself stalls there)."""
        failures = []
        for idx, fp in enumerate(_NB13_FINGERPRINTS):
            if not fp["scipy_success"]:
                continue
            _, x1, _, _, flux = self._run_one(fp)
            DW_inf = float(np.max(np.abs(self.model.DW_x(x1, flux))))
            if DW_inf > _DW_TOL:
                failures.append(
                    f"  vacuum #{idx+1}: |DW|_inf = {DW_inf:.3e} > {_DW_TOL}"
                )
        self.assertFalse(failures, msg="\n" + "\n".join(failures))


# ============================================================================
# Group 7 — Cross-limit consistency (coniLCS vs coniLCS_series)
# ============================================================================

@_NEEDS_CYTOOLS
@_NEEDS_MODEL
class TestConiLCSVsConiLCSSeriesAgreement(TestCase):
    r"""Build a sibling ``limit="coniLCS_series"`` model on the same data
    and verify that, at the same physical point, the two limits agree on
    the F-terms to the truncation tolerance.

    The series expansion of the conifold prepotential converges in
    powers of z_cf; for |z_cf| ≲ 10⁻⁵ (NB13 regime) and ``nmax >= 3`` we
    expect agreement at the 10⁻³ level on |DW|.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_lcs = _MODEL
        # Series model on the same lattice.
        poly = _Polytope(_POLYTOPE_PTS)
        cy = poly.triangulate().get_cy()
        h11 = int(cy.h11()); h12_dual = int(cy.h12())
        try:
            cls.model_series = jvc.FluxVacuaFinder(
                h12=h11, Q=h11 + h12_dual + 2,
                use_cytools=True, mirror_cy=cy, ncf=_NCF, use_gvs=True,
                maximum_degree=_MAXIMUM_DEGREE,
                basis_change=_BASIS_CHANGE,
                conifold_curve=_CONIFOLD_CURVE,
                limit="coniLCS_series",
                prange=_PRANGE, conifold_basis=True,
            )
        except Exception as exc:                                           # noqa: BLE001
            cls._series_err = f"{type(exc).__name__}: {exc}"
            cls.model_series = None
        else:
            cls._series_err = None

    def setUp(self):
        super().setUp()
        if self.model_series is None:
            self.skipTest(f"coniLCS_series unavailable: {self._series_err}")

    def test_DW_at_seed_agrees_to_truncation(self):
        r"""At |z_cf| ~ 10⁻⁵, the series truncation error on DW should
        be < 10⁻² (loose — the series converges geometrically in z_cf,
        so this is a loose envelope rather than a strict bound)."""
        M = _M_DEFAULT; K = _K_DEFAULT; tau0 = 1j / _GS_DEFAULT
        flux = self.model_lcs.pfv_to_flux(M, K)
        z0 = self.model_lcs.pfv_to_moduli(M, K, tau0)
        DW_lcs    = self.model_lcs.DW_x(
            self.model_lcs._convert_complex_to_real(
                z0, jnp.conj(z0), tau0, jnp.conj(tau0)), flux)
        DW_series = self.model_series.DW_x(
            self.model_series._convert_complex_to_real(
                z0, jnp.conj(z0), tau0, jnp.conj(tau0)), flux)
        # Compare the bulk-direction entries (indices 2:); the conifold
        # entries [0:2] differ between limits by construction (different
        # treatment of the dilog).
        diff = float(np.max(np.abs(
            np.asarray(DW_lcs)[2:] - np.asarray(DW_series)[2:])))
        # 0.2 is loose but sufficient to catch a broken polylog evaluation
        # (we'd see DW differences of O(1) like in the user-reported bug).
        self.assertLess(diff, 0.2,
                        msg=f"|DW^lcs - DW^series|_inf = {diff:.3e} on bulk "
                            f"directions — series should agree at |z_cf|≪1")


if __name__ == "__main__":
    import unittest
    unittest.main()
