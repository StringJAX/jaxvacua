# Copyright 2024 Andreas Schachner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------------------------------------------------------------------------------
# Tests for the conifold bulk-EFT block in jaxvacua/conifold_utils.py.
#
# Covers (with conifold_basis=True):
#   - `_ConifoldGated` conditional-loading semantics (test #8 from plan)
#   - `conifold_fluxes` round-trip (#4)
#   - `W_log_coeff(mode="manual") ≡ W_log_coeff(mode="autodiff")` (#1, primary)
#   - `compute_zcf` dispatcher: mode + apply_correction + invalid mode (#2, #5)
#   - Kähler correction composition (#3)
#   - `zcf_handling` shape + null-mode (#6)
#   - `DWbulk_x` slice consistency (#7)
#   - per-period (`F_coniLCS_exp_per`) vs per-modulus (`F_coniLCS_exp`) parity (#10)
#
# `conifold_basis=False` parity is exercised separately in Phase 1.5.
# ------------------------------------------------------------------------------

import os
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore")

sys.path.append("./../")
import jaxvacua as jvc
from util import TestCase

# ---------------------------------------------------------------------------
# Test fixture: pre-pickled "aule" coniLCS model wired up via the
# `vp.PromotionModels` helper (the only proven path that exercises all three
# coniLCS limits — bulk / series / conilcs — from a single lcs_tree).
# Tests skip gracefully when that helper or the pickled data isn't reachable.
# ---------------------------------------------------------------------------
_NAME  = "aule"
_MVEC0 = np.array([20, 4, 8, -18, -20])
_KVEC0 = np.array([-5, -1, 0, 1, -1])
_PVEC0 = np.array([0.0, 0.020833333333333332, 0.041666666666666664,
                   0.020833333333333332, 0.0])
_TAU0  = 1j / 0.04317129968232153
_ATOL  = 1e-10

_MODELS = None
_PFV = None
_LOAD_ERROR = None


def _try_load_models():
    """Build the (bulk, series, conilcs) PromotionModels triple from the
    'aule' lcs_tree, plus a PFV seed point. Returns (models, pfv) on success,
    or raises with a descriptive error."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    promo_dir = os.path.join(repo_root, "private", "promotion")
    if promo_dir not in sys.path:
        sys.path.insert(0, promo_dir)
    import vacuum_promotion as vp  # noqa: E402

    lcs_tree = jvc.periods(h12=len(_MVEC0), model_ID=_NAME, limit="coniLCS").lcs_tree
    models = vp.PromotionModels.from_lcs_tree(
        lcs_tree, conifold_basis=True, ncf=2, prange=20, maximum_degree=2,
    )
    pfv = vp.PFV.from_quantum_numbers(
        models, M_vec=_MVEC0, K_vec=_KVEC0, p_vec=_PVEC0, tau=_TAU0,
        metadata={"model_name": _NAME},
    )
    return models, pfv


try:
    _MODELS, _PFV = _try_load_models()
except Exception as exc:
    _LOAD_ERROR = f"{type(exc).__name__}: {exc}"


_NEEDS_MODEL = pytest.mark.skipif(
    _MODELS is None,
    reason=f"PromotionModels unavailable ({_LOAD_ERROR})",
)


# Convenience accessors used inside test methods.
def _model(limit_name):
    return getattr(_MODELS, limit_name)


def _x_full():
    return jnp.asarray(_PFV.x)


def _flux():
    return jnp.asarray(_PFV.flux)


# ==============================================================================
#  Test #8 — `_ConifoldGated` conditional-loading semantics
# ==============================================================================
@_NEEDS_MODEL
class TestConditionalLoading(TestCase):
    r"""
    `_ConifoldGated` is a class-level descriptor that surfaces conifold-only
    methods on a model only when ``model.limit`` belongs to the coniLCS family.
    Plain LCS limits must not expose them; coniLCS variants must.
    """

    def test_conilcs_variants_expose_compute_zcf(self):
        for name in ("bulk", "series", "conilcs"):
            m = _model(name)
            self.assertTrue(hasattr(m, "compute_zcf"),
                            msg=f"{name}: compute_zcf not exposed")
            self.assertTrue(callable(getattr(m, "compute_zcf")),
                            msg=f"{name}: compute_zcf not callable")

    def test_plain_lcs_construction_rejects_conifold_data(self):
        # Updated behaviour: constructing a plain LCS instance from a model
        # whose pickled data carries conifold info now raises ValueError in
        # `lcs_tree.__init__` ("Input for conifold curve provided but limit is
        # not coniLCS!"). The previous assertion that hasattr(plain,
        # "compute_zcf") returns False is no longer reachable for the "aule"
        # fixture because the plain instance can never be built; the
        # `_ConifoldGated` negative case is instead implied by the validation.
        with self.assertRaises(ValueError):
            jvc.periods(h12=len(_MVEC0), model_ID=_NAME, limit="LCS")


# ==============================================================================
#  Test #4 — `conifold_fluxes` round-trip
# ==============================================================================
@_NEEDS_MODEL
class TestConifoldFluxes(TestCase):
    r"""
    `conifold_fluxes` splits a flux vector into 12 conifold-vs-bulk
    projections. With ``conifold_basis=True`` the splits reduce to plain
    array slicing of the underlying ``_split_fluxes`` output.
    """

    def test_roundtrip_basis_true(self):
        m = _model("bulk")
        flux = _flux()
        f1, f2, h1, h2 = m._split_fluxes(flux)
        out = m.conifold_fluxes(flux)
        # Names match the unpacking in conifold_fluxes.
        M0, H0, M1, H1, Malpha, Halpha, P1, K1, Palpha, Kalpha, P0, K0 = out
        self.assertAllClose(M0, f2[0], atol=_ATOL)
        self.assertAllClose(H0, h2[0], atol=_ATOL)
        self.assertAllClose(M1, f2[1], atol=_ATOL)
        self.assertAllClose(H1, h2[1], atol=_ATOL)
        self.assertAllClose(Malpha, f2[2:], atol=_ATOL)
        self.assertAllClose(Halpha, h2[2:], atol=_ATOL)
        self.assertAllClose(P0, f1[0], atol=_ATOL)
        self.assertAllClose(K0, h1[0], atol=_ATOL)
        self.assertAllClose(P1, f1[1], atol=_ATOL)
        self.assertAllClose(K1, h1[1], atol=_ATOL)
        self.assertAllClose(Palpha, f1[2:], atol=_ATOL)
        self.assertAllClose(Kalpha, h1[2:], atol=_ATOL)


# ==============================================================================
#  Test #1 (primary verification target) —
#       W_log_coeff(mode="manual") ≡ W_log_coeff(mode="autodiff")
# ==============================================================================
@_NEEDS_MODEL
class TestWLogCoeff(TestCase):
    r"""
    The two W̃₁ routes must produce the same number:
      - "manual"   = closed-form kappa/a_matrix/b_vector + Li sums (replaces the
                     pre-Phase-1 ``compute_zcf_explicit`` body).
      - "autodiff" = via the css-side ``F_coniLCS_exp`` + ``dF_coniLCS_exp``
                     (replaces ``compute_zcf_compact``).
    """

    def test_modes_agree_at_seed(self):
        for name in ("bulk", "series", "conilcs"):
            m = _model(name)
            z, _, tau, _ = m._convert_real_to_complex(_x_full())
            w_man = m.W_log_coeff(z[1:], tau, _flux(), mode="manual",   conj=False)
            w_aut = m.W_log_coeff(z[1:], tau, _flux(), mode="autodiff", conj=False)
            self.assertAllClose(w_man, w_aut, atol=_ATOL,
                                msg=f"{name}: manual vs autodiff disagree")

    def test_modes_agree_at_perturbed_points(self):
        m = _model("series")
        z0, _, tau0, _ = m._convert_real_to_complex(_x_full())
        # Perturb |z| in Im-direction by random small jitters.
        rng = np.random.default_rng(0)
        for _ in range(4):
            jitter = 1.0 + 0.1 * rng.standard_normal(z0.shape[0] - 1)
            z_pert = jnp.asarray(np.asarray(z0[1:]) * jitter)
            w_man = m.W_log_coeff(z_pert, tau0, _flux(), mode="manual",   conj=False)
            w_aut = m.W_log_coeff(z_pert, tau0, _flux(), mode="autodiff", conj=False)
            self.assertAllClose(w_man, w_aut, atol=_ATOL)

    def test_invalid_mode_raises(self):
        m = _model("bulk")
        z, _, tau, _ = m._convert_real_to_complex(_x_full())
        with self.assertRaises(ValueError):
            m.W_log_coeff(z[1:], tau, _flux(), mode="bogus", conj=False)


# ==============================================================================
#  Test #2 + #5 — `compute_zcf` dispatcher
# ==============================================================================
@_NEEDS_MODEL
class TestComputeZcfDispatcher(TestCase):
    r"""
    Dispatcher contract:
      - mode="manual" / "autodiff" produce the same z_cf at apply_correction=False
      - apply_correction=True/False give different z_cf (correction is non-trivial)
      - mode="bogus" raises ValueError
    """

    def test_manual_equals_autodiff(self):
        for name in ("bulk", "series", "conilcs"):
            m = _model(name)
            z_m = m.compute_zcf_x(_x_full()[2:], _flux(), mode="manual",
                                apply_correction=False, conj=False)
            z_a = m.compute_zcf_x(_x_full()[2:], _flux(), mode="autodiff",
                                apply_correction=False, conj=False)
            self.assertAllClose(z_m, z_a, atol=_ATOL,
                                msg=f"{name}: dispatcher manual vs autodiff")

    def test_apply_correction_changes_value(self):
        m = _model("bulk")
        z0 = m.compute_zcf_x(_x_full()[2:], _flux(), mode="manual",
                           apply_correction=False, conj=False)
        z1 = m.compute_zcf_x(_x_full()[2:], _flux(), mode="manual",
                           apply_correction=True,  conj=False)
        diff = abs(complex(z1) - complex(z0))
        self.assertTrue(
            diff > 1e-12 * abs(complex(z0)),
            msg=f"apply_correction toggle had no effect (|Δ|={diff:.2e})",
        )

    def test_invalid_mode_raises(self):
        m = _model("bulk")
        with self.assertRaises(ValueError):
            m.compute_zcf_x(_x_full()[2:], _flux(), mode="bogus",
                          apply_correction=False, conj=False)


# ==============================================================================
#  Test #3 — Kähler correction composes correctly
# ==============================================================================
@_NEEDS_MODEL
class TestKaehlerCorrectionComposition(TestCase):
    r"""
    The corrected solver multiplies z_cf by ``exp(-log_coeff_K_corr / log_prefactor)``.
    Equivalently:
        compute_zcf(apply_correction=True) / compute_zcf(apply_correction=False)
        ==  exp(-log_coeff_K_corr / log_prefactor).
    """

    def test_correction_factor_consistent(self):
        m = _model("bulk")
        x = _x_full()
        flux = _flux()
        z, cz, tau, ctau = m._convert_real_to_complex(x)

        zcf0 = complex(m.compute_zcf_x(x[2:], flux, mode="manual",
                                     apply_correction=False, conj=False))
        zcf1 = complex(m.compute_zcf_x(x[2:], flux, mode="manual",
                                     apply_correction=True,  conj=False))
        ratio_actual = zcf1 / zcf0

        log_corr = m.log_coeff_K_corr(z[1:], cz[1:], tau, ctau, flux, conj=False)
        log_pref = m.log_prefactor(tau, flux, conj=False)
        ratio_predicted = complex(jnp.exp(-log_corr / log_pref))

        self.assertAllClose(ratio_actual, ratio_predicted, atol=_ATOL)


# ==============================================================================
#  Test #6 — `zcf_handling` shape + null-mode
# ==============================================================================
@_NEEDS_MODEL
class TestZcfHandling(TestCase):
    r"""
    `zcf_handling` builds the full real-vector ``[Re(zcf*), Im(zcf*), *x]``.
    With ``mode=None`` the conifold prefix collapses to ``[0, 0]``.
    """

    def test_null_mode_zero_prefix(self):
        m = _model("bulk")
        h12 = len(_MVEC0)
        x_bulk_real = _x_full()[2:]
        out = m.zcf_handling(x_bulk_real, _flux(), mode=None)
        self.assertEqual(out.shape[0], x_bulk_real.shape[0] + 2)
        self.assertAllClose(out[:2], jnp.zeros(2), atol=_ATOL)
        self.assertAllClose(out[2:], x_bulk_real, atol=_ATOL)

    def test_manual_mode_prepends_zcf(self):
        m = _model("bulk")
        h12 = len(_MVEC0)
        x_bulk_real = _x_full()[2:]
        out = m.zcf_handling(x_bulk_real, _flux(), mode="manual")
        # The full real-vector built internally is [zeros(2), *x_bulk_real].
        # zcf_handling computes z_cf at that point and returns
        #   [Re(zcf), Im(zcf), *x_bulk_real].
        zcf = m.compute_zcf_x(x_bulk_real, _flux(),
                            mode="manual", apply_correction=False, conj=False)
        self.assertAllClose(out[0], jnp.real(zcf), atol=_ATOL)
        self.assertAllClose(out[1], jnp.imag(zcf), atol=_ATOL)
        self.assertAllClose(out[2:], x_bulk_real, atol=_ATOL)


# ==============================================================================
#  Test #7 — ``ConifoldFreezer.DW_x_light`` ≡ ``DW_x[2:]`` slice consistency
# ==============================================================================
@_NEEDS_MODEL
class TestDWxLight(TestCase):
    r"""
    ``freezer.DW_x_light(x_bulk, flux, mode="manual")`` integrates out z_cf
    via the freezer's ``_real_light_to_full`` and projects the light slice.
    Equivalent direct construction:
      x_full = [Re(zcf*), Im(zcf*), *x_bulk]
      → DW_x(x_full, flux)[2:].

    Replaces the pre-2026-05-01 ``TestDWbulkX::test_slice_equivalence`` test
    after ``model.DWbulk_x`` / ``model.dDWbulk_x`` were hard-removed.
    """

    def test_slice_equivalence(self):
        from jaxvacua.freezer import ConifoldFreezer
        for name in ("bulk", "series", "conilcs"):
            m = _model(name)
            freezer = ConifoldFreezer(m)
            x_bulk_real = _x_full()[2:]
            dw_bulk = freezer.DW_x_light(x_bulk_real, _flux(), mode="manual")
            zcf = complex(m.compute_zcf_x(
                x_bulk_real, _flux(),
                mode="manual", apply_correction=False, conj=False,
            ))
            x_full_recon = jnp.concatenate(
                [jnp.array([zcf.real, zcf.imag]), x_bulk_real]
            )
            dw_full_sliced = m.DW_x(x_full_recon, _flux())[2:]
            self.assertAllClose(dw_bulk, dw_full_sliced, atol=_ATOL,
                                msg=f"{name}: DW_x_light vs DW_x[2:]")


# ==============================================================================
#  Test #10 — per-period (on `periods`) vs per-modulus (on `css`) prepotential
# ==============================================================================
@_NEEDS_MODEL
class TestPerPeriodVsPerModulus(TestCase):
    r"""
    The css-side helpers ``F_coniLCS_exp(zbulk, n)`` (used by
    ``W_log_coeff(mode="autodiff")``) must equal the periods-side
    ``F_coniLCS_exp_per`` evaluated at ``XPer = (1, z_cf, *zbulk)``.

    With ``z_cf = 0`` and ``XPer[0] = 1``, the n-th Taylor coefficient is
    purely a function of zbulk, so the two reductions must agree.
    """

    def test_F_coniLCS_exp_zero_zcf(self):
        m = _model("series")
        z_full, _, _, _ = m._convert_real_to_complex(_x_full())
        z_bulk = z_full[1:]
        # The css-side helper goes:
        #   moduli  = [0, *z_bulk]
        #   XPer    = moduli_to_periods(moduli)
        #   F_coniLCS_exp_per(X0=XPer[0], XConi=1, XPerBulk=XPer[2:], n=n)
        # So we replicate the same chain on the periods side.
        moduli = jnp.append(jnp.zeros(1), z_bulk)
        XPer = m.moduli_to_periods(moduli, conj=False)
        X0 = XPer[0]
        XConi = jnp.asarray(1.0 + 0.0j)
        XPerBulk = XPer[2:]
        for n in (0, 1, 2):
            f_css = m.F_coniLCS_exp(z_bulk, conj=False, n=n)
            f_per = m.periods.F_coniLCS_exp_per(X0, XConi, XPerBulk,
                                                 conj=False, n=n)
            self.assertAllClose(f_css, f_per, atol=_ATOL,
                                msg=f"n={n}: css vs periods disagree")