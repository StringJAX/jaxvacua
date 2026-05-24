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

"""Tests for one-modulus hypergeometric model support.

Purpose
-------
Validate the hypergeometric model registry, prepotential builders and basic
Newton-solver compatibility for special one-modulus limits.

Main public API
---------------
- ``TestHypergeometricModels``: registry and construction checks.
- ``TestHypergeometricPrepotentials``: LCS, K-point and C-point prepotential
  behaviour.
- ``TestNewtonSolveSmokeLCS``: smoke tests for solver compatibility.

Design notes
------------
The tests use compact one-modulus fixtures to catch convention drift in the
closed-form model data.
"""

import sys, os, warnings
import jax
import jax.numpy as jnp
import numpy as np
import chex
from functools import partial
from util import *

jax.config.update("jax_enable_x64", True)

sys.path.append("./../")
from jaxvacua import HypergeometricModels
from jaxvacua.hypergeometric_models import (
    HYPERGEOMETRIC_MODELS, KPOINT_DATA, CPOINT_DATA,
    make_prepot_LCS, make_prepot_Kpoint, make_prepot_Cpoint,
)

warnings.filterwarnings("ignore")


# ==============================================================================
#  TestHypergeometricModels — the static registry / factory
# ==============================================================================

class TestHypergeometricModels(TestCase):
    r"""
    **Description:**
    Test suite for :class:`~jaxvacua.HypergeometricModels`. Verifies the static
    registry methods (``list``, ``available_limits``, ``lcs_data``,
    ``boundary_data``) and the ``build`` factory at LCS / Kpoint / Cpoint.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.m_lcs = HypergeometricModels.build("X33", limit="LCS")
        cls.m_kpt = HypergeometricModels.build("X33", limit="Kpoint")
        cls.m_cpt = HypergeometricModels.build("X42", limit="Cpoint")

    # ------------------------------------------------------------------
    # Registry / metadata
    # ------------------------------------------------------------------

    def test_list_includes_known_labels(self):
        r"""**Description:** :meth:`HypergeometricModels.list` must return all
        18 registered labels including the canonical hypergeometric quintics."""
        labels = HypergeometricModels.list()
        for expected in ["X5", "X33", "X44", "X42"]:
            self.assertIn(expected, labels)
        # Total registry size: 14 hypergeometric + 4 additional = 18.
        self.assertEqual(len(labels), 18)

    def test_available_limits(self):
        r"""**Description:** ``available_limits`` returns a subset of ``["LCS",
        "Kpoint", "Cpoint"]`` matching the registry contents."""
        self.assertEqual(HypergeometricModels.available_limits("X33"),
                         ["LCS", "Kpoint"])
        self.assertEqual(HypergeometricModels.available_limits("X42"),
                         ["LCS", "Cpoint"])
        # X66 is filtered out of the Kpoint list (degenerate B1=0).
        self.assertNotIn("Kpoint", HypergeometricModels.available_limits("X66"))

    def test_lcs_data_keys(self):
        r"""**Description:** ``lcs_data`` returns a dict with the topological
        data needed to reconstruct an LCS prepotential."""
        d = HypergeometricModels.lcs_data("X5")
        for k in ["kappa", "c2", "chi"]:
            self.assertIn(k, d)
        self.assertIsInstance(d["kappa"], int)

    def test_unknown_label_raises(self):
        r"""**Description:** Querying an unknown model label must raise
        ``KeyError`` from registry methods and ``ValueError`` from ``build``."""
        with self.assertRaises(KeyError):
            HypergeometricModels.lcs_data("NOT_A_MODEL")
        with self.assertRaises(KeyError):
            HypergeometricModels.build("NOT_A_MODEL", limit="LCS")

    def test_x66_kpoint_raises(self):
        r"""**Description:** X66 K-point requires the degenerate ``B1 = 0``
        formula (arXiv:2306.01059 §5); not implemented yet."""
        with self.assertRaises(NotImplementedError):
            HypergeometricModels.build("X66", limit="Kpoint")
        # X66 LCS still loads.
        m = HypergeometricModels.build("X66", limit="LCS")
        self.assertEqual(m.h12, 1)

    # ------------------------------------------------------------------
    # Build → FluxVacuaFinder shape
    # ------------------------------------------------------------------

    def test_build_returns_flux_vacua_finder(self):
        r"""**Description:** Each :meth:`build` call returns a fully-formed
        ``FluxVacuaFinder`` with the standard API (``DW_x``, ``dDW_x``,
        ``superpotential``)."""
        from jaxvacua import FluxVacuaFinder
        for m in [self.m_lcs, self.m_kpt, self.m_cpt]:
            self.assertIsInstance(m, FluxVacuaFinder)
            self.assertEqual(m.h12, 1)
            for attr in ("DW_x", "dDW_x", "superpotential"):
                self.assertTrue(hasattr(m, attr))

    def test_non_lcs_auto_detect_tags_model_type(self):
        r"""**Description:** When the user constructs a ``FluxVacuaFinder``
        directly with ``limit ∈ {Kpoint, Cpoint}`` and a registered
        ``model_ID``, ``periods`` auto-detects the model and stamps
        ``model_type='hypergeometric'`` plus the closed-form prepotential."""
        for limit in ("Kpoint", "Cpoint"):
            label = "X33" if limit == "Kpoint" else "X42"
            m = HypergeometricModels.build(label, limit=limit)
            self.assertEqual(m.periods.limit, limit)
            self.assertEqual(m.periods.model_type, "hypergeometric")
            self.assertTrue(m.periods._prepotential_input_used)

    def test_auto_detect_rejects_h12_neq_1(self):
        r"""**Description:** ``periods`` rejects K/C-point limits when
        ``h12 != 1`` because the registry only covers one-modulus models."""
        from jaxvacua import FluxVacuaFinder
        with self.assertRaisesRegex(ValueError, "one-modulus"):
            FluxVacuaFinder(h12=2, model_ID="X33", limit="Kpoint")

    def test_auto_detect_rejects_unknown_model_ID(self):
        r"""**Description:** ``periods`` rejects K/C-point limits with an
        unregistered ``model_ID`` and points the user at the registry."""
        from jaxvacua import FluxVacuaFinder
        with self.assertRaisesRegex(ValueError, "registered one-modulus"):
            FluxVacuaFinder(h12=1, model_ID="NOT_A_MODEL", limit="Kpoint")


# ==============================================================================
#  TestHypergeometricPrepotentials — closed-form callables
# ==============================================================================

class TestHypergeometricPrepotentials(TestCase):
    r"""
    **Description:**
    Test suite for the closed-form prepotential factories
    (:func:`make_prepot_LCS`, :func:`make_prepot_Kpoint`,
    :func:`make_prepot_Cpoint`).  Verifies scalar shape, finiteness,
    holomorphic differentiability, period-vector isotropicity and
    conjugation consistency.
    """

    # Wrap the closures in a tuple so attribute access on the class doesn't
    # trigger the function-descriptor protocol (otherwise self.F_lcs binds
    # `self` as the first arg of F).
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        d33 = HYPERGEOMETRIC_MODELS["X33"]
        bd_kpt = KPOINT_DATA["X33"]
        bd_cpt = CPOINT_DATA["X42"]
        cls._F = {
            "lcs": make_prepot_LCS(d33["kappa"], d33["c2"], d33["chi"]),
            "kpt": make_prepot_Kpoint(
                tau=bd_kpt["tau"], gamma=bd_kpt["gamma"], delta=bd_kpt["delta"],
                c=bd_kpt["c"], B1=bd_kpt["B1"], B2=bd_kpt["B2"], B3=bd_kpt["B3"],
            ),
            "cpt": make_prepot_Cpoint(
                tau=bd_cpt["tau"], gamma=bd_cpt["gamma"], delta=bd_cpt["delta"],
                k=bd_cpt["k"], A1=bd_cpt["A1"], A2=bd_cpt["A2"],
            ),
        }
        cls.X_lcs = jnp.array([1.0 + 0j, 0.01 + 0.05j])
        cls.X_kpt = jnp.array([1.0 + 0j, 0.3 + 0.4j])
        cls.X_cpt = jnp.array([1.0 + 0j, 0.01 + 0.02j])

    @chex.variants(with_jit=True, without_jit=True)
    def test_prepotential_returns_scalar(self):
        r"""**Description:** :math:`F(X)` returns a rank-0 complex scalar."""
        for tag, X in [("lcs", self.X_lcs),
                       ("kpt", self.X_kpt),
                       ("cpt", self.X_cpt)]:
            F = self._F[tag]
            val = self.variant(lambda x: F(x, conj=False))(X)
            self.assertEqual(val.shape, ())
            chex.assert_type(val, complex)

    @chex.variants(with_jit=True, without_jit=True)
    def test_prepotential_holomorphic_grad(self):
        r"""**Description:** :math:`\partial F/\partial X^I` exists and is
        finite — the upper half of the period vector
        :math:`\Pi = (\partial_I F, X^I)`."""
        for tag, X in [("lcs", self.X_lcs),
                       ("kpt", self.X_kpt),
                       ("cpt", self.X_cpt)]:
            F = self._F[tag]
            grad_F = self.variant(
                jax.grad(lambda x: F(x, conj=False), holomorphic=True)
            )(X)
            chex.assert_shape(grad_F, (2,))
            self.assertTrue(jnp.all(jnp.isfinite(grad_F)))

    @chex.variants(with_jit=True, without_jit=True)
    def test_period_vector_isotropicity(self):
        r"""**Description:** Period vector
        :math:`\Pi = (\partial_I F, X^I)` satisfies
        :math:`\Pi^T \Sigma \Pi = 0`."""
        sigma = jnp.array([[0, 0, 1, 0], [0, 0, 0, 1],
                           [-1, 0, 0, 0], [0, -1, 0, 0]], dtype=complex)

        def make_Pi(F):
            def Pi(x):
                gF = jax.grad(lambda y: F(y, conj=False), holomorphic=True)(x)
                return jnp.concatenate([gF, x])
            return Pi

        for tag, X in [("lcs", self.X_lcs), ("kpt", self.X_kpt)]:
            F = self._F[tag]
            Pi_val = self.variant(make_Pi(F))(X)
            self.assertAllClose(Pi_val @ sigma @ Pi_val, 0.0 + 0j, atol=1e-11)

    @chex.variants(with_jit=True, without_jit=True)
    def test_prepotential_conjugation(self):
        r"""**Description:** Reality condition
        :math:`\bar F(\bar X) = \overline{F(X)}` — required for a real
        Kähler potential."""
        for tag, X in [("lcs", self.X_lcs),
                       ("kpt", self.X_kpt),
                       ("cpt", self.X_cpt)]:
            F = self._F[tag]
            F_orig = self.variant(lambda x: F(x, conj=False))(X)
            F_conj = self.variant(lambda x: F(x, conj=True))(jnp.conj(X))
            self.assertAllClose(F_conj, jnp.conj(F_orig), atol=1e-12)


# ==============================================================================
#  TestNewtonSolveSmokeLCS — end-to-end FluxVacuaFinder integration
# ==============================================================================

class TestNewtonSolveSmokeLCS(TestCase):
    r"""**Description:** Smoke-test the standard `FluxVacuaFinder` pipeline
    on a hypergeometric LCS model: build → ``DW_x`` evaluates to a finite
    real vector of the expected shape."""

    def test_lcs_DW_x_finite(self):
        for label in ["X5", "X33", "X42"]:
            m = HypergeometricModels.build(label, limit="LCS")
            z0 = jnp.array([3.0 + 1.0j])
            tau0 = 0.5 + 2.0j
            n_fl = 2 * m.n_fluxes
            flux = jnp.zeros(n_fl, dtype=jnp.float64).at[
                jnp.array([0, m.n_fluxes])
            ].set(1.0)
            x0 = m._convert_complex_to_real(
                z0, jnp.conj(z0), tau0, jnp.conj(tau0)
            )
            DW = m.DW_x(x0, flux)
            self.assertEqual(DW.shape, (4,))   # 2*(h12+1) = 4 real comps
            self.assertTrue(jnp.all(jnp.isfinite(DW)))


if __name__ == "__main__":
    import unittest
    unittest.main()
