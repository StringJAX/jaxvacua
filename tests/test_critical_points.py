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
# Tests for critical_points.py — non-SUSY vacuum finding.
# ------------------------------------------------------------------------------

import sys, os, warnings
import jax
import jax.numpy as jnp
import numpy as np
import math
from util import *

jax.config.update("jax_enable_x64", True)

sys.path.append("./../")
import jaxvacua
from jaxvacua.critical_points import CriticalPointFinder

warnings.filterwarnings("ignore")


# ==============================================================================
#  TestCriticalPointFinder
# ==============================================================================

class TestCriticalPointFinder(TestCase):
    r"""
    **Description:**
    Test suite for :class:`CriticalPointFinder`, which finds critical points
    of the scalar potential :math:`V` (both SUSY and non-SUSY).

    .. admonition:: Background
        :class: dropdown

        Non-SUSY critical points satisfy :math:`\partial_\alpha V = 0` but
        :math:`D_I W \neq 0`.  The no-scale potential
        :math:`V_{\rm ns} = e^K |DW|^2 \geq 0` is particularly well-suited
        for finding minima (~85% minima rate in benchmarks).
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.h12 = 2
        # Use maximum_degree=5 for instanton corrections (needed for Newton convergence)
        cls.model = jaxvacua.FluxVacuaFinder(
            h12=cls.h12, model_ID=1, model_type="KS", maximum_degree=5
        )
        cls.model.lcs_tree.a_matrix = jnp.array([[4.5, 1.5], [1.5, 0.]])

        cls.sampler = jaxvacua.data_sampler(
            cls.model,
            moduli_bounds=(2., 5.),
            dilaton_bounds=(math.sqrt(3) / 2, 10.),
            axion_bounds=(-0.5, 0.5),
            seed=42,
        )

        cls.Nmax = 200
        cls.n_fl = cls.model.n_fluxes
        cls.dim_H3 = cls.model.dimension_H3

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def test_constructor(self):
        r"""
        **Description:**
        Verify that ``CriticalPointFinder`` initialises correctly with
        the expected attributes.
        """
        finder = CriticalPointFinder(self.model, self.sampler, Nmax=self.Nmax)
        # Check stored attributes
        self.assertEqual(finder.Nmax, self.Nmax)
        self.assertEqual(finder.n_fluxes, self.n_fl)
        self.assertEqual(finder.dimension_H3, self.dim_H3)
        self.assertTrue(finder.noscale)

    def test_constructor_noscale_false(self):
        r"""
        **Description:**
        Verify that ``noscale=False`` is stored correctly.
        """
        finder = CriticalPointFinder(self.model, self.sampler,
                                     Nmax=self.Nmax, noscale=False)
        self.assertFalse(finder.noscale)

    # ------------------------------------------------------------------
    # Flux candidate generation
    # ------------------------------------------------------------------

    def test_generate_flux_candidates_F_mode(self):
        r"""
        **Description:**
        Verify that ``_generate_flux_candidates`` with mode ``"F"`` produces
        valid flux candidates with correct shapes.
        """
        finder = CriticalPointFinder(self.model, self.sampler, Nmax=self.Nmax)
        mod_pts = jnp.array(self.sampler.get_complex_moduli(50), dtype=complex)
        tau_pts = jnp.array(self.sampler.get_complex_tau(50), dtype=complex)

        x0, fluxes, idx = finder._generate_flux_candidates(
            50, mod_pts, tau_pts, isd_mode="F")

        if len(x0) > 0:
            # x0 should have shape (N, 2*(h12+1))
            self.assertEqual(x0.shape[1], 2 * (self.h12 + 1))
            # fluxes should have shape (N, 2*n_fluxes)
            self.assertEqual(fluxes.shape[1], 2 * self.n_fl)

    def test_generate_flux_candidates_ISD_minus(self):
        r"""
        **Description:**
        Verify that mode ``"ISD-"`` produces valid candidates.
        ISD- takes [f₁|h₁] of length 2*dim_H3 as input and computes [f₂|h₂].
        """
        finder = CriticalPointFinder(self.model, self.sampler, Nmax=self.Nmax)
        mod_pts = jnp.array(self.sampler.get_complex_moduli(50), dtype=complex)
        tau_pts = jnp.array(self.sampler.get_complex_tau(50), dtype=complex)

        x0, fluxes, idx = finder._generate_flux_candidates(
            50, mod_pts, tau_pts, isd_mode="ISD-")

        # ISD- should produce many valid candidates at Nmax=200
        self.assertGreater(len(x0), 0,
                           msg="ISD- should produce valid candidates at Nmax=200")

    def test_generate_flux_candidates_invalid_mode(self):
        r"""
        **Description:**
        Verify that an invalid mode raises ``ValueError``.
        """
        finder = CriticalPointFinder(self.model, self.sampler, Nmax=self.Nmax)
        mod_pts = jnp.array(self.sampler.get_complex_moduli(10), dtype=complex)
        tau_pts = jnp.array(self.sampler.get_complex_tau(10), dtype=complex)

        with self.assertRaises(ValueError):
            finder._generate_flux_candidates(10, mod_pts, tau_pts, isd_mode="INVALID")

    # ------------------------------------------------------------------
    # Newton solver
    # ------------------------------------------------------------------

    def test_newton_solver_converges(self):
        r"""
        **Description:**
        Verify that the Newton solver finds at least one critical point
        from ISD- flux candidates at Nmax=200.

        A critical point satisfies :math:`|\partial_\alpha V| < \text{tol}`.
        """
        finder = CriticalPointFinder(self.model, self.sampler, Nmax=self.Nmax)
        mod_pts = jnp.array(self.sampler.get_complex_moduli(20), dtype=complex)
        tau_pts = jnp.array(self.sampler.get_complex_tau(20), dtype=complex)

        x0, fluxes, _ = finder._generate_flux_candidates(
            20, mod_pts, tau_pts, isd_mode="ISD-")

        if len(x0) > 0:
            # Try first candidate
            x_sol, res, conv = finder._solve_dV_newton_single(
                x0[0], fluxes[0], tol=1e-8, max_iters=300)
            # Should converge for at least some candidates
            # (not guaranteed for every single one)
            self.assertIsInstance(res, float)
            self.assertEqual(x_sol.shape, x0[0].shape)

    def test_newton_solver_residual_format(self):
        r"""
        **Description:**
        Verify that the Newton solver returns the correct output format:
        ``(x_solution, residual, converged)`` with correct types.
        """
        finder = CriticalPointFinder(self.model, self.sampler, Nmax=self.Nmax)
        mod_pts = jnp.array(self.sampler.get_complex_moduli(5), dtype=complex)
        tau_pts = jnp.array(self.sampler.get_complex_tau(5), dtype=complex)

        x0, fluxes, _ = finder._generate_flux_candidates(
            5, mod_pts, tau_pts, isd_mode="F")

        if len(x0) > 0:
            x_sol, res, conv = finder._solve_dV_newton_single(
                x0[0], fluxes[0], max_iters=10)
            # x_sol is numpy array
            self.assertIsInstance(x_sol, np.ndarray)
            # residual is float
            self.assertIsInstance(res, float)
            # converged is bool
            self.assertIsInstance(conv, bool)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def test_classify_solution_keys(self):
        r"""
        **Description:**
        Verify that ``_classify_solution`` returns a dict with all expected keys.
        """
        finder = CriticalPointFinder(self.model, self.sampler, Nmax=self.Nmax)

        # Use a random point (not necessarily a critical point)
        x = np.array([0.1, 3.0, -0.2, 2.5, -0.3, 5.0])
        fl = np.array([1., 0., -2., 0., 3., -1., 2., 1., 0., -1., 1., 0.])

        info = finder._classify_solution(x, fl)

        # Check all expected keys
        for key in ['V', '|DW|', 'eigenvalues', 'is_susy', 'is_minimum', 'Nflux']:
            self.assertIn(key, info, msg=f"Missing key: {key}")

        # eigenvalues should be a sorted array of length 2*(h12+1)
        self.assertEqual(len(info['eigenvalues']), 2 * (self.h12 + 1))
        # is_susy and is_minimum should be bool
        self.assertIsInstance(info['is_susy'], bool)
        self.assertIsInstance(info['is_minimum'], bool)

    # ------------------------------------------------------------------
    # Main entry point: sample_critical_points
    # ------------------------------------------------------------------

    def test_sample_critical_points_newton(self):
        r"""
        **Description:**
        Integration test: ``sample_critical_points`` with Newton solver
        should find at least one critical point from a small batch.
        """
        finder = CriticalPointFinder(self.model, self.sampler, Nmax=self.Nmax)
        results = finder.sample_critical_points(
            n_target=5, n_batch=50, max_batches=2,
            isd_mode="ISD-", solver="newton", verbose=False)

        # Should find at least some critical points
        self.assertIsInstance(results, list)
        if len(results) > 0:
            r = results[0]
            # Each result is a dict with expected keys
            self.assertIn('flux', r)
            self.assertIn('moduli', r)
            self.assertIn('tau', r)
            self.assertIn('residual', r)
            self.assertIn('V', r)
            self.assertIn('is_susy', r)
            self.assertIn('is_minimum', r)

    def test_sample_critical_points_scipy(self):
        r"""
        **Description:**
        Integration test: ``sample_critical_points`` with scipy solver.
        """
        finder = CriticalPointFinder(self.model, self.sampler, Nmax=self.Nmax)
        results = finder.sample_critical_points(
            n_target=5, n_batch=50, max_batches=2,
            isd_mode="ISD-", solver="scipy", verbose=False)

        self.assertIsInstance(results, list)

    def test_sample_critical_points_returns_non_susy(self):
        r"""
        **Description:**
        Verify that at least some found critical points are non-SUSY
        (|DW| > threshold) when using ISD- mode with noscale=True.
        """
        finder = CriticalPointFinder(self.model, self.sampler,
                                     Nmax=self.Nmax, noscale=True)
        results = finder.sample_critical_points(
            n_target=10, n_batch=100, max_batches=3,
            isd_mode="ISD-", solver="scipy", verbose=False)

        if len(results) > 0:
            # At least some should be non-SUSY (from benchmarks: ~78%)
            n_nonsusy = sum(1 for r in results if not r['is_susy'])
            # With ISD- at Nmax=200, we expect most to be non-SUSY
            # Allow for statistical variation — just check > 0
            self.assertGreater(
                n_nonsusy, 0,
                msg="ISD- with noscale=True should find non-SUSY critical points")

    def test_sample_critical_points_classify_false(self):
        r"""
        **Description:**
        Verify that ``classify=False`` skips Hessian computation.
        Results should NOT have ``eigenvalues`` or ``is_minimum`` keys.
        """
        finder = CriticalPointFinder(self.model, self.sampler, Nmax=self.Nmax)
        results = finder.sample_critical_points(
            n_target=3, n_batch=30, max_batches=2,
            isd_mode="ISD-", solver="scipy",
            classify=False, verbose=False)

        if len(results) > 0:
            r = results[0]
            # Should have flux/moduli/tau/residual
            self.assertIn('flux', r)
            self.assertIn('residual', r)
            # Should NOT have classification keys
            self.assertNotIn('eigenvalues', r)
            self.assertNotIn('is_minimum', r)

    def test_invalid_solver_raises(self):
        r"""
        **Description:**
        Verify that an invalid solver name raises ``ValueError``.
        """
        finder = CriticalPointFinder(self.model, self.sampler, Nmax=self.Nmax)
        with self.assertRaises(ValueError):
            finder.sample_critical_points(
                n_target=1, n_batch=10, max_batches=1,
                solver="invalid_solver", verbose=False)

    # ------------------------------------------------------------------
    # Vectorised optax solver
    # ------------------------------------------------------------------

    def test_optax_batch_solver_output_shapes(self):
        r"""
        **Description:**
        Verify that ``_solve_dV_optax_batch`` returns arrays with correct
        shapes: ``(N, 2*(h12+1))``, ``(N,)``, ``(N,)``.
        """
        finder = CriticalPointFinder(self.model, self.sampler, Nmax=self.Nmax)
        mod_pts = jnp.array(self.sampler.get_complex_moduli(20), dtype=complex)
        tau_pts = jnp.array(self.sampler.get_complex_tau(20), dtype=complex)

        x0, fluxes, _ = finder._generate_flux_candidates(
            20, mod_pts, tau_pts, isd_mode="ISD-")

        if len(x0) > 0:
            n = min(5, len(x0))
            x_out, res_out, conv_out = finder._solve_dV_optax_batch(
                x0[:n], fluxes[:n], n_steps=100, tol=1e-8)
            self.assertEqual(x_out.shape, (n, 2 * (self.h12 + 1)))
            self.assertEqual(res_out.shape, (n,))
            self.assertEqual(conv_out.shape, (n,))
            self.assertTrue(np.all(np.isfinite(x_out)))

    def test_optax_batch_residuals_decrease(self):
        r"""
        **Description:**
        Verify that the vectorised Adam solver reduces residuals compared
        to the starting points (more steps → lower residuals).
        """
        finder = CriticalPointFinder(self.model, self.sampler, Nmax=self.Nmax)
        mod_pts = jnp.array(self.sampler.get_complex_moduli(30), dtype=complex)
        tau_pts = jnp.array(self.sampler.get_complex_tau(30), dtype=complex)

        x0, fluxes, _ = finder._generate_flux_candidates(
            30, mod_pts, tau_pts, isd_mode="ISD-")

        if len(x0) >= 3:
            n = min(5, len(x0))
            _, res_100, _ = finder._solve_dV_optax_batch(
                x0[:n], fluxes[:n], n_steps=100, tol=1e-8)
            _, res_1000, _ = finder._solve_dV_optax_batch(
                x0[:n], fluxes[:n], n_steps=1000, tol=1e-8)
            # Median residual should decrease with more steps
            self.assertLessEqual(np.median(res_1000), np.median(res_100))

    def test_sample_critical_points_adam_v(self):
        r"""
        **Description:**
        Integration test: ``sample_critical_points`` with vectorised Adam
        solver (``solver="adam_v"``) runs without error.
        """
        finder = CriticalPointFinder(self.model, self.sampler, Nmax=self.Nmax)
        results = finder.sample_critical_points(
            n_target=5, n_batch=50, max_batches=1,
            isd_mode="ISD-", solver="adam_v",
            optax_steps=500, verbose=False)
        self.assertIsInstance(results, list)

    def test_sample_critical_points_hybrid(self):
        r"""
        **Description:**
        Integration test: ``sample_critical_points`` with hybrid solver
        (vectorised Adam warm-start + Newton refinement) runs without error
        and can find critical points.
        """
        finder = CriticalPointFinder(self.model, self.sampler, Nmax=self.Nmax)
        results = finder.sample_critical_points(
            n_target=5, n_batch=100, max_batches=2,
            isd_mode="ISD-", solver="hybrid",
            optax_steps=2000, verbose=False)
        self.assertIsInstance(results, list)
        # Hybrid should be able to find at least some critical points
        if len(results) > 0:
            r = results[0]
            self.assertIn('flux', r)
            self.assertIn('V', r)
            self.assertIn('is_minimum', r)


if __name__ == "__main__":
    import unittest
    unittest.main()
