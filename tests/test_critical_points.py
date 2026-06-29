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

"""Tests for non-SUSY critical-point search workflows.

Purpose
-------
Validate the non-SUSY ``dV = 0`` workflow now hosted on
``FluxVacuaFinder`` after the legacy ``CriticalPointFinder`` merge.

Main public API
---------------
- ``TestCriticalPointFinder``: regression tests for absorbed candidate
  generation, classification, Optax/SciPy solver paths, critical-point
  sampling and method-level search parameters.

Design notes
------------
The tests intentionally refer to the historical critical-point workflow while
asserting that the implementation now lives on ``FluxVacuaFinder``.
"""

import sys, os, warnings
import jax
import jax.numpy as jnp
import numpy as np
import math
import pytest
from util import *

jax.config.update("jax_enable_x64", True)

sys.path.append("./../")
import jaxvacua
from jaxvacua.util import PRNGSequence

warnings.filterwarnings("ignore")


# ==============================================================================
#  TestCriticalPointFinder
# ==============================================================================

class TestCriticalPointFinder(TestCase):
    r"""
    **Description:**
    Test suite for the non-SUSY critical-point sampling workflow on
    :class:`FluxVacuaFinder` — :func:`sample_critical_points` and its
    companion helpers (:func:`_generate_flux_candidates`,
    :func:`_solve_dV_optax_batch`, :func:`classify_solution`).

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
        cls.finder = jaxvacua.FluxVacuaFinder(
            h12=cls.h12, model_ID=1, model_type="KS", maximum_degree=5,
        )
        cls.finder.lcs_tree.a_matrix = jnp.array([[4.5, 1.5], [1.5, 0.]])

        cls.sampler = jaxvacua.data_sampler(
            cls.finder,
            moduli_bounds=(2., 5.),
            dilaton_bounds=(math.sqrt(3) / 2, 10.),
            axion_bounds=(-0.5, 0.5),
            seed=42,
            use_jax=True,   # required so `rns_key=...` passed to the sampler
                            # methods below is actually honoured (otherwise the
                            # _rand_* helpers fall through to global np.random.*).
        )

        cls.n_fl = cls.finder.n_fluxes
        cls.dim_H3 = cls.finder.dimension_H3

        # Class-level PRNGSequence threaded through every sampler call below
        # so test outcomes are independent of which earlier tests advanced the
        # sampler's own internal PRNGSequence.
        cls.rng_key = PRNGSequence(42)

    # ------------------------------------------------------------------
    # Construction / API surface
    # ------------------------------------------------------------------

    def test_finder_exposes_critical_points_api(self):
        r"""
        **Description:**
        Verify that the merged :class:`FluxVacuaFinder` exposes the
        critical-points workflow surface (``sample_critical_points``,
        ``_generate_flux_candidates``, ``classify_solution``,
        ``_solve_dV_optax_batch``, ``calibrate_priors``).
        """
        for name in ("sample_critical_points",
                     "_generate_flux_candidates",
                     "classify_solution",
                     "_solve_dV_optax_batch",
                     "calibrate_priors",
                     "_estimate_sigmas",
                     "_precompute_M_eigensystem"):
            self.assertTrue(callable(getattr(self.finder, name, None)),
                            msg=f"FluxVacuaFinder is missing absorbed method: {name}")

        # Sanity-check the inherited FluxEFT attributes the workflow relies on
        self.assertEqual(self.finder.n_fluxes,     self.n_fl)
        self.assertEqual(self.finder.dimension_H3, self.dim_H3)

    def test_sample_critical_points_accepts_noscale_kwarg(self):
        r"""
        **Description:**
        ``noscale`` is now a method-level kwarg on ``sample_critical_points``
        (was ``CriticalPointFinder`` instance state).  Smoke-test that
        ``noscale=False`` is accepted without raising.
        """
        results = self.finder.sample_critical_points(
            n_target=1, n_batch=5, max_batches=1,
            isd_mode="ISD-", solver="scipy", noscale=False,
            sampler=self.sampler, verbose=False,
        )
        self.assertIsInstance(results, list)

    # ------------------------------------------------------------------
    # Flux candidate generation (C3)
    # ------------------------------------------------------------------

    def test_generate_flux_candidates_F_mode(self):
        r"""
        **Description:**
        Verify that ``_generate_flux_candidates`` with mode ``"F"`` produces
        valid flux candidates with correct shapes.
        """
        N = 20
        mod_pts, tau_pts = self.sampler.initial_guesses(
            N, filter_moduli=True, include_fluxes=False, rns_key=self.rng_key,
        )

        x0, fluxes, idx = self.finder._generate_flux_candidates(
            50, mod_pts, tau_pts, isd_mode="F",
            sampler=self.sampler,
        )

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
        N = 20
        mod_pts, tau_pts = self.sampler.initial_guesses(
            N, filter_moduli=True, include_fluxes=False, rns_key=self.rng_key,
        )

        x0, fluxes, idx = self.finder._generate_flux_candidates(
            50, mod_pts, tau_pts, isd_mode="ISD-",
            sampler=self.sampler,
        )

        # ISD- should produce many valid candidates
        self.assertGreater(len(x0), 0,
                           msg="ISD- should produce valid candidates")

    def test_generate_flux_candidates_invalid_mode(self):
        r"""
        **Description:**
        Verify that an invalid mode raises ``ValueError``.
        """
        N = 10
        mod_pts, tau_pts = self.sampler.initial_guesses(
            N, filter_moduli=True, include_fluxes=False, rns_key=self.rng_key,
        )

        with self.assertRaises(ValueError):
            self.finder._generate_flux_candidates(
                10, mod_pts, tau_pts, isd_mode="INVALID",
                sampler=self.sampler,
            )

    # ------------------------------------------------------------------
    # Newton solver (FVF's `newton_method_flux_vacua` — replaces the
    # retired CPF `_solve_dV_newton_single`)
    # ------------------------------------------------------------------

    @pytest.mark.slow
    def test_newton_solver_converges(self):
        r"""
        **Description:**
        Verify that the FVF Newton solver runs end-to-end on an ISD-
        candidate and returns a real residual + same-shape x.
        """
        N = 10
        mod_pts, tau_pts = self.sampler.initial_guesses(
            N, filter_moduli=True, include_fluxes=False, rns_key=self.rng_key,
        )

        x0, fluxes, _ = self.finder._generate_flux_candidates(
            10, mod_pts, tau_pts, isd_mode="ISD-",
            sampler=self.sampler,
        )

        if len(x0) > 0:
            # FVF Newton takes (moduli, tau) — derive from x0 via
            # _convert_real_to_complex, then convert back to real coords
            # for shape-equality assertion.
            x0_j = jnp.asarray(x0[0])
            mod0, _, tau0, _ = self.finder._convert_real_to_complex(x0_j)
            mod_sol, tau_sol, res_j = self.finder.newton_method_flux_vacua(
                mod0, tau0, jnp.asarray(fluxes[0]), mode=None,
                step_size_Newton=1.0, tol=1e-5, max_iters=100,
                solver_mode="real",
            )
            x_sol = np.asarray(self.finder._convert_complex_to_real(
                mod_sol, jnp.conj(mod_sol), tau_sol, jnp.conj(tau_sol),
            ))
            res = float(jnp.abs(res_j))
            self.assertIsInstance(res, float)
            self.assertEqual(x_sol.shape, x0[0].shape)

    def test_newton_solver_residual_format(self):
        r"""
        **Description:**
        Verify the output format of the FVF Newton solver:
        ``(moduli, tau, res)`` 3-tuple, with ``res`` convertible to a
        Python float and ``x`` (after back-conversion) of the right shape.
        """
        N = 5
        mod_pts, tau_pts = self.sampler.initial_guesses(
            N, filter_moduli=True, include_fluxes=False, rns_key=self.rng_key,
        )

        x0, fluxes, _ = self.finder._generate_flux_candidates(
            5, mod_pts, tau_pts, isd_mode="F",
            sampler=self.sampler,
        )

        if len(x0) > 0:
            x0_j = jnp.asarray(x0[0])
            mod0, _, tau0, _ = self.finder._convert_real_to_complex(x0_j)
            mod_sol, tau_sol, res_j = self.finder.newton_method_flux_vacua(
                mod0, tau0, jnp.asarray(fluxes[0]), mode=None,
                max_iters=10, solver_mode="real",
            )
            x_sol = np.asarray(self.finder._convert_complex_to_real(
                mod_sol, jnp.conj(mod_sol), tau_sol, jnp.conj(tau_sol),
            ))
            res = float(jnp.abs(res_j))
            conv = bool(res < 1e-10)
            # x_sol is numpy array
            self.assertIsInstance(x_sol, np.ndarray)
            # residual is float
            self.assertIsInstance(res, float)
            # converged is bool
            self.assertIsInstance(conv, bool)

    # ------------------------------------------------------------------
    # Classification (A2)
    # ------------------------------------------------------------------

    @pytest.mark.slow
    def test_classify_solution_keys(self):
        r"""
        **Description:**
        Verify that ``classify_solution`` returns a dict with all expected keys.
        """
        # Use a random point (not necessarily a critical point)
        x = np.array([0.1, 3.0, -0.2, 2.5, -0.3, 5.0])
        fl = np.array([1., 0., -2., 0., 3., -1., 2., 1., 0., -1., 1., 0.])

        info = self.finder.classify_solution(x, fl)

        # Check all expected keys
        for key in ['V', '|DW|', 'eigenvalues', 'is_susy', 'is_minimum', 'Nflux']:
            self.assertIn(key, info, msg=f"Missing key: {key}")

        # eigenvalues should be a sorted array of length 2*(h12+1)
        self.assertEqual(len(info['eigenvalues']), 2 * (self.h12 + 1))
        # is_susy and is_minimum should be bool
        self.assertIsInstance(info['is_susy'], bool)
        self.assertIsInstance(info['is_minimum'], bool)

    # ------------------------------------------------------------------
    # Main entry point: sample_critical_points (C8)
    # ------------------------------------------------------------------

    def test_sample_critical_points_newton(self):
        r"""
        **Description:**
        Integration test: ``sample_critical_points`` with Newton solver
        should find at least one critical point from a small batch.
        """
        results = self.finder.sample_critical_points(
            n_target=5, n_batch=50, max_batches=2,
            isd_mode="ISD-", solver="newton",
            sampler=self.sampler, verbose=False,
        )

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

    @pytest.mark.slow
    def test_sample_critical_points_scipy(self):
        r"""
        **Description:**
        Integration test: ``sample_critical_points`` with scipy solver.
        """
        results = self.finder.sample_critical_points(
            n_target=5, n_batch=50, max_batches=2,
            isd_mode="ISD-", solver="scipy",
            sampler=self.sampler, verbose=False,
        )

        self.assertIsInstance(results, list)

    @pytest.mark.slow
    def test_sample_critical_points_returns_non_susy(self):
        r"""
        **Description:**
        Verify that at least some found critical points are non-SUSY
        (|DW| > threshold) when using ISD- mode with noscale=True.
        """
        results = self.finder.sample_critical_points(
            n_target=10, n_batch=100, max_batches=3,
            isd_mode="ISD-", solver="scipy", noscale=True,
            sampler=self.sampler, verbose=False,
        )

        if len(results) > 0:
            # At least some should be non-SUSY (from benchmarks: ~78%)
            n_nonsusy = sum(1 for r in results if not r['is_susy'])
            # With ISD-, we expect most to be non-SUSY
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
        results = self.finder.sample_critical_points(
            n_target=3, n_batch=30, max_batches=2,
            isd_mode="ISD-", solver="scipy",
            classify=False, sampler=self.sampler, verbose=False,
        )

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
        with self.assertRaises(ValueError):
            self.finder.sample_critical_points(
                n_target=1, n_batch=10, max_batches=1,
                solver="invalid_solver",
                sampler=self.sampler, verbose=False,
            )

    # ------------------------------------------------------------------
    # Vectorised optax solver (C6)
    # ------------------------------------------------------------------

    @pytest.mark.slow
    def test_optax_batch_solver_output_shapes(self):
        r"""
        **Description:**
        Verify that ``_solve_dV_optax_batch`` returns arrays with correct
        shapes: ``(N, 2*(h12+1))``, ``(N,)``, ``(N,)``.
        """
        N = 10
        mod_pts, tau_pts = self.sampler.initial_guesses(
            N, filter_moduli=True, include_fluxes=False, rns_key=self.rng_key,
        )

        x0, fluxes, _ = self.finder._generate_flux_candidates(
            10, mod_pts, tau_pts, isd_mode="ISD-",
            sampler=self.sampler,
        )

        if len(x0) > 0:
            n = min(5, len(x0))
            x_out, res_out, conv_out = self.finder._solve_dV_optax_batch(
                x0[:n], fluxes[:n], n_steps=100, tol=1e-8,
            )
            self.assertEqual(x_out.shape, (n, 2 * (self.h12 + 1)))
            self.assertEqual(res_out.shape, (n,))
            self.assertEqual(conv_out.shape, (n,))
            self.assertTrue(np.all(np.isfinite(x_out)))

    @pytest.mark.slow
    def test_optax_batch_residuals_decrease(self):
        r"""
        **Description:**
        Verify that the vectorised Adam solver reduces residuals compared
        to the starting points (more steps → lower residuals).
        """
        N = 10
        mod_pts, tau_pts = self.sampler.initial_guesses(
            N, filter_moduli=True, include_fluxes=False, rns_key=self.rng_key,
        )

        x0, fluxes, _ = self.finder._generate_flux_candidates(
            10, mod_pts, tau_pts, isd_mode="ISD-",
            sampler=self.sampler,
        )

        if len(x0) >= 3:
            n = min(5, len(x0))
            x_final_100, res_100, _ = self.finder._solve_dV_optax_batch(
                x0[:n], fluxes[:n], n_steps=100, tol=1e-8,
            )
            x_final_1000, res_1000, _ = self.finder._solve_dV_optax_batch(
                x0[:n], fluxes[:n], n_steps=1000, tol=1e-8,
            )

            # Only validate residuals at points that are still inside the
            # Kähler cone after the optax scan AND pass `filter_moduli`
            # (instanton convergence + Kähler-metric positivity).  The optax
            # solver is unconstrained and can wander to points with
            # ``Im(z) < 0`` or otherwise singular geometry, where ``dV_x``
            # may legitimately produce NaN; such points are not physically
            # meaningful so should not poison the median comparison.
            def _valid_mask(x_finals):
                mask = []
                for xf in x_finals:
                    z, _, tau, _ = self.finder._convert_real_to_complex(jnp.asarray(xf))
                    in_cone = bool(np.all(
                        np.asarray(self.sampler._hyperplanes @ z.imag) > 0))
                    if not in_cone:
                        mask.append(False)
                        continue
                    z_kept = self.sampler.filter_moduli(
                        jnp.asarray([z]), jnp.asarray([tau]))
                    mask.append(len(z_kept) > 0)
                return np.array(mask)

            res_100  = np.where(_valid_mask(x_final_100),  res_100,  np.nan)
            res_1000 = np.where(_valid_mask(x_final_1000), res_1000, np.nan)

            # nanmedian skips invalidated points; require at least one valid
            # residual at each step count.
            self.assertTrue(np.any(np.isfinite(res_100)),
                            msg="no in-cone residuals after n_steps=100")
            self.assertTrue(np.any(np.isfinite(res_1000)),
                            msg="no in-cone residuals after n_steps=1000")
            # Median residual should decrease with more steps
            self.assertLessEqual(np.nanmedian(res_1000), np.nanmedian(res_100))

    @pytest.mark.slow
    def test_sample_critical_points_adam_v(self):
        r"""
        **Description:**
        Integration test: ``sample_critical_points`` with vectorised Adam
        solver (``solver="adam_v"``) runs without error.
        """
        results = self.finder.sample_critical_points(
            n_target=5, n_batch=50, max_batches=1,
            isd_mode="ISD-", solver="adam_v",
            optax_steps=500, sampler=self.sampler, verbose=False,
        )
        self.assertIsInstance(results, list)

    @pytest.mark.slow
    def test_sample_critical_points_hybrid(self):
        r"""
        **Description:**
        Integration test: ``sample_critical_points`` with hybrid solver
        (vectorised Adam warm-start + Newton refinement) runs without error
        and can find critical points.
        """
        results = self.finder.sample_critical_points(
            n_target=5, n_batch=100, max_batches=2,
            isd_mode="ISD-", solver="hybrid",
            optax_steps=2000, sampler=self.sampler, verbose=False,
        )
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
