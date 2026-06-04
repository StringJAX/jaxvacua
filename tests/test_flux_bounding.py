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

"""Tests for the flux-bounding workflow.

Purpose
-------
Validate the ``bounded_fluxes`` algorithm, local/global bounds, flux
candidate processing and cluster round-trip helpers.

Main public API
---------------
- ``TestFluxBounding``: checks construction, bound evaluation, candidate
  enumeration and filtering behaviour.
- ``TestClusterRoundTrip``: checks export/import behaviour for cluster search
  chunks.

Design notes
------------
Tests keep the geometry small so the performance-sensitive code paths are
covered without requiring large enumeration jobs.
"""

import sys, os, warnings
import jax
import numpy as np
import jax.numpy as jnp
import pytest
from functools import partial
from util import *

sys.path.append("./../")
import jaxvacua
from jaxvacua.flux_bounding import bounded_fluxes


# ---------------------------------------------------------------------------
# Diagnostic autouse fixture (temporary).  Writes START/END markers around
# every test directly to fd 2, bypassing pytest's stdout/stderr capture
# (combine with ``-s`` and ``PYTHONUNBUFFERED=1`` on the CI command), so
# CI logs show exactly which test was running when a hang occurs.  Flip
# ``autouse=True`` -> ``autouse=False`` to disable once diagnosed.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _ci_test_marker(request):
    os.write(2, f">>> START {request.node.nodeid}\n".encode())
    yield
    os.write(2, f">>> END   {request.node.nodeid}\n".encode())


# ==============================================================================
#  TestFluxBounding
# ==============================================================================

class TestFluxBounding(TestCase):
    r"""
    **Description:**
    Test suite for the :class:`bounded_fluxes` class in
    :mod:`jaxvacua.flux_bounding`, which implements the flux-bounding
    algorithm of arXiv:2501.03984 for systematic enumeration of Type IIB
    flux vacua in finite regions of moduli space.

    The tests cover the constructor, global eigenvalue bounds, bounding box
    computation, h-candidate enumeration, Newton refinement at a known SUSY
    vacuum, ISD refinement, flux splitting, and tadpole computation.

    Attributes:
        model (jaxvacua.FluxVacuaFinder): Flux vacua finder model with
            ``h12=2``, ``model_ID=1``, ``model_type="KS"``,
            ``maximum_degree=0``.
        sampler (jaxvacua.data_sampler): Data sampler with moduli bounds
            ``(2., 5.)``, dilaton bounds ``(2., 10.)``, axion bounds
            ``(-0.5, 0.5)``, and ``seed=42``.
        bf (bounded_fluxes): Bounded-fluxes instance with ``Nmax=4``.
        h12 (int): Number of complex structure moduli.
        n_fluxes (int): Length of a half-flux vector, :math:`2(h^{1,2}+1)`.
        dimension_H3 (int): Dimension of the :math:`H^3_+` sector,
            :math:`h^{1,2}+1`.
        f_solution (jnp.ndarray): Known SUSY-minimum flux vector.
        zsol (jnp.ndarray): Complex structure moduli at the known SUSY
            minimum.
        tausol (complex): Axio-dilaton at the known SUSY minimum.
    """

    # --------------------------------------------------------------------------

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        super().setUpClass()
        import sys
        def _mark(s): print(f">>> [setUpClass] {s}", file=sys.stderr, flush=True)
        _mark("start")
        cls.h12 = 2
        cls.model = jaxvacua.FluxVacuaFinder(
            h12=cls.h12, model_ID=1, model_type="KS", maximum_degree=2,
        )
        _mark("model built")
        cls.model.lcs_tree.a_matrix = jnp.array([[4.5, 1.5], [1.5, 0.]])
        cls.sampler = jaxvacua.data_sampler(
            cls.model, moduli_bounds=(2., 3.), dilaton_bounds=(2., 5.),
            axion_bounds=(-0.5, 0.5), seed=42,
        )
        _mark("sampler built")
        cls.bf = bounded_fluxes(cls.model, sampler=cls.sampler, Nmax=4)
        _mark("bf built")
        """
        cls.h12 = 2

        # Build model and override the a_matrix
        # maximum_degree=2 includes instanton corrections needed for
        # Newton convergence at the known SUSY solution
        cls.model = jaxvacua.FluxVacuaFinder(
            h12=cls.h12, model_ID=1, model_type="KS", maximum_degree=2,prange=20
        )
        cls.model.lcs_tree.a_matrix = jnp.array([[4.5, 1.5], [1.5, 0.]])

        # Build sampler
        cls.sampler = jaxvacua.data_sampler(
            cls.model,
            moduli_bounds=(2., 3.),
            dilaton_bounds=(2., 5.),
            axion_bounds=(-0.5, 0.5),
            seed=42,
        )

        # Build bounded_fluxes instance
        cls.bf = bounded_fluxes(cls.model, sampler=cls.sampler, Nmax=4)
        """
        # Convenience attributes
        cls.n_fluxes = cls.model.n_fluxes          # = 2*(h12+1) = 6
        cls.dimension_H3 = cls.model.dimension_H3  # = h12+1 = 3

        # Known SUSY minimum solution (from test_css.py)
        cls.f_solution = jnp.array(
            [7, 3, -24, 0, -16, 50, 0, 3, -4, 0, 0, 0], dtype=float
        )
        u1sol = 2.74215479602462524879172086700112955631 * 1j
        u2sol = 2.05661613496943436323419976712599580262 * 1j
        tausol = 6.85540179778358427172610564536555609784 * 1j

        cls.zsol = jnp.array([u1sol, u2sol])
        cls.tausol = tausol

        _mark("done")

    # ==========================================================================
    #  1. Constructor
    # ==========================================================================

    def test_constructor_Nmax(self):
        r"""**Description:**
        Verify that the ``Nmax`` attribute is correctly set from the
        constructor argument.
        """
        self.assertEqual(
            self.bf.Nmax, 4,
            msg="bf.Nmax should equal the value passed to the constructor",
        )

    def test_constructor_n_fluxes(self):
        r"""**Description:**
        Verify that ``n_fluxes`` equals :math:`2(h^{1,2}+1)`.
        """
        expected = 2 * (self.h12 + 1)
        self.assertEqual(
            self.bf.n_fluxes, expected,
            msg=f"bf.n_fluxes should be 2*(h12+1) = {expected}",
        )

    def test_constructor_dimension_H3(self):
        r"""**Description:**
        Verify that ``dimension_H3`` equals :math:`h^{1,2}+1`.
        """
        expected = self.h12 + 1
        self.assertEqual(
            self.bf.dimension_H3, expected,
            msg=f"bf.dimension_H3 should be h12+1 = {expected}",
        )


    # ==========================================================================
    #  2. bounds_initialized Property
    # ==========================================================================

    def test_bounds_not_initialized_initially(self):
        r"""**Description:**
        Before any call to :func:`compute_eigenvalue_bounds` or
        :func:`compute_bounding_box`, the :attr:`bounds_initialized`
        property should return ``False``.
        """
        # Create a fresh instance to avoid pollution from other tests
        bf_fresh = bounded_fluxes(self.model, sampler=self.sampler, Nmax=4)
        self.assertFalse(
            bf_fresh.bounds_initialized,
            msg="bounds_initialized should be False before any bound computation",
        )

    @pytest.mark.slow
    def test_bounds_initialized_after_compute(self):
        r"""**Description:**
        After calling :func:`compute_eigenvalue_bounds`, the
        :attr:`bounds_initialized` property should return ``True``.
        """
        # Use a small sample to keep the test fast
        bf_test = bounded_fluxes(self.model, sampler=self.sampler, Nmax=4)
        bf_test.compute_eigenvalue_bounds(n_sample=50, verbose=False)
        self.assertTrue(
            bf_test.bounds_initialized,
            msg="bounds_initialized should be True after compute_eigenvalue_bounds",
        )


    # ==========================================================================
    #  3. compute_eigenvalue_bounds
    # ==========================================================================

    def test_compute_eigenvalue_bounds_returns_positive(self):
        r"""**Description:**
        Verifies that :func:`compute_eigenvalue_bounds` returns a 3-tuple
        ``(h1_box, h2_box, h_box)`` of positive floats, and that the
        global eigenvalue extrema ``lambda_max_gl`` and ``mu_min_gl``
        are positive after the call.

        The bounding box radii are defined as

        .. math::
            h_{1,\rm box} = \sqrt{\frac{N_{\max}}{s_{\min}\,\tilde\mu_{\min}^{\rm gl}}}\,,\quad
            h_{2,\rm box} = \sqrt{\frac{N_{\max}}{s_{\min}\,\mu_{\min}^{\rm gl}}}\,,\quad
            h_{\rm box} = \sqrt{\frac{2\,\lambda_{\max}^{\rm gl}\,N_{\max}}{s_{\min}}}\,.
        """
        bf_test = bounded_fluxes(self.model, sampler=self.sampler, Nmax=4)
        h1_box, h2_box, h_box = bf_test.compute_eigenvalue_bounds(
            n_sample=100, verbose=False,
        )

        # All box dimensions must be positive floats
        self.assertIsInstance(h1_box, float)
        self.assertIsInstance(h2_box, float)
        self.assertIsInstance(h_box, float)
        self.assertGreater(h1_box, 0., msg="h1_box must be positive")
        self.assertGreater(h2_box, 0., msg="h2_box must be positive")
        self.assertGreater(h_box, 0., msg="h_box must be positive")

        # Global eigenvalue extrema must be positive after computation
        self.assertGreater(
            bf_test.lambda_max_gl, 0.,
            msg="lambda_max_gl must be positive after compute_eigenvalue_bounds",
        )
        self.assertGreater(
            bf_test.mu_min_gl, 0.,
            msg="mu_min_gl must be positive after compute_eigenvalue_bounds",
        )


    # ==========================================================================
    #  4. compute_bounding_box
    # ==========================================================================

    def test_compute_bounding_box_returns_positive(self):
        r"""**Description:**
        Verifies that :func:`compute_bounding_box` accepts explicit
        ``moduli_sample`` and ``tau_sample`` arrays and returns three
        positive floats.

        A small batch of random moduli is drawn from the sampler and
        passed directly to :func:`compute_bounding_box`.
        """
        bf_test = bounded_fluxes(self.model, sampler=self.sampler, Nmax=4)

        # Draw a small sample of moduli and tau
        #moduli_sample = self.sampler.get_complex_moduli(50)
        #tau_sample = self.sampler.get_complex_tau(50)
        N = 20
        moduli_sample, tau_sample = self.sampler.initial_guesses(N,filter_moduli=True,include_fluxes=False)

        h1_box, h2_box, h_box = bf_test.compute_bounding_box(
            moduli_sample, tau_sample,
        )

        # All three box dimensions must be positive
        self.assertGreater(h1_box, 0., msg="h1_box must be positive")
        self.assertGreater(h2_box, 0., msg="h2_box must be positive")
        self.assertGreater(h_box, 0., msg="h_box must be positive")


    # ==========================================================================
    #  5. get_h_candidates
    # ==========================================================================

    def test_get_h_candidates_shape_and_type(self):
        r"""**Description:**
        Verifies that the streaming h-candidate generator produces 2D
        integer chunks of shape ``(N, n_fluxes)`` after the bounding box
        has been computed.

        Each row of the returned array is a candidate NSNS-flux vector
        :math:`h = [h_1 \mid h_2]` satisfying the :math:`L^2`-norm
        constraints :math:`\|h_1\|^2 \leq h_{1,\rm box}^2` and
        :math:`\|h_2\|^2 \leq h_{2,\rm box}^2`.

        The full enumeration via :func:`get_h_candidates` can materialise
        billions of candidates for typical bounding boxes (up to 96 GB),
        so this test uses the streaming path :func:`_iter_h_chunks_streaming`
        to read only the first chunk — enough to validate shape/dtype
        without exhausting memory.
        """
        # Ensure bounding box is computed
        bf_test = bounded_fluxes(self.model, sampler=self.sampler, Nmax=4)
        bf_test.compute_eigenvalue_bounds(n_sample=100, verbose=False)
        _, _, h_box = bf_test.get_h_box()

        # Stream the first chunk of h-candidates instead of materialising all
        h_iter = bf_test._iter_h_chunks_streaming(h_box ** 2)
        h_chunk, _ = next(h_iter)

        # Must be a 2D numpy integer array
        self.assertIsInstance(h_chunk, np.ndarray)
        self.assertEqual(
            h_chunk.ndim, 2,
            msg="h chunk must be a 2D array",
        )
        self.assertTrue(
            np.issubdtype(h_chunk.dtype, np.integer),
            msg="h chunk must have integer dtype",
        )

        # Shape: (N, n_fluxes) with N > 0
        N, n_fl = h_chunk.shape
        self.assertGreater(N, 0, msg="Must have at least one h-candidate")
        self.assertEqual(
            n_fl, self.n_fluxes,
            msg=f"h chunk columns should equal n_fluxes={self.n_fluxes}",
        )

    def test_get_h_candidates_norm_bounds(self):
        r"""**Description:**
        Verifies that every streamed h-candidate satisfies the individual
        :math:`L^2`-norm bounds on :math:`h_1` and :math:`h_2`.

        Uses :func:`_iter_h_chunks_streaming` to sample a few chunks
        without materialising the full Cartesian product (which can be
        tens of GB).
        """
        bf_test = bounded_fluxes(self.model, sampler=self.sampler, Nmax=4)
        bf_test.compute_eigenvalue_bounds(n_sample=100, verbose=False)

        h1_box, h2_box, h_box = bf_test.get_h_box()
        dim = self.dimension_H3

        # Iterate through a handful of chunks, validating each.  No need to
        # check all of them — the constraint is enforced at construction
        # time inside the generator.
        h_iter = bf_test._iter_h_chunks_streaming(h_box ** 2)
        n_checked = 0
        for h_chunk, _ in h_iter:
            h1_all = h_chunk[:, :dim]
            h2_all = h_chunk[:, dim:]

            h1_norms_sq = np.sum(h1_all ** 2, axis=1)
            h2_norms_sq = np.sum(h2_all ** 2, axis=1)

            self.assertAllTrue(
                h1_norms_sq <= h1_box ** 2 + 1e-10,
                msg="All h1 sub-vectors must satisfy ||h1||^2 <= h1_box^2",
            )
            self.assertAllTrue(
                h2_norms_sq <= h2_box ** 2 + 1e-10,
                msg="All h2 sub-vectors must satisfy ||h2||^2 <= h2_box^2",
            )

            n_checked += len(h_chunk)
            if n_checked >= 10_000:
                break


    # ==========================================================================
    #  6. newton_refine_batch
    # ==========================================================================

    @pytest.mark.slow
    def test_newton_refine_batch_shapes(self):
        r"""**Description:**
        Verifies that :func:`newton_refine_batch` returns the correct
        output shapes ``(N, h12)``, ``(N,)``, and ``(N,)`` for
        ``moduli_out``, ``tau_out``, and ``residuals`` respectively.
        """
        N = 2  # batch size
        h12 = self.h12

        # Create a small batch by repeating the known solution
        moduli_batch = jnp.tile(self.zsol, (N, 1))
        tau_batch = jnp.array([self.tausol] * N)
        flux_batch = jnp.tile(self.f_solution, (N, 1))

        moduli_out, tau_out, residuals = self.bf.newton_refine_batch(
            moduli_batch, tau_batch, flux_batch,
            step_size=1.0, tol=1e-10, max_iters=100,
        )

        # Check output shapes
        chex.assert_shape(moduli_out, (N, h12))
        chex.assert_shape(tau_out, (N,))
        chex.assert_shape(residuals, (N,))

    @pytest.mark.slow
    def test_newton_refine_batch_convergence(self):
        r"""**Description:**
        Verifies convergence of :func:`newton_refine_batch` using the
        known SUSY vacuum solution:

        .. math::
            f = [7, 3, -24, 0, -16, 50, 0, 3, -4, 0, 0, 0]\,,\quad
            z \approx (2.742\,i,\; 2.057\,i)\,,\quad
            \tau \approx 6.855\,i\,.

        Starting from the known solution, Newton refinement should converge
        to a residual :math:`\sum |D_I W| < 10^{-8}`, and the converged
        moduli and axio-dilaton should agree with the starting point to
        high precision.
        """
        N = 1

        # Start at the known SUSY vacuum
        moduli_batch = self.zsol[None, :]     # shape (1, h12)
        tau_batch = jnp.array([self.tausol])  # shape (1,)
        flux_batch = self.f_solution[None, :] # shape (1, 2*n_fluxes)

        moduli_out, tau_out, residuals = self.bf.newton_refine_batch(
            moduli_batch, tau_batch, flux_batch,
            step_size=1.0, tol=1e-12, max_iters=100,
        )

        # Residual should be extremely small at the known vacuum
        self.assertAllTrue(
            residuals < 1e-8,
            msg="Newton refinement should converge to small residual at known vacuum",
        )

        # Converged moduli should match the starting point (already at minimum)
        self.assertAllClose(
            moduli_out[0], self.zsol, atol=1e-6,
            msg="Converged moduli should match the known SUSY vacuum",
        )

        # Converged tau should match the starting point
        self.assertAllClose(
            tau_out[0], self.tausol, atol=1e-6,
            msg="Converged tau should match the known SUSY vacuum",
        )

    def test_newton_refine_batch_from_perturbed_start(self):
        r"""**Description:**
        Verifies that Newton refinement converges to the known SUSY vacuum
        even when starting from a slightly perturbed initial guess.

        The starting moduli and axio-dilaton are shifted by a small
        imaginary perturbation, and the **damped** Newton solver
        (``step_size=0.5``) should recover the exact vacuum to high
        accuracy.  Full-step Newton (``step_size=1.0``) overshoots from
        this perturbation magnitude and diverges to NaN — production
        code in ``vacuum_promotion.py`` uses damped Newton for the
        same reason.
        """
        N = 1

        # Perturb the known solution slightly in the imaginary direction
        z_start = self.zsol + 0.05j
        tau_start = self.tausol + 0.1j

        moduli_batch = z_start[None, :]
        tau_batch = jnp.array([tau_start])
        flux_batch = self.f_solution[None, :]

        moduli_out, tau_out, residuals = self.bf.newton_refine_batch(
            moduli_batch, tau_batch, flux_batch,
            step_size=0.5, tol=1e-12, max_iters=200,
        )

        # Should converge to a small residual
        self.assertAllTrue(
            residuals < 1e-6,
            msg="Newton refinement should converge from a perturbed start",
        )

        # Converged moduli should be close to the known vacuum
        self.assertAllClose(
            moduli_out[0], self.zsol, atol=1e-4,
            msg="Converged moduli should recover the known SUSY vacuum",
        )


    # ==========================================================================
    #  7. isd_refine_batch
    # ==========================================================================

    @pytest.mark.slow
    def test_isd_refine_batch_shapes(self):
        r"""**Description:**
        Verifies that :func:`isd_refine_batch` returns arrays with
        correct shapes.  The output is a 3-tuple
        ``(mod_out, tau_out, flux_out)``.

        When no candidates pass all constraints, the output arrays are
        empty.  When candidates do pass, the shapes are
        ``(K, h12)``, ``(K,)``, ``(K, 2*n_fluxes)`` where ``K`` is the
        number of surviving candidates.
        """
        # Ensure bounding box is computed first
        bf_test = bounded_fluxes(self.model, sampler=self.sampler, Nmax=4)
        bf_test.compute_eigenvalue_bounds(n_sample=100, verbose=False)

        # Use the known h-flux from the known solution: h = f_solution[n_fluxes:]
        h_known = self.f_solution[self.n_fluxes:]  # shape (n_fluxes,)
        h_batch = h_known[None, :]                  # shape (1, n_fluxes)

        # Use a small set of moduli starting points
        n_pts = 2
        #moduli_pts = self.sampler.get_complex_moduli(n_pts)
        #tau_pts = self.sampler.get_complex_tau(n_pts)
        moduli_pts, tau_pts = self.sampler.initial_guesses(n_pts,filter_moduli=True,include_fluxes=False)

        mod_out, tau_out, flux_out = bf_test.isd_refine_batch(
            jnp.array(h_batch, dtype=float),
            jnp.array(moduli_pts),
            jnp.array(tau_pts),
            n_iters=2,
            h_sub_batch=10,
        )

        # Output should be numpy arrays (possibly empty)
        self.assertIsInstance(mod_out, np.ndarray)
        self.assertIsInstance(tau_out, np.ndarray)
        self.assertIsInstance(flux_out, np.ndarray)

        # If non-empty, shapes should be consistent
        if mod_out.size > 0:
            K = mod_out.shape[0]
            self.assertEqual(mod_out.shape[-1], self.h12)
            self.assertEqual(flux_out.shape[-1], 2 * self.n_fluxes)
            self.assertEqual(tau_out.shape[0], K)
            self.assertEqual(flux_out.shape[0], K)


    # ==========================================================================
    #  8. get_fh
    # ==========================================================================

    def test_get_fh_split(self):
        r"""**Description:**
        Verifies that :func:`get_fh` correctly splits a full flux vector
        ``[f | h]`` of length ``2 * n_fluxes`` into its RR-flux ``f``
        and NSNS-flux ``h`` components, each of length ``n_fluxes``.
        """
        flux = self.f_solution  # length 2*n_fluxes = 12

        f_part, h_part = self.bf.get_fh(flux)

        # Each part should have length n_fluxes
        self.assertEqual(
            len(f_part), self.n_fluxes,
            msg=f"f should have length n_fluxes={self.n_fluxes}",
        )
        self.assertEqual(
            len(h_part), self.n_fluxes,
            msg=f"h should have length n_fluxes={self.n_fluxes}",
        )

        # Reconstruction: concatenation should recover the original
        reconstructed = np.concatenate([f_part, h_part])
        self.assertAllClose(
            reconstructed, np.asarray(flux).real, atol=1e-15,
            msg="Concatenating f and h should recover the original flux vector",
        )

    def test_get_fh_known_values(self):
        r"""**Description:**
        Verifies the numerical values of the split for the known flux
        vector ``f_solution = [7, 3, -24, 0, -16, 50, 0, 3, -4, 0, 0, 0]``.

        The RR-flux is ``f = [7, 3, -24, 0, -16, 50]`` and the NSNS-flux
        is ``h = [0, 3, -4, 0, 0, 0]``.
        """
        f_part, h_part = self.bf.get_fh(self.f_solution)

        expected_f = np.array([7., 3., -24., 0., -16., 50.])
        expected_h = np.array([0., 3., -4., 0., 0., 0.])

        self.assertAllClose(
            f_part, expected_f, atol=1e-15,
            msg="RR-flux f should match expected values",
        )
        self.assertAllClose(
            h_part, expected_h, atol=1e-15,
            msg="NSNS-flux h should match expected values",
        )


    # ==========================================================================
    #  9. compute_tadpole_batch
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_compute_tadpole_batch_shape(self):
        r"""**Description:**
        Verifies that :func:`compute_tadpole_batch` returns an array of
        shape ``(N,)`` for a batch of ``N`` flux vectors.
        """
        N = 3
        flux_batch = jnp.tile(self.f_solution, (N, 1))  # (N, 2*n_fluxes)

        tadpoles = self.variant(self.bf.compute_tadpole_batch)(flux_batch)

        chex.assert_shape(tadpoles, (N,))

    @chex.variants(with_jit=True, without_jit=True)
    def test_compute_tadpole_batch_positive(self):
        r"""**Description:**
        Verifies that the tadpole :math:`N_{\rm flux} = |f^T \Sigma h|`
        is a positive real number for the known flux solution.
        """
        flux_batch = self.f_solution[None, :]  # (1, 2*n_fluxes)

        tadpoles = self.variant(self.bf.compute_tadpole_batch)(flux_batch)

        # Tadpole should be a real positive number
        self.assertGreater(
            float(tadpoles[0].real), 0.,
            msg="Tadpole N_flux should be positive for the known solution",
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_compute_tadpole_batch_consistency(self):
        r"""**Description:**
        Verifies that :func:`compute_tadpole_batch` gives the same result
        as a single call to ``model.tadpole`` for each element in the batch.
        """
        N = 3

        # Create distinct flux vectors by scaling the known solution
        flux_batch = jnp.stack([
            self.f_solution,
            2.0 * self.f_solution,
            0.5 * self.f_solution,
        ])  # (3, 2*n_fluxes)

        # Batch computation
        tadpoles_batch = self.variant(self.bf.compute_tadpole_batch)(flux_batch)

        # Single-element computation for reference
        for i in range(N):
            tad_single = self.model.tadpole(flux_batch[i])
            self.assertAllClose(
                tadpoles_batch[i], tad_single, atol=1e-12,
                msg=f"Batch tadpole[{i}] should match single-element tadpole",
            )

    # ==========================================================================
    #  10. Flux utility helpers
    # ==========================================================================

    def test_get_nflux(self):
        r"""
        **Description:**
        Verify that :func:`get_nflux` returns the D3-tadpole charge
        :math:`N_{\rm flux} = |f^T \Sigma h|` as a positive scalar,
        and agrees with :func:`model.tadpole`.
        """
        Nfl = self.bf.get_nflux(self.f_solution)
        # Must be a positive scalar
        self.assertGreater(float(Nfl), 0.)
        # Must agree with model.tadpole
        tad = float(self.model.tadpole(self.f_solution).real)
        self.assertAllClose(Nfl, tad, atol=1e-10)

    def test_get_flux_split(self):
        r"""
        **Description:**
        Verify that :func:`get_flux_split` splits a half-flux vector of
        length :math:`n_{\rm fluxes}` into two sub-vectors of length
        :math:`\dim H^3 = h^{1,2}+1`.
        """
        # Extract h from the known solution
        _, h = self.bf.get_fh(self.f_solution)
        h1, h2 = self.bf.get_flux_split(h)
        # Each sub-vector has length dimension_H3
        self.assertEqual(len(h1), self.dimension_H3)
        self.assertEqual(len(h2), self.dimension_H3)

    def test_get_subvector(self):
        r"""
        **Description:**
        Verify that :func:`get_subvector` returns four sub-vectors
        ``(h1, h2, f1, f2)`` from a full flux vector, each of length
        :math:`\dim H^3`.
        """
        h1, h2, f1, f2 = self.bf.get_subvector(self.f_solution)
        for vec, name in [(h1, 'h1'), (h2, 'h2'), (f1, 'f1'), (f2, 'f2')]:
            self.assertEqual(len(vec), self.dimension_H3,
                             msg=f"{name} has length {len(vec)}, expected {self.dimension_H3}")

    def test_compute_norm(self):
        r"""
        **Description:**
        Verify that :func:`compute_norm` returns the squared Euclidean norm
        :math:`\|v\|^2 = \sum_i v_i^2`.
        """
        v = np.array([3., 4.])
        # ||[3,4]||² = 25
        self.assertAllClose(self.bf.compute_norm(v), 25.0, atol=1e-14)

    # ==========================================================================
    #  11. Eigenvalue computation
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_compute_evs(self):
        r"""
        **Description:**
        Verify that :func:`compute_evs` returns a 5-tuple of scalar eigenvalue
        quantities :math:`(\lambda_{\max}, \mu_{\min}, \mu_{\max},
        \tilde\mu_{\min}, \tilde\mu_{\max})` for a single moduli point.
        """
        z = jnp.array([0.1 + 3j, -0.2 + 2.5j])
        fn = self.variant(self.bf.compute_evs)
        result = fn(z)
        # Must return 5 scalars
        self.assertEqual(len(result), 5)
        for i, name in enumerate(['lam_max', 'mu_min', 'mu_max', 'tmu_min', 'tmu_max']):
            # Each eigenvalue must be a finite positive scalar
            val = float(result[i])
            self.assertTrue(np.isfinite(val), msg=f"{name} is not finite")
            self.assertGreater(val, 0., msg=f"{name} must be positive")

    @chex.variants(with_jit=True, without_jit=True)
    def test_compute_evs_vmap(self):
        r"""
        **Description:**
        Verify that :func:`compute_evs_vmap` returns a 5-tuple of arrays,
        each of shape ``(N,)``, for a batch of moduli points.
        """
        N = 10
        #moduli_batch = jnp.array(self.sampler.get_complex_moduli(N), dtype=complex)
        moduli_batch, _ = self.sampler.initial_guesses(N,filter_moduli=True,include_fluxes=False)
        fn = self.variant(self.bf.compute_evs_vmap)
        result = fn(moduli_batch)
        self.assertEqual(len(result), 5)
        for i, name in enumerate(['lam_max', 'mu_min', 'mu_max', 'tmu_min', 'tmu_max']):
            # Each must be an array of shape (N,)
            chex.assert_shape(result[i], (N,))

    # ==========================================================================
    #  12. Precompute ISD data
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    @pytest.mark.slow
    def test_precompute_isd_data(self):
        r"""
        **Description:**
        Verify that :func:`precompute_isd_data` returns three arrays
        ``(M0_all, dM_all, dM_c_all)`` with correct shapes for a batch
        of moduli points.

        - ``M0_all``: ISD matrices, shape ``(n_pts, n_fl, n_fl)``
        - ``dM_all``: holomorphic derivative, shape ``(n_pts, n_fl, n_fl, h12)``
        - ``dM_c_all``: antiholomorphic derivative, shape ``(n_pts, n_fl, n_fl, h12)``
        """
        n_pts = 5
        moduli_pts, _ = self.sampler.initial_guesses(n_pts,filter_moduli=True,include_fluxes=False)
        fn = self.variant(self.bf.precompute_isd_data)
        M0, dM, dM_c = fn(moduli_pts)
        nfl = self.n_fluxes
        h12 = self.h12
        # ISD matrix: (n_pts, n_fl, n_fl)
        chex.assert_shape(M0, (n_pts, nfl, nfl))
        # Derivatives: (n_pts, n_fl, n_fl, h12)
        chex.assert_shape(dM, (n_pts, nfl, nfl, h12))
        chex.assert_shape(dM_c, (n_pts, nfl, nfl, h12))

    # ==========================================================================
    #  13. Reset and update methods
    # ==========================================================================

    def test_reset_eigenvalue_bounds(self):
        r"""
        **Description:**
        Verify that :func:`reset_eigenvalue_bounds` clears the cached bounds
        so that :attr:`bounds_initialized` returns ``False``.
        """
        # Create a fresh instance and compute bounds
        bf_tmp = bounded_fluxes(self.model, sampler=self.sampler, Nmax=4)
        bf_tmp.compute_eigenvalue_bounds(50, verbose=False)
        self.assertTrue(bf_tmp.bounds_initialized)
        # Reset must clear the cache
        bf_tmp.reset_eigenvalue_bounds()
        self.assertFalse(bf_tmp.bounds_initialized)

    def test_update_local(self):
        r"""
        **Description:**
        Verify that :func:`update_local` stores local eigenvalue data and
        flux information for a specific field-space point.
        """
        bf_tmp = bounded_fluxes(self.model, sampler=self.sampler, Nmax=4)
        bf_tmp.compute_eigenvalue_bounds(50, verbose=False)
        # update_local stores local state (no return value, just no error)
        bf_tmp.update_local(self.zsol, self.tausol, self.f_solution)

    def test_update_evs(self):
        r"""
        **Description:**
        Verify that :func:`update_evs` computes and caches eigenvalues for
        a specific moduli point.
        """
        bf_tmp = bounded_fluxes(self.model, sampler=self.sampler, Nmax=4)
        bf_tmp.compute_eigenvalue_bounds(50, verbose=False)
        # update_evs stores eigenvalue cache (no error)
        bf_tmp.update_evs(self.zsol)

    # ==========================================================================
    #  14. Bound checking — individual bounds
    # ==========================================================================

    @pytest.mark.slow
    def test_bound_checking_at_solution(self):
        r"""
        **Description:**
        Verify that all individual bound-checking methods return boolean
        results without error when called at the known SUSY solution.

        The bound checks verify local and global consistency of the flux
        vector with the eigenvalue-based bounding box. At a valid vacuum,
        all bounds should be satisfied.
        """
        bf_tmp = bounded_fluxes(self.model, sampler=self.sampler, Nmax=4)
        bf_tmp.compute_eigenvalue_bounds(200, verbose=False)
        bf_tmp.update_local(self.zsol, self.tausol, self.f_solution)

        # Each bound method returns (result_tuple, label_string)
        bound_methods = [
            'bound_s_local', 'bound_s_global',
            'bound_h_local', 'bound_h_global',
            'bound_f_local', 'bound_f_global',
        ]
        for name in bound_methods:
            fn = getattr(bf_tmp, name)
            result = fn()
            # Must return a tuple (checks, label)
            self.assertIsInstance(result, tuple, msg=f"{name} must return a tuple")
            self.assertEqual(len(result), 2, msg=f"{name} must return (checks, label)")

    def test_check_bounds_at_solution(self):
        r"""
        **Description:**
        Verify that :func:`check_bounds` returns a list of bound check results
        for a given (moduli, tau, flux) point.
        """
        bf_tmp = bounded_fluxes(self.model, sampler=self.sampler, Nmax=4)
        bf_tmp.compute_eigenvalue_bounds(200, verbose=False)
        results = bf_tmp.check_bounds(self.zsol, self.tausol, self.f_solution)
        # Must return a list of tuples
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_check_bounds_flat(self):
        r"""
        **Description:**
        Verify that :func:`check_bounds_flat` returns ``(all_pass, results)``
        where ``all_pass`` is a boolean and ``results`` is a list.
        """
        bf_tmp = bounded_fluxes(self.model, sampler=self.sampler, Nmax=4)
        bf_tmp.compute_eigenvalue_bounds(200, verbose=False)
        bf_tmp.update_local(self.zsol, self.tausol, self.f_solution)
        all_pass, results = bf_tmp.check_bounds_flat()
        # all_pass must be boolean
        self.assertIsInstance(all_pass, (bool, np.bool_))
        # results must be a list
        self.assertIsInstance(results, list)

    @chex.variants(with_jit=True, without_jit=True)
    def test_check_bounds_batch(self):
        r"""
        **Description:**
        Verify that :func:`check_bounds_batch` returns a boolean array of
        shape ``(N,)`` for a batch of flux vectors.
        """
        bf_tmp = bounded_fluxes(self.model, sampler=self.sampler, Nmax=4)
        bf_tmp.compute_eigenvalue_bounds(200, verbose=False)

        N = 5
        #moduli_batch = jnp.array(self.sampler.get_complex_moduli(N), dtype=complex)
        #tau_batch = jnp.array(self.sampler.get_complex_tau(N), dtype=complex)
        
        moduli_batch, tau_batch = self.sampler.initial_guesses(N,filter_moduli=True,include_fluxes=False)
        evs_batch = bf_tmp.compute_evs_vmap(moduli_batch)

        # Create a batch of identical flux vectors for testing
        flux_batch = jnp.tile(self.f_solution, (N, 1))

        fn = self.variant(bf_tmp.check_bounds_batch)
        result = fn(
            evs_batch, tau_batch, flux_batch,
            bf_tmp.lambda_max_gl, bf_tmp.mu_min_gl, bf_tmp.mu_max_gl,
            bf_tmp.tilde_mu_min_gl, bf_tmp.tilde_mu_max_gl,
            bf_tmp.dil_min, bf_tmp.dil_max, float(bf_tmp.Nmax),
        )
        # Must return boolean array of shape (N,)
        chex.assert_shape(result, (N,))

    # ==========================================================================
    #  15. Patch membership
    # ==========================================================================

    def test_in_patch_batch(self):
        r"""
        **Description:**
        Verify that :func:`in_patch_batch` returns a boolean array indicating
        whether each ``(moduli, tau)`` pair lies inside the sampler's moduli
        patch.

        ``_in_patch`` enforces a **per-component box** check
        (:math:`\operatorname{Im}(z_i) \in [\texttt{moduli\_lower},
        \texttt{moduli\_upper}]` for every :math:`i`), which matches the
        patch geometry assumed by the flux-bounding pipeline.
        ``initial_guesses`` defaults to ``moduli_sampling_mode="cone"``
        whose per-component bounds can fall outside the box, so we pass
        ``moduli_sampling_mode="box"`` here to sample from the matching
        domain.
        """
        N = 10
        moduli_batch, tau_batch = self.sampler.initial_guesses(
            N,
            filter_moduli=True,
            include_fluxes=False,
            moduli_sampling_mode="box",
        )

        result = self.bf.in_patch_batch(moduli_batch, tau_batch)
        chex.assert_shape(result, (len(moduli_batch),))
        # Points sampled from the box-mode sampler must all be in-patch.
        self.assertTrue(
            np.all(np.asarray(result)),
            msg="Points from box-mode sampler should all be in-patch",
        )

    # ==========================================================================
    #  16. Bounding box convergence
    # ==========================================================================

    def test_compute_bounding_box_converged(self):
        r"""
        **Description:**
        Verify that :func:`compute_bounding_box_converged` returns three
        positive floats and runs without error.

        This method iteratively samples moduli until the bounding box
        dimensions converge within a specified tolerance.
        """
        bf_tmp = bounded_fluxes(self.model, sampler=self.sampler, Nmax=4)
        h1, h2, h = bf_tmp.compute_bounding_box_converged(
            batch_size=50, max_batches=10, tol=0.1, min_batches=2, verbose=False
        )
        # All dimensions must be positive
        self.assertGreater(h1, 0.)
        self.assertGreater(h2, 0.)
        self.assertGreater(h, 0.)

    # ==========================================================================
    #  17. enumerate_fluxes (integration test)
    # ==========================================================================

    def test_enumerate_fluxes_small(self):
        r"""
        **Description:**
        Integration test: run :func:`enumerate_fluxes` on a very small
        bounding box (Nmax=2, tight dilaton bounds) and verify the output
        format. This tests the full pipeline: eigenvalue computation →
        h-candidate enumeration → ISD completion → Newton refinement.
        """
        sampler_tight = jaxvacua.data_sampler(
            self.model,
            moduli_bounds=(3., 4.),
            dilaton_bounds=(3., 5.),
            axion_bounds=(-0.3, 0.3),
            seed=99,
        )
        bf_small = bounded_fluxes(self.model, sampler=sampler_tight, Nmax=2)

        results = bf_small.enumerate_fluxes(
            n_sample=10, n_isd_per_h=5, max_h_candidates=10_000,
            refine=False, return_moduli=False, verbose=False,
            confirm_streaming=False,
        )
        # Must return a list (possibly empty for Nmax=2)
        self.assertIsInstance(results, list)

    @pytest.mark.slow
    def test_enumerate_fluxes_return_moduli(self):
        r"""
        **Description:**
        Verify that ``return_moduli=True`` returns a list of dicts with
        keys ``"flux"``, ``"moduli"``, ``"tau"``.
        """
        sampler_tight = jaxvacua.data_sampler(
            self.model,
            moduli_bounds=(3., 4.),
            dilaton_bounds=(3., 5.),
            axion_bounds=(-0.3, 0.3),
            seed=99,
        )
        bf_small = bounded_fluxes(self.model, sampler=sampler_tight, Nmax=2)

        results = bf_small.enumerate_fluxes(
            n_sample=10, n_isd_per_h=5, max_h_candidates=100_000,
            refine=False, return_moduli=True, verbose=False,
            confirm_streaming=False,
        )
        self.assertIsInstance(results, list)
        # If any results found, check dict structure
        if len(results) > 0:
            r = results[0]
            self.assertIn("flux", r)
            self.assertIn("moduli", r)
            self.assertIn("tau", r)

    # ==========================================================================
    #  18. sample_bounded_fluxes (integration test)
    # ==========================================================================

    def test_sample_bounded_fluxes_small(self):
        r"""
        **Description:**
        Integration test: run :func:`sample_bounded_fluxes` with a small
        target count and verify the output format.
        """
        sampler_tight = jaxvacua.data_sampler(
            self.model,
            moduli_bounds=(3., 4.),
            dilaton_bounds=(3., 5.),
            axion_bounds=(-0.3, 0.3),
            seed=99,
        )
        bf_small = bounded_fluxes(self.model, sampler=sampler_tight, Nmax=4)

        results = bf_small.sample_bounded_fluxes(
            n_target=10, n_batch=1000, n_sample=50, n_mod=5,
            max_batches=3, refine=False, return_moduli=False, verbose=False,
        )
        # Must return a list
        self.assertIsInstance(results, list)


# ==============================================================================
#  TestClusterRoundTrip
# ==============================================================================

class TestClusterRoundTrip(TestCase):
    r"""
    **Description:**
    Round-trip tests for the cluster-parallel flux pipeline: export a small
    enumeration/sample job, process chunks sequentially in-process, then
    merge — verifying the result matches direct `enumerate_fluxes` output
    and that the optional database/designate path writes correctly to the
    vault.

    Attributes:
        model (jaxvacua.FluxVacuaFinder): Model with ``h12=2``,
            ``model_ID=1``, ``maximum_degree=0`` (fast test geometry).
        sampler (jaxvacua.data_sampler): Sampler with tight bounds.
        bf (bounded_fluxes): Bounded-fluxes instance with ``Nmax=4``.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        import tempfile
        cls._tmp_root = tempfile.mkdtemp(prefix="jvc_cluster_test_")
        # Isolate the vault to this test's scratch directory so designate
        # calls don't pollute the user's real vault.
        os.environ["JAXVACUA_VAULT"]    = os.path.join(cls._tmp_root, "vault")
        os.environ["JAXVACUA_DATA_DIR"] = os.path.join(cls._tmp_root, "cache")

        cls.model = jaxvacua.FluxVacuaFinder(
            h12=2, model_ID=1, model_type="KS", maximum_degree=0,
        )
        cls.sampler = jaxvacua.data_sampler(
            cls.model,
            moduli_bounds=(2., 5.),
            dilaton_bounds=(2., 10.),
            axion_bounds=(-0.5, 0.5),
            seed=42,
        )
        cls.bf = bounded_fluxes(cls.model, sampler=cls.sampler, Nmax=4)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        import shutil
        shutil.rmtree(cls._tmp_root, ignore_errors=True)
        # Clean up our env overrides so later tests don't see them.
        os.environ.pop("JAXVACUA_VAULT", None)
        os.environ.pop("JAXVACUA_DATA_DIR", None)

    def _run_cluster_pipeline(self, run_dir, mode="enumerate",
                              chunk_size=100, n_total_samples=300,
                              seed=42):
        r"""Export → process-all-chunks-sequentially → merge. Returns
        (info_dict, merge_results).

        Defaults are deliberately small (well below the library defaults
        of ``chunk_size=100_000`` and ``n_total_samples=5_000_000``) so
        the roundtrip tests exercise the multi-chunk pipeline mechanics
        on test scale.  ``300 / 100 = 3 chunks`` keeps
        :meth:`test_missing_chunks` from skipping its multi-chunk path.
        """
        info = self.bf.export_cluster_job(
            output_dir=run_dir, mode=mode, chunk_size=chunk_size,
            n_total_samples=n_total_samples, seed=seed, verbose=False,
        )
        for i in range(info["n_chunks"]):
            bounded_fluxes.process_chunk_from_disk(
                output_dir=run_dir, chunk_id=i,
                model=self.model, sampler=self.sampler, verbose=False,
            )
        results = bounded_fluxes.merge_cluster_results(
            run_dir, model=self.model, sampler=self.sampler,
            refine=False, verbose=False,
        )
        return info, results

    @pytest.mark.slow
    def test_enumerate_roundtrip(self):
        r"""
        **Description:**
        Export → process all chunks → merge should produce a non-empty
        flux set whose every entry respects the tadpole bound
        :math:`|f^T\,\sigma\,h|\le N_{\max}` and the shape
        :math:`2\,(h^{1,2}+1)` dictated by the pipeline.

        We intentionally do **not** compare the merged set against a
        direct ``enumerate_fluxes`` call: the two code paths dedupe at
        different granularities (direct dedupes within a chunk; cluster
        dedupes across all merged chunks after sampler re-sampling per
        worker), and the random-sampling component inside the ISD
        moduli scan is not bit-reproducible across two independent
        pipeline invocations.  The roundtrip invariants that *must*
        hold are tested directly below.
        """
        import os, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run")
            _, merged = self._run_cluster_pipeline(run_dir, mode="enumerate")

            self.assertGreater(
                len(merged), 0, "cluster merge returned 0 results",
            )

            expected_len = 2 * self.bf.n_fluxes
            # All merged fluxes must satisfy the tadpole constraint and
            # have the expected flux-vector length.
            for r in merged[:10]:
                fl = np.asarray(r["flux"])
                self.assertEqual(
                    len(fl), expected_len,
                    f"flux length {len(fl)} != 2*n_fluxes={expected_len}",
                )
                tad = abs(float(
                    jnp.real(self.model.tadpole(jnp.asarray(fl)))
                ))
                self.assertLessEqual(
                    tad, self.bf.Nmax + 1e-9,
                    f"merged flux violates tadpole: {tad} > Nmax={self.bf.Nmax}",
                )

    def test_sample_mode_roundtrip(self):
        r"""
        **Description:**
        Sample mode: export → process → merge should produce fluxes that
        satisfy the tadpole constraint.
        """
        import os, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run")
            info, merged = self._run_cluster_pipeline(
                run_dir, mode="sample", n_total_samples=500,
            )
            # Tadpole: |f^T σ h| ≤ Nmax for every merged flux
            for r in merged[:10]:  # spot-check 10 to keep the test fast
                tad = abs(float(
                    jnp.real(self.model.tadpole(jnp.asarray(r["flux"])))
                ))
                self.assertLessEqual(
                    tad, self.bf.Nmax + 1e-9,
                    f"flux violates tadpole: {tad} > Nmax={self.bf.Nmax}",
                )

    @pytest.mark.slow
    def test_missing_chunks(self):
        r"""
        **Description:**
        When only some chunks are processed, ``merge_cluster_results``
        should report the missing chunks and still return a result list
        from the present ones.
        """
        import os, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run")
            info = self.bf.export_cluster_job(
                output_dir=run_dir, mode="enumerate", chunk_size=100,
                n_total_samples=300, verbose=False,
            )
            if info["n_chunks"] < 2:
                self.skipTest("need at least 2 chunks to test missing-chunk handling")
            # Process only the first chunk
            bounded_fluxes.process_chunk_from_disk(
                output_dir=run_dir, chunk_id=0,
                model=self.model, sampler=self.sampler, verbose=False,
            )
            results = bounded_fluxes.merge_cluster_results(
                run_dir, model=self.model, sampler=self.sampler,
                refine=False, verbose=False,
            )
            self.assertIsInstance(results, list)
            # Partial results should still be a list (possibly empty)

    @pytest.mark.slow
    def test_merge_with_database_designate(self):
        r"""
        **Description:**
        Merge with ``database=..., designate=True`` promotes merged
        results to the vault.  The vault should then contain a shard
        with the expected rows loadable via ``load_local_vacua``.

        Skipped when ``stringforge`` is not installed — the database +
        vault layer was extracted to the stringforge package on
        2026-05-01, so this test only runs in environments that have
        the optional sibling installed.
        """
        import os, tempfile
        # ``importorskip`` produces a clean ``SKIPPED`` (with reason) in
        # pytest output rather than failing with ``ModuleNotFoundError``.
        # ``exc_type=ImportError`` silences a pytest 9.1 deprecation
        # warning and locks the skip semantics to "module not found"
        # only — any other exception during import is re-raised.
        LCSDatabase = pytest.importorskip(
            "stringforge.lcs_database",
            reason="stringforge is not installed; skipping vault round-trip "
                   "test. Install stringforge via "
                   "`pip install git+https://github.com/AndreasSchachner/stringforge`.",
            exc_type=ImportError,
        ).LCSDatabase
        with tempfile.TemporaryDirectory() as tmp:
            # Point the strict ``_resolve_vault_dir`` at this scratch
            # directory; otherwise it raises LookupError and the test
            # can't designate / load_local_vacua.  Restore on exit so
            # subsequent tests aren't affected.
            prev_vault = os.environ.get("STRINGFORGE_VAULT")
            os.environ["STRINGFORGE_VAULT"] = os.path.join(tmp, "vault")
            run_dir = os.path.join(tmp, "run")
            info = self.bf.export_cluster_job(
                output_dir=run_dir, mode="enumerate", chunk_size=20,
                n_total_samples=50, verbose=False,
            )
            for i in range(info["n_chunks"]):
                bounded_fluxes.process_chunk_from_disk(
                    output_dir=run_dir, chunk_id=i,
                    model=self.model, sampler=self.sampler, verbose=False,
                )
            db = LCSDatabase()
            results = bounded_fluxes.merge_cluster_results(
                run_dir, model=self.model, sampler=self.sampler,
                refine=False, verbose=False,
                database=db,
                designate=True,
                label="test_roundtrip",
                committed_by="test",
                validate_before_designate=False,
            )
            self.assertGreater(len(results), 0,
                               "cluster merge produced no results")
            df = db.load_local_vacua(model=self.model, label="test_roundtrip")
            self.assertIsNotNone(df)
            self.assertGreater(len(df), 0,
                               "designate=True did not promote any rows to the vault")
            # Restore the prior STRINGFORGE_VAULT (if any).
            if prev_vault is None:
                os.environ.pop("STRINGFORGE_VAULT", None)
            else:
                os.environ["STRINGFORGE_VAULT"] = prev_vault
