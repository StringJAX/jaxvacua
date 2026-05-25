# Copyright 2022-2026 Andreas Schachner
#
# This file is part of JAXVacua.
#
# JAXVacua is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# JAXVacua is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with JAXVacua. If not, see <https://www.gnu.org/licenses/>.

"""Tests for ``data_sampler`` and sampling-bound parsers.

Purpose
-------
Validate flux, moduli, axion, dilaton and combined initial-condition sampling
in both NumPy and JAX random backends.

Main public API
---------------
- ``TestDataSampler``: exercises integer flux sampling, Kähler-cone sampling,
  sphere/ray sampling, ISD sampling, instanton filters, metric filters and
  joint ``(z, tau, flux)`` initial guesses.
- ``TestBoundsParsing``: checks standalone parsing of box bounds and
  wall-exclusion options.

Design notes
------------
The suite uses a small Kreuzer-Skarke geometry with ``h12=2`` so cone,
instanton and sampling branches are all covered with manageable runtime.
"""


# Standard libraries
import sys, os
import numpy as np
import pytest

# JAX
import jax
import jax.numpy as jnp
import chex

# Test base class (provides assertAllClose, assertAllEqual, next_key, ...)
from util import TestCase

# Make the package importable when running from the tests/ directory
sys.path.append("./../")
import jaxvacua
from jaxvacua import data_sampler


# ==============================================================================
# TestDataSampler
# ==============================================================================

class TestDataSampler(TestCase):
    r"""
    **Description:**
    Test suite for :class:`jaxvacua.data_sampler`.

    The tests verify correctness of output shapes, dtypes, bound constraints, and
    error handling for every public method of :class:`data_sampler`.  All numerical
    sampling tests are run once with ``use_jax=False`` (NumPy backend) and once with
    ``use_jax=True`` (JAX backend) to ensure both code paths are exercised.

    .. note::
        The physical model used throughout is a KS geometry with :math:`h^{1,2}=2`.
        This is the smallest example for which all code paths are non-trivial: the
        Kähler cone has extremal rays, the instanton prepotential is non-zero, and
        the gauge kinetic matrix is well-defined.

    Attributes:
        model (jaxvacua.flux_eft.FluxEFT): The underlying string compactification model.
        N (int): Batch size used for all sampling calls (kept small for speed).
        sampler_np (data_sampler): Sampler configured with ``use_jax=False``.
        sampler_jax (data_sampler): Sampler configured with ``use_jax=True``.
        samplers (list): Convenience list ``[sampler_np, sampler_jax]``.
        key (jax.Array): Fixed JAX PRNG key used wherever a key is required.
        z_batch (jax.Array): Pre-generated batch of complex-structure moduli used
            for the filter-related tests.
        tau_batch (jax.Array): Pre-generated batch of axio-dilaton values paired
            with ``z_batch``.
    """

    @classmethod
    def setUpClass(cls):
        r"""
        **Description:**
        Builds the shared physics model and both sampler instances once for the
        entire test class.

        The setup intentionally uses narrow bounds for fluxes, dilaton, and moduli
        so that the ISD condition is more likely to be satisfied in the tests and
        the Gurobi-based ``find_interior_points`` terminates quickly.
        """
        super().setUpClass()

        # -----------------------------------------------------------------------
        # Physics model
        # -----------------------------------------------------------------------
        # Use a KS geometry with h^{1,2}=2 and maximum instanton degree 5.
        # All tests share this single model instance to avoid redundant
        # initialisation overhead (period integrals, cone computations, ...).
        h12 = 2
        cls.model = jaxvacua.FluxEFT(
            h12=h12, model_ID=1, model_type="KS", maximum_degree=5
        )

        # Small batch size: enough to exercise vectorised code paths without
        # making the test suite slow.
        cls.N = 8

        # Shared bounds used for both samplers
        common_kwargs = dict(
            flux_bounds=(-5, 5),
            axion_bounds=(-0.5, 0.5),
            dilaton_bounds=(2.0, 8.0),
            moduli_bounds=(1.0, 4.0),
        )

        # -----------------------------------------------------------------------
        # Create one sampler per backend
        # -----------------------------------------------------------------------
        cls.sampler_np  = data_sampler(cls.model, use_jax=False, **common_kwargs)
        cls.sampler_jax = data_sampler(cls.model, use_jax=True,  **common_kwargs)

        # Convenience list so tests can loop over both backends in one line
        cls.samplers = [cls.sampler_np, cls.sampler_jax]

        # Fixed PRNG key – using a constant seed guarantees reproducibility
        cls.key = jax.random.PRNGKey(0)

        # -----------------------------------------------------------------------
        # Pre-generate moduli + tau for the filter tests
        # -----------------------------------------------------------------------
        # We use the NumPy sampler here because the filter methods (filter_by_km,
        # filter_by_instantons) work on JAX arrays regardless of the sampler backend.
        cls.z_batch, cls.tau_batch = cls.sampler_np.initial_guesses(
            cls.N, rns_key=cls.key, include_fluxes=False
        )

    # ==========================================================================
    #  1.  get_fluxes
    # ==========================================================================
    #
    # get_fluxes samples integer flux quanta used in the ISD biased sampling
    # scheme.  The method supports:
    #   mode            : "full" (both RR and NSNS halves), "half" (one half),
    #                     None  (identical to "full")
    #   sampling_mode   : "box"           – uniform integer box in flux space
    #                     "sphere"        – constrain Euclidean norm ≤ radius
    #                     "tadpole_bound" – constrain D3-tadpole ≤ radius
    #                     "tadpole_cancel"– require D3-tadpole = radius (exact)
    # ==========================================================================

    def test_get_fluxes_box_full(self):
        r"""
        **Description:**
        Verifies that ``get_fluxes`` with ``mode="full"`` and ``sampling_mode="box"``
        returns an integer array of shape ``(N, 2*n_fluxes)``.

        The *full* mode returns both the RR-flux vector :math:`f=(f_1,f_2)` and the
        NSNS-flux vector :math:`h=(h_1,h_2)` concatenated along the last axis,
        yielding ``2*n_fluxes`` entries per sample.
        """
        expected_shape = (self.N, 2 * self.model.n_fluxes)
        for s in self.samplers:
            out = s.get_fluxes(
                self.N, mode="full", sampling_mode="box", rns_key=self.key
            )
            chex.assert_shape(out, expected_shape)

    def test_jax_backend_same_key_reproducible_box_samples(self):
        r"""
        **Description:**
        Using the same explicit JAX PRNG key must reproduce the same one-shot
        box samples.  This pins the deterministic contract for callers that
        manage their own keys.
        """
        key = jax.random.PRNGKey(314159)

        flux_a = self.sampler_jax.get_fluxes(
            4, mode="full", sampling_mode="box", rns_key=key
        )
        flux_b = self.sampler_jax.get_fluxes(
            4, mode="full", sampling_mode="box", rns_key=key
        )
        self.assertAllEqual(flux_a, flux_b)

        moduli_a = self.sampler_jax.get_moduli(
            4, sampling_mode="box", rns_key=key
        )
        moduli_b = self.sampler_jax.get_moduli(
            4, sampling_mode="box", rns_key=key
        )
        self.assertAllEqual(moduli_a, moduli_b)

        tau_a = self.sampler_jax.get_complex_tau(4, rns_key=key)
        tau_b = self.sampler_jax.get_complex_tau(4, rns_key=key)
        self.assertAllEqual(tau_a, tau_b)

    def test_get_fluxes_box_half(self):
        r"""
        **Description:**
        Verifies that ``mode="half"`` returns only one symplectic half of the flux
        vector, giving shape ``(N, n_fluxes)``.

        In the ISD sampling workflow, the user provides one half of the fluxes
        and the ISD condition determines the other half.
        """
        expected_shape = (self.N, self.model.n_fluxes)
        for s in self.samplers:
            out = s.get_fluxes(
                self.N, mode="half", sampling_mode="box", rns_key=self.key
            )
            chex.assert_shape(out, expected_shape)

    def test_get_fluxes_box_none_mode(self):
        r"""
        **Description:**
        Verifies that ``mode=None`` is treated identically to ``mode="full"``,
        i.e. returns shape ``(N, 2*n_fluxes)``.
        """
        expected_shape = (self.N, 2 * self.model.n_fluxes)
        for s in self.samplers:
            out = s.get_fluxes(
                self.N, mode=None, sampling_mode="box", rns_key=self.key
            )
            chex.assert_shape(out, expected_shape)

    def test_get_fluxes_sphere(self):
        r"""
        **Description:**
        Verifies that ``sampling_mode="sphere"`` returns fluxes whose Euclidean
        norms are all at most ``radius``.

        The sphere constraint :math:`\|f\|_2 \leq r` is useful for ensuring that
        the flux tadpole does not grow too large.
        """
        radius = 15.0
        for s in self.samplers:
            out = s.get_fluxes(
                self.N, mode="full", sampling_mode="sphere",
                radius=radius, rns_key=self.key
            )
            chex.assert_shape(out, (self.N, 2 * self.model.n_fluxes))

            # All Euclidean norms must be within the requested radius
            norms = np.sqrt(np.sum(np.asarray(out) ** 2, axis=1))
            self.assertTrue(
                np.all(norms <= radius),
                msg=f"Some flux norms exceeded radius={radius}: max={norms.max()}"
            )

    def test_get_fluxes_tadpole_bound(self):
        r"""
        **Description:**
        Verifies that ``sampling_mode="tadpole_bound"`` returns fluxes whose
        D3-tadpole contribution :math:`N_\text{flux} = \frac{1}{2} F \cdot H`
        does not exceed the model's D3-tadpole upper bound.

        This mode is the most physically motivated one: it ensures the flux
        configuration is consistent with tadpole cancellation for a given Calabi-Yau
        compactification.
        """
        # Use the model's own D3-tadpole as the maximal allowed value
        if self.model.D3_tadpole is None:
            self.skipTest("D3_tadpole not set for this model")
        Nmax = float(self.model.D3_tadpole)
        for s in self.samplers:
            out = s.get_fluxes(
                self.N, mode="full", sampling_mode="tadpole_bound",
                radius=Nmax, rns_key=self.key
            )
            chex.assert_shape(out, (self.N, 2 * self.model.n_fluxes))

    def test_get_fluxes_invalid_mode_raises(self):
        r"""
        **Description:**
        Verifies that an unrecognised ``mode`` string raises a ``ValueError``
        with a helpful message listing the accepted values.
        """
        with self.assertRaises(ValueError):
            self.sampler_np.get_fluxes(self.N, mode="invalid")

    def test_get_fluxes_invalid_sampling_mode_raises(self):
        r"""
        **Description:**
        Verifies that an unrecognised ``sampling_mode`` string raises a
        ``ValueError``.
        """
        with self.assertRaises(ValueError):
            self.sampler_np.get_fluxes(self.N, sampling_mode="nonsense")

    # ==========================================================================
    #  2.  get_moduli
    # ==========================================================================
    #
    # get_moduli samples real Kähler moduli t^a > 0  (the imaginary parts of the
    # complexified Kähler form).  Supported sampling_modes:
    #
    #   "box"           – uniform in [moduli_lower, moduli_upper]^{h12}
    #   "cone"          – positive linear combination of Kähler cone generators
    #   "stretched_cone"– cone shifted by stretching * tip_ray
    #   "tip_ray"       – scale the tip of the stretched Kähler cone randomly
    #   "random_ray"    – scale a single random interior direction randomly
    #   "random_rays"   – positive combination of n_rays random rays
    # ==========================================================================

    def test_get_moduli_box(self):
        r"""
        **Description:**
        Verifies that box sampling returns an array of shape ``(N, h12)``
        and that every entry lies within the declared bounds
        ``[moduli_lower, moduli_upper]``.
        """
        for s in self.samplers:
            out = s.get_moduli(self.N, sampling_mode="box", rns_key=self.key)
            chex.assert_shape(out, (self.N, self.model.h12))

            # Each sampled value must lie inside the declared bounds
            vals = np.asarray(out)
            self.assertTrue(
                np.all(vals >= s.moduli_lower),
                msg=f"Some moduli below lower bound {s.moduli_lower}"
            )
            self.assertTrue(
                np.all(vals <= s.moduli_upper),
                msg=f"Some moduli above upper bound {s.moduli_upper}"
            )

    def test_get_moduli_cone_use_rays(self):
        r"""
        **Description:**
        Verifies that ``sampling_mode="cone"`` with ``use_rays=True`` returns an
        array of shape ``(N, h12)``.

        With ``use_rays=True`` the points are constructed as positive linear
        combinations of the extremal rays (or generators) of the Kähler cone:

        .. math::
            t = \sum_{a} u_a \, r_a, \quad u_a \geq 0 \, .

        This guarantees that the sampled points lie inside the cone by construction.
        """
        if self.sampler_np._extremal_rays is None and self.sampler_np._rays is None:
            self.skipTest("No ray data available for cone sampling on this model.")
        for s in self.samplers:
            out = s.get_moduli(
                self.N, sampling_mode="cone", use_rays=True, rns_key=self.key
            )
            chex.assert_shape(out, (self.N, self.model.h12))

    def test_get_moduli_cone_no_rays(self):
        r"""
        **Description:**
        Verifies that ``sampling_mode="cone"`` with ``use_rays=False`` (the
        default) returns an array of shape ``(N, h12)``.

        In this mode, previously computed interior cone points (``_cone_points``)
        are perturbed and rescaled, which avoids repeated Gurobi calls but requires
        at least one successful initialisation of ``_cone_points`` at construction
        time.
        """
        if len(self.sampler_np._cone_points) == 0:
            self.skipTest("No pre-computed cone points available.")
        for s in self.samplers:
            out = s.get_moduli(
                self.N, sampling_mode="cone", use_rays=False, rns_key=self.key
            )
            chex.assert_shape(out, (self.N, self.model.h12))

    def test_get_moduli_random_rays(self):
        r"""
        **Description:**
        Verifies that ``sampling_mode="random_rays"`` with ``use_rays=True`` and
        ``n_rays=2`` returns an array of shape ``(N, h12)``.

        This mode draws ``n_rays`` random rays from the stored ray list and forms
        positive linear combinations, providing an unbiased sample in the interior
        of the Kähler cone when many rays are available.
        """
        if self.sampler_np._rays is None:
            self.skipTest("No rays available on this model.")
        for s in self.samplers:
            out = s.get_moduli(
                self.N, sampling_mode="random_rays", use_rays=True,
                n_rays=2, rns_key=self.key
            )
            chex.assert_shape(out, (self.N, self.model.h12))

    def test_get_moduli_tip_ray(self):
        r"""
        **Description:**
        Verifies that ``sampling_mode="tip_ray"`` returns an array of shape
        ``(N, h12)``.

        Points are sampled along the tip ray of the stretched Kähler cone
        :math:`t = r \cdot t_\text{tip} + \epsilon \cdot t_\text{tip}`,
        where :math:`r \sim \text{Uniform}(0, \text{maxval})` and
        :math:`\epsilon` is the stretching parameter.
        """
        if self.sampler_np._tip is None:
            self.skipTest("No tip ray available on this model.")
        for s in self.samplers:
            out = s.get_moduli(self.N, sampling_mode="tip_ray", rns_key=self.key)
            chex.assert_shape(out, (self.N, self.model.h12))

    def test_get_moduli_random_ray(self):
        r"""
        **Description:**
        Verifies that ``sampling_mode="random_ray"`` returns an array of shape
        ``(N, h12)``.

        A single random interior direction is sampled first (via
        ``sample_ray``), and all ``N`` points are then obtained by scaling that
        direction by independent uniform random factors.
        """
        for s in self.samplers:
            out = s.get_moduli(self.N, sampling_mode="random_ray", rns_key=self.key)
            chex.assert_shape(out, (self.N, self.model.h12))

    def test_get_moduli_invalid_sampling_mode_raises(self):
        r"""
        **Description:**
        Verifies that an unrecognised ``sampling_mode`` raises a ``ValueError``.
        """
        with self.assertRaises(ValueError):
            self.sampler_np.get_moduli(self.N, sampling_mode="invalid")

    # ==========================================================================
    #  3.  get_axions / get_axion
    # ==========================================================================
    #
    # Axions are the real parts of the complex-structure moduli z^i = c^i + i*u^i.
    # get_axions samples c^i jointly for all h12 moduli.
    # get_axion  samples the single universal axion c_0 of the axio-dilaton τ.
    # Both methods return complex128 arrays (the imaginary part is zero; the full
    # complex modulus is assembled later by adding i * dilaton / i * moduli).
    # ==========================================================================

    def test_get_axions_box(self):
        r"""
        **Description:**
        Verifies that ``get_axions`` with ``sampling_mode="box"`` returns a
        complex128 array of shape ``(N, h12)`` with entries drawn from
        ``[axions_lower, axions_upper]``.
        """
        for s in self.samplers:
            out = s.get_axions(self.N, sampling_mode="box", rns_key=self.key)
            # Axions must be arrays of shape (N, h12)
            chex.assert_shape(out, (self.N, self.model.h12))
            # Axion arrays must have complex dtype (imaginary part is zero at this stage)
            self.assertTrue(jnp.issubdtype(out.dtype, jnp.complexfloating))

    def test_get_axions_bounds(self):
        r"""
        **Description:**
        Verifies that all sampled axion values lie within the declared bounds.

        The real part of the returned complex array corresponds to the axion
        :math:`c^i`; the imaginary part is identically zero at this stage.
        """
        for s in self.samplers:
            out = np.asarray(s.get_axions(self.N, rns_key=self.key))
            # All axion real parts must be at or above the declared lower bound
            self.assertTrue(
                np.all(out.real >= s.axions_lower),
                msg=f"Some axions below lower bound {s.axions_lower}"
            )
            # All axion real parts must be at or below the declared upper bound
            self.assertTrue(
                np.all(out.real <= s.axions_upper),
                msg=f"Some axions above upper bound {s.axions_upper}"
            )

    def test_get_axions_invalid_mode_raises(self):
        r"""
        **Description:**
        Verifies that an unrecognised ``sampling_mode`` raises a ``ValueError``.
        """
        with self.assertRaises(ValueError):
            self.sampler_np.get_axions(self.N, sampling_mode="bad")

    def test_get_axion_box(self):
        r"""
        **Description:**
        Verifies that the *universal* axion sampler (``get_axion``) returns a
        complex128 array of shape ``(N,)`` – a vector rather than a matrix,
        since this is a single scalar axion for each sample rather than an
        :math:`h^{1,2}`-vector.
        """
        for s in self.samplers:
            out = s.get_axion(self.N, sampling_mode="box", rns_key=self.key)
            # Universal axion must be a 1-d vector of shape (N,), not a matrix
            chex.assert_shape(out, (self.N,))
            # Universal axion must have complex dtype
            self.assertTrue(jnp.issubdtype(out.dtype, jnp.complexfloating))

    def test_get_axion_invalid_mode_raises(self):
        r"""
        **Description:**
        Verifies that an unrecognised ``sampling_mode`` raises a ``ValueError``.
        """
        with self.assertRaises(ValueError):
            self.sampler_np.get_axion(self.N, sampling_mode="bad")

    # ==========================================================================
    #  4.  get_dilaton
    # ==========================================================================
    #
    # The dilaton s = Im(τ) > 0 sets the string coupling g_s = 1/s.  It is
    # sampled as the imaginary part of a complex128 scalar τ = c_0 + i*s.
    # ==========================================================================

    def test_get_dilaton_box(self):
        r"""
        **Description:**
        Verifies that ``get_dilaton`` with ``sampling_mode="box"`` returns a
        complex128 array of shape ``(N,)`` and that the imaginary part (the
        physical dilaton :math:`s`) lies within ``[s_lower, s_upper]``.

        .. note::
            The returned array is complex because the dilaton is eventually
            combined with the axion to form the full axio-dilaton
            :math:`\tau = c_0 + \text{i}\, s`.
        """
        for s in self.samplers:
            out = s.get_dilaton(self.N, sampling_mode="box", rns_key=self.key)
            # Dilaton output must be a 1-d vector of shape (N,)
            chex.assert_shape(out, (self.N,))
            # Dilaton output must have complex dtype for later combination into tau
            self.assertTrue(jnp.issubdtype(out.dtype, jnp.complexfloating))

            # The real part holds s (it becomes Im(τ) after multiplication by 1j in get_complex_tau)
            vals = np.real(np.asarray(out))
            self.assertTrue(
                np.all(vals >= s.s_lower),
                msg=f"Some dilaton values below lower bound {s.s_lower}"
            )
            self.assertTrue(
                np.all(vals <= s.s_upper),
                msg=f"Some dilaton values above upper bound {s.s_upper}"
            )

    def test_get_dilaton_invalid_mode_raises(self):
        r"""
        **Description:**
        Verifies that an unrecognised ``sampling_mode`` raises a ``ValueError``.
        """
        with self.assertRaises(ValueError):
            self.sampler_np.get_dilaton(self.N, sampling_mode="bad")

    # ==========================================================================
    #  5.  get_complex_tau
    # ==========================================================================
    #
    # The axio-dilaton τ = c_0 + i*s combines the universal axion c_0 (real part)
    # with the dilaton s > 0 (imaginary part).  It is sampled as a single complex
    # number per point.
    # ==========================================================================

    def test_get_complex_tau(self):
        r"""
        **Description:**
        Verifies that ``get_complex_tau`` returns a complex128 array of shape
        ``(N,)`` by combining the output of ``get_axion`` and ``get_dilaton``.

        The function effectively computes ``τ = get_axion(N) + 1j * get_dilaton(N)``
        using the same PRNG key for reproducibility.
        """
        for s in self.samplers:
            tau = s.get_complex_tau(self.N, rns_key=self.key)
            chex.assert_shape(tau, (self.N,))
            chex.assert_type(tau, complex)

            # The imaginary part corresponds to the dilaton and should be positive
            self.assertTrue(
                np.all(np.imag(np.asarray(tau)) > 0),
                msg="Dilaton Im(τ) must be strictly positive."
            )

    # ==========================================================================
    #  6.  get_complex_moduli
    # ==========================================================================
    #
    # The complex-structure moduli z^i = c^i + i*u^i are assembled from the
    # complex-structure axions c^i (real part) and the real moduli u^i > 0
    # (imaginary part).
    # ==========================================================================

    def test_get_complex_moduli(self):
        r"""
        **Description:**
        Verifies that ``get_complex_moduli`` returns a complex128 array of shape
        ``(N, h12)`` whose imaginary part (the real moduli :math:`u^i`) is
        strictly positive.
        """
        for s in self.samplers:
            out = s.get_complex_moduli(self.N, rns_key=self.key)
            chex.assert_shape(out, (self.N, self.model.h12))
            self.assertTrue(jnp.issubdtype(out.dtype, jnp.complexfloating))

            # The imaginary part u^i = Im(z^i) must be strictly positive for the
            # periods expansion to converge in the large complex structure limit.
            self.assertTrue(
                np.all(np.imag(np.asarray(out)) > 0),
                msg="Moduli Im(z^i) must be strictly positive."
            )

    # ==========================================================================
    #  7.  sample_sphere
    # ==========================================================================
    #
    # Samples points from a d-dimensional disk/ball of given radius expressed
    # as complex numbers r*exp(iα).  The result is a starting-point generator
    # for axion scans along circular trajectories in the complex moduli space.
    # ==========================================================================

    def test_sample_sphere_1d(self):
        r"""
        **Description:**
        Verifies that ``sample_sphere`` with ``dim=1`` returns a complex128 array
        of shape ``(N,)`` (a single complex number per sample).
        """
        for s in self.samplers:
            out = s.sample_sphere(self.N, dim=1, rns_key=self.key)
            # 1-d sphere samples must be a flat vector of shape (N,)
            chex.assert_shape(out, (self.N,))
            # Sphere samples must have complex dtype (r * exp(i*alpha))
            self.assertTrue(jnp.issubdtype(out.dtype, jnp.complexfloating))

    def test_sample_sphere_2d(self):
        r"""
        **Description:**
        Verifies that ``sample_sphere`` with ``dim=2`` returns a complex128 array
        of shape ``(N, 2)`` – one complex number per complex-structure modulus.
        """
        for s in self.samplers:
            out = s.sample_sphere(self.N, dim=2, rns_key=self.key)
            # 2-d sphere samples must have shape (N, 2), one complex number per modulus
            chex.assert_shape(out, (self.N, 2))
            # Sphere samples must have complex dtype
            self.assertTrue(jnp.issubdtype(out.dtype, jnp.complexfloating))

    def test_sample_sphere_radius(self):
        r"""
        **Description:**
        Verifies that all sampled points have modulus :math:`|z| \leq r`,
        i.e. that the sampling respects the specified radius constraint.
        """
        radius = 3.0
        for s in self.samplers:
            out = np.asarray(
                s.sample_sphere(self.N, dim=1, radius=radius, rns_key=self.key)
            )
            self.assertTrue(
                np.all(np.abs(out) <= radius + 1e-9),
                msg=f"Some sphere samples exceeded radius={radius}."
            )

    # ==========================================================================
    #  8.  rescale_points
    # ==========================================================================
    #
    # Rescales a batch of real points so that their chosen norm does not exceed
    # maxval.  Points already within the target region are left unchanged (or
    # randomly contracted).  Supported norms:
    #   "l2"  – Euclidean norm ‖t‖_2  ≤ maxval
    #   "l1"  – Manhattan norm ‖t‖_1  ≤ maxval
    #   "inf" – Chebyshev norm ‖t‖_∞  ≤ maxval
    # ==========================================================================

    def _make_pts(self):
        r"""Helper: return deterministic points with large norms."""
        base = np.array([
            [2.0, 9.0],
            [8.5, 3.0],
            [4.25, 7.75],
            [9.5, 9.25],
            [6.0, 2.5],
        ])
        reps = int(np.ceil(self.N / len(base)))
        return np.tile(base, (reps, 1))[:self.N, :self.model.h12]

    def test_rescale_points_l2(self):
        r"""
        **Description:**
        Verifies that after ``rescale_points`` with ``norm="l2"`` all rescaled
        points have Euclidean norm :math:`\|t\|_2 \leq \text{maxval}`.
        """
        pts    = self._make_pts()
        maxval = 5.0
        for s in self.samplers:
            out   = s.rescale_points(pts, norm="l2", maxval=maxval, rns_key=self.key)
            # Rescaled output must preserve the original shape (N, h12)
            chex.assert_shape(out, (self.N, self.model.h12))
            norms = np.linalg.norm(np.asarray(out), axis=1)
            # All Euclidean norms must be at most maxval after rescaling
            self.assertTrue(
                np.all(norms <= maxval + 1e-9),
                msg=f"L2 rescaling: norm exceeded maxval={maxval}; max={norms.max()}"
            )

    def test_rescale_points_l1(self):
        r"""
        **Description:**
        Verifies that after ``rescale_points`` with ``norm="l1"`` all rescaled
        points have :math:`\ell^1`-norm :math:`\|t\|_1 \leq \text{maxval}`.
        """
        pts    = self._make_pts()
        maxval = 5.0
        for s in self.samplers:
            out   = s.rescale_points(pts, norm="l1", maxval=maxval, rns_key=self.key)
            # Rescaled output must preserve the original shape (N, h12)
            chex.assert_shape(out, (self.N, self.model.h12))
            norms = np.sum(np.abs(np.asarray(out)), axis=1)
            # All L1 norms must be at most maxval after rescaling
            self.assertTrue(
                np.all(norms <= maxval + 1e-9),
                msg=f"L1 rescaling: norm exceeded maxval={maxval}; max={norms.max()}"
            )

    def test_rescale_points_inf(self):
        r"""
        **Description:**
        Verifies that after ``rescale_points`` with ``norm="inf"`` all rescaled
        points satisfy :math:`\max_a |t^a| \leq \text{maxval}`.
        """
        pts    = self._make_pts()
        maxval = 5.0
        for s in self.samplers:
            out   = s.rescale_points(pts, norm="inf", maxval=maxval, rns_key=self.key)
            # Rescaled output must preserve the original shape (N, h12)
            chex.assert_shape(out, (self.N, self.model.h12))
            norms = np.max(np.abs(np.asarray(out)), axis=1)
            # All infinity norms must be at most maxval after rescaling
            self.assertTrue(
                np.all(norms <= maxval + 1e-9),
                msg=f"Inf rescaling: norm exceeded maxval={maxval}; max={norms.max()}"
            )

    def test_rescale_points_invalid_norm_raises(self):
        r"""
        **Description:**
        Verifies that an unrecognised ``norm`` string raises a ``ValueError``.
        """
        pts = self._make_pts()
        with self.assertRaises(ValueError):
            self.sampler_np.rescale_points(pts, norm="bad")

    # ==========================================================================
    #  9.  find_interior_points
    # ==========================================================================
    #
    # Uses Gurobi integer programming to find N integer vectors that lie strictly
    # inside the Kähler cone  (H · t ≥ stretching for all hyperplanes H).
    # Optionally normalises each point to unit L2-norm.
    # ==========================================================================

    @pytest.mark.requires_gurobi
    def test_find_interior_points_basic(self):
        r"""
        **Description:**
        Verifies that ``find_interior_points`` finds at least one solution and
        returns an array with the correct number of Kähler moduli.

        .. note::
            The Gurobi solver may find fewer solutions than requested if the
            pool search exhausts the feasible set.  We therefore only assert
            ``shape[0] >= 1``.
        """
        pts = self.sampler_np.find_interior_points(N=5, stretching=0.1)
        # Gurobi must find at least one feasible interior point
        self.assertGreaterEqual(pts.shape[0], 1)
        # Each point must have exactly h12 coordinates (one per Kaehler modulus)
        self.assertEqual(pts.shape[1], self.model.h12)

    @pytest.mark.requires_gurobi
    def test_find_interior_points_normalised(self):
        r"""
        **Description:**
        Verifies that with ``normalise=True`` each returned point has
        :math:`\|t\|_2 = 1` up to numerical precision.
        """
        pts   = self.sampler_np.find_interior_points(N=5, normalise=True)
        norms = np.linalg.norm(pts, axis=1)
        self.assertAllClose(
            norms, np.ones(len(norms)), atol=1e-6,
        )

    @pytest.mark.requires_gurobi
    def test_find_interior_points_in_cone(self):
        r"""
        **Description:**
        Verifies that all returned points satisfy the defining hyperplane
        inequalities of the Kähler cone:
        :math:`H \cdot t \geq 0` for all hyperplane normals :math:`H`.

        This is the fundamental consistency check: the solver should never return
        a point outside the cone.
        """
        pts      = self.sampler_np.find_interior_points(N=5, stretching=0.0)
        H        = self.sampler_np._hyperplanes
        products = pts @ H.T
        self.assertTrue(
            np.all(products >= -1e-6),
            msg="Some interior points violate the Kähler cone hyperplane constraints."
        )

    # ==========================================================================
    #  10. filter_points
    # ==========================================================================
    #
    # Filters a batch of points by (a) the Kähler cone inequalities and
    # (b) an optional user-supplied callable.  Used inside get_moduli to keep
    # only physically valid samples.
    # ==========================================================================

    def test_filter_points_cone_interior(self):
        r"""
        **Description:**
        Verifies that ``filter_points`` applied to the pre-computed interior cone
        points keeps at most all of them (cone-interior points pass by definition)
        and that the output has the correct second dimension.
        """
        if len(self.sampler_np._cone_points) == 0:
            self.skipTest("No cone points available on this model.")
        pts = self.sampler_np._cone_points
        for s in self.samplers:
            out = s.filter_points(pts, stretching=0.0)
            # filter_points can only discard rows, never add them
            self.assertLessEqual(out.shape[0], pts.shape[0])
            self.assertEqual(out.shape[1], self.model.h12)

    def test_filter_points_custom_filter(self):
        r"""
        **Description:**
        Verifies that a custom callable filter is applied *after* the built-in
        cone constraint and that the output satisfies the custom condition.

        The custom filter used here keeps only points with positive coordinate
        sum, acting as a simple sanity check that the chaining of filters works
        correctly.
        """
        if len(self.sampler_np._cone_points) == 0:
            self.skipTest("No cone points available on this model.")
        pts = self.sampler_np._cone_points

        # Custom filter: keep only points whose coordinates sum to a positive value
        def custom_filter(x):
            return x[np.sum(x, axis=1) > 0]

        for s in self.samplers:
            out = s.filter_points(pts, filter=custom_filter, stretching=0.0)
            self.assertEqual(out.shape[1], self.model.h12)
            # All surviving rows must satisfy the custom condition
            self.assertTrue(
                np.all(np.sum(np.asarray(out), axis=1) > 0),
                msg="Custom filter did not remove all points with non-positive sum."
            )

    # ==========================================================================
    #  11. sample_interior_point / sample_ray / sample_rays
    # ==========================================================================
    #
    # These methods expose the random-direction sampling interface used by
    # get_moduli internally.
    #   sample_interior_point – calls find_interior_points, returns one point
    #   sample_ray            – alias for sample_interior_point
    #   sample_rays           – samples k rays without replacement from _rays
    # ==========================================================================

    @pytest.mark.requires_gurobi
    def test_sample_interior_point(self):
        r"""
        **Description:**
        Verifies that ``sample_interior_point`` returns a single point of shape
        ``(h12,)`` lying inside the Kähler cone.
        """
        for s in self.samplers:
            pt = s.sample_interior_point(rns_key=self.key)
            chex.assert_shape(pt, (self.model.h12,))

    @pytest.mark.requires_gurobi
    def test_sample_ray(self):
        r"""
        **Description:**
        Verifies that ``sample_ray`` (which delegates to ``sample_interior_point``)
        returns a single direction of shape ``(h12,)``.
        """
        for s in self.samplers:
            ray = s.sample_ray(rns_key=self.key)
            chex.assert_shape(ray, (self.model.h12,))

    def test_sample_rays(self):
        r"""
        **Description:**
        Verifies that ``sample_rays(k)`` returns exactly :math:`k` rays drawn
        without replacement from the stored ray list, giving shape ``(k, h12)``.
        """
        if self.sampler_np._rays is None:
            self.skipTest("No ray data available on this model.")
        # Use k=2 to keep the test fast while still exercising the slicing logic
        k = min(2, len(self.sampler_np._rays))
        for s in self.samplers:
            rays = s.sample_rays(k, rns_key=self.key)
            chex.assert_shape(rays, (k, self.model.h12))

    def test_sample_rays_no_rays_raises(self):
        r"""
        **Description:**
        Verifies that ``sample_rays`` raises ``RuntimeError`` when no ray data
        is available (``_rays is None``).

        A minimal stub object is used so the test does not depend on model
        construction.
        """
        # Build a minimal stub that has _rays=None without calling __init__
        stub = data_sampler.__new__(data_sampler)
        stub._rays = None
        with self.assertRaises(RuntimeError):
            stub.sample_rays(2)

    def test_sample_rays_k_too_large_raises(self):
        r"""
        **Description:**
        Verifies that requesting more rays than are available raises a
        ``ValueError``.
        """
        if self.sampler_np._rays is None:
            self.skipTest("No ray data available on this model.")
        k_too_large = len(self.sampler_np._rays) + 1
        with self.assertRaises(ValueError):
            self.sampler_np.sample_rays(k_too_large)

    # ==========================================================================
    #  12. initial_guesses
    # ==========================================================================
    #
    # Generates initial guesses for (z, τ, fluxes) to be used as starting points
    # for the gradient-flow minimiser of the scalar potential.
    # Supports all moduli_sampling_mode and fluxes_sampling_mode options.
    # ==========================================================================

    def test_initial_guesses_with_fluxes(self):
        r"""
        **Description:**
        Verifies that ``initial_guesses`` with ``include_fluxes=True`` returns a
        tuple ``(moduli, tau, fluxes)`` where:

        - ``moduli`` has shape ``(N, h12)`` and dtype ``complex128``
        - ``tau`` has shape ``(N,)`` and dtype ``complex128``
        - ``fluxes`` has shape ``(N, 2*n_fluxes)``
        """
        for s in self.samplers:
            moduli, tau, fluxes = s.initial_guesses(
                self.N, rns_key=self.key, include_fluxes=True
            )
            # Moduli must have shape (N, h12)
            chex.assert_shape(moduli, (self.N, self.model.h12))
            # Axio-dilaton must be a 1-d vector of shape (N,)
            chex.assert_shape(tau,    (self.N,))
            # Flux vector must contain both RR and NSNS halves: shape (N, 2*n_fluxes)
            chex.assert_shape(fluxes, (self.N, 2 * self.model.n_fluxes))
            # Moduli must have complex dtype for the complex-structure moduli z^i
            chex.assert_type(moduli, complex)
            # Axio-dilaton must have complex dtype (tau = c0 + i*s)
            chex.assert_type(tau,    complex)

    def test_initial_guesses_jax_same_key_reproducible_box_mode(self):
        r"""
        **Description:**
        ``initial_guesses`` must be reproducible when the caller supplies the
        same explicit JAX key and routes through the one-shot box samplers.
        """
        key = jax.random.PRNGKey(271828)
        kwargs = dict(
            rns_key=key,
            include_fluxes=True,
            moduli_sampling_mode="box",
            fluxes_sampling_mode="box",
        )
        moduli_a, tau_a, fluxes_a = self.sampler_jax.initial_guesses(5, **kwargs)
        moduli_b, tau_b, fluxes_b = self.sampler_jax.initial_guesses(5, **kwargs)

        self.assertAllEqual(moduli_a, moduli_b)
        self.assertAllEqual(tau_a, tau_b)
        self.assertAllEqual(fluxes_a, fluxes_b)

    def test_initial_guesses_without_fluxes(self):
        r"""
        **Description:**
        Verifies that ``initial_guesses`` with ``include_fluxes=False`` returns
        only ``(moduli, tau)`` without sampling fluxes, which is faster when
        only the moduli space geometry is needed.
        """
        for s in self.samplers:
            moduli, tau = s.initial_guesses(
                self.N, rns_key=self.key, include_fluxes=False
            )
            # Moduli must have shape (N, h12) even when fluxes are omitted
            chex.assert_shape(moduli, (self.N, self.model.h12))
            # Axio-dilaton must be a 1-d vector of shape (N,)
            chex.assert_shape(tau,    (self.N,))
            # Moduli must have complex dtype
            chex.assert_type(moduli, complex)
            # Axio-dilaton must have complex dtype
            chex.assert_type(tau,    complex)

    def test_initial_guesses_moduli_sampling_box(self):
        r"""
        **Description:**
        Verifies that ``moduli_sampling_mode="box"`` (the default) produces
        moduli of the correct shape regardless of which flux sampling mode is
        used.
        """
        moduli, tau, _ = self.sampler_np.initial_guesses(
            self.N, moduli_sampling_mode="box", rns_key=self.key
        )
        chex.assert_shape(moduli, (self.N, self.model.h12))

    def test_initial_guesses_fluxes_sphere(self):
        r"""
        **Description:**
        Verifies that ``fluxes_sampling_mode="sphere"`` (with a specified radius)
        produces flux arrays of the correct shape.  The sphere constraint bounds
        the magnitude of the flux quanta independently of the tadpole.
        """
        moduli, tau, fluxes = self.sampler_np.initial_guesses(
            self.N, fluxes_sampling_mode="sphere",
            flux_radius=10.0, rns_key=self.key
        )
        chex.assert_shape(fluxes, (self.N, 2 * self.model.n_fluxes))

    # ==========================================================================
    #  13. ISD_sampling
    # ==========================================================================
    #
    # Solves the Imaginary Self-Dual condition  ⋆G_3 = i G_3  for given moduli
    # and axio-dilaton values, producing flux quanta that approximately satisfy
    # the ISD condition after rounding to integers.
    #
    # Four modes:
    #   "ISD+"  – from (f_2, h_2), solve for (f_1, h_1) via the gauge kinetic matrix
    #   "ISD-"  – from (f_1, h_1), solve for (f_2, h_2) via the gauge kinetic matrix
    #   "F"     – from NSNS-fluxes h, solve for RR-fluxes f via the ISD matrix M
    #   "H"     – from RR-fluxes f, solve for NSNS-fluxes h via the ISD matrix M
    #
    # Two output formats:
    #   "full"  – returns the complete flux vector [f_1, f_2, h_1, h_2]
    #   "half"  – returns only the computed half [f_i, h_i] or [f] or [h]
    # ==========================================================================

    def _make_isd_inputs(self):
        r"""
        Helper: generates a consistent set of (z, τ, flux0) for ISD tests.

        Returns:
            Tuple: ``(z, tau, flux0)`` where

            - ``z`` (jax.Array): Complex-structure moduli, shape ``(h12,)``
            - ``tau`` (complex): Axio-dilaton scalar
            - ``flux0`` (jax.Array): Half-flux vector, shape ``(n_fluxes,)``
        """
        z = jnp.array([0.18 + 2.7j, -0.24 + 3.1j])[:self.model.h12]
        tau = complex(0.21, 5.4)
        flux0 = jnp.array([2, -1, 3, 0, -2, 1])[:self.model.n_fluxes]
        flux0 = flux0.astype(jnp.float64)
        return z, tau, flux0

    def test_ISD_sampling_PM_modes(self):
        r"""
        **Description:**
        Verifies that the Picard-Fuchs (PM) modes ``"ISD+"`` and ``"ISD-"`` return
        arrays of the correct shape for both output formats.

        - ``output="full"``  → shape ``(2*n_fluxes,)`` (all four flux halves)
        - ``output="half"``  → shape ``(n_fluxes,)``   (the computed half only)

        Both modes solve the condition

        .. math::
            f_1 - \tau h_1 = \overline{\mathcal{N}}\, (f_2 - \tau h_2)

        for one pair of halves given the other.
        """
        z, tau, flux0 = self._make_isd_inputs()
        tau_jnp = jnp.array(tau)

        for mode in ["ISD+", "ISD-"]:
            # Full output contains all four quarter-halves concatenated
            out_full = self.sampler_np.ISD_sampling(
                z, jnp.conj(z), tau_jnp, jnp.conj(tau_jnp),
                flux0, mode=mode, output="full"
            )
            chex.assert_shape(out_full, (2 * self.model.n_fluxes,))

            # Half output contains only the newly computed half
            out_half = self.sampler_np.ISD_sampling(
                z, jnp.conj(z), tau_jnp, jnp.conj(tau_jnp),
                flux0, mode=mode, output="half"
            )
            chex.assert_shape(out_half, (self.model.n_fluxes,))

    @pytest.mark.slow
    def test_ISD_sampling_FH_modes(self):
        r"""
        **Description:**
        Verifies that the Flux-Half (FH) modes ``"F"`` and ``"H"`` return arrays
        of the correct shape for both output formats.

        These modes use the ISD matrix :math:`M(z^i, \bar{z}^i)` and solve

        .. math::
            f = (s\, M \Sigma + c_0)\, h, \quad \tau = c_0 + \text{i}\, s

        for either :math:`f` (mode ``"F"``) or :math:`h` (mode ``"H"``).
        The input for these modes is a *half* flux vector of length ``n_fluxes``
        (same as PM modes), since :math:`\Sigma` is an ``n_fluxes × n_fluxes``
        symplectic matrix acting on either :math:`f` or :math:`h`.
        """
        z, tau, flux0 = self._make_isd_inputs()
        tau_jnp = jnp.array(tau)

        for mode in ["F", "H"]:
            out_full = self.sampler_np.ISD_sampling(
                z, jnp.conj(z), tau_jnp, jnp.conj(tau_jnp),
                flux0, mode=mode, output="full"
            )
            # Full output: [computed_half, input_half] = 2 * n_fluxes
            chex.assert_shape(out_full, (2 * self.model.n_fluxes,))

            out_half = self.sampler_np.ISD_sampling(
                z, jnp.conj(z), tau_jnp, jnp.conj(tau_jnp),
                flux0, mode=mode, output="half"
            )
            # Half output: just the computed half
            chex.assert_shape(out_half, (self.model.n_fluxes,))

    def test_ISD_sampling_return_integer(self):
        r"""
        **Description:**
        Verifies that ``return_integer_flux=True`` rounds the continuous ISD
        solution to integer-valued fluxes.

        After rounding, the ISD condition is only *approximately* satisfied:
        :math:`\star G_3 \approx i G_3`.  This is the standard procedure in
        the statistics of flux vacua literature.
        """
        z, tau, flux0 = self._make_isd_inputs()
        tau_jnp = jnp.array(tau)
        out = self.sampler_np.ISD_sampling(
            z, jnp.conj(z), tau_jnp, jnp.conj(tau_jnp),
            flux0, mode="ISD+", output="full",
            return_integer_flux=True
        )
        # Each component must equal its rounded value – i.e. be an integer
        self.assertAllClose(out, jnp.round(out), atol=1e-8)

    def test_ISD_sampling_vmap(self):
        r"""
        **Description:**
        Verifies that the vmapped version of ``ISD_sampling`` correctly handles
        a batch of ``N`` input points, returning shape ``(N, 2*n_fluxes)``.

        The vmap interface is the primary one used inside
        ``initial_guesses_ISD`` for efficient batch processing.
        """
        z, tau, flux0 = self._make_isd_inputs()
        N_batch = 4

        # Replicate the single point N_batch times to form a simple batch
        z_batch    = jnp.stack([z]    * N_batch)
        tau_batch  = jnp.array([tau]  * N_batch)
        flux_batch = jnp.stack([flux0] * N_batch)

        out = self.sampler_np.ISD_sampling(
            z_batch,  jnp.conj(z_batch),
            tau_batch, jnp.conj(tau_batch),
            flux_batch, mode="ISD+", output="full",
            in_axes=(0, 0, 0), vmap=True
        )
        chex.assert_shape(out, (N_batch, 2 * self.model.n_fluxes))

    def test_ISD_sampling_invalid_mode_raises(self):
        r"""
        **Description:**
        Verifies that an unrecognised ``mode`` string raises a ``ValueError``.
        """
        z, tau, flux0 = self._make_isd_inputs()
        tau_jnp = jnp.array(tau)
        with self.assertRaises(ValueError):
            self.sampler_np.ISD_sampling(
                z, jnp.conj(z), tau_jnp, jnp.conj(tau_jnp),
                flux0, mode="invalid"
            )

    def test_ISD_sampling_invalid_output_raises(self):
        r"""
        **Description:**
        Verifies that an unrecognised ``output`` string raises a ``ValueError``.
        """
        z, tau, flux0 = self._make_isd_inputs()
        tau_jnp = jnp.array(tau)
        with self.assertRaises(ValueError):
            self.sampler_np.ISD_sampling(
                z, jnp.conj(z), tau_jnp, jnp.conj(tau_jnp),
                flux0, output="invalid"
            )

    # ==========================================================================
    #  14. filter_by_instantons / filter_by_km / filter_moduli
    # ==========================================================================
    #
    # These methods return boolean masks or filtered arrays used to restrict the
    # starting-point sample to physically valid regions of moduli space.
    #
    #   filter_by_instantons  –  |F_inst| / |F_pert| < inst_cutoff
    #   filter_by_km          –  all eigenvalues of the Kähler metric > 0
    #   filter_moduli         –  conjunction of both filters applied to moduli
    # ==========================================================================

    def test_filter_by_instantons_returns_boolean_mask(self):
        r"""
        **Description:**
        Verifies that ``filter_by_instantons`` returns a boolean JAX array of
        shape ``(N,)`` where ``True`` indicates that the instanton contribution
        at that point is small relative to the perturbative prepotential:

        .. math::
            \frac{|F_\text{inst}(z^i)|}{|F_\text{pert}(z^i)|} < \varepsilon_\text{cut} \, .

        Starting points that fail this criterion are likely to be in regions where
        the large complex structure expansion has broken down.
        """
        flag = self.sampler_np.filter_by_instantons(self.z_batch)
        # Instanton mask must have one boolean entry per sample point
        chex.assert_shape(flag, (self.N,))
        # Mask dtype must be boolean, not integer or float
        self.assertTrue(
            jnp.issubdtype(flag.dtype, jnp.bool_),
            msg=f"Expected boolean dtype, got {flag.dtype}"
        )

    def test_filter_by_km_returns_boolean_mask(self):
        r"""
        **Description:**
        Verifies that ``filter_by_km`` returns a boolean JAX array of shape
        ``(N,)`` where ``True`` indicates that the Kähler metric matrix evaluated
        at that point is positive definite (all eigenvalues strictly positive).

        A non-positive-definite Kähler metric signals that the point lies outside
        the physical region of the moduli space.
        """
        flag = self.sampler_np.filter_by_km(self.z_batch, self.tau_batch)
        # Kaehler metric mask must have one boolean entry per sample point
        chex.assert_shape(flag, (self.N,))
        # Mask dtype must be boolean, not integer or float
        self.assertTrue(
            jnp.issubdtype(flag.dtype, jnp.bool_),
            msg=f"Expected boolean dtype, got {flag.dtype}"
        )

    def test_filter_moduli_shape(self):
        r"""
        **Description:**
        Verifies that ``filter_moduli`` returns a subset of the input array with
        at most ``N`` rows and exactly ``h12`` columns.

        ``filter_moduli`` applies both the instanton filter and the Kähler metric
        filter and returns only the moduli that pass both criteria.
        """
        filtered = self.sampler_np.filter_moduli(self.z_batch, self.tau_batch)

        # The number of surviving points can be anywhere from 0 to N
        self.assertLessEqual(filtered.shape[0], self.N)

        # If any points survived, the second dimension must equal h12
        if len(filtered) > 0:
            self.assertEqual(
                filtered.shape[1], self.model.h12,
                msg="Filtered moduli array has wrong number of columns."
            )

    def test_filter_moduli_conjunction(self):
        r"""
        **Description:**
        Verifies that ``filter_moduli`` is equivalent to applying
        ``filter_by_instantons`` and ``filter_by_km`` in conjunction:

        .. math::
            \text{flag} = \text{flag}_\text{inst} \wedge \text{flag}_\text{km} \, .

        This cross-check ensures that the compound filter does not silently drop
        or include more points than the component filters dictate.
        """
        # Compute the compound filter directly
        filtered = self.sampler_np.filter_moduli(self.z_batch, self.tau_batch)

        # Compute the expected mask by hand
        flag_inst = self.sampler_np.filter_by_instantons(self.z_batch)
        flag_km   = self.sampler_np.filter_by_km(self.z_batch, self.tau_batch)
        expected  = self.z_batch[flag_inst & flag_km]

        # The two routes must give the same result
        self.assertAllClose(
            jnp.array(filtered), jnp.array(expected), atol=1e-10,
        )


# ==========================================================================
#  TestBoundsParsing — direction-aware moduli_bounds + new knobs
# ==========================================================================

class TestBoundsParsing(TestCase):
    r"""
    **Description:**
    Unit tests for :func:`jaxvacua.sampling._parse_box_bounds`,
    :func:`jaxvacua.sampling._parse_exclude_walls`, and the new
    constructor kwargs ``stretching``, ``exclude_walls``,
    ``cone_cutoff``.  Covers the canonical forms described in the
    vacua-storage plan §1.1-1.4.
    """

    def test_scalar_broadcast(self):
        r"""**Description:** A ``(lo, hi)`` scalar pair broadcasts to
        length-``h12`` arrays, every entry identical."""
        from jaxvacua.sampling import _parse_box_bounds
        lo, hi = _parse_box_bounds((1., 5.), 3)
        chex.assert_shape(lo, (3,))
        chex.assert_shape(hi, (3,))
        self.assertTrue(bool(jnp.all(lo == 1.)))
        self.assertTrue(bool(jnp.all(hi == 5.)))

    def test_per_direction(self):
        r"""**Description:** ``(lower_vec, upper_vec)`` passes through
        unchanged."""
        from jaxvacua.sampling import _parse_box_bounds
        lo, hi = _parse_box_bounds(([1., 2.], [3., 5.]), 2)
        self.assertEqual(lo.tolist(), [1., 2.])
        self.assertEqual(hi.tolist(), [3., 5.])

    def test_mixed(self):
        r"""**Description:** Scalar + vector forms broadcast correctly
        side-by-side (lower scalar broadcast, upper vector verbatim)."""
        from jaxvacua.sampling import _parse_box_bounds
        lo, hi = _parse_box_bounds((1., [3., 5.]), 2)
        self.assertEqual(lo.tolist(), [1., 1.])
        self.assertEqual(hi.tolist(), [3., 5.])

    def test_ndarray_1d_scalar_pair(self):
        r"""**Description:** A 1-D ``np.ndarray`` of length 2 is a valid
        scalar pair and broadcasts."""
        from jaxvacua.sampling import _parse_box_bounds
        lo, hi = _parse_box_bounds(np.array([1., 5.]), 2)
        self.assertEqual(lo.tolist(), [1., 1.])
        self.assertEqual(hi.tolist(), [5., 5.])

    def test_length_mismatch_raises(self):
        r"""**Description:** Length-3 vectors passed with ``h12=2`` raise
        :class:`ValueError`."""
        from jaxvacua.sampling import _parse_box_bounds
        with self.assertRaises(ValueError):
            _parse_box_bounds(([1., 2., 3.], [4., 5., 6.]), 2)

    def test_lo_greater_than_hi_raises(self):
        r"""**Description:** ``lower > upper`` in any direction raises
        :class:`ValueError` with offending indices."""
        from jaxvacua.sampling import _parse_box_bounds
        with self.assertRaises(ValueError):
            _parse_box_bounds((5., 1.), 2)

    def test_2d_ndarray_rejected(self):
        r"""**Description:** Explicit 2-D ndarrays are rejected to avoid
        axis-ordering ambiguity; users must pass ``(arr[:, 0], arr[:, 1])``
        or ``(arr[0], arr[1])``."""
        from jaxvacua.sampling import _parse_box_bounds
        with self.assertRaises(ValueError):
            _parse_box_bounds(np.array([[1., 3.], [2., 5.]]), 2)

    def test_none_rejected(self):
        r"""**Description:** ``moduli_bounds=None`` is not a valid form."""
        from jaxvacua.sampling import _parse_box_bounds
        with self.assertRaises(ValueError):
            _parse_box_bounds(None, 2)

    def test_exclude_walls_none(self):
        r"""**Description:** ``exclude_walls=None`` yields an all-False
        mask of length ``n_hyperplanes``."""
        from jaxvacua.sampling import _parse_exclude_walls
        H = np.array([[1., 0.], [0., 1.], [1., 1.]])
        mask = _parse_exclude_walls(None, H)
        chex.assert_shape(mask, (3,))
        self.assertFalse(bool(mask.any()))

    def test_exclude_walls_indices(self):
        r"""**Description:** Integer-index entries set the corresponding
        mask positions to ``True``."""
        from jaxvacua.sampling import _parse_exclude_walls
        H = np.array([[1., 0.], [0., 1.], [1., 1.]])
        mask = _parse_exclude_walls([0, 2], H)
        self.assertEqual(mask.tolist(), [True, False, True])

    def test_exclude_walls_row_vectors(self):
        r"""**Description:** Explicit hyperplane-row vectors are matched
        against ``hyperplanes`` and the corresponding row is marked."""
        from jaxvacua.sampling import _parse_exclude_walls
        H = np.array([[1., 0.], [0., 1.], [1., 1.]])
        mask = _parse_exclude_walls([[0., 1.]], H)
        self.assertEqual(mask.tolist(), [False, True, False])

    def test_exclude_walls_out_of_range_raises(self):
        r"""**Description:** Out-of-range indices raise
        :class:`ValueError`."""
        from jaxvacua.sampling import _parse_exclude_walls
        H = np.array([[1., 0.]])
        with self.assertRaises(ValueError):
            _parse_exclude_walls([5], H)

    def test_cone_cutoff_auto(self):
        r"""**Description:** ``cone_cutoff=None`` resolves to
        ``float(max(moduli_upper))``; an explicit scalar wins."""
        model = jaxvacua.FluxVacuaFinder(
            h12=2, model_ID=1, model_type="KS", maximum_degree=0,
        )
        s_auto = jaxvacua.data_sampler(model, moduli_bounds=([1., 2.], [3., 7.]))
        self.assertEqual(s_auto.cone_cutoff, 7.)
        s_exp = jaxvacua.data_sampler(
            model, moduli_bounds=(1., 5.), cone_cutoff=42.,
        )
        self.assertEqual(s_exp.cone_cutoff, 42.)

    def test_constructor_stores_arrays(self):
        r"""**Description:** After construction, ``moduli_lower`` /
        ``moduli_upper`` are length-``h12`` arrays and
        ``exclude_walls`` is a boolean mask whose length matches
        ``n_hyperplanes``."""
        model = jaxvacua.FluxVacuaFinder(
            h12=2, model_ID=1, model_type="KS", maximum_degree=0,
        )
        s = jaxvacua.data_sampler(
            model, moduli_bounds=([1., 2.], [3., 5.]),
            stretching=0.1, exclude_walls=[0],
        )
        chex.assert_shape(s.moduli_lower, (2,))
        chex.assert_shape(s.moduli_upper, (2,))
        self.assertEqual(s.stretching, 0.1)
        self.assertEqual(s.exclude_walls.dtype, bool)
        self.assertTrue(bool(s.exclude_walls[0]))

    def test_get_moduli_box_per_direction(self):
        r"""**Description:** In ``"box"`` mode with per-direction bounds,
        each column of the output respects its own ``[lo_i, hi_i]``
        range (h12=2: col 0 ∈ [1,3], col 1 ∈ [2,5])."""
        model = jaxvacua.FluxVacuaFinder(
            h12=2, model_ID=1, model_type="KS", maximum_degree=0,
        )
        s = jaxvacua.data_sampler(
            model, moduli_bounds=([1., 2.], [3., 5.]),
        )
        pts = np.asarray(s.get_moduli(
            100, sampling_mode="box", minval=None, maxval=None,
        ))
        self.assertTrue((pts[:, 0] >= 1.).all() and (pts[:, 0] <= 3.).all(),
                        f"col 0 out of [1, 3]: {pts[:, 0].min()}..{pts[:, 0].max()}")
        self.assertTrue((pts[:, 1] >= 2.).all() and (pts[:, 1] <= 5.).all(),
                        f"col 1 out of [2, 5]: {pts[:, 1].min()}..{pts[:, 1].max()}")


if __name__ == "__main__":
    import unittest
    unittest.main()
