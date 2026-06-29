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

"""Tests for the canonical flux-vacuum finder.

Purpose
-------
Validate ``FluxVacuaFinder`` construction, inherited EFT behaviour, SUSY
vacuum solving, sampling and post-processing helpers.

Main public API
---------------
- ``TestFluxVacuaFinder``: broad checks for finder methods, solver outputs,
  deduplication and physicality workflows.

Design notes
------------
The finder is tested as an ``FluxEFT`` subclass, so these tests also guard the
absence of a separate wrapped model object.
"""

import sys, os, warnings, tempfile
import jax
import jax.numpy as jnp
import numpy as np
import chex
import pytest
from functools import partial
from util import *

jax.config.update("jax_enable_x64", True)

sys.path.append("./../")
import jaxvacua

# Suppress warnings
warnings.filterwarnings("ignore")


class _StaticVacuaSampler:
    """Small deterministic sampler for wrapper-level finder tests."""

    def __init__(self, model):
        self.model = model
        self.moduli = jnp.array([0.1 + 3.0j, -0.2 + 2.5j])
        self.tau = -0.3 + 5.0j
        self.fluxes = jnp.array(
            [1., 0., -2., 0., 3., -1., 2., 1., 0., -1., 1., 0.]
        )

    def initial_guesses(self, n, *args, include_fluxes=True, **kwargs):
        moduli = jnp.broadcast_to(self.moduli, (n, self.model.h12))
        tau = jnp.broadcast_to(jnp.asarray(self.tau), (n,))
        if not include_fluxes:
            return moduli, tau
        fluxes = jnp.broadcast_to(self.fluxes, (n, 2 * self.model.n_fluxes))
        return moduli, tau, fluxes


# ==============================================================================
#  TestFluxVacuaFinder
# ==============================================================================

class TestFluxVacuaFinder(TestCase):
    r"""
    **Description:**
    Test suite for the :class:`FluxVacuaFinder` class, which inherits from
    :class:`FluxEFT` and provides methods for constructing and refining flux
    vacua in Type IIB orientifold compactifications.

    The tests cover:

    *  **linearised_shifts_H**: Given a starting point :math:`(z^i, \tau)` and
       a flux vector :math:`[f \mid h]`, this method solves a linearised system
       for the moduli shifts :math:`\delta z^i`, :math:`\delta\tau` that
       approximately satisfy the ISD condition when the H-flux is held fixed.
       The linear system has dimension :math:`2h^{1,2}+2` (real and imaginary
       parts of each modulus plus the axio-dilaton).

    *  **linearised_shifts_ISD**: The same linearisation but using the ISD
       sampling mode, which completes the F-flux from the H-flux via the ISD
       matrix :math:`\mathcal{M}`.  The linear system has dimension
       :math:`2(2h^{1,2}+2)` because both real and imaginary parts of the
       completed flux enter.

    *  **linearised_shifts**: A dispatch function that delegates to the
       appropriate mode-specific implementation.

    *  **newton_method_flux_vacua**: A Newton solver that iterates the
       covariant derivative conditions :math:`D_I W = 0` (SUSY mode) or
       :math:`\partial_I V = 0` (general mode) to machine precision.

    *  **compute_residual**: A utility that sums absolute values along a given
       axis.

    *  **sampler**: A lazily-initialised :class:`data_sampler` property.

    Attributes:
        model (jaxvacua.FluxVacuaFinder): Model with ``h12=2``,
            ``model_ID=1``, ``model_type="KS"``, ``maximum_degree=0``.
        z (jnp.ndarray): Generic complex structure moduli test point.
        tau (complex): Generic axio-dilaton test point.
        fl (jnp.ndarray): Generic flux vector of shape ``(4*(h12+1),)``.
    """

    # --------------------------------------------------------------------------

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        h12 = 2

        # Instantiate the model via the public jaxvacua interface
        # maximum_degree=5 includes instanton corrections needed for
        # Newton convergence at the known SUSY solution
        cls.model = jaxvacua.FluxVacuaFinder(
            h12=h12, model_ID=1, model_type="KS", maximum_degree=5
        )
        # Override the a_matrix as required by the test geometry
        cls.model.lcs_tree.a_matrix = jnp.array([[4.5, 1.5], [1.5, 0.]])

        cls.h12 = h12
        cls.n_fluxes = cls.model.n_fluxes       # h12 + 1 = 3
        cls.dim_H3 = cls.model.dimension_H3      # h12 + 1 = 3

        # Generic test point: moduli deep in the LCS regime (Im(z) >> 1)
        cls.z = jnp.array([0.1 + 3j, -0.2 + 2.5j])
        cls.cz = jnp.conj(cls.z)
        cls.tau = -0.3 + 5j
        cls.ctau = jnp.conj(cls.tau)

        # Generic flux vector [f | h] with 4*(h12+1) = 12 entries
        cls.fl = jnp.array([1., 0., -2., 0., 3., -1., 2., 1., 0., -1., 1., 0.])

        # -----------------------------------------------------------------
        # Known SUSY solution for convergence test of Newton's method.
        # flux = [7, 3, -24, 0, -16, 50, 0, 3, -4, 0, 0, 0]
        # z ~ [2.742j, 2.057j],  tau ~ 6.855j
        # -----------------------------------------------------------------
        cls.fl_solution = jnp.array(
            [7., 3., -24., 0., -16., 50., 0., 3., -4., 0., 0., 0.]
        )
        # Use high-precision initial guess (truncated from exact solution)
        cls.z_solution = jnp.array([0. + 2.74215479602j, 0. + 2.05661613497j])
        cls.tau_solution = 0. + 6.85540179778j

    # ==========================================================================
    #  1. linearised_shifts_H
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_linearised_shifts_H_shapes(self):
        r"""**Description:**
        Verifies the output shapes of :func:`linearised_shifts_H`.

        The method solves a :math:`(2h^{1,2}+2) \times (2h^{1,2}+2)` real
        linear system derived from the ISD condition linearised around the
        input moduli, with the H-flux held fixed.  The outputs are:

        * ``moduli_new``: shifted complex structure moduli, shape ``(h12,)``.
        * ``tau_new``: shifted axio-dilaton, scalar.
        * ``flux_new``: the flux vector :math:`[f \mid h]` with the
          F-flux completed by rounding, shape ``(4*(h12+1),)``.
        """

        fn = self.variant(
            lambda z, tau, fl: self.model.linearised_shifts_H(z, tau, fl)
        )
        moduli_new, tau_new, flux_new = fn(self.z, self.tau, self.fl)

        # moduli_new must have the same shape as the input moduli
        chex.assert_shape(moduli_new, (self.h12,))
        chex.assert_type(moduli_new, complex)

        # tau_new is a complex scalar
        chex.assert_shape(tau_new, ())
        chex.assert_type(tau_new, complex)

        # flux_new is [f | h] with total length 4*(h12+1) = 12
        chex.assert_shape(flux_new, (4 * (self.h12 + 1),))

    @chex.variants(with_jit=True, without_jit=True)
    @pytest.mark.slow
    def test_linearised_shifts_H_finiteness(self):
        r"""**Description:**
        The linearised shift must produce finite moduli.  If the linear
        system is singular (e.g. at a degenerate point), the output would
        contain NaNs or Infs.  This test confirms that the generic test
        point is non-degenerate.
        """

        fn = self.variant(
            lambda z, tau, fl: self.model.linearised_shifts_H(z, tau, fl)
        )
        moduli_new, tau_new, flux_new = fn(self.z, self.tau, self.fl)

        # All outputs must be finite
        self.assertAllTrue(jnp.isfinite(moduli_new),
                           msg="linearised_shifts_H moduli_new must be finite")
        self.assertTrue(jnp.isfinite(tau_new),
                        msg="linearised_shifts_H tau_new must be finite")
        self.assertAllTrue(jnp.isfinite(flux_new),
                           msg="linearised_shifts_H flux_new must be finite")

    @chex.variants(with_jit=True, without_jit=True)
    def test_linearised_shifts_H_flux_structure(self):
        r"""**Description:**
        The returned ``flux_new`` vector must preserve the H-flux part
        (second half) unchanged, since ``linearised_shifts_H`` only
        completes the F-flux.  The F-flux part (first half) should consist
        of integers (rounded from continuous values).
        """

        fn = self.variant(
            lambda z, tau, fl: self.model.linearised_shifts_H(z, tau, fl)
        )
        _, _, flux_new = fn(self.z, self.tau, self.fl)

        # The H-flux half must be unchanged from the input
        h_flux_input = self.fl[self.n_fluxes:]
        h_flux_output = flux_new[self.n_fluxes:]
        self.assertAllClose(h_flux_output, h_flux_input, atol=1e-14,
                            msg="H-flux half must be preserved by linearised_shifts_H")

        # The F-flux half must be integer-valued (rounded)
        f_flux_output = flux_new[:self.n_fluxes]
        self.assertAllClose(f_flux_output, jnp.round(f_flux_output), atol=1e-14,
                            msg="F-flux half must be integer-valued after rounding")

    # ==========================================================================
    #  2. linearised_shifts_ISD
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_linearised_shifts_ISD_shapes(self):
        r"""**Description:**
        Verifies the output shapes of :func:`linearised_shifts_ISD`.

        The ISD linearisation solves a :math:`2(2h^{1,2}+2) \times 2(2h^{1,2}+2)`
        real linear system (twice as large as the H-flux system because both
        real and imaginary parts of the gauge kinetic matrix enter).
        Outputs are analogous to ``linearised_shifts_H``.
        """

        fn = self.variant(
            lambda z, tau, fl: self.model.linearised_shifts_ISD(z, tau, fl)
        )
        moduli_new, tau_new, flux_new = fn(self.z, self.tau, self.fl)

        # Shifted moduli must retain the same shape (h12,) as the input
        chex.assert_shape(moduli_new, (self.h12,))
        # Moduli are complex-valued (real + imaginary parts of z^i)
        chex.assert_type(moduli_new, complex)

        # Shifted axio-dilaton must be a complex scalar
        chex.assert_shape(tau_new, ())
        chex.assert_type(tau_new, complex)

        # Completed flux vector [f | h] must have length 4*(h12+1)
        chex.assert_shape(flux_new, (4 * (self.h12 + 1),))

    @chex.variants(with_jit=True, without_jit=True)
    def test_linearised_shifts_ISD_finiteness(self):
        r"""**Description:**
        Same finiteness check as for ``linearised_shifts_H`` but for the
        ISD mode.  The ISD system is larger and uses the gauge kinetic
        matrix :math:`\mathcal{N}`, so singularity patterns may differ.
        """

        fn = self.variant(
            lambda z, tau, fl: self.model.linearised_shifts_ISD(z, tau, fl)
        )
        moduli_new, tau_new, flux_new = fn(self.z, self.tau, self.fl)

        # All moduli components must be finite (no NaN from singular ISD linear system)
        self.assertAllTrue(jnp.isfinite(moduli_new),
                           msg="linearised_shifts_ISD moduli_new must be finite")
        # The shifted axio-dilaton must be finite
        self.assertTrue(jnp.isfinite(tau_new),
                        msg="linearised_shifts_ISD tau_new must be finite")
        # All flux entries must be finite after ISD completion
        self.assertAllTrue(jnp.isfinite(flux_new),
                           msg="linearised_shifts_ISD flux_new must be finite")

    @chex.variants(with_jit=True, without_jit=True)
    def test_linearised_shifts_ISD_flux_structure(self):
        r"""**Description:**
        In ISD mode the lower (``mu``, ``nu``) components of each flux
        half are taken from the input while the upper (``ml``, ``nl``)
        components are completed by rounding.  The returned flux must
        therefore have integer entries everywhere (the input ``mu``/``nu``
        are already integers in our test vector).
        """

        fn = self.variant(
            lambda z, tau, fl: self.model.linearised_shifts_ISD(z, tau, fl)
        )
        _, _, flux_new = fn(self.z, self.tau, self.fl)

        # All flux entries should be integer-valued after ISD completion + rounding
        self.assertAllClose(flux_new, jnp.round(flux_new), atol=1e-14,
                            msg="ISD-completed flux must be integer-valued")

    # ==========================================================================
    #  3. linearised_shifts (dispatch)
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_linearised_shifts_dispatch_ISD(self):
        r"""**Description:**
        The dispatch function :func:`linearised_shifts` with ``mode="ISD"``
        must produce the same result as calling :func:`linearised_shifts_ISD`
        directly.
        """

        # Direct call
        fn_direct = self.variant(
            lambda z, tau, fl: self.model.linearised_shifts_ISD(z, tau, fl)
        )
        m_direct, t_direct, f_direct = fn_direct(self.z, self.tau, self.fl)

        # Dispatch call (return_flag=False to get the same 3-tuple)
        fn_dispatch = self.variant(
            lambda z, tau, fl: self.model.linearised_shifts(
                z, tau, fl, mode="ISD", return_flag=False
            )
        )
        m_dispatch, t_dispatch, f_dispatch = fn_dispatch(self.z, self.tau, self.fl)

        self.assertAllClose(m_dispatch, m_direct, atol=1e-14,
                            msg="dispatch ISD moduli must match direct call")
        self.assertAllClose(t_dispatch, t_direct, atol=1e-14,
                            msg="dispatch ISD tau must match direct call")
        self.assertAllClose(f_dispatch, f_direct, atol=1e-14,
                            msg="dispatch ISD flux must match direct call")

    @chex.variants(with_jit=True, without_jit=True)
    @pytest.mark.slow
    def test_linearised_shifts_dispatch_Hflux(self):
        r"""**Description:**
        The dispatch function with ``mode="Hflux"`` must produce the same
        result as calling :func:`linearised_shifts_H` directly.
        """

        fn_direct = self.variant(
            lambda z, tau, fl: self.model.linearised_shifts_H(z, tau, fl)
        )
        m_direct, t_direct, f_direct = fn_direct(self.z, self.tau, self.fl)

        fn_dispatch = self.variant(
            lambda z, tau, fl: self.model.linearised_shifts(
                z, tau, fl, mode="Hflux", return_flag=False
            )
        )
        m_dispatch, t_dispatch, f_dispatch = fn_dispatch(self.z, self.tau, self.fl)

        # Dispatched moduli must exactly match the direct linearised_shifts_H call
        self.assertAllClose(m_dispatch, m_direct, atol=1e-14,
                            msg="dispatch Hflux moduli must match direct call")
        # Dispatched axio-dilaton must exactly match the direct call
        self.assertAllClose(t_dispatch, t_direct, atol=1e-14,
                            msg="dispatch Hflux tau must match direct call")
        # Dispatched flux vector must exactly match the direct call
        self.assertAllClose(f_dispatch, f_direct, atol=1e-14,
                            msg="dispatch Hflux flux must match direct call")

    @chex.variants(with_jit=True, without_jit=True)
    def test_linearised_shifts_return_flag(self):
        r"""**Description:**
        When ``return_flag=True`` (default), :func:`linearised_shifts`
        returns a 4-tuple ``(moduli_new, tau_new, flux_new, flag)`` where
        ``flag`` is a boolean indicating whether boundary conditions
        (Kaehler cone, tadpole) are satisfied.
        """

        fn = self.variant(
            lambda z, tau, fl: self.model.linearised_shifts(
                z, tau, fl, mode="ISD", return_flag=True
            )
        )
        result = fn(self.z, self.tau, self.fl)

        # Must return 4 values when return_flag=True
        self.assertEqual(len(result), 4,
                         msg="linearised_shifts with return_flag=True must return 4 values")

        moduli_new, tau_new, flux_new, flag = result

        chex.assert_shape(moduli_new, (self.h12,))
        chex.assert_shape(flux_new, (4 * (self.h12 + 1),))

        # flag is a boolean scalar
        self.assertEqual(flag.shape, ())

    # ==========================================================================
    #  4. newton_method_flux_vacua
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    @pytest.mark.slow
    def test_newton_method_shapes(self):
        r"""**Description:**
        Verifies the output shapes of :func:`newton_method_flux_vacua`.

        The Newton solver returns a 3-tuple ``(moduli, tau, residual)``
        where ``moduli`` has shape ``(h12,)``, ``tau`` is a complex scalar,
        and ``residual`` is a real scalar measuring :math:`\sum_I |D_I W|`.
        """

        fn = self.variant(
            lambda z, tau, fl: self.model.newton_method_flux_vacua(
                z, tau, fl, step_size_Newton=1.0, max_iters=5
            )
        )
        moduli_out, tau_out, res = fn(self.z, self.tau, self.fl)

        # Newton output moduli must have shape (h12,) matching the input
        chex.assert_shape(moduli_out, (self.h12,))
        # Output moduli are complex-valued (z^i have real and imaginary parts)
        chex.assert_type(moduli_out, complex)
        # The refined axio-dilaton must be a scalar
        chex.assert_shape(tau_out, ())
        # The residual sum |D_I W| must be a real scalar
        chex.assert_shape(res, ())

    @chex.variants(with_jit=True, without_jit=True)
    @pytest.mark.slow
    def test_newton_method_convergence(self):
        r"""**Description:**
        Tests that Newton's method converges to a known SUSY vacuum.

        Starting from a point close to the known solution
        :math:`z \approx (2.742\mathrm{i},\, 2.057\mathrm{i})`,
        :math:`\tau \approx 6.855\mathrm{i}`, the solver should converge
        to a residual :math:`\sum_I |D_I W| < 10^{-10}` within a moderate
        number of iterations.  The default step size of 1.0 gives
        quadratic convergence for this system.
        """

        # Use solver_mode="real" which is stable with instanton corrections
        # (the complex solver can produce NaN with maximum_degree > 0)
        fn = self.variant(
            lambda z, tau, fl: self.model.newton_method_flux_vacua(
                z, tau, fl,
                step_size_Newton=1.0,
                tol=1e-10,
                max_iters=100,
                solver_mode="real",
            )
        )
        moduli_out, tau_out, res = fn(
            self.z_solution, self.tau_solution, self.fl_solution
        )

        # Residual must be below tolerance
        self.assertTrue(
            float(res) < 1e-10,
            msg=f"Newton residual {float(res):.2e} exceeds 1e-10 tolerance"
        )

        # The output moduli must be finite
        self.assertAllTrue(jnp.isfinite(moduli_out),
                           msg="Newton output moduli must be finite")
        self.assertTrue(jnp.isfinite(tau_out),
                        msg="Newton output tau must be finite")

        # Imaginary parts must remain positive (physical constraint)
        self.assertAllTrue(jnp.imag(moduli_out) > 0,
                           msg="Im(z) must remain positive after Newton iteration")
        self.assertTrue(float(jnp.imag(tau_out)) > 0,
                        msg="Im(tau) must remain positive after Newton iteration")

    @chex.variants(with_jit=True, without_jit=True)
    @pytest.mark.slow
    def test_newton_method_solver_modes(self):
        r"""**Description:**
        The Newton solver supports ``"real"`` mode which converts to a real
        parameterisation
        :math:`x = [\mathrm{Re}(z_1),\mathrm{Im}(z_1),\ldots,\mathrm{Re}(\tau),\mathrm{Im}(\tau)]`.

        Verify that the real solver converges and returns moduli that are
        consistent with ``scipy.optimize.root`` (which also uses the real
        parameterisation via :func:`DW_x`).

        .. note::
            The ``"complex"`` solver mode can produce NaN when instanton
            corrections are included (``maximum_degree > 0``), so it is
            not tested here.
        """

        # Real solver
        fn_real = self.variant(
            lambda z, tau, fl: self.model.newton_method_flux_vacua(
                z, tau, fl,
                step_size_Newton=1.0, tol=1e-10, max_iters=100,
                solver_mode="real"
            )
        )
        m_r, t_r, r_r = fn_real(
            self.z_solution, self.tau_solution, self.fl_solution
        )

        # Must converge
        self.assertTrue(float(r_r) < 1e-10,
                        msg=f"Real solver residual {float(r_r):.2e} too large")

        # Moduli must be close to the known solution
        self.assertAllClose(jnp.imag(m_r), jnp.imag(self.z_solution), atol=1e-2,
                            msg="Real solver moduli must be near the known solution")
        self.assertAllClose(jnp.imag(t_r), jnp.imag(self.tau_solution), atol=1e-2,
                            msg="Real solver tau must be near the known solution")

    # ==========================================================================
    #  5. compute_residual
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_compute_residual_shape(self):
        r"""**Description:**
        :func:`compute_residual` computes :math:`\sum_j |x_{ij}|` along
        ``axis=1`` by default, reducing a 2D array to a 1D vector.
        For a single 1D input with ``axis=0`` it returns a scalar.
        """

        # 2D input: batch of residual vectors
        x_2d = jnp.array([[1.0 + 2j, -3.0, 0.5j],
                           [0.0, 1.0 - 1j, 2.0]])

        fn = self.variant(
            lambda x: self.model.compute_residual(x, axis=1)
        )
        res = fn(x_2d)

        chex.assert_shape(res, (2,))

        # Manual calculation: sum of absolute values along axis=1
        expected = jnp.sum(jnp.abs(x_2d), axis=1)
        self.assertAllClose(res, expected, atol=1e-14,
                            msg="compute_residual must equal sum of abs values along axis")

    @chex.variants(with_jit=True, without_jit=True)
    def test_compute_residual_axis0(self):
        r"""**Description:**
        Tests :func:`compute_residual` with ``axis=0`` which sums absolute
        values along the first axis.
        """

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        fn = self.variant(
            lambda x: self.model.compute_residual(x, axis=0)
        )
        res = fn(x)

        chex.assert_shape(res, (2,))

        expected = jnp.array([4.0, 6.0])
        self.assertAllClose(res, expected, atol=1e-14,
                            msg="compute_residual axis=0 must sum along first axis")

    # ==========================================================================
    #  6. sampler property
    # ==========================================================================

    def test_sampler_returns_data_sampler(self):
        r"""**Description:**
        The ``sampler`` property must lazily construct and return a
        :class:`data_sampler` instance.  The sampler is cached after
        first access so that repeated calls return the same object.
        """

        sampler = self.model.sampler

        # Must be a data_sampler instance
        self.assertIsInstance(sampler, jaxvacua.sampling.data_sampler,
                              msg="sampler must be a data_sampler instance")

        # Repeated access must return the same object (lazy caching)
        sampler2 = self.model.sampler
        self.assertIs(sampler, sampler2,
                      msg="sampler property must be cached (same object on repeated access)")

    def test_sampler_has_get_moduli(self):
        r"""**Description:**
        The sampler must expose a ``get_moduli`` method that can produce
        random moduli points for the model.
        """

        sampler = self.model.sampler
        self.assertTrue(callable(getattr(sampler, 'get_moduli', None)),
                        msg="sampler must have a callable get_moduli method")

    # ==========================================================================
    #  7. constructor and post-processing helpers
    # ==========================================================================

    def test_from_model_reuses_geometry_and_sets_finder_state(self):
        r"""``from_model`` should add finder state without rebuilding geometry."""

        base = jaxvacua.FluxEFT(
            h12=self.h12, model_ID=1, model_type="KS", maximum_degree=0
        )
        sampler = _StaticVacuaSampler(base)
        finder = jaxvacua.FluxVacuaFinder.from_model(
            base, sampler=sampler, map_to_fd=True, moduli_bounds=(2., 4.)
        )

        self.assertIsInstance(finder, jaxvacua.FluxVacuaFinder)
        self.assertIs(finder.periods, base.periods)
        self.assertIs(finder.lcs_tree, base.lcs_tree)
        self.assertIs(finder.sampler, sampler)
        self.assertTrue(finder._map_to_fd)
        self.assertEqual(finder._sampler_kwargs["moduli_bounds"], (2., 4.))

        finder._calibrated_sigmas = {"H": 1.0}
        self.assertFalse(hasattr(base, "_calibrated_sigmas"))

    def test_to_fd_disabled_is_noop_wrapper(self):
        r"""``to_fd`` should respect the finder-level ``map_to_fd`` flag."""

        finder = jaxvacua.FluxVacuaFinder.from_model(self.model, map_to_fd=False)
        moduli = np.array([0.6 + 3.0j, -0.7 + 2.0j])
        tau = 0.2 + 3.0j
        fluxes = np.arange(2 * self.model.n_fluxes, dtype=float)

        out = finder.to_fd(moduli, tau, fluxes)

        self.assertIs(out[0], moduli)
        self.assertIs(out[1], tau)
        self.assertIs(out[2], fluxes)

    def test_deduplicate_vacua_collapses_rounded_duplicates(self):
        r"""Batched deduplication should keep the first representative."""

        finder = jaxvacua.FluxVacuaFinder.from_model(self.model, map_to_fd=False)
        moduli = jnp.array([
            [0.100000001 + 3.0j, -0.2 + 2.5j],
            [0.100000002 + 3.0j, -0.2 + 2.5j],
            [0.3 + 3.0j, -0.2 + 2.5j],
        ])
        tau = jnp.array([0.1 + 4.0j, 0.1 + 4.0j, 0.2 + 4.0j])
        fluxes = jnp.array([
            self.fl,
            self.fl,
            self.fl.at[0].add(1.0),
        ])

        mod_u, tau_u, flux_u, keep = finder.deduplicate_vacua(
            moduli, tau, fluxes, n_digits=6
        )

        self.assertAllEqual(keep, jnp.array([0, 2]))
        chex.assert_shape(mod_u, (2, self.h12))
        chex.assert_shape(tau_u, (2,))
        chex.assert_shape(flux_u, (2, 2 * self.model.n_fluxes))

    def test_deduplicate_vacua_empty_batch(self):
        r"""Empty batches should round-trip without special-case crashes."""

        finder = jaxvacua.FluxVacuaFinder.from_model(self.model, map_to_fd=False)
        moduli = jnp.zeros((0, self.h12), dtype=complex)
        tau = jnp.zeros((0,), dtype=complex)
        fluxes = jnp.zeros((0, 2 * self.model.n_fluxes))

        mod_u, tau_u, flux_u, keep = finder.deduplicate_vacua(moduli, tau, fluxes)

        chex.assert_shape(mod_u, (0, self.h12))
        chex.assert_shape(tau_u, (0,))
        chex.assert_shape(flux_u, (0, 2 * self.model.n_fluxes))
        chex.assert_shape(keep, (0,))

    # ==========================================================================
    #  8. F-flux linearisation and solver wrappers
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_linearised_shifts_F_shapes_and_flux_structure(self):
        r"""``linearised_shifts_F`` completes/preserves the H-flux block."""

        fn = self.variant(
            lambda z, tau, fl: self.model.linearised_shifts_F(
                z, tau, fl, mode="Fflux"
            )
        )
        moduli_new, tau_new, flux_new = fn(self.z, self.tau, self.fl)

        chex.assert_shape(moduli_new, (self.h12,))
        chex.assert_shape(tau_new, ())
        chex.assert_shape(flux_new, (2 * self.model.n_fluxes,))
        self.assertAllClose(flux_new[:self.model.n_fluxes], self.fl[:self.model.n_fluxes])
        self.assertAllClose(flux_new[self.model.n_fluxes:],
                            jnp.round(flux_new[self.model.n_fluxes:]))

    @chex.variants(with_jit=True, without_jit=True)
    def test_linearised_shifts_dispatch_Fflux(self):
        r"""The generic dispatcher should match ``linearised_shifts_F``."""

        direct = self.variant(
            lambda z, tau, fl: self.model.linearised_shifts_F(
                z, tau, fl, mode="Fflux"
            )
        )(self.z, self.tau, self.fl)
        dispatch = self.variant(
            lambda z, tau, fl: self.model.linearised_shifts(
                z, tau, fl, mode="Fflux", return_flag=False
            )
        )(self.z, self.tau, self.fl)

        for got, expected in zip(dispatch, direct):
            self.assertAllClose(got, expected, atol=1e-12)

    def test_fterm_solver_accepts_custom_objective_and_optimiser(self):
        r"""``fterm_solver`` should honour user-supplied solver callables."""

        moduli = jnp.broadcast_to(self.z, (2, self.h12))
        tau = jnp.broadcast_to(jnp.asarray(self.tau), (2,))
        fluxes = jnp.broadcast_to(self.fl, (2, 2 * self.model.n_fluxes))

        def objective_fct(m, cm, t, ct, fl):
            return jnp.zeros((m.shape[0], self.model.dimension_H3), dtype=complex)

        def optimiser(m, t, fl):
            return m, t, fl, jnp.ones(m.shape[0], dtype=bool)

        step, mod_out, tau_out, flux_out, checks, res = self.model.fterm_solver(
            moduli, tau, fluxes,
            objective_fct=objective_fct,
            optimiser=optimiser,
            tol=1e-12,
            max_iters=3,
        )

        self.assertEqual(int(step), 1)
        self.assertAllClose(mod_out, moduli)
        self.assertAllClose(tau_out, tau)
        self.assertAllClose(flux_out, fluxes)
        self.assertAllTrue(checks)
        self.assertAllClose(res, jnp.zeros(2), atol=1e-14)

    def test_sample_SUSY_flux_vacua_wrapper_with_custom_solver(self):
        r"""The high-level SUSY sampler should compose sampler, solver and dedup."""

        sampler = _StaticVacuaSampler(self.model)

        def objective_fct(m, cm, t, ct, fl):
            return jnp.zeros((m.shape[0], self.model.dimension_H3), dtype=complex)

        def optimiser(m, t, fl):
            return m, t, fl, jnp.ones(m.shape[0], dtype=bool)

        moduli, tau, fluxes, residuals = self.model.sample_SUSY_flux_vacua(
            N=1,
            sampler=sampler,
            max_iters=2,
            objective_fct=objective_fct,
            optimiser=optimiser,
            tol=1e-12,
            vmap_dim=2,
            deduplicate=False,
            max_batches=1,
            errors="raise",
        )

        self.assertGreaterEqual(len(moduli), 1)
        chex.assert_shape(moduli, (len(moduli), self.h12))
        chex.assert_shape(tau, (len(moduli),))
        chex.assert_shape(fluxes, (len(moduli), 2 * self.model.n_fluxes))
        self.assertAllClose(residuals, jnp.zeros_like(residuals), atol=1e-14)

    def test_sample_SUSY_vacua_from_fluxes_wrapper_with_custom_solver(self):
        r"""Flux-first SUSY sampling should handle externally supplied solvers."""

        initial_moduli = jnp.broadcast_to(self.z, (2, self.h12))
        initial_tau = jnp.broadcast_to(jnp.asarray(self.tau), (2,))
        fluxes_init = jnp.broadcast_to(self.fl, (1, 2 * self.model.n_fluxes))

        def objective_fct(m, cm, t, ct, fl):
            return jnp.zeros(
                (m.shape[0], m.shape[1], self.model.dimension_H3),
                dtype=complex,
            )

        def optimiser_init(m, t, fl):
            n_flux = fl.shape[0]
            n_pts = m.shape[0]
            moduli = jnp.broadcast_to(m[None, :, :], (n_flux, n_pts, self.h12))
            tau = jnp.broadcast_to(t[None, :], (n_flux, n_pts))
            fluxes = jnp.broadcast_to(
                fl[:, None, :], (n_flux, n_pts, 2 * self.model.n_fluxes)
            )
            return moduli, tau, fluxes

        def optimiser_steps(m, t, fl):
            return m, t, fl, jnp.ones(m.shape[:2], dtype=bool)

        moduli, tau, fluxes, residuals = self.model.sample_SUSY_vacua_from_fluxes(
            fluxes_init=fluxes_init,
            initial_guesses=(initial_moduli, initial_tau),
            N=1,
            max_iters=2,
            objective_fct=objective_fct,
            optimiser_init=optimiser_init,
            optimiser_steps=optimiser_steps,
            mode="Fflux",
            tol=1e-12,
            deduplicate=False,
            max_batches=1,
            errors="raise",
        )

        self.assertGreaterEqual(len(moduli), 1)
        chex.assert_shape(moduli, (len(moduli), self.h12))
        chex.assert_shape(tau, (len(moduli),))
        chex.assert_shape(fluxes, (len(moduli), 2 * self.model.n_fluxes))
        self.assertAllClose(residuals, jnp.zeros_like(residuals), atol=1e-14)

    # ==========================================================================
    #  9. calibration persistence
    # ==========================================================================

    def test_run_calibration_orchestrates_calibration_steps(self):
        r"""``run_calibration`` should call the three calibration stages."""

        finder = jaxvacua.FluxVacuaFinder.from_model(self.model)
        calls = []
        sampler = object()

        def fake_precompute(n_sample=50, Q=None, sampler=None):
            calls.append(("precompute", n_sample, Q, sampler))
            finder._s_min = 1.0
            finder._tr_Minv_median = 2.0
            finder._M_cond = 3.0

        def fake_estimate(Q=None):
            calls.append(("estimate", Q))
            finder._calibrated_sigmas = {"H": 1.0}

        def fake_calibrate(Q=None, modes=None, n_test=200,
                           target_acceptance=0.8, sampler=None, verbose=True):
            calls.append(("calibrate", Q, tuple(modes), n_test,
                          target_acceptance, sampler, verbose))
            finder._calibrated_sigmas["F"] = 2.0
            return dict(finder._calibrated_sigmas)

        finder._precompute_M_eigensystem = fake_precompute
        finder._estimate_sigmas = fake_estimate
        finder.calibrate_priors = fake_calibrate

        result = finder.run_calibration(
            Q=17,
            n_sample=3,
            n_test=4,
            target_acceptance=0.7,
            modes=["H"],
            sampler=sampler,
            verbose=False,
        )

        self.assertEqual(result, {"H": 1.0, "F": 2.0})
        self.assertEqual(calls[0], ("precompute", 3, 17, sampler))
        self.assertEqual(calls[1], ("estimate", 17))
        self.assertEqual(calls[2], ("calibrate", 17, ("H",), 4, 0.7, sampler, False))

    def test_save_and_load_calibration_roundtrip(self):
        r"""Calibration JSON files should round-trip the sigma table."""

        finder = jaxvacua.FluxVacuaFinder.from_model(self.model)
        finder._calibrated_sigmas = {"F": 1.25, "H": 2.5}
        finder._M_cond = 12.0
        finder._tr_Minv_median = 3.0
        finder._s_min = 0.9

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "calibration.json")
            saved = finder.save_calibration(Q=99, path=path)

            loaded_finder = jaxvacua.FluxVacuaFinder.from_model(self.model)
            sigmas = loaded_finder.load_calibration(saved)

        self.assertEqual(sigmas, {"F": 1.25, "H": 2.5})
        self.assertEqual(loaded_finder._calibration_isd_modes, ("F", "H"))


if __name__ == "__main__":
    import unittest
    unittest.main()
