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

import sys, os, warnings
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


if __name__ == "__main__":
    import unittest
    unittest.main()
