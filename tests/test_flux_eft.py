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
# Tests for the flux_sector class (jaxvacua/flux_sector.py).
#
# Test groups:
#   1.  test_kahler_metric       – Kähler metric shape, block-diagonal form,
#                                  and equivalence of the two implementations
#   2.  test_W                   – superpotential scalar, dtype, conjugation
#   3.  test_W_gradients         – numerical gradient check for W and conj(W)
#   4.  test_W_gauge_invariant   – gauge-invariant |W|; SL(2,Z) gauge invariance;
#                                  value at known SUSY solution
#   5.  test_dW                  – first derivatives ∂_z W and ∂_τ W; dW wrapper;
#                                  real-valued F-terms; Jacobian; zero at SUSY
#   6.  test_ddW                 – second derivatives ∂_z∂_z W, ∂_z∂_τ W; shape/dtype
#   7.  test_DW                  – Kähler covariant derivatives D_I W; F-terms;
#                                  F-term Jacobian; canonical form; zero at SUSY
#   8.  test_dDW                 – first derivatives of the covariant F-terms ∂_I D_J W;
#                                  all cross-derivatives; conjugation relations
#   9.  test_DDW                 – second Kähler covariant derivatives D_I D_J W;
#                                  all cross-derivatives; both conj=False/True
#  10.  test_V                   – scalar potential V; real/imaginary parts;
#                                  gradient and Hessian; solution tests;
#                                  SUSY/general Hessian equivalence
#  11.  test_tadpole             – D3-tadpole scalar, dtype, solution value
#  12.  test_map_to_FD_tau       – SL(2,Z) fundamental domain: Im(τ)≥√3/2,
#                                  |Re(τ)|≤1/2, |τ|≥1; flux orbit invariance
#  13.  test_dV_complex          – complex partial derivatives of V: shape, dtype,
#                                  conjugation, zero gradient at SUSY minimum
#  14.  test_ddV_complex         – second complex partial derivatives of V:
#                                  shape, dtype, self-consistency
#  15.  test_DDW_matrix          – DDW_matrix_SUSY and DDW_matrix_general:
#                                  shape, Hermitian, conj relations
#  16.  test_mass_matrix         – shape, Hermitian, real non-negative
#                                  eigenvalues at SUSY minimum
#  17.  test_ISD_condition       – |ISD(z,τ,f)|≈0 at the SUSY solution; shape
#  18.  test_projection_fluxes   – Hodge component shapes: (2n,) for all modes
# ------------------------------------------------------------------------------


# Standard libraries
import sys, os, warnings
import jax
import chex
from functools import partial
from scipy.optimize import root
from util import *

sys.path.append("./../")
import jaxvacua


# ==============================================================================
# TestFluxEFT
# ==============================================================================

class TestFluxEFT(TestCase):
    r"""
    **Description:**
    Test suite for :class:`jaxvacua.flux_sector`.

    The tests verify correctness of output shapes, dtypes, algebraic identities,
    conjugation relations, and exact numerical values at a known SUSY critical
    point for every public method of :class:`flux_sector`.

    All derivative tests are run both with and without JIT compilation via the
    ``@chex.variants`` decorator.  Methods that depend on ``conj`` are tested
    with both ``conj=False`` and ``conj=True`` to verify the conjugation
    symmetry :math:`[Q(z,\\tau)]^* = Q^*(\\bar{z},\\bar{\\tau})`.

    .. note::
        The underlying physical model is a KS (Kreuzer-Skarke) geometry with
        :math:`h^{1,2}=2`.  This is the smallest example for which all code
        paths are non-trivial: the gauge kinetic matrix is non-diagonal, the
        instanton prepotential is non-zero, and the scalar-potential Hessian has
        only positive eigenvalues at the SUSY locus.

    Attributes:
        model (jaxvacua.flux_sector): Shared physics model.
        z (jax.Array): Random complex-structure moduli, shape ``(h12,)``.
        cz (jax.Array): Complex conjugate of ``z``.
        tau (complex): Axio-dilaton scalar.
        ctau (complex): Complex conjugate of ``tau``.
        f (jax.Array): Random flux vector, shape ``(4*(h12+1),)``.
        x (jax.Array): Real encoding ``[Re(z), Im(z), Re(τ), Im(τ)]``, shape
            ``(2*(h12+1),)``.
        tau_fd (complex): Axio-dilaton mapped to the SL(2,Z) fundamental domain.
        f_fd (jax.Array): Flux vector transformed consistently with ``tau_fd``.
        ctau_fd (complex): Complex conjugate of ``tau_fd``.
        f_solution (jax.Array): Known integer flux vector admitting a SUSY minimum.
        zsol (jax.Array): SUSY minimum moduli found by ``scipy.optimize.root``.
        tausol (complex): SUSY minimum axio-dilaton.
        solution (jax.Array): Real encoding of the SUSY minimum, shape
            ``(2*(h12+1),)``.
    """

    @classmethod
    def setUpClass(cls):
        r"""
        **Description:**
        Builds the shared physics model and all test fixtures once for the
        entire test class.

        The SUSY solution is found by solving the F-term equations
        :math:`D_I W = 0` via ``scipy.optimize.root`` starting from the
        analytic large-complex-structure values recorded in the source.  The
        test suite fails at setup if the root finder does not converge.
        """
        super().setUpClass()

        h12 = 2

        cls.model = jaxvacua.FluxEFT(
            h12=h12, model_ID=1, model_type="KS", maximum_degree=5
        )

        cls.model.lcs_tree.a_matrix = jnp.array([[4.5,1.5],[1.5,0.]])

        rng = np.random.default_rng(12345)
        cls.z   = jnp.array(
            rng.uniform(-0.5, 0.5, h12) + 1j * rng.uniform(2, 10, h12)
        )
        cls.cz   = jnp.conj(cls.z)
        cls.tau  = rng.uniform(-0.5, 0.5) + 1j * rng.uniform(2, 10)
        cls.ctau = jnp.conj(cls.tau)
        cls.f    = jnp.array(
            rng.integers(-10, 11, 4 * (h12 + 1))
        ).astype(float)

        # Real encoding used by fterms / gradV / gradV_jacobian
        cls.x = jnp.array(
            np.append(
                np.append([cls.z.real], [cls.z.imag], axis=0).T.flatten(),
                [cls.tau.real, cls.tau.imag]
            )
        )

        # SL(2,Z)-fundamental-domain map
        cls.tau_fd, cls.f_fd = cls.model.map_to_FD_tau(cls.tau, cls.f)
        cls.f_fd = jnp.array(cls.f_fd).astype(float)
        cls.ctau_fd = jnp.conj(cls.tau_fd)

        # -----------------------------------------------------------------------
        # Known SUSY solution
        # -----------------------------------------------------------------------
        cls.f_solution = jnp.array([7, 3, -24, 0, -16, 50, 0, 3, -4, 0, 0, 0])

        u1sol  = 2.74215479602462524879172086700112955631003945168828832743217138983767 * 1j
        u2sol  = 2.05661613496943436323419976712599580262262253939859294519039244649420 * 1j
        tausol = 6.85540179778358427172610564536555609784128313762349971439377181031816 * 1j

        x0  = jnp.array([0., u1sol.imag, 0., u2sol.imag, 0., tausol.imag])
        res = root(
            cls.model.DW_x, x0=x0,
            args=(cls.f_solution,),
            method="hybr",
            jac=cls.model.dDW_x
        )

        if not res.success:
            raise ValueError("Unable to find minimum using `scipy.optimize.root`!")

        x = res.x
        cls.tausol  = x[4] + 1j * x[5]
        cls.zsol    = jnp.array([x[0] + 1j * x[1], x[2] + 1j * x[3]])
        cls.czsol   = jnp.conj(cls.zsol)
        cls.ctausol = jnp.conj(cls.tausol)
        cls.solution = jnp.array(x)

    # ==========================================================================
    #  1.  Kähler metric
    # ==========================================================================
    #
    # The Kähler metric K_{I\bar{J}} is the Hermitian matrix of second
    # derivatives of the Kähler potential K with respect to the moduli.
    # For the no-scale structure it has a block-diagonal form:
    #   K_{i\bar{j}}  (complex-structure block, size h12 × h12)
    #   K_{τ\bar{τ}}  (dilaton entry, scalar)
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_kahler_metric(self):
        r"""
        **Description:**
        Verifies that both implementations of the Kähler metric return the
        correct shape :math:`(h^{1,2}+1, h^{1,2}+1)` and that the generic
        and block-diagonal implementations agree.

        The Kähler metric is defined as:

        .. math::
            K_{I\bar{J}} = \partial_I \partial_{\bar{J}} K(z^i, \bar{z}^i, \tau, \bar{\tau})

        where the block-diagonal form follows from the no-scale condition
        :math:`K_{i\bar{\tau}} = 0`.
        """
        KM    = self.variant(
            lambda x, y, z, u: self.model.kahler_metric(x, y, z, u, mode=None)
        )(self.z, self.cz, self.tau, self.ctau)
        KM_bd = self.variant(
            lambda x, y, z, u: self.model.kahler_metric(x, y, z, u, mode="block diagonal")
        )(self.z, self.cz, self.tau, self.ctau)

        chex.assert_shape(KM,    (self.model.h12 + 1, self.model.h12 + 1))
        chex.assert_shape(KM_bd, (self.model.h12 + 1, self.model.h12 + 1))
        self.assertAllClose(KM, KM_bd)

    # ==========================================================================
    #  2.  Superpotential
    # ==========================================================================
    #
    # The Gukov-Vafa-Witten superpotential is
    #   W = ∫ G_3 ∧ Ω = f · Π - τ h · Π
    # where Π is the period vector, f are the RR-fluxes, h the NSNS-fluxes.
    # It is a holomorphic function of z^i and τ.
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_W(self):
        r"""
        **Description:**
        Verifies that the superpotential :math:`W` is a complex scalar and that
        its conjugate satisfies :math:`\overline{W(z,\tau)} = W^*(\bar{z},\bar{\tau})`.

        The Gukov-Vafa-Witten superpotential is:

        .. math::
            W = \int G_3 \wedge \Omega = f \cdot \Pi(z^i) - \tau \, h \cdot \Pi(z^i)

        where :math:`\Pi` is the period vector, :math:`f` the RR-flux vector,
        and :math:`h` the NSNS-flux vector.
        """
        W  = self.variant(
            lambda x, y, z: self.model.superpotential(x, y, z, conj=False)
        )(self.z, self.tau, self.f)
        cW = self.variant(
            lambda x, y, z: self.model.superpotential(x, y, z, conj=True)
        )(self.cz, self.ctau, self.f)

        chex.assert_type(W, complex)
        chex.assert_shape(W, ())
        chex.assert_equal(jnp.conj(W), cW)

    # ==========================================================================
    #  3.  Superpotential gradients
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_W_gradients(self):
        r"""
        **Description:**
        Verifies that the analytical superpotential and its conjugate pass a
        numerical gradient check to second order using ``chex.assert_numerical_grads``.

        The gradient condition tested is:

        .. math::
            \partial_{z^i} W \approx \left.\frac{W(z^i + h) - W(z^i - h)}{2h}\right|_{h \to 0}

        and analogously for second-order finite differences and for :math:`\partial_\tau W`.
        """
        W  = self.variant(
            lambda x, y: self.model.superpotential(x, y, self.f, conj=False)
        )
        cW = self.variant(
            lambda x, y: self.model.superpotential(x, y, self.f, conj=True)
        )

        chex.assert_numerical_grads(f=W,  f_args=(self.z, self.tau),   order=1, atol=1e-10)
        chex.assert_numerical_grads(f=cW, f_args=(self.cz, self.ctau), order=1, atol=1e-10)
        chex.assert_numerical_grads(f=W,  f_args=(self.z, self.tau),   order=2, atol=1e-10)
        chex.assert_numerical_grads(f=cW, f_args=(self.cz, self.ctau), order=2, atol=1e-10)

    # ==========================================================================
    #  4.  Gauge-invariant superpotential
    # ==========================================================================
    #
    # Under the SL(2,Z) Moebius transformation τ → (aτ+b)/(cτ+d) the
    # superpotential transforms as W → (cτ+d) W.  The gauge-invariant
    # combination |W| / Im(τ)^{1/2} is preserved.
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_W_gauge_invariant(self):
        r"""
        **Description:**
        Verifies that the gauge-invariant superpotential satisfies
        :math:`|W_\text{gi}(z, \tau, f)| = |W_\text{gi}(z, \tau_\text{FD}, f_\text{FD})|`
        under SL(2,Z) transformation.

        Also verifies the known numerical value at the SUSY solution:

        .. math::
            W_\text{gi}(z^\text{sol}, \tau^\text{sol}, f^\text{sol})
            \approx -2.037 \times 10^{-8}

        confirming that :math:`|W|` is extremely small at the minimum but does
        not vanish (the minimum breaks SUSY weakly via non-perturbative effects).

        .. note::
            The superpotential is not itself gauge invariant, but the combination
            :math:`e^{K/2} W` used in the scalar potential is gauge invariant
            (up to a Kähler transformation).
        """
        W  = self.variant(
            lambda x, y, z: self.model.superpotential_gauge_invariant(x, y, z, conj=False)
        )(self.z, self.tau, self.f)
        cW = self.variant(
            lambda x, y, z: self.model.superpotential_gauge_invariant(x, y, z, conj=True)
        )(self.cz, self.ctau, self.f)

        chex.assert_type(W, complex)
        chex.assert_shape(W, ())
        chex.assert_equal(jnp.conj(W), cW)

        # Test SL(2,Z) gauge invariance: |W| is invariant under τ → τ_FD
        W_fd  = self.variant(
            lambda x, y, z: self.model.superpotential_gauge_invariant(x, y, z, conj=False)
        )(self.z, self.tau_fd, self.f_fd)
        cW_fd = self.variant(
            lambda x, y, z: self.model.superpotential_gauge_invariant(x, y, z, conj=True)
        )(self.cz, self.ctau_fd, self.f_fd)

        self.assertAllClose(jnp.abs(W),  jnp.abs(W_fd))
        self.assertAllClose(jnp.abs(cW), jnp.abs(cW_fd))

        # Test known value at SUSY solution
        Wsol = self.variant(
            lambda x, y, z: self.model.superpotential_gauge_invariant(x, y, z, conj=False)
        )(self.zsol, self.tausol, self.f_solution)

        self.assertAllClose(Wsol, -2.037e-08, rtol=1e-11, atol=1e-11)

    # ==========================================================================
    #  5.  First derivatives of W
    # ==========================================================================
    #
    # dW_z  : ∂_{z^i} W,  shape (h12,)
    # dW_tau: ∂_τ W,      scalar
    # dW    : stacked [∂_{z^i} W, ∂_τ W], shape (h12+1,)
    # fterms: real encoding [Re(DW_i), Im(DW_i), Re(DW_τ), Im(DW_τ)],
    #         shape (2*(h12+1),) — used by scipy.optimize.root
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_dW(self):
        r"""
        **Description:**
        Verifies shapes, dtypes, and conjugation relations for the first
        derivatives of the superpotential.

        The stacked derivative vector is:

        .. math::
            \partial W = (\partial_{z^1} W, \ldots, \partial_{z^{h^{1,2}}} W,
                          \partial_\tau W)

        and the real-valued encoding for the root finder is:

        .. math::
            [\mathrm{Re}(\partial_{z^i} W), \mathrm{Im}(\partial_{z^i} W),
             \mathrm{Re}(\partial_\tau W), \mathrm{Im}(\partial_\tau W)]

        Also verifies that the F-term Jacobian has the correct shape and that
        all F-terms vanish at the known SUSY solution.
        """
        # ∂_z W and ∂_{z̄} W
        dWz   = self.variant(
            lambda x, y, z: self.model.dW_z(x, y, z, conj=False)
        )(self.z, self.tau, self.f)
        dWcz  = self.variant(
            lambda x, y, z: self.model.dW_z(x, y, z, conj=True)
        )(self.cz, self.ctau, self.f)

        chex.assert_shape(dWz,  (self.model.h12,))
        chex.assert_shape(dWcz, (self.model.h12,))
        self.assertAllClose(jnp.conj(dWz), dWcz)

        # ∂_τ W and ∂_{τ̄} W
        dWtau   = self.variant(
            lambda x, y, z: self.model.dW_tau(x, y, z, conj=False)
        )(self.z, self.tau, self.f)
        dWctau  = self.variant(
            lambda x, y, z: self.model.dW_tau(x, y, z, conj=True)
        )(self.cz, self.ctau, self.f)

        chex.assert_type(dWtau,  complex)
        chex.assert_type(dWctau, complex)
        chex.assert_shape(dWtau,  ())
        chex.assert_shape(dWctau, ())
        chex.assert_equal(jnp.conj(dWtau), dWctau)

        # Stacked dW wrapper
        dW   = self.variant(
            lambda x, y, z: self.model.dW(x, y, z, conj=False)
        )(self.z, self.tau, self.f)
        dWc  = self.variant(
            lambda x, y, z: self.model.dW(x, y, z, conj=True)
        )(self.cz, self.ctau, self.f)

        chex.assert_shape(dW,  (self.model.h12 + 1,))
        chex.assert_shape(dWc, (self.model.h12 + 1,))
        self.assertAllClose(jnp.conj(dW), dWc)
        self.assertAllClose(dW,  jnp.append(dWz,  dWtau))
        self.assertAllClose(dWc, jnp.append(dWcz, dWctau))

        # Real-valued encoding: DW_x computes D_I W (covariant derivative),
        # not ∂_I W (plain gradient). So we compare against model.DW, not model.dW.
        DW_real  = self.variant(self.model.DW_x)(self.x, self.f)
        DW_cov   = self.model.DW(self.z, self.cz, self.tau, self.ctau, self.f)
        DW_test  = jnp.array(
            np.append([DW_cov.real], [DW_cov.imag], axis=0).T.flatten()
        )

        chex.assert_type(DW_real,  float)
        chex.assert_shape(DW_real, (2 * (self.model.h12 + 1),))
        self.assertAllClose(DW_real, DW_test)

        # Jacobian of F-terms
        dDW_real = self.variant(self.model.dDW_x)(self.x, self.f)

        chex.assert_type(dDW_real,  float)
        chex.assert_shape(dDW_real, (
            2 * (self.model.h12 + 1),
            2 * (self.model.h12 + 1)
        ))

        # Zero at SUSY solution
        DW_sol = self.variant(self.model.DW_x)(self.solution, self.f_solution)
        self.assertAllClose(DW_sol, jnp.zeros(2 * (self.model.h12 + 1)))

    # ==========================================================================
    #  6.  Second derivatives of W
    # ==========================================================================
    #
    # ddW_z_z  : ∂_{z^i} ∂_{z^j} W,   shape (h12, h12)
    # ddW_z_tau: ∂_{z^i} ∂_τ W,        shape (h12,)
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_ddW(self):
        r"""
        **Description:**
        Verifies shapes, dtypes, and conjugation relations for the second
        partial derivatives of the superpotential.

        The second derivatives are:

        .. math::
            \partial^2_{z^i z^j} W \quad (h^{1,2} \times h^{1,2}),
            \qquad
            \partial^2_{z^i \tau} W \quad (h^{1,2},)
        """
        # ∂²_{zz} W
        ddWzz    = self.variant(
            lambda x, y, z: self.model.ddW_z_z(x, y, z, conj=False)
        )(self.z, self.tau, self.f)
        ddWczcz  = self.variant(
            lambda x, y, z: self.model.ddW_z_z(x, y, z, conj=True)
        )(self.cz, self.ctau, self.f)

        chex.assert_shape(ddWzz,   (self.model.h12, self.model.h12))
        chex.assert_shape(ddWczcz, (self.model.h12, self.model.h12))
        self.assertAllClose(jnp.conj(ddWzz), ddWczcz)

        # ∂²_{z τ} W
        ddWztau   = self.variant(
            lambda x, y, z: self.model.ddW_z_tau(x, y, z, conj=False)
        )(self.z, self.tau, self.f)
        ddWczctau = self.variant(
            lambda x, y, z: self.model.ddW_z_tau(x, y, z, conj=True)
        )(self.cz, self.ctau, self.f)

        chex.assert_shape(ddWztau,   (self.model.h12,))
        chex.assert_shape(ddWczctau, (self.model.h12,))
        self.assertAllClose(jnp.conj(ddWztau), ddWczctau)

    # ==========================================================================
    #  7.  Kähler covariant F-terms  D_I W
    # ==========================================================================
    #
    # The Kähler covariant derivative is
    #   D_I W = ∂_I W + (∂_I K) W
    # For a no-scale model D_τ W = ∂_τ K · W + ∂_τ W.
    # canonical_fterms returns the canonically normalised F-terms.
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_DW(self):
        r"""
        **Description:**
        Verifies shapes, dtypes, conjugation, consistency with components, and
        the real-valued encoding of the Kähler covariant F-terms.

        The Kähler covariant derivative is defined as:

        .. math::
            D_I W = \partial_I W + (\partial_I K)\, W

        where :math:`I = (z^1, \ldots, z^{h^{1,2}}, \tau)`.  The wrapper
        ``dW`` collects all F-terms into a vector of shape :math:`(h^{1,2}+1,)`.

        Also verifies that the canonical F-terms and all F-terms vanish at the
        known SUSY solution :math:`D_I W = 0`.
        """
        args = (self.z, self.cz, self.tau, self.ctau, self.f)

        # D_{z^i} W
        DWz   = self.variant(
            lambda x, y, z, u, v: self.model.DW_z(x, y, z, u, v, conj=False)
        )(*args)
        DWcz  = self.variant(
            lambda x, y, z, u, v: self.model.DW_z(x, y, z, u, v, conj=True)
        )(*args)

        chex.assert_shape(DWz,  (self.model.h12,))
        chex.assert_shape(DWcz, (self.model.h12,))
        self.assertAllClose(jnp.conj(DWz), DWcz)

        # D_τ W
        DWtau   = self.variant(
            lambda x, y, z, u, v: self.model.DW_tau(x, y, z, u, v, conj=False)
        )(*args)
        DWctau  = self.variant(
            lambda x, y, z, u, v: self.model.DW_tau(x, y, z, u, v, conj=True)
        )(*args)

        chex.assert_type(DWtau,  complex)
        chex.assert_type(DWctau, complex)
        chex.assert_shape(DWtau,  ())
        chex.assert_shape(DWctau, ())
        chex.assert_equal(jnp.conj(DWtau), DWctau)

        # Stacked DW wrapper
        DW   = self.variant(
            lambda x, y, z, u, v: self.model.DW(x, y, z, u, v, conj=False)
        )(*args)
        DWc  = self.variant(
            lambda x, y, z, u, v: self.model.DW(x, y, z, u, v, conj=True)
        )(*args)

        chex.assert_shape(DW,  (self.model.h12 + 1,))
        chex.assert_shape(DWc, (self.model.h12 + 1,))
        self.assertAllClose(jnp.conj(DW), DWc)
        self.assertAllClose(DW,  jnp.append(DWz,  DWtau))
        self.assertAllClose(DWc, jnp.append(DWcz, DWctau))

        # Canonical F-terms
        cDW1,  cDW2  = self.variant(
            lambda x, y, z, u, v: self.model.canonical_fterms(x, y, z, u, v, conj=False)
        )(*args)
        cDWc1, cDWc2 = self.variant(
            lambda x, y, z, u, v: self.model.canonical_fterms(x, y, z, u, v, conj=True)
        )(*args)

        chex.assert_shape(cDW1,  (self.model.h12 + 1,))
        chex.assert_shape(cDWc1, (self.model.h12 + 1,))
        chex.assert_shape(cDW2,  (self.model.h12 + 1,))
        chex.assert_shape(cDWc2, (self.model.h12 + 1,))
        self.assertAllClose(jnp.conj(cDW1), cDWc1)
        self.assertAllClose(jnp.conj(cDW2), cDWc2)

        # Real-valued F-term encoding
        DW_real = self.variant(self.model.DW_x)(self.x, self.f)
        DW_test = jnp.array(
            np.append([DW.real], [DW.imag], axis=0).T.flatten()
        )

        chex.assert_type(DW_real,  float)
        chex.assert_shape(DW_real, (2 * (self.model.h12 + 1),))
        self.assertAllClose(DW_real, DW_test)

        # Jacobian of F-terms
        dDW_real = self.variant(self.model.dDW_x)(self.x, self.f)

        chex.assert_type(dDW_real,  float)
        chex.assert_shape(dDW_real, (
            2 * (self.model.h12 + 1),
            2 * (self.model.h12 + 1)
        ))

        # SUSY solution: DW must vanish at (zsol, tausol, f_solution)
        DW_sol = self.variant(self.model.DW_x)(self.solution, self.f_solution)
        self.assertAllClose(DW_sol, jnp.zeros(2 * (self.model.h12 + 1)))

    # ==========================================================================
    #  8.  Derivatives of the F-terms  ∂_I D_J W
    # ==========================================================================
    #
    # The derivatives ∂_I D_J W and ∂_{\bar I} D_J W appear in the scalar
    # potential Hessian.  All cross-combinations are tested for shape, dtype,
    # and the conjugation identity:
    #   ∂_{\bar I} D_{\bar J} W^* = conj( ∂_I D_J W )
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_dDW(self):
        r"""
        **Description:**
        Verifies shapes, dtypes, and conjugation relations for the first
        derivatives of the covariant F-terms :math:`\partial_I D_J W` and
        :math:`\partial_{\bar I} D_J W`.

        The full set of cross-derivatives tested is:

        .. math::
            \partial_\tau D_\tau W, \quad
            \partial_\tau D_{z^i} W, \quad
            \partial_{\bar\tau} D_\tau W, \quad
            \partial_{\bar\tau} D_{z^i} W, \quad
            \partial_{z^j} D_\tau W, \quad
            \partial_{z^j} D_{z^i} W, \quad
            \partial_{\bar z^j} D_\tau W, \quad
            \partial_{\bar z^j} D_{z^i} W

        All pairs satisfy the symmetry
        :math:`[\partial_I D_J W]^* = \partial_{\bar I} D_{\bar J} W^*`.
        """
        args = (self.z, self.cz, self.tau, self.ctau, self.f)

        # --- conj = False ---
        conj = False
        dDW_tau_ctau = self.variant(
            lambda x, y, z, u, v: self.model.dDW_tau_ctau(x, y, z, u, v, conj=conj)
        )(*args)
        dDW_tau_tau  = self.variant(
            lambda x, y, z, u, v: self.model.dDW_tau_tau(x, y, z, u, v, conj=conj)
        )(*args)
        # d_tau(D_ctau W) must be complex-valued
        chex.assert_type(dDW_tau_ctau, complex)
        # d_tau(D_ctau W) is a scalar — single dilaton derivative of a single F-term
        chex.assert_shape(dDW_tau_ctau, ())
        # d_tau(D_tau W) must be complex-valued
        chex.assert_type(dDW_tau_tau,  complex)
        # d_tau(D_tau W) is a scalar — both indices are the dilaton
        chex.assert_shape(dDW_tau_tau,  ())

        dDW_tau_z  = self.variant(
            lambda x, y, z, u, v: self.model.dDW_tau_z(x, y, z, u, v, conj=conj)
        )(*args)
        dDW_tau_cz = self.variant(
            lambda x, y, z, u, v: self.model.dDW_tau_cz(x, y, z, u, v, conj=conj)
        )(*args)
        # d_tau(D_z W) must be complex-valued
        chex.assert_type(dDW_tau_z, complex)
        # d_tau(D_z W) has shape (h12,) — one entry per complex structure modulus
        chex.assert_shape(dDW_tau_z,  (self.model.h12,))
        # d_tau(D_cz W) must be complex-valued
        chex.assert_type(dDW_tau_cz, complex)
        # d_tau(D_cz W) has shape (h12,) — tau derivative of each conj z-sector F-term
        chex.assert_shape(dDW_tau_cz, (self.model.h12,))

        dDW_z_tau  = self.variant(
            lambda x, y, z, u, v: self.model.dDW_z_tau(x, y, z, u, v, conj=conj)
        )(*args)
        dDW_z_ctau = self.variant(
            lambda x, y, z, u, v: self.model.dDW_z_ctau(x, y, z, u, v, conj=conj)
        )(*args)
        # d_z(D_tau W) must be complex-valued
        chex.assert_type(dDW_z_tau, complex)
        # d_z(D_tau W) has shape (h12,) — z-gradient of the dilaton F-term
        chex.assert_shape(dDW_z_tau,  (self.model.h12,))
        # d_z(D_ctau W) must be complex-valued
        chex.assert_type(dDW_z_ctau, complex)
        # d_z(D_ctau W) has shape (h12,) — z-gradient of the conj-dilaton F-term
        chex.assert_shape(dDW_z_ctau, (self.model.h12,))

        dDW_z_cz   = self.variant(
            lambda x, y, z, u, v: self.model.dDW_z_cz(x, y, z, u, v, conj=conj)
        )(*args)
        dDW_z_z    = self.variant(
            lambda x, y, z, u, v: self.model.dDW_z_z(x, y, z, u, v, conj=conj)
        )(*args)
        # d_z(D_cz W) must be complex-valued
        chex.assert_type(dDW_z_cz, complex)
        # d_z(D_cz W) has shape (h12, h12) — Jacobian of conj z F-terms w.r.t. z
        chex.assert_shape(dDW_z_cz, (self.model.h12, self.model.h12))
        # d_z(D_z W) must be complex-valued
        chex.assert_type(dDW_z_z,  complex)
        # d_z(D_z W) has shape (h12, h12) — Jacobian of z F-terms w.r.t. z
        chex.assert_shape(dDW_z_z,  (self.model.h12, self.model.h12))

        dDW_z   = self.variant(
            lambda x, y, z, u, v: self.model.dDW_z(x, y, z, u, v, conj=conj)
        )(*args)
        dDW_cz  = self.variant(
            lambda x, y, z, u, v: self.model.dDW_cz(x, y, z, u, v, conj=conj)
        )(*args)
        # Combined d_z block must be complex-valued
        chex.assert_type(dDW_z,  complex)
        # d_z(D_I W) has shape (n, h12) — all n F-terms differentiated w.r.t. h12 z-moduli
        chex.assert_shape(dDW_z,  (self.model.h12 + 1, self.model.h12))
        # Combined d_cz block must be complex-valued
        chex.assert_type(dDW_cz, complex)
        # d_cz(D_I W) has shape (n, h12) — all n F-terms differentiated w.r.t. h12 conj z
        chex.assert_shape(dDW_cz, (self.model.h12 + 1, self.model.h12))

        dDW_tau  = self.variant(
            lambda x, y, z, u, v: self.model.dDW_tau(x, y, z, u, v, conj=conj)
        )(*args)
        dDW_ctau = self.variant(
            lambda x, y, z, u, v: self.model.dDW_ctau(x, y, z, u, v, conj=conj)
        )(*args)
        # Combined d_tau block must be complex-valued
        chex.assert_type(dDW_tau,  complex)
        # d_tau(D_I W) has shape (n,) — tau derivative of all n F-terms
        chex.assert_shape(dDW_tau,  (self.model.h12 + 1,))
        # Combined d_ctau block must be complex-valued
        chex.assert_type(dDW_ctau, complex)
        # d_ctau(D_I W) has shape (n,) — conj-tau derivative of all n F-terms
        chex.assert_shape(dDW_ctau, (self.model.h12 + 1,))

        dDW   = self.variant(
            lambda x, y, z, u, v: self.model.dDW(x, y, z, u, v, conj=conj)
        )(*args)
        dDW_c = self.variant(
            lambda x, y, z, u, v: self.model.dDW_c(x, y, z, u, v, conj=conj)
        )(*args)
        # Full holomorphic dDW matrix must be complex-valued
        chex.assert_type(dDW,   complex)
        # dDW has shape (n, n) — full Jacobian d_I(D_J W) over all n fields
        chex.assert_shape(dDW,  (self.model.h12 + 1, self.model.h12 + 1))
        # Full anti-holomorphic dDW_c matrix must be complex-valued
        chex.assert_type(dDW_c, complex)
        # dDW_c has shape (n, n) — full Jacobian d_cI(D_J W) over all n fields
        chex.assert_shape(dDW_c, (self.model.h12 + 1, self.model.h12 + 1))

        # --- conj = True ---
        conj = True
        dDW_ctau_tau   = self.variant(
            lambda x, y, z, u, v: self.model.dDW_tau_ctau(x, y, z, u, v, conj=conj)
        )(*args)
        dDW_ctau_ctau  = self.variant(
            lambda x, y, z, u, v: self.model.dDW_tau_tau(x, y, z, u, v, conj=conj)
        )(*args)
        # Conjugate d_ctau(D_tau W*) must be complex-valued
        chex.assert_type(dDW_ctau_tau,  complex)
        # d_ctau(D_tau W*) is a scalar — single conj-dilaton derivative
        chex.assert_shape(dDW_ctau_tau,  ())
        # Conjugate d_ctau(D_ctau W*) must be complex-valued
        chex.assert_type(dDW_ctau_ctau, complex)
        # d_ctau(D_ctau W*) is a scalar — both indices are conj-dilaton
        chex.assert_shape(dDW_ctau_ctau, ())

        dDW_ctau_cz = self.variant(
            lambda x, y, z, u, v: self.model.dDW_tau_z(x, y, z, u, v, conj=conj)
        )(*args)
        dDW_ctau_z  = self.variant(
            lambda x, y, z, u, v: self.model.dDW_tau_cz(x, y, z, u, v, conj=conj)
        )(*args)
        # Conjugate d_ctau(D_cz W*) must be complex-valued
        chex.assert_type(dDW_ctau_cz, complex)
        # d_ctau(D_cz W*) has shape (h12,) — conj-tau derivative of each conj-z F-term*
        chex.assert_shape(dDW_ctau_cz, (self.model.h12,))
        # Conjugate d_ctau(D_z W*) must be complex-valued
        chex.assert_type(dDW_ctau_z,  complex)
        # d_ctau(D_z W*) has shape (h12,) — conj-tau derivative of each z F-term*
        chex.assert_shape(dDW_ctau_z,  (self.model.h12,))

        dDW_cz_ctau = self.variant(
            lambda x, y, z, u, v: self.model.dDW_z_tau(x, y, z, u, v, conj=conj)
        )(*args)
        dDW_cz_tau  = self.variant(
            lambda x, y, z, u, v: self.model.dDW_z_ctau(x, y, z, u, v, conj=conj)
        )(*args)
        # Conjugate d_cz(D_ctau W*) must be complex-valued
        chex.assert_type(dDW_cz_ctau, complex)
        # d_cz(D_ctau W*) has shape (h12,) — conj-z gradient of conj-tau F-term*
        chex.assert_shape(dDW_cz_ctau, (self.model.h12,))
        # Conjugate d_cz(D_tau W*) must be complex-valued
        chex.assert_type(dDW_cz_tau,  complex)
        # d_cz(D_tau W*) has shape (h12,) — conj-z gradient of tau F-term*
        chex.assert_shape(dDW_cz_tau,  (self.model.h12,))

        dDW_cz_z  = self.variant(
            lambda x, y, z, u, v: self.model.dDW_z_cz(x, y, z, u, v, conj=conj)
        )(*args)
        dDW_cz_cz = self.variant(
            lambda x, y, z, u, v: self.model.dDW_z_z(x, y, z, u, v, conj=conj)
        )(*args)
        # Conjugate d_cz(D_z W*) must be complex-valued
        chex.assert_type(dDW_cz_z,  complex)
        # d_cz(D_z W*) has shape (h12, h12) — Jacobian of conj z F-terms* w.r.t. conj z
        chex.assert_shape(dDW_cz_z,  (self.model.h12, self.model.h12))
        # Conjugate d_cz(D_cz W*) must be complex-valued
        chex.assert_type(dDW_cz_cz, complex)
        # d_cz(D_cz W*) has shape (h12, h12) — Jacobian of z F-terms* w.r.t. conj z
        chex.assert_shape(dDW_cz_cz, (self.model.h12, self.model.h12))

        dDW_cz_c = self.variant(
            lambda x, y, z, u, v: self.model.dDW_z(x, y, z, u, v, conj=conj)
        )(*args)
        dDW_z_c  = self.variant(
            lambda x, y, z, u, v: self.model.dDW_cz(x, y, z, u, v, conj=conj)
        )(*args)
        # Conjugate combined d_cz block must be complex-valued
        chex.assert_type(dDW_cz_c, complex)
        # d_cz(D_I W*) has shape (n, h12) — all n conj F-terms differentiated w.r.t. conj z
        chex.assert_shape(dDW_cz_c, (self.model.h12 + 1, self.model.h12))
        # Conjugate combined d_z block must be complex-valued
        chex.assert_type(dDW_z_c,  complex)
        # d_z(D_I W*) has shape (n, h12) — all n conj F-terms differentiated w.r.t. z
        chex.assert_shape(dDW_z_c,  (self.model.h12 + 1, self.model.h12))

        dDW_ctau_c = self.variant(
            lambda x, y, z, u, v: self.model.dDW_tau(x, y, z, u, v, conj=conj)
        )(*args)
        dDW_tau_c  = self.variant(
            lambda x, y, z, u, v: self.model.dDW_ctau(x, y, z, u, v, conj=conj)
        )(*args)
        # Conjugate combined d_ctau block must be complex-valued
        chex.assert_type(dDW_ctau_c, complex)
        # d_ctau(D_I W*) has shape (n,) — conj-tau derivative of all n conj F-terms
        chex.assert_shape(dDW_ctau_c, (self.model.h12 + 1,))
        # Conjugate combined d_tau block must be complex-valued
        chex.assert_type(dDW_tau_c,  complex)
        # d_tau(D_I W*) has shape (n,) — tau derivative of all n conj F-terms
        chex.assert_shape(dDW_tau_c,  (self.model.h12 + 1,))

        dDW_cc   = self.variant(
            lambda x, y, z, u, v: self.model.dDW(x, y, z, u, v, conj=conj)
        )(*args)
        dDW_c_cc = self.variant(
            lambda x, y, z, u, v: self.model.dDW_c(x, y, z, u, v, conj=conj)
        )(*args)
        # Full conjugate holomorphic dDW matrix must be complex-valued
        chex.assert_type(dDW_cc,   complex)
        # d_cI(D_cJ W*) has shape (n, n) — full conj Jacobian over all n fields
        chex.assert_shape(dDW_cc,  (self.model.h12 + 1, self.model.h12 + 1))
        # Full conjugate anti-holomorphic dDW_c matrix must be complex-valued
        chex.assert_type(dDW_c_cc, complex)
        # d_I(D_cJ W*) has shape (n, n) — full mixed conj Jacobian over all n fields
        chex.assert_shape(dDW_c_cc, (self.model.h12 + 1, self.model.h12 + 1))

        # Conjugation relations: d_cI(D_cJ W*) = conj(d_I(D_J W))
        # d_cz(D_z W*) = conj(d_z(D_cz W)) — z-cz block conjugation symmetry
        self.assertAllClose(dDW_cz_z,    jnp.conj(dDW_z_cz),    rtol=1e-11, atol=1e-11)
        # d_cz(D_cz W*) = conj(d_z(D_z W)) — pure z block conjugation symmetry
        self.assertAllClose(dDW_cz_cz,   jnp.conj(dDW_z_z),     rtol=1e-11, atol=1e-11)
        # d_cz(D_tau W*) = conj(d_z(D_ctau W)) — z-tau mixed block conjugation
        self.assertAllClose(dDW_cz_tau,  jnp.conj(dDW_z_ctau),  rtol=1e-11, atol=1e-11)
        # d_cz(D_ctau W*) = conj(d_z(D_tau W)) — z-tau mixed block conjugation
        self.assertAllClose(dDW_cz_ctau, jnp.conj(dDW_z_tau),   rtol=1e-11, atol=1e-11)
        # d_ctau(D_z W*) = conj(d_tau(D_cz W)) — tau-z mixed block conjugation
        self.assertAllClose(dDW_ctau_z,  jnp.conj(dDW_tau_cz),  rtol=1e-11, atol=1e-11)
        # d_ctau(D_cz W*) = conj(d_tau(D_z W)) — tau-z mixed block conjugation
        self.assertAllClose(dDW_ctau_cz, jnp.conj(dDW_tau_z),   rtol=1e-11, atol=1e-11)
        # d_ctau(D_tau W*) = conj(d_tau(D_ctau W)) — pure tau block conjugation
        self.assertAllClose(dDW_ctau_tau,  jnp.conj(dDW_tau_ctau), rtol=1e-11, atol=1e-11)
        # d_ctau(D_ctau W*) = conj(d_tau(D_tau W)) — pure tau block conjugation
        self.assertAllClose(dDW_ctau_ctau, jnp.conj(dDW_tau_tau),  rtol=1e-11, atol=1e-11)

        print("ARE THESE CORRECT? IF SO; WHAT DOES IT MEAN?")
        # Combined conj-z block of W* equals conjugate of z block of W
        self.assertAllClose(dDW_cz_c,   jnp.conj(dDW_z),   rtol=1e-11, atol=1e-11)
        # Combined z block of W* equals conjugate of conj-z block of W
        self.assertAllClose(dDW_z_c,    jnp.conj(dDW_cz),  rtol=1e-11, atol=1e-11)
        # Combined conj-tau block of W* equals conjugate of tau block of W
        self.assertAllClose(dDW_ctau_c, jnp.conj(dDW_tau), rtol=1e-11, atol=1e-11)
        # Combined tau block of W* equals conjugate of conj-tau block of W
        self.assertAllClose(dDW_tau_c,  jnp.conj(dDW_ctau),rtol=1e-11, atol=1e-11)
        # Full conj-hol dDW(W*) equals conjugate of full hol dDW(W)
        self.assertAllClose(dDW_cc,     jnp.conj(dDW),     rtol=1e-11, atol=1e-11)
        # Full conj anti-hol dDW_c(W*) equals conjugate of full anti-hol dDW_c(W)
        self.assertAllClose(dDW_c_cc,   jnp.conj(dDW_c),   rtol=1e-11, atol=1e-11)

    # ==========================================================================
    #  9.  Second Kähler covariant derivatives  D_I D_J W
    # ==========================================================================
    #
    # The second Kähler covariant derivative of the superpotential appears in
    # the mass matrix of the scalar fields.  Two variants:
    #   DDW_general : general point in moduli space
    #   DDW_SUSY    : specialised form valid at the SUSY locus
    # and the mixed Kähler derivative DcDW_general.
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_DDW(self):
        r"""
        **Description:**
        Verifies shapes, dtypes, conjugation relations, and self-consistency of
        the second Kähler covariant derivatives of the superpotential
        :math:`D_I D_J W` (both holomorphic and anti-holomorphic).

        The block structure of the full matrix is:

        .. math::
            (D_I D_J W)_{I,J \in \{z^1,\ldots,z^{h^{1,2}},\tau\}}

        and the conjugate is:

        .. math::
            (D_{\bar I} D_{\bar J} W^*)_{I,J} = \overline{D_I D_J W}

        Also tests the wrapper ``DDW`` with ``mode="SUSY"`` and ``mode=None``
        and verifies that the SUSY and general implementations agree at the
        SUSY locus.
        """
        args = (self.z, self.cz, self.tau, self.ctau, self.f)

        # --- conj = False ---
        conj = False
        DDW_tau_ctau = self.variant(
            lambda x, y, z, u, v: self.model.DDW_tau_ctau(x, y, z, u, v, conj=conj)
        )(*args)
        DDW_tau_tau  = self.variant(
            lambda x, y, z, u, v: self.model.DDW_tau_tau(x, y, z, u, v, conj=conj)
        )(*args)
        # D_tau(D_ctau W) must be complex-valued
        chex.assert_type(DDW_tau_ctau, complex)
        # D_tau(D_ctau W) is a scalar — second covariant derivative with both dilaton indices
        chex.assert_shape(DDW_tau_ctau, ())
        # D_tau(D_tau W) must be complex-valued
        chex.assert_type(DDW_tau_tau,  complex)
        # D_tau(D_tau W) is a scalar — purely holomorphic dilaton second derivative
        chex.assert_shape(DDW_tau_tau,  ())

        DDW_tau_z  = self.variant(
            lambda x, y, z, u, v: self.model.DDW_tau_z(x, y, z, u, v, conj=conj)
        )(*args)
        DDW_tau_cz = self.variant(
            lambda x, y, z, u, v: self.model.DDW_tau_cz(x, y, z, u, v, conj=conj)
        )(*args)
        # D_tau(D_z W) must be complex-valued
        chex.assert_type(DDW_tau_z, complex)
        # D_tau(D_z W) has shape (h12,) — covariant tau-z mixed second derivative
        chex.assert_shape(DDW_tau_z,  (self.model.h12,))
        # D_tau(D_cz W) must be complex-valued
        chex.assert_type(DDW_tau_cz, complex)
        # D_tau(D_cz W) has shape (h12,) — covariant tau-conj(z) mixed second derivative
        chex.assert_shape(DDW_tau_cz, (self.model.h12,))

        DDW_z_tau  = self.variant(
            lambda x, y, z, u, v: self.model.DDW_z_tau(x, y, z, u, v, conj=conj)
        )(*args)
        DDW_z_ctau = self.variant(
            lambda x, y, z, u, v: self.model.DDW_z_ctau(x, y, z, u, v, conj=conj)
        )(*args)
        # D_z(D_tau W) must be complex-valued
        chex.assert_type(DDW_z_tau, complex)
        # D_z(D_tau W) has shape (h12,) — z-gradient of the covariant dilaton derivative
        chex.assert_shape(DDW_z_tau,  (self.model.h12,))
        # D_z(D_ctau W) must be complex-valued
        chex.assert_type(DDW_z_ctau, complex)
        # D_z(D_ctau W) has shape (h12,) — z-gradient of the conj-tau covariant derivative
        chex.assert_shape(DDW_z_ctau, (self.model.h12,))

        DDW_z_cz = self.variant(
            lambda x, y, z, u, v: self.model.DDW_z_cz(x, y, z, u, v, conj=conj)
        )(*args)
        DDW_z_z  = self.variant(
            lambda x, y, z, u, v: self.model.DDW_z_z(x, y, z, u, v, conj=conj)
        )(*args)
        # D_z(D_cz W) must be complex-valued
        chex.assert_type(DDW_z_cz, complex)
        # D_z(D_cz W) has shape (h12, h12) — mixed hol/anti-hol z-sector mass matrix block
        chex.assert_shape(DDW_z_cz, (self.model.h12, self.model.h12))
        # D_z(D_z W) must be complex-valued
        chex.assert_type(DDW_z_z,  complex)
        # D_z(D_z W) has shape (h12, h12) — purely holomorphic z-sector mass matrix block
        chex.assert_shape(DDW_z_z,  (self.model.h12, self.model.h12))

        DDW_gen  = self.variant(
            lambda x, y, z, u, v: self.model.DDW_general(x, y, z, u, v, conj=conj)
        )(*args)
        DDW_SUSY = self.variant(
            lambda x, y, z, u, v: self.model.DDW_SUSY(x, y, z, u, v, conj=conj)
        )(*args)
        # DDW_general must be complex-valued
        chex.assert_type(DDW_gen,  complex)
        # DDW_general has shape (n, n) — full D_I D_J W matrix over all n = h12+1 fields
        chex.assert_shape(DDW_gen,  (self.model.h12 + 1, self.model.h12 + 1))
        # DDW_SUSY must be complex-valued
        chex.assert_type(DDW_SUSY, complex)
        # DDW_SUSY has shape (n, n) — simplified form valid at the SUSY locus D_I W = 0
        chex.assert_shape(DDW_SUSY, (self.model.h12 + 1, self.model.h12 + 1))

        DDW_wrap_SUSY = self.variant(
            lambda x, y, z, u, v: self.model.DDW(x, y, z, u, v, conj=conj, mode="SUSY")
        )(*args)
        DDW_wrap_gen  = self.variant(
            lambda x, y, z, u, v: self.model.DDW(x, y, z, u, v, conj=conj, mode=None)
        )(*args)
        # DDW wrapper with mode="SUSY" must be complex-valued
        chex.assert_type(DDW_wrap_SUSY, complex)
        # DDW wrapper (SUSY mode) has shape (n, n) — same as direct DDW_SUSY call
        chex.assert_shape(DDW_wrap_SUSY, (self.model.h12 + 1, self.model.h12 + 1))
        # DDW wrapper with mode=None must be complex-valued
        chex.assert_type(DDW_wrap_gen,  complex)
        # DDW wrapper (general mode) has shape (n, n) — same as direct DDW_general call
        chex.assert_shape(DDW_wrap_gen,  (self.model.h12 + 1, self.model.h12 + 1))

        # DDW wrapper with mode="SUSY" must match the direct DDW_SUSY implementation
        self.assertAllClose(DDW_wrap_SUSY, DDW_SUSY, rtol=1e-11, atol=1e-11)
        # DDW wrapper with mode=None must match the direct DDW_general implementation
        self.assertAllClose(DDW_wrap_gen,  DDW_gen,  rtol=1e-11, atol=1e-11)

        # Component consistency: assemble DDW from individual blocks [[D_z D_z, D_z D_tau], [D_tau D_z, D_tau D_tau]]
        a = jnp.hstack((
            jnp.asarray(DDW_z_z),
            jnp.asarray(DDW_z_tau).reshape(self.model.h12, 1)
        ))
        b = jnp.hstack((
            jnp.asarray(DDW_tau_z),
            jnp.asarray(DDW_tau_tau)
        ))
        DDW_gen_comp = jnp.vstack((a, b))
        # Assembled holomorphic DDW matrix must be complex-valued
        chex.assert_type(DDW_gen_comp, complex)
        # Assembled matrix has shape (n, n) — matches full DDW_general output
        chex.assert_shape(DDW_gen_comp, (self.model.h12 + 1, self.model.h12 + 1))
        # Block-assembled DDW must agree with DDW_general — verifies internal consistency
        self.assertAllClose(DDW_gen_comp, DDW_gen, rtol=1e-11, atol=1e-11)

        DcDW = self.variant(
            lambda x, y, z, u, v: self.model.DcDW(x, y, z, u, v, conj=conj)
        )(*args)
        # Mixed Kahler derivative D_I D_cJ W must be complex-valued
        chex.assert_type(DcDW,  complex)
        # DcDW has shape (n, n) — mixed hol/anti-hol covariant second derivative matrix
        chex.assert_shape(DcDW, (self.model.h12 + 1, self.model.h12 + 1))

        # Assemble DcDW from mixed blocks [[D_z D_cz, D_z D_ctau], [D_tau D_cz, D_tau D_ctau]]
        a = jnp.hstack((
            jnp.asarray(DDW_z_cz),
            jnp.asarray(DDW_z_ctau).reshape(self.model.h12, 1)
        ))
        b = jnp.hstack((
            jnp.asarray(DDW_tau_cz),
            jnp.asarray(DDW_tau_ctau)
        ))
        DcDW_gen = jnp.vstack((a, b))
        # Assembled mixed DcDW matrix must be complex-valued
        chex.assert_type(DcDW_gen,  complex)
        # Assembled mixed matrix has shape (n, n)
        chex.assert_shape(DcDW_gen, (self.model.h12 + 1, self.model.h12 + 1))
        # Block-assembled DcDW must agree with DcDW — verifies mixed-derivative consistency
        self.assertAllClose(DcDW_gen, DcDW, rtol=1e-11, atol=1e-11)

        # --- conj = True ---
        conj = True
        DDW_ctau_tau  = self.variant(
            lambda x, y, z, u, v: self.model.DDW_tau_ctau(x, y, z, u, v, conj=conj)
        )(*args)
        DDW_ctau_ctau = self.variant(
            lambda x, y, z, u, v: self.model.DDW_tau_tau(x, y, z, u, v, conj=conj)
        )(*args)
        # Conjugate D_ctau(D_tau W*) must be complex-valued
        chex.assert_type(DDW_ctau_tau,  complex)
        # D_ctau(D_tau W*) is a scalar — conj dilaton second covariant derivative
        chex.assert_shape(DDW_ctau_tau,  ())
        # Conjugate D_ctau(D_ctau W*) must be complex-valued
        chex.assert_type(DDW_ctau_ctau, complex)
        # D_ctau(D_ctau W*) is a scalar — purely anti-hol dilaton second derivative
        chex.assert_shape(DDW_ctau_ctau, ())

        DDW_ctau_cz = self.variant(
            lambda x, y, z, u, v: self.model.DDW_tau_z(x, y, z, u, v, conj=conj)
        )(*args)
        DDW_ctau_z  = self.variant(
            lambda x, y, z, u, v: self.model.DDW_tau_cz(x, y, z, u, v, conj=conj)
        )(*args)
        # Conjugate D_ctau(D_cz W*) must be complex-valued
        chex.assert_type(DDW_ctau_cz, complex)
        # D_ctau(D_cz W*) has shape (h12,) — conj-tau derivative of conj-z covariant terms
        chex.assert_shape(DDW_ctau_cz, (self.model.h12,))
        # Conjugate D_ctau(D_z W*) must be complex-valued
        chex.assert_type(DDW_ctau_z,  complex)
        # D_ctau(D_z W*) has shape (h12,) — conj-tau derivative of z covariant terms
        chex.assert_shape(DDW_ctau_z,  (self.model.h12,))

        DDW_cz_ctau = self.variant(
            lambda x, y, z, u, v: self.model.DDW_z_tau(x, y, z, u, v, conj=conj)
        )(*args)
        DDW_cz_tau  = self.variant(
            lambda x, y, z, u, v: self.model.DDW_z_ctau(x, y, z, u, v, conj=conj)
        )(*args)
        # Conjugate D_cz(D_ctau W*) must be complex-valued
        chex.assert_type(DDW_cz_ctau, complex)
        # D_cz(D_ctau W*) has shape (h12,) — conj-z gradient of conj-tau covariant deriv
        chex.assert_shape(DDW_cz_ctau, (self.model.h12,))
        # Conjugate D_cz(D_tau W*) must be complex-valued
        chex.assert_type(DDW_cz_tau,  complex)
        # D_cz(D_tau W*) has shape (h12,) — conj-z gradient of tau covariant deriv
        chex.assert_shape(DDW_cz_tau,  (self.model.h12,))

        DDW_cz_z  = self.variant(
            lambda x, y, z, u, v: self.model.DDW_z_cz(x, y, z, u, v, conj=conj)
        )(*args)
        DDW_cz_cz = self.variant(
            lambda x, y, z, u, v: self.model.DDW_z_z(x, y, z, u, v, conj=conj)
        )(*args)
        # Conjugate D_cz(D_z W*) must be complex-valued
        chex.assert_type(DDW_cz_z,  complex)
        # D_cz(D_z W*) has shape (h12, h12) — conj z-sector mixed mass matrix block
        chex.assert_shape(DDW_cz_z,  (self.model.h12, self.model.h12))
        # Conjugate D_cz(D_cz W*) must be complex-valued
        chex.assert_type(DDW_cz_cz, complex)
        # D_cz(D_cz W*) has shape (h12, h12) — purely anti-hol z-sector mass matrix block
        chex.assert_shape(DDW_cz_cz, (self.model.h12, self.model.h12))

        DDW_gen_c  = self.variant(
            lambda x, y, z, u, v: self.model.DDW_general(x, y, z, u, v, conj=conj)
        )(*args)
        DDW_SUSY_c = self.variant(
            lambda x, y, z, u, v: self.model.DDW_SUSY(x, y, z, u, v, conj=conj)
        )(*args)
        # Conjugate DDW_general must be complex-valued
        chex.assert_type(DDW_gen_c,  complex)
        # Conjugate DDW_general has shape (n, n) — D_cI D_cJ W* matrix
        chex.assert_shape(DDW_gen_c,  (self.model.h12 + 1, self.model.h12 + 1))
        # Conjugate DDW_SUSY must be complex-valued
        chex.assert_type(DDW_SUSY_c, complex)
        # Conjugate DDW_SUSY has shape (n, n) — SUSY-specialised D_cI D_cJ W* matrix
        chex.assert_shape(DDW_SUSY_c, (self.model.h12 + 1, self.model.h12 + 1))

        DDW_wrap_SUSY_c = self.variant(
            lambda x, y, z, u, v: self.model.DDW(x, y, z, u, v, conj=conj, mode="SUSY")
        )(*args)
        DDW_wrap_gen_c  = self.variant(
            lambda x, y, z, u, v: self.model.DDW(x, y, z, u, v, conj=conj, mode=None)
        )(*args)
        # Conjugate DDW wrapper (SUSY mode) must be complex-valued
        chex.assert_type(DDW_wrap_SUSY_c, complex)
        # Conjugate DDW wrapper (SUSY mode) has shape (n, n)
        chex.assert_shape(DDW_wrap_SUSY_c, (self.model.h12 + 1, self.model.h12 + 1))
        # Conjugate DDW wrapper (general mode) must be complex-valued
        chex.assert_type(DDW_wrap_gen_c,   complex)
        # Conjugate DDW wrapper (general mode) has shape (n, n)
        chex.assert_shape(DDW_wrap_gen_c,  (self.model.h12 + 1, self.model.h12 + 1))

        # Conj DDW wrapper (SUSY) must match the direct conj DDW_SUSY implementation
        self.assertAllClose(DDW_wrap_SUSY_c, DDW_SUSY_c, rtol=1e-11, atol=1e-11)
        # Conj DDW wrapper (general) must match the direct conj DDW_general implementation
        self.assertAllClose(DDW_wrap_gen_c,  DDW_gen_c,  rtol=1e-11, atol=1e-11)

        a = jnp.hstack((
            jnp.asarray(DDW_cz_cz),
            jnp.asarray(DDW_cz_ctau).reshape(self.model.h12, 1)
        ))
        b = jnp.hstack((
            jnp.asarray(DDW_ctau_cz),
            jnp.asarray(DDW_ctau_ctau)
        ))
        DDW_gen_comp_c = jnp.vstack((a, b))
        # Assembled conj DDW matrix must be complex-valued
        chex.assert_type(DDW_gen_comp_c, complex)
        # Assembled conj DDW has shape (n, n) — matches full conj DDW_general output
        chex.assert_shape(DDW_gen_comp_c, (self.model.h12 + 1, self.model.h12 + 1))
        # Block-assembled conj DDW must agree with conj DDW_general — internal consistency
        self.assertAllClose(DDW_gen_comp_c, DDW_gen_c, rtol=1e-11, atol=1e-11)

        DcDW_c = self.variant(
            lambda x, y, z, u, v: self.model.DcDW(x, y, z, u, v, conj=conj)
        )(*args)
        # Conjugate mixed Kahler derivative D_cI D_J W* must be complex-valued
        chex.assert_type(DcDW_c,  complex)
        # Conjugate DcDW has shape (n, n) — mixed anti-hol/hol covariant second derivative
        chex.assert_shape(DcDW_c, (self.model.h12 + 1, self.model.h12 + 1))

        # Assemble conj DcDW from blocks [[D_cz D_z, D_cz D_tau], [D_ctau D_z, D_ctau D_tau]]
        a = jnp.hstack((
            jnp.asarray(DDW_cz_z),
            jnp.asarray(DDW_cz_tau).reshape(self.model.h12, 1)
        ))
        b = jnp.hstack((
            jnp.asarray(DDW_ctau_z),
            jnp.asarray(DDW_ctau_tau)
        ))
        DcDW_gen_c = jnp.vstack((a, b))
        # Assembled conj DcDW matrix must be complex-valued
        chex.assert_type(DcDW_gen_c,  complex)
        # Assembled conj DcDW has shape (n, n)
        chex.assert_shape(DcDW_gen_c, (self.model.h12 + 1, self.model.h12 + 1))
        # Block-assembled conj DcDW must agree with conj DcDW — mixed-derivative consistency
        self.assertAllClose(DcDW_gen_c, DcDW_c, rtol=1e-11, atol=1e-11)

        # Conjugation relations: D_cI(D_cJ W*) = conj(D_I(D_J W))
        # D_cz(D_z W*) = conj(D_z(D_cz W)) — z-cz block conjugation symmetry
        self.assertAllClose(DDW_cz_z,    jnp.conj(DDW_z_cz),    rtol=1e-11, atol=1e-11)
        # D_cz(D_cz W*) = conj(D_z(D_z W)) — pure z block conjugation symmetry
        self.assertAllClose(DDW_cz_cz,   jnp.conj(DDW_z_z),     rtol=1e-11, atol=1e-11)
        # D_cz(D_tau W*) = conj(D_z(D_ctau W)) — z-tau mixed block conjugation
        self.assertAllClose(DDW_cz_tau,  jnp.conj(DDW_z_ctau),  rtol=1e-11, atol=1e-11)
        # D_cz(D_ctau W*) = conj(D_z(D_tau W)) — z-tau mixed block conjugation
        self.assertAllClose(DDW_cz_ctau, jnp.conj(DDW_z_tau),   rtol=1e-11, atol=1e-11)
        # D_ctau(D_z W*) = conj(D_tau(D_cz W)) — tau-z mixed block conjugation
        self.assertAllClose(DDW_ctau_z,  jnp.conj(DDW_tau_cz),  rtol=1e-11, atol=1e-11)
        # D_ctau(D_cz W*) = conj(D_tau(D_z W)) — tau-z mixed block conjugation
        self.assertAllClose(DDW_ctau_cz, jnp.conj(DDW_tau_z),   rtol=1e-11, atol=1e-11)
        # D_ctau(D_tau W*) = conj(D_tau(D_ctau W)) — pure tau block conjugation
        self.assertAllClose(DDW_ctau_tau,  jnp.conj(DDW_tau_ctau), rtol=1e-11, atol=1e-11)
        # D_ctau(D_ctau W*) = conj(D_tau(D_tau W)) — pure tau block conjugation
        self.assertAllClose(DDW_ctau_ctau, jnp.conj(DDW_tau_tau),  rtol=1e-11, atol=1e-11)

        # NOT true in general:
        # self.assertAllClose(DDW_cz_z, DDW_z_cz.T, rtol=1e-11, atol=1e-11)

        # Full conj DDW_SUSY equals conjugate of DDW_SUSY — master conjugation identity
        self.assertAllClose(DDW_SUSY_c,     jnp.conj(DDW_SUSY),     rtol=1e-11, atol=1e-11)
        # Full conj DDW_general equals conjugate of DDW_general
        self.assertAllClose(DDW_gen_c,      jnp.conj(DDW_gen),      rtol=1e-11, atol=1e-11)
        # Conj DDW wrapper (SUSY) equals conjugate of DDW wrapper (SUSY)
        self.assertAllClose(DDW_wrap_SUSY_c, jnp.conj(DDW_wrap_SUSY), rtol=1e-11, atol=1e-11)
        # Conj DDW wrapper (general) equals conjugate of DDW wrapper (general)
        self.assertAllClose(DDW_wrap_gen_c,  jnp.conj(DDW_wrap_gen),  rtol=1e-11, atol=1e-11)

        # Block-assembled conj DDW equals conjugate of block-assembled DDW
        self.assertAllClose(DDW_gen_comp_c, jnp.conj(DDW_gen_comp), rtol=1e-11, atol=1e-11)
        # Redundant cross-check: conj DDW_general equals conjugate of DDW_general
        self.assertAllClose(DDW_gen_c,      jnp.conj(DDW_gen),      rtol=1e-11, atol=1e-11)

        # Block-assembled conj DcDW equals conjugate of block-assembled DcDW
        self.assertAllClose(DcDW_gen_c, jnp.conj(DcDW_gen), rtol=1e-11, atol=1e-11)
        # Conj DcDW equals conjugate of DcDW — mixed covariant derivative conjugation
        self.assertAllClose(DcDW_c,     jnp.conj(DcDW),     rtol=1e-11, atol=1e-11)

        print("SUSY TESTS MISSING!")

    # ==========================================================================
    #  10.  Scalar potential V
    # ==========================================================================
    #
    # The no-scale scalar potential in type IIB flux compactifications is:
    #   V = e^K ( K^{I\bar{J}} D_I W \overline{D_J W} - 3|W|^2 )
    # where the no-scale condition ensures the −3|W|^2 term is cancelled.
    # V must be real and non-negative; it vanishes at SUSY minima.
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_V(self):
        r"""
        **Description:**
        Verifies correctness of the scalar potential and all its derivatives.

        The scalar potential is:

        .. math::
            V = e^K K^{I\bar{J}} D_I W \overline{D_J W}

        where the no-scale condition sets the :math:`-3|W|^2` term to zero.
        Required properties:

        - :math:`V \geq 0` for all field configurations.
        - :math:`V = 0` at the SUSY locus :math:`D_I W = 0`.
        - The Hessian :math:`\partial^2 V / \partial \phi^i \partial \phi^j` is
          positive semi-definite at the SUSY minimum.
        - The SUSY and general Hessian implementations agree at the minimum.
        - The eigenvalues of the real-coordinate Hessian are twice those of the
          complex-coordinate Hessian (due to the factor-of-2 convention).

        agrees with ``scalar_potential``, and that the gradient
        :math:`\partial_\phi V = 0` at the minimum.
        """
        args = (self.z, self.cz, self.tau, self.ctau, self.f)

        # -----------------------------------------------------------------------
        # Random point
        # -----------------------------------------------------------------------
        V = self.variant(self.model.scalar_potential)(*args)

        chex.assert_type(V, complex)
        chex.assert_shape(V, ())
        # V is real-valued: Im(V) ≈ 0 up to floating-point rounding.
        # Use atol proportional to |V| since the imaginary part arises from
        # cancellation of large terms at O(|V| * machine_epsilon).
        self.assertAllClose(V.imag, 0., rtol=1e-11, atol=max(1e-11, abs(float(V.real)) * 1e-14))

        # Real encoding
        V_real = self.variant(self.model.V_x)(self.x, self.f)

        chex.assert_type(V_real, float)
        chex.assert_shape(V_real, ())

        # Gradient
        dV_real = self.variant(self.model.dV_x)(self.x, self.f)

        chex.assert_type(dV_real,  float)
        chex.assert_shape(dV_real, (2 * (self.model.h12 + 1),))

        # Hessian (real coordinates)
        ddV_real = self.variant(self.model.ddV_x)(self.x, self.f)

        chex.assert_type(ddV_real,  float)
        chex.assert_shape(ddV_real, (
            2 * (self.model.h12 + 1),
            2 * (self.model.h12 + 1)
        ))

        # Complex Hessians
        # hessian() dispatches to _hessian_general by default
        hess_gen  = self.variant(self.model._hessian_general)(*args)

        # The general Hessian must be Hermitian (V is real-valued)
        self.assertAllClose(hess_gen,  jnp.conj(hess_gen.T),  rtol=1e-11, atol=1e-11)

        # Eigenvalues of the general Hessian should be real
        eigvals_gen  = jnp.linalg.eigvals(hess_gen)
        self.assertAllClose(eigvals_gen.imag,  0., rtol=1e-10, atol=1e-10)

        # Real-coordinate Hessian eigenvalues should also be real
        eigvals      = jnp.linalg.eigvals(ddV_real)
        self.assertAllClose(eigvals.imag,      0., rtol=1e-10, atol=1e-10)

        # Factor-2 relationship between real and complex eigenvalues
        eigvals_gen_sorted = jnp.sort(eigvals_gen.real)
        eigvals_sorted     = jnp.sort(eigvals.real)
        self.assertAllClose(eigvals_gen_sorted * 2, eigvals_sorted, rtol=1e-09, atol=1e-09)

        # NOTE: _hessian_SUSY is only valid at a SUSY point (D_IW=0),
        # so we do NOT test it at a generic point.

        # -----------------------------------------------------------------------
        # SUSY solution
        # -----------------------------------------------------------------------
        args_sol = (self.zsol, self.czsol, self.tausol, self.ctausol, self.f_solution)

        V_sol = self.variant(self.model.scalar_potential)(*args_sol)
        chex.assert_type(V_sol, complex)
        chex.assert_shape(V_sol, ())
        self.assertAllClose(V_sol.imag, 0., rtol=1e-11, atol=1e-11)
        self.assertAllClose(V_sol.real, 0., rtol=1e-11, atol=1e-11)

        V_real_sol = self.variant(self.model.V_x)(
            self.solution, self.f_solution
        )
        chex.assert_type(V_real_sol, float)
        chex.assert_shape(V_real_sol, ())
        self.assertAllClose(V_real_sol, 0., rtol=1e-11, atol=1e-11)

        dV_real_sol = self.variant(self.model.dV_x)(self.solution, self.f_solution)
        chex.assert_type(dV_real_sol,  float)
        chex.assert_shape(dV_real_sol, (2 * (self.model.h12 + 1),))
        self.assertAllClose(dV_real_sol, 0., rtol=1e-08, atol=1e-08)

        ddV_real_sol = self.variant(self.model.ddV_x)(
            self.solution, self.f_solution
        )
        chex.assert_type(ddV_real_sol,  float)
        chex.assert_shape(ddV_real_sol, (
            2 * (self.model.h12 + 1),
            2 * (self.model.h12 + 1)
        ))

        # Hessian must be positive semi-definite at SUSY minimum
        eigvals_sol = jnp.linalg.eigvals(ddV_real_sol)
        self.assertAllClose(eigvals_sol.imag, 0., rtol=1e-11, atol=1e-11)
        self.assertAllClose(
            jnp.min(eigvals_sol.real) / jnp.min(jnp.abs(eigvals_sol.real)),
            1., rtol=1e-11, atol=1e-11,
        )
        self.assertAllClose(
            jnp.sign(eigvals_sol.real), jnp.ones(len(eigvals_sol)),
            rtol=1e-11, atol=1e-11
        )

        # At the SUSY locus, SUSY and general Hessians must agree
        hess_SUSY_sol = self.variant(self.model._hessian_SUSY)(*args_sol)
        hess_gen_sol  = self.variant(self.model._hessian_general)(*args_sol)
        self.assertAllClose(hess_SUSY_sol, hess_gen_sol, rtol=1e-09, atol=1e-09)

        eigvals_SUSY_sol = jnp.linalg.eigvals(hess_SUSY_sol)
        eigvals_gen_sol  = jnp.linalg.eigvals(hess_gen_sol)

        self.assertAllClose(eigvals_sol.imag,      0., rtol=1e-11, atol=1e-11)
        self.assertAllClose(eigvals_SUSY_sol.imag, 0., rtol=1e-11, atol=1e-11)
        self.assertAllClose(eigvals_gen_sol.imag,  0., rtol=1e-11, atol=1e-11)

        eigvals_SUSY_sol = jnp.sort(eigvals_SUSY_sol.real)
        eigvals_gen_sol  = jnp.sort(eigvals_gen_sol.real)
        eigvals_sol_s    = jnp.sort(eigvals_sol.real)

        self.assertAllClose(eigvals_SUSY_sol * 2, eigvals_sol_s, rtol=1e-09, atol=1e-09)
        self.assertAllClose(eigvals_SUSY_sol, eigvals_gen_sol,   rtol=1e-09, atol=1e-09)
        self.assertAllClose(eigvals_gen_sol  * 2, eigvals_sol_s, rtol=1e-09, atol=1e-09)

        print("Add Tests for dV_real, ddV_real, ddV_real_real!")

    # ==========================================================================
    #  11.  D3-tadpole
    # ==========================================================================
    #
    # The D3-tadpole induced by the flux G_3 is
    #   N_flux = (1/2) ∫ H_3 ∧ F_3 = (1/2) f·h  (in the symplectic basis)
    # It must satisfy the tadpole cancellation condition N_flux ≤ N_D3^max.
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_tadpole(self):
        r"""
        **Description:**
        Verifies that the D3-tadpole function returns a scalar and that the known
        solution flux vector gives a finite non-negative tadpole value.

        The D3-tadpole is:

        .. math::
            N_\text{flux} = \frac{1}{2} \int H_3 \wedge F_3
            = \frac{1}{2} f \cdot h

        where :math:`f` and :math:`h` are the RR and NSNS flux vectors,
        respectively.  Tadpole cancellation requires
        :math:`N_\text{flux} \leq N_{D3}^\text{max}`.
        """
        tadpole_rand = self.variant(self.model.tadpole)(self.f)

        # Must be a scalar
        chex.assert_shape(tadpole_rand, ())
        # Must be real (integer flux → integer tadpole)
        self.assertAllClose(
            jnp.imag(jnp.array(tadpole_rand, dtype=complex)), 0.,
            atol=1e-10,
            # msg is only supported in assertAllClose via atol/rtol
        )

        # Solution tadpole must be non-negative and finite
        tadpole_sol = self.variant(self.model.tadpole)(
            self.f_solution.astype(float)
        )
        chex.assert_shape(tadpole_sol, ())
        self.assertTrue(
            float(tadpole_sol) >= 0,
            msg=f"Solution tadpole must be non-negative, got {float(tadpole_sol)}"
        )
        self.assertTrue(
            float(tadpole_sol) <= float(self.model.D3_tadpole) + 1e-6,
            msg=(
                f"Solution tadpole {float(tadpole_sol)} exceeds "
                f"model bound {float(self.model.D3_tadpole)}"
            )
        )

    # ==========================================================================
    #  12.  SL(2,Z) fundamental domain map
    # ==========================================================================
    #
    # Any axio-dilaton τ can be mapped to the SL(2,Z) fundamental domain
    #   F = { τ ∈ H : |Re(τ)| ≤ 1/2, |τ| ≥ 1 }
    # by a sequence of T und S transformations.  The associated flux vector
    # transforms accordingly so that the physics does not change.
    # ==========================================================================

    def test_map_to_FD_tau(self):
        r"""
        **Description:**
        Verifies that ``map_to_FD_tau`` maps an arbitrary axio-dilaton to the
        SL(2,Z) fundamental domain satisfying:

        .. math::
            |\mathrm{Re}(\tau_\text{FD})| \leq \frac{1}{2}, \quad
            |\tau_\text{FD}| \geq 1, \quad
            \mathrm{Im}(\tau_\text{FD}) > 0.

        and that the gauge-invariant quantity :math:`|e^{K/2} W|` is preserved
        under this transformation.

        .. note::
            The test uses the SL(2,Z) orbit relation
            :math:`|W_\text{gi}(z, \tau, f)| = |W_\text{gi}(z, \tau_\text{FD}, f_\text{FD})|`.
        """
        tau_fd, f_fd = self.model.map_to_FD_tau(self.tau, self.f)

        # Fundamental domain conditions
        self.assertTrue(
            abs(tau_fd.real) <= 0.5 + 1e-10,
            msg=f"|Re(τ_FD)| = {abs(tau_fd.real):.6f} exceeds 1/2."
        )
        self.assertTrue(
            abs(tau_fd) >= 1.0 - 1e-10,
            msg=f"|τ_FD| = {abs(tau_fd):.6f} is less than 1."
        )
        self.assertTrue(
            tau_fd.imag > 0,
            msg=f"Im(τ_FD) = {tau_fd.imag:.6f} must be positive."
        )

        # The gauge-invariant superpotential must be preserved
        W_orig = self.model.superpotential_gauge_invariant(
            self.z, self.tau, self.f, conj=False
        )
        W_fd   = self.model.superpotential_gauge_invariant(
            self.z, tau_fd, jnp.array(f_fd, dtype=float), conj=False
        )
        self.assertAllClose(
            jnp.abs(W_orig), jnp.abs(W_fd), rtol=1e-10, atol=1e-10
        )

    # ==========================================================================
    #  13.  Complex derivatives of the scalar potential  ∂_I V
    # ==========================================================================
    #
    # dV_z  : ∂_{z^i} V,  shape (h12,)  (or (h12+1,) for dV)
    # dV_tau: ∂_τ V,      scalar
    # dV    : stacked [∂_{z^i} V, ∂_τ V], shape (h12+1,)
    # The conjugation identity is  ∂_I V = \overline{∂_{\bar I} V}
    # and V must be stationary at the SUSY minimum: ∂_I V = 0.
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_dV_complex(self):
        r"""
        **Description:**
        Verifies shapes, dtypes, conjugation relations, and the stationarity
        condition for the complex partial derivatives of the scalar potential.

        The gradient relationships tested are:

        .. math::
            \partial_{z^i} V \in \mathbb{C}^{h^{1,2}}, \quad
            \partial_\tau V \in \mathbb{C}, \quad
            \partial_I V \in \mathbb{C}^{h^{1,2}+1}

        with the conjugation:

        .. math::
            \overline{\partial_{z^i} V} = \partial_{\bar z^i} V,
            \quad
            \overline{\partial_\tau V} = \partial_{\bar\tau} V.

        Also verifies that all complex gradients vanish at the SUSY solution:

        .. math::
            \partial_I V \big|_\text{SUSY} = 0.
        """
        args = (self.z, self.cz, self.tau, self.ctau, self.f)

        # ∂_z V
        dVz  = self.variant(
            lambda x, y, z, u, v: self.model.dV_z(x, y, z, u, v, conj=False)
        )(*args)
        dVcz = self.variant(
            lambda x, y, z, u, v: self.model.dV_z(x, y, z, u, v, conj=True)
        )(*args)

        chex.assert_shape(dVz,  (self.model.h12,))
        chex.assert_shape(dVcz, (self.model.h12,))
        self.assertAllClose(
            jnp.conj(dVz), dVcz, rtol=1e-10, atol=1e-10
        )

        # ∂_τ V
        dVtau  = self.variant(
            lambda x, y, z, u, v: self.model.dV_tau(x, y, z, u, v, conj=False)
        )(*args)
        dVctau = self.variant(
            lambda x, y, z, u, v: self.model.dV_tau(x, y, z, u, v, conj=True)
        )(*args)

        chex.assert_shape(dVtau,  ())
        chex.assert_shape(dVctau, ())
        self.assertAllClose(
            jnp.conj(dVtau), dVctau, rtol=1e-10, atol=1e-10
        )

        # Stacked dV wrapper (complex mode)
        dV  = self.variant(
            lambda x, y, z, u, v: self.model.dV(x, y, z, u, v, conj=False, mode="complex")
        )(*args)
        dVc = self.variant(
            lambda x, y, z, u, v: self.model.dV(x, y, z, u, v, conj=True,  mode="complex")
        )(*args)

        chex.assert_shape(dV,  (self.model.h12 + 1,))
        chex.assert_shape(dVc, (self.model.h12 + 1,))
        self.assertAllClose(jnp.conj(dV), dVc, rtol=1e-10, atol=1e-10)

        # Consistency: dV = [dV_z, dV_tau]
        self.assertAllClose(
            dV, jnp.append(dVz, dVtau), rtol=1e-10, atol=1e-10
        )

        # Zero gradient at SUSY solution
        args_sol = (self.zsol, self.czsol, self.tausol, self.ctausol, self.f_solution)

        dV_sol = self.variant(
            lambda x, y, z, u, v: self.model.dV(x, y, z, u, v, conj=False, mode="complex")
        )(*args_sol)

        self.assertAllClose(
            jnp.abs(dV_sol), jnp.zeros(self.model.h12 + 1),
            rtol=1e-08, atol=1e-08
        )

    # ==========================================================================
    #  14.  Second complex derivatives of the scalar potential  ∂_I ∂_J V
    # ==========================================================================
    #
    # The second complex partial derivatives form the complex Hessian.
    # They are used to construct the mass matrix of the scalar fields.
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_ddV_complex(self):
        r"""
        **Description:**
        Verifies shapes and dtypes for the second complex partial derivatives
        of the scalar potential.

        The second derivatives tested are:

        .. math::
            \partial_I \partial_{z^j} V \in \mathbb{C}^{(h^{1,2}+1) \times h^{1,2}},
            \quad
            \partial_I \partial_{\bar z^j} V \in \mathbb{C}^{(h^{1,2}+1) \times h^{1,2}},
            \quad
            \partial_I \partial_\tau V \in \mathbb{C}^{h^{1,2}+1},
            \quad
            \partial_I \partial_{\bar\tau} V \in \mathbb{C}^{h^{1,2}+1}

        where :math:`I = (z^1,\ldots,z^{h^{1,2}},\tau)`.  All are complex
        arrays arising from the second holomorphic / anti-holomorphic
        derivatives of the real potential :math:`V`.
        """
        args = (self.z, self.cz, self.tau, self.ctau, self.f)

        # ∂_I ∂_z V   (shape: (h12+1, h12))
        ddV_z  = self.variant(
            lambda x, y, z, u, v: self.model.ddV_z(x, y, z, u, v, conj=False)
        )(*args)
        ddV_cz = self.variant(
            lambda x, y, z, u, v: self.model.ddV_cz(x, y, z, u, v, conj=False)
        )(*args)

        chex.assert_shape(ddV_z,  (self.model.h12 + 1, self.model.h12))
        chex.assert_shape(ddV_cz, (self.model.h12 + 1, self.model.h12))
        chex.assert_type(ddV_z,  complex)
        chex.assert_type(ddV_cz, complex)

        # ∂_I ∂_τ V   (shape: (h12+1,))
        ddV_tau  = self.variant(
            lambda x, y, z, u, v: self.model.ddV_tau(x, y, z, u, v, conj=False)
        )(*args)
        ddV_ctau = self.variant(
            lambda x, y, z, u, v: self.model.ddV_ctau(x, y, z, u, v, conj=False)
        )(*args)

        chex.assert_shape(ddV_tau,  (self.model.h12 + 1,))
        chex.assert_shape(ddV_ctau, (self.model.h12 + 1,))
        chex.assert_type(ddV_tau,  complex)
        chex.assert_type(ddV_ctau, complex)

        # Full second derivative via ddV wrapper (complex mode):
        # Returns [∂_I∂_JV | ∂_I∂_J̄V] stacked horizontally → shape (n, 2n)
        n = self.model.h12 + 1
        ddV = self.variant(
            lambda x, y, z, u, v: self.model.ddV(x, y, z, u, v, conj=False, mode="complex")
        )(*args)

        chex.assert_shape(ddV, (n, 2 * n))
        chex.assert_type(ddV, complex)

    # ==========================================================================
    #  15.  DDW matrix
    # ==========================================================================
    #
    # The matrix of second Kähler covariant derivatives appears in the
    # expression for the mass matrix.  Two implementations:
    #   DDW_matrix_SUSY    – valid at the SUSY locus (simpler formula)
    #   DDW_matrix_general – valid everywhere
    # Both must be Hermitian and conjugation must map one to the other.
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_DDW_matrix(self):
        r"""
        **Description:**
        Verifies shape, dtype, and Hermiticity of the full
        :math:`(2(h^{1,2}+1)) \times (2(h^{1,2}+1))` DDW matrix assembling the
        second Kähler covariant derivatives.

        The block structure is:

        .. math::
            \mathcal{M}_{IJ} = \begin{pmatrix}
                D_I D_J W         & D_I D_{\bar J} W \\
                D_{\bar I} D_J W  & D_{\bar I} D_{\bar J} W
            \end{pmatrix}

        and the Hermiticity condition is
        :math:`\mathcal{M} = \mathcal{M}^\dagger`.
        """
        args = (self.z, self.cz, self.tau, self.ctau, self.f)

        DDW_SUSY = self.variant(self.model.DDW_matrix_SUSY)(*args)
        DDW_gen  = self.variant(self.model.DDW_matrix_general)(*args)

        n = self.model.h12 + 1
        dim = 2 * n
        # Shape: 2n × 2n block matrix
        chex.assert_shape(DDW_SUSY, (dim, dim))
        chex.assert_shape(DDW_gen,  (dim, dim))
        chex.assert_type(DDW_SUSY, complex)
        chex.assert_type(DDW_gen,  complex)

        # Block conjugation: M = [[A, B], [C, D]] where
        #   A = D_I D_J W,  B = D_I D_J̄ W,  C = D_Ī D_J W̄,  D = D_Ī D_J̄ W̄
        # Relations: D = conj(A), C = conj(B)
        A_gen = DDW_gen[:n, :n]
        B_gen = DDW_gen[:n, n:]
        C_gen = DDW_gen[n:, :n]
        D_gen = DDW_gen[n:, n:]
        # (1,1)* = (2,2): conj(D_ID_JW) = D_ĪD_J̄W̄
        self.assertAllClose(D_gen, jnp.conj(A_gen), rtol=1e-10, atol=1e-10)
        # (1,2)* = (2,1): conj(D_ID_J̄W) = D_ĪD_JW̄
        self.assertAllClose(C_gen, jnp.conj(B_gen), rtol=1e-10, atol=1e-10)

        # At the SUSY locus (D_IW = 0), both implementations must agree
        args_sol = (self.zsol, self.czsol, self.tausol, self.ctausol, self.f_solution)
        DDW_SUSY_sol = self.variant(self.model.DDW_matrix_SUSY)(*args_sol)
        DDW_gen_sol  = self.variant(self.model.DDW_matrix_general)(*args_sol)
        # SUSY and general implementations agree at the SUSY locus
        self.assertAllClose(DDW_SUSY_sol, DDW_gen_sol, rtol=1e-09, atol=1e-09)

    # ==========================================================================
    #  16.  Mass matrix
    # ==========================================================================
    #
    # The canonically normalised mass matrix M is defined via:
    #   M = K^{-1} V_{IJ}
    # where V_{IJ} is the scalar potential Hessian and K is the Kähler metric.
    # At a SUSY minimum, M must be positive semi-definite (no tachyons).
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_mass_matrix(self):
        r"""
        **Description:**
        Verifies that the canonically normalised mass matrix has the correct
        shape and is Hermitian, and that its eigenvalues are real and
        non-negative at the known SUSY minimum.

        The mass matrix is defined as:

        .. math::
            \mathcal{M}_{IJ} = K^{I\bar{K}} \partial_{\bar{K}} \partial_{\bar{J}} V

        and canonically normalised to account for the field-space metric.
        At a SUSY minimum, all eigenvalues :math:`m^2 \geq 0` (no tachyons).
        """
        args = (self.z, self.cz, self.tau, self.ctau, self.f)

        dim = 2 * (self.model.h12 + 1)
        M = self.variant(self.model.mass_matrix)(*args)

        # Shape and type at a generic point
        chex.assert_shape(M, (dim, dim))
        chex.assert_type(M, complex)

        # At SUSY minimum: Hermitian, real eigenvalues, non-negative (no tachyons)
        args_sol = (self.zsol, self.czsol, self.tausol, self.ctausol, self.f_solution)
        M_sol = self.variant(self.model.mass_matrix)(*args_sol)

        chex.assert_shape(M_sol, (dim, dim))
        eigvals_sol = jnp.linalg.eigvals(M_sol)

        self.assertAllClose(
            eigvals_sol.imag, jnp.zeros(dim),
            rtol=1e-09, atol=1e-09
        )
        self.assertTrue(
            bool(jnp.all(eigvals_sol.real >= -1e-8)),
            msg=(
                f"Mass matrix has negative eigenvalues at SUSY minimum: "
                f"min = {float(jnp.min(eigvals_sol.real)):.4e}"
            )
        )

    # ==========================================================================
    #  17.  ISD condition  ⋆ G_3 = i G_3
    # ==========================================================================
    #
    # At the SUSY minimum the3-form flux G_3 = F_3 - τ H_3 must be ISD:
    #   ⋆_6 G_3 = i G_3
    # This is equivalent to the F-term conditions D_I W = 0.
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_ISD_condition(self):
        r"""
        **Description:**
        Verifies that the ISD (Imaginary Self-Dual) condition
        :math:`\star_6 G_3 = \text{i} G_3` is approximately satisfied at the
        known SUSY solution, and that the output has the correct shape.

        The ISD condition is:

        .. math::
            \text{ISD}(z^i, \tau, f) = G_3^{(0,3)} = 0
            \iff
            f_1 - \tau h_1 = \bar{\mathcal{N}}(z^i, \bar{z}^i)(f_2 - \tau h_2)

        where :math:`\mathcal{N}` is the gauge-kinetic matrix.  At the SUSY
        locus, the residual :math:`|\text{ISD}|` must be zero.
        """
        args = (self.z, self.cz, self.tau, self.ctau, self.f)

        # Shape and type at a generic point
        # ISD_condition returns a complex vector of length n = h12+1
        isd = self.variant(self.model.ISD_condition)(*args)
        n = self.model.h12 + 1
        chex.assert_shape(isd, (n,))

        # ISD condition must be (nearly) zero at the SUSY solution
        # since SUSY ⟹ ISD (G_3^{(0,3)} = 0)
        args_sol = (self.zsol, self.czsol, self.tausol, self.ctausol, self.f_solution)
        isd_sol  = self.variant(self.model.ISD_condition)(*args_sol)

        chex.assert_shape(isd_sol, (n,))
        self.assertAllClose(
            jnp.abs(isd_sol), jnp.zeros(n),
            rtol=1e-06, atol=1e-06
        )

    # ==========================================================================
    #  18.  Hodge decomposition of flux
    # ==========================================================================
    #
    # Any 3-form flux can be decomposed into four Hodge components:
    #   N_{3,0}: (3,0) component  — ISD
    #   N_{2,1}: (2,1) component  — IASD (imaginary anti-self-dual)
    #   N_{1,2}: (1,2) component  — ISD
    #   N_{0,3}: (0,3) component  — IASD
    # The projection_fluxes method computes all four (or the two for SUSY).
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_projection_fluxes(self):
        r"""
        **Description:**
        Verifies that ``projection_fluxes`` returns correctly shaped arrays for
        both the full Hodge decomposition (mode=None) and the SUSY mode.

        The flux vector :math:`G_3 = F_3 - \tau H_3` admits the Hodge
        decomposition:

        .. math::
            G_3 = G_3^{(3,0)} + G_3^{(2,1)} + G_3^{(1,2)} + G_3^{(0,3)}

        The ISD condition :math:`\star_6 G_3 = \text{i} G_3` selects
        :math:`G_3 \in H^{2,1} \oplus H^{0,3}`.  The ``"SUSY"`` mode returns
        only the SUSY-relevant components.
        """
        # Full Hodge decomposition
        N_30, N_21, N_12, N_03 = self.variant(
            lambda x, y, z: self.model.projection_fluxes(x, y, z, mode=None)
        )(self.z, self.tau, self.f)

        # Each Hodge component is a flux vector [f|h] of length 2*n_fluxes
        for component, name in [
            (N_30, "N_30"), (N_21, "N_21"),
            (N_12, "N_12"), (N_03, "N_03")
        ]:
            chex.assert_shape(
                component, (2 * self.model.n_fluxes,),
            )

        # SUSY mode returns only SUSY-relevant components: (2,1) and (0,3)
        N_21_susy, N_03_susy = self.variant(
            lambda x, y, z: self.model.projection_fluxes(x, y, z, mode="SUSY")
        )(self.z, self.tau, self.f)

        chex.assert_shape(N_21_susy, (2 * self.model.n_fluxes,))
        chex.assert_shape(N_03_susy, (2 * self.model.n_fluxes,))

        # Full and SUSY mode must agree on (2,1) and (0,3) components
        self.assertAllClose(N_21, N_21_susy, rtol=1e-10, atol=1e-10)
        self.assertAllClose(N_03, N_03_susy, rtol=1e-10, atol=1e-10)


# ==============================================================================
#  TestHessianSUGRA — tests for _hessian_SUGRA and ddDW (no scipy.root needed)
# ==============================================================================

class TestHessianSUGRA(TestCase):
    r"""
    **Description:**
    Test suite for :func:`_hessian_SUGRA` and :func:`ddDW` on :class:`FluxEFT`.

    .. admonition:: Background
        :class: dropdown

        The function :func:`_hessian_SUGRA` computes the Hessian of the
        :math:`\mathcal{N}=1` SUGRA scalar potential

        .. math::
            V = \mathrm{e}^K\bigl(D_{\bar I}\bar W\,K^{I\bar J}\,D_J W - \lambda\,|W|^2\bigr)

        by explicitly expanding the 9 product-rule terms of
        :math:`\partial_A\partial_{\bar B}S` (and similarly for :math:`\partial_A\partial_B S`),
        where :math:`S = D_{\bar I}\bar W\,K^{I\bar J}\,D_J W`.  The Christoffel
        symbols :math:`\Gamma^E_{AC}` appear in the :math:`\partial_A(K^{-1})` terms
        and the Riemann tensor :math:`R_{i\bar jk\bar l}` in the
        :math:`\partial_A\partial_{\bar B}(K^{-1})` term.

        :func:`ddDW` computes :math:`\partial_B\partial_A(D_I W)` via ``jacrev``
        of :func:`dDW`.

    These tests do **not** require finding a SUSY vacuum.

    Attributes:
        model (FluxEFT): Physics model with :math:`h^{1,2}=2`.
        test_points (list): Three random ``(z, zc, tau, tauc, fl)`` tuples.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        h12 = 2
        cls.model = jaxvacua.FluxEFT(
            h12=h12, model_ID=1, model_type="KS", maximum_degree=0
        )
        cls.n = h12 + 1

        rng = np.random.default_rng(99)
        cls.test_points = []
        for _ in range(3):
            z = jnp.array(
                rng.uniform(-0.3, 0.3, h12) + 1j * rng.uniform(2.5, 4.5, h12)
            )
            tau = complex(rng.uniform(-0.4, 0.4) + 1j * rng.uniform(2.0, 7.0))
            fl = jnp.array(rng.integers(-4, 5, 2 * cls.model.n_fluxes), dtype=float)
            if jnp.all(fl == 0):
                fl = fl.at[0].set(1.0)
            cls.test_points.append(
                (z, jnp.conj(z), tau, jnp.conj(tau), fl)
            )

        cls.z, cls.cz, cls.tau, cls.ctau, cls.fl = cls.test_points[0]

    # ------------------------------------------------------------------
    # ddDW — second derivative of the F-terms ∂_B ∂_A (D_I W)
    # ------------------------------------------------------------------

    @chex.variants(with_jit=True, without_jit=True)
    def test_ddDW_shape(self):
        r"""
        **Description:**
        Verify that :func:`ddDW` returns a rank-3 tensor of shape ``(n, n, n)``
        for both ``conj=False`` (holomorphic :math:`\partial_B\partial_A D_I W`)
        and ``conj=True`` (mixed :math:`\partial_{\bar B}\partial_A D_I W`).
        """
        fn_hol = self.variant(lambda: self.model.ddDW(
            self.z, self.cz, self.tau, self.ctau, self.fl, conj=False))
        fn_mix = self.variant(lambda: self.model.ddDW(
            self.z, self.cz, self.tau, self.ctau, self.fl, conj=True))
        # Holomorphic ddDW must be rank-3: (n, n, n) for d_B d_A (D_I W) with B, A, I over n fields
        chex.assert_shape(fn_hol(), (self.n, self.n, self.n))
        # Mixed ddDW (conj=True) must also be rank-3: d_cB d_A (D_I W) with same shape
        chex.assert_shape(fn_mix(), (self.n, self.n, self.n))

    @chex.variants(with_jit=True, without_jit=True)
    def test_ddDW_vs_jacrev(self):
        r"""
        **Description:**
        Verify that :func:`ddDW` matches an explicit ``jacrev`` of :func:`dDW`.

        For ``conj=False``, computes :math:`\partial_B(\partial_A(D_I W))` by
        differentiating :func:`dDW` w.r.t. :math:`(z, \tau)`.
        For ``conj=True``, differentiates w.r.t. :math:`(\bar z, \bar\tau)`.
        Both should agree to machine precision with :func:`ddDW`.
        """
        def _cat(a, b):
            return jnp.concatenate([a, b[..., None]], axis=-1)

        fn_hol = self.variant(lambda: self.model.ddDW(
            self.z, self.cz, self.tau, self.ctau, self.fl, conj=False))
        fn_mix = self.variant(lambda: self.model.ddDW(
            self.z, self.cz, self.tau, self.ctau, self.fl, conj=True))

        # Holomorphic reference: jacrev of dDW w.r.t. (z, tau)
        dz = jax.jacrev(lambda z_: self.model.dDW(z_, self.cz, self.tau, self.ctau, self.fl),
                        holomorphic=True)(self.z)
        dt = jax.jacrev(lambda t_: self.model.dDW(self.z, self.cz, t_, self.ctau, self.fl),
                        holomorphic=True)(self.tau)
        ref_hol = _cat(dz, dt)
        self.assertAllClose(fn_hol(), ref_hol, atol=1e-10)

        # Mixed reference
        dzc = jax.jacrev(lambda zc_: self.model.dDW(self.z, zc_, self.tau, self.ctau, self.fl),
                         holomorphic=True)(self.cz)
        dtc = jax.jacrev(lambda tc_: self.model.dDW(self.z, self.cz, self.tau, tc_, self.fl),
                         holomorphic=True)(self.ctau)
        ref_mix = _cat(dzc, dtc)
        self.assertAllClose(fn_mix(), ref_mix, atol=1e-10)

    # ------------------------------------------------------------------
    # _hessian_SUGRA via hessian(mode="SUGRA")
    # ------------------------------------------------------------------

    @chex.variants(with_jit=True, without_jit=True)
    def test_hessian_SUGRA_vs_general_noscale(self):
        r"""
        **Description:**
        Verify that the SUGRA Hessian (built from explicit 9-term product-rule
        expansion of :math:`V = e^K S`) agrees with the general Hessian
        (built from ``jacrev`` of :math:`V`) for the no-scale potential
        (:math:`\lambda = 0`).

        Tested at 3 random field-space points.  Agreement to ``atol=1e-10``
        validates the Christoffel, Riemann, and SUGRA building block algebra
        in :func:`_hessian_SUGRA`.
        """
        for z, cz, tau, ctau, fl in self.test_points:
            fn_ref = self.variant(lambda: self.model.hessian(
                z, cz, tau, ctau, fl, noscale=True, mode=None))
            fn_sugra = self.variant(lambda: self.model.hessian(
                z, cz, tau, ctau, fl, noscale=True, mode="SUGRA"))
            self.assertAllClose(fn_ref(), fn_sugra(), atol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_hessian_SUGRA_vs_general_full(self):
        r"""
        **Description:**
        Same as :func:`test_hessian_SUGRA_vs_general_noscale` but for the full
        :math:`\mathcal{N}=1` SUGRA potential with the :math:`-3|W|^2` term
        (:math:`\lambda = 3`, ``noscale=False``).

        This additionally validates the :math:`G = |W|^2` derivative terms.
        """
        for z, cz, tau, ctau, fl in self.test_points:
            fn_ref = self.variant(lambda: self.model.hessian(
                z, cz, tau, ctau, fl, noscale=False, mode=None))
            fn_sugra = self.variant(lambda: self.model.hessian(
                z, cz, tau, ctau, fl, noscale=False, mode="SUGRA"))
            self.assertAllClose(fn_ref(), fn_sugra(), atol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_hessian_SUGRA_shape(self):
        r"""
        **Description:**
        Verify the SUGRA Hessian has shape ``(2n, 2n)`` where :math:`n = h^{1,2}+1`.

        The Hessian is assembled as a :math:`2n \times 2n` block matrix:

        .. math::
            H = \begin{pmatrix}
            V_{A\bar B} & V_{AB} \\
            \overline{V_{AB}} & \overline{V_{A\bar B}}
            \end{pmatrix}
        """
        fn = self.variant(lambda: self.model.hessian(
            self.z, self.cz, self.tau, self.ctau, self.fl,
            noscale=True, mode="SUGRA"))
        H = fn()
        chex.assert_shape(H, (2 * self.n, 2 * self.n))

    @chex.variants(with_jit=True, without_jit=True)
    def test_hessian_SUGRA_hermitian_mixed_block(self):
        r"""
        **Description:**
        Verify that the mixed block :math:`V_{A\bar B}` is Hermitian:
        :math:`V_{A\bar B} = \overline{V_{B\bar A}}`.

        This is a necessary consequence of :math:`V` being real-valued: the
        mixed second derivatives satisfy :math:`(\partial_A\partial_{\bar B}V)^*
        = \partial_{\bar A}\partial_B V = \partial_B\partial_{\bar A}V`.
        """
        fn = self.variant(lambda: self.model.hessian(
            self.z, self.cz, self.tau, self.ctau, self.fl,
            noscale=True, mode="SUGRA"))
        H = fn()
        V_mixed = H[: self.n, : self.n]
        self.assertAllClose(V_mixed, jnp.conj(V_mixed.T), atol=1e-10,
                            msg="V_{AB̄} must be Hermitian")

    # ------------------------------------------------------------------
    # ddDW_x — real-coordinate second derivative of F-terms
    # ------------------------------------------------------------------

    @chex.variants(with_jit=True, without_jit=True)
    def test_ddDW_x_shape(self):
        r"""
        **Description:**
        Verify that :func:`ddDW_x` returns the real-coordinate Hessian of the
        F-terms with shape ``(2n, 2n, 2n)`` where :math:`n = h^{1,2}+1`.

        This is :math:`\partial_{\phi^\alpha}\partial_{\phi^\beta}(D_I W)` in real
        coordinates :math:`\phi^\alpha = (a^1,v^1,\ldots,c_0,s)`.
        """
        n = self.model.h12 + 1
        x = jnp.array(np.append(
            np.append([self.z.real], [self.z.imag], axis=0).T.flatten(),
            [self.tau.real, self.tau.imag]
        ))
        # ddDW_x = jacrev(dDW_x) = second derivative of the F-term vector.
        # dDW_x has shape (2n, 2n), so ddDW_x = jacrev of that w.r.t. x → (2n, 2n, 2n)
        result = self.variant(lambda: self.model.ddDW_x(x, self.fl))()
        chex.assert_shape(result, (2 * n, 2 * n, 2 * n))
        chex.assert_type(result, float)

    # ------------------------------------------------------------------
    # DDW_matrix — wrapper dispatching to SUSY/general
    # ------------------------------------------------------------------

    @chex.variants(with_jit=True, without_jit=True)
    def test_DDW_matrix_dispatch(self):
        r"""
        **Description:**
        Verify that :func:`DDW_matrix` dispatches correctly to
        :func:`DDW_matrix_SUSY` (``mode="SUSY"``) and
        :func:`DDW_matrix_general` (``mode=None``).
        """
        args = (self.z, self.cz, self.tau, self.ctau, self.fl)
        n = self.model.h12 + 1

        # mode="SUSY" → DDW_matrix_SUSY
        M_susy = self.variant(lambda: self.model.DDW_matrix(*args, mode="SUSY"))()
        M_susy_direct = self.model.DDW_matrix_SUSY(*args)
        # Fermionic mass matrix is (2n × 2n)
        chex.assert_shape(M_susy, (2 * n, 2 * n))
        self.assertAllClose(M_susy, M_susy_direct, atol=1e-12)

        # mode=None → DDW_matrix_general
        M_gen = self.variant(lambda: self.model.DDW_matrix(*args, mode=None))()
        M_gen_direct = self.model.DDW_matrix_general(*args)
        chex.assert_shape(M_gen, (2 * n, 2 * n))
        self.assertAllClose(M_gen, M_gen_direct, atol=1e-12)

    # ------------------------------------------------------------------
    # Aliases: W, W0, V, H, ddW
    # ------------------------------------------------------------------

    @chex.variants(with_jit=True, without_jit=True)
    def test_aliases(self):
        r"""
        **Description:**
        Verify that convenience aliases return the same values as the methods
        they wrap:

        - ``W`` = ``superpotential``
        - ``W0`` = ``superpotential_gauge_invariant``
        - ``V`` = ``scalar_potential``
        - ``ddW`` assembles ``ddW_z_z``, ``ddW_z_tau``, ``ddW_tau_tau``
        """
        # W = superpotential
        W_alias = self.variant(lambda: self.model.W(self.z, self.tau, self.fl))()
        W_full  = self.model.superpotential(self.z, self.tau, self.fl)
        self.assertAllClose(W_alias, W_full, atol=1e-14)

        # W0 = superpotential_gauge_invariant
        W0_alias = self.variant(lambda: self.model.W0(self.z, self.tau, self.fl))()
        W0_full  = self.model.superpotential_gauge_invariant(self.z, self.tau, self.fl)
        self.assertAllClose(W0_alias, W0_full, atol=1e-14)

        # V = scalar_potential
        args = (self.z, self.cz, self.tau, self.ctau, self.fl)
        V_alias = self.variant(lambda: self.model.V(*args))()
        V_full  = self.model.scalar_potential(*args)
        self.assertAllClose(V_alias, V_full, atol=1e-14)

        # ddW assembles the component blocks into (n, n)
        n = self.model.h12 + 1
        ddW_full = self.variant(lambda: self.model.ddW(self.z, self.tau, self.fl))()
        chex.assert_shape(ddW_full, (n, n))
        # Check it matches the component blocks
        ddW_zz = self.model.ddW_z_z(self.z, self.tau, self.fl)
        ddW_zt = self.model.ddW_z_tau(self.z, self.tau, self.fl)
        ddW_tt = self.model.ddW_tau_tau(self.z, self.tau, self.fl)
        # Top-left: ddW_z_z
        self.assertAllClose(ddW_full[:n-1, :n-1], ddW_zz, atol=1e-12)
        # Top-right column: ddW_z_tau
        self.assertAllClose(ddW_full[:n-1, n-1], ddW_zt, atol=1e-12)
        # Bottom-right: ddW_tau_tau
        self.assertAllClose(ddW_full[n-1, n-1], ddW_tt, atol=1e-12)


# ==============================================================================
#  TestMapToFD
# ==============================================================================

class TestMapToFD(TestCase):
    r"""
    **Description:**
    Test suite for ``map_to_FD``, ``apply_monodromy``, and ``axion_FD``.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = jaxvacua.FluxEFT(
            h12=2, model_ID=1, maximum_degree=2, limit="LCS", model_type="KS"
        )
        cls.lo, cls.hi = cls.model.axion_FD
        cls.tau = 0.1 + 3j
        cls.fl = jnp.array(np.random.default_rng(42).integers(-5, 6, 12).astype(float))

    def test_axion_FD_default(self):
        r"""Default axion_FD is (-0.5, 0.5)."""
        self.assertEqual(self.model.axion_FD, (-0.5, 0.5))

    def test_axion_FD_settable(self):
        r"""axion_FD can be changed and restored."""
        old = self.model.axion_FD
        self.model.axion_FD = (0, 1)
        self.assertEqual(self.model.axion_FD, (0, 1))
        self.model.axion_FD = old

    def test_map_to_FD_basic_range(self):
        r"""Re(z) is in (lo, hi] after FD mapping."""
        rng = np.random.default_rng(42)
        for _ in range(30):
            z = jnp.array(rng.uniform(-5, 5, 2) + 1j * rng.uniform(1, 5, 2))
            tau = complex(rng.uniform(-3, 3), rng.uniform(1, 5))
            fl = jnp.array(rng.integers(-8, 9, 12).astype(float))
            m, t, f = self.model.map_to_FD(z, tau, fl)
            re = np.array(m.real)
            self.assertTrue(np.all(re > self.lo - 1e-10))
            self.assertTrue(np.all(re <= self.hi + 1e-10))

    def test_map_to_FD_custom_range(self):
        r"""Custom axion_FD=(0,1) is respected."""
        z = jnp.array([0.7 + 3j, -0.3 + 2j])
        m, _, _ = self.model.map_to_FD(z, self.tau, self.fl, axion_FD=(0, 1))
        self.assertTrue(np.all(np.array(m.real) > -1e-10))
        self.assertTrue(np.all(np.array(m.real) <= 1 + 1e-10))

    def test_boundary_snap_exact_lo(self):
        r"""Re(z) = lo maps to hi."""
        z = jnp.array([-0.5 + 3j, 0.2 + 2.5j])
        m, _, _ = self.model.map_to_FD(z, self.tau, self.fl)
        self.assertAlmostEqual(float(m.real[0]), 0.5, places=10)

    def test_boundary_snap_near_lo(self):
        r"""Re(z) within boundary_tol of lo snaps to hi."""
        z = jnp.array([-0.5 + 1e-9 + 3j, 0.2 + 2.5j])
        m, _, _ = self.model.map_to_FD(z, self.tau, self.fl)
        self.assertAlmostEqual(float(m.real[0]), 0.5, places=10)

    def test_boundary_no_snap_outside_tol(self):
        r"""Re(z) outside boundary_tol of lo is NOT snapped."""
        z = jnp.array([-0.5 + 1e-7 + 3j, 0.2 + 2.5j])
        m, _, _ = self.model.map_to_FD(z, self.tau, self.fl)
        self.assertAlmostEqual(float(m.real[0]), -0.5 + 1e-7, places=10)

    def test_boundary_custom_tol(self):
        r"""Custom boundary_tol works."""
        z = jnp.array([-0.5 + 1e-6 + 3j, 0.2 + 2.5j])
        m, _, _ = self.model.map_to_FD(z, self.tau, self.fl, boundary_tol=1e-5)
        self.assertAlmostEqual(float(m.real[0]), 0.5, places=10)

    def test_W0_gauge_invariant_preserved(self):
        r"""|W0_gi| is preserved by map_to_FD."""
        rng = np.random.default_rng(42)
        max_err = 0
        for _ in range(30):
            z = jnp.array(rng.uniform(-3, 3, 2) + 1j * rng.uniform(1, 5, 2))
            tau = complex(rng.uniform(-2, 2), rng.uniform(1, 5))
            fl = jnp.array(rng.integers(-5, 6, 12).astype(float))
            W0_b = float(jnp.abs(self.model.superpotential_gauge_invariant(z, tau, fl)))
            m, t, f = self.model.map_to_FD(z, tau, fl)
            W0_a = float(jnp.abs(self.model.superpotential_gauge_invariant(m, t, f)))
            max_err = max(max_err, abs(W0_b - W0_a) / max(W0_b, 1e-30))
        self.assertLess(max_err, 1e-10)

    def test_tadpole_preserved(self):
        r"""N_flux is preserved by map_to_FD."""
        rng = np.random.default_rng(42)
        for _ in range(30):
            z = jnp.array(rng.uniform(-3, 3, 2) + 1j * rng.uniform(1, 5, 2))
            tau = complex(rng.uniform(-2, 2), rng.uniform(1, 5))
            fl = jnp.array(rng.integers(-5, 6, 12).astype(float))
            N_b = float(self.model.tadpole(fl))
            _, _, f_fd = self.model.map_to_FD(z, tau, fl)
            N_a = float(self.model.tadpole(f_fd))
            self.assertAlmostEqual(N_b, N_a, places=8)

    def test_flux_integrality(self):
        r"""Fluxes remain integer after map_to_FD."""
        rng = np.random.default_rng(42)
        for _ in range(30):
            z = jnp.array(rng.uniform(-5, 5, 2) + 1j * rng.uniform(1, 5, 2))
            tau = complex(rng.uniform(-3, 3), rng.uniform(1, 5))
            fl = jnp.array(rng.integers(-8, 9, 12).astype(float))
            _, _, f_fd = self.model.map_to_FD(z, tau, fl)
            self.assertLess(float(jnp.max(jnp.abs(f_fd - jnp.round(f_fd)))), 1e-10)

    def test_idempotent(self):
        r"""Mapping to FD twice gives the same result."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            z = jnp.array(rng.uniform(-5, 5, 2) + 1j * rng.uniform(1, 5, 2))
            tau = complex(rng.uniform(-3, 3), rng.uniform(1, 5))
            fl = jnp.array(rng.integers(-8, 9, 12).astype(float))
            m1, t1, f1 = self.model.map_to_FD(z, tau, fl)
            m2, t2, f2 = self.model.map_to_FD(m1, t1, f1)
            self.assertAllClose(m1, m2, atol=1e-12)
            self.assertAlmostEqual(complex(t1), complex(t2), places=10)
            self.assertAllClose(f1, f2, atol=1e-10)

    def test_apply_monodromy_zero_shift(self):
        r"""Zero shift is identity."""
        z = jnp.array([0.1 + 3j, 0.2 + 2.5j])
        m, f = self.model.apply_monodromy(z, self.fl, [0, 0])
        self.assertAllClose(m, z, atol=1e-14)
        self.assertAllClose(f, self.fl, atol=1e-10)

    def test_apply_monodromy_W0_invariant(self):
        r"""|W0_gi| is preserved by apply_monodromy."""
        z = jnp.array([0.1 + 3j, 0.2 + 2.5j])
        W0_b = float(jnp.abs(self.model.superpotential_gauge_invariant(z, self.tau, self.fl)))
        m_s, f_s = self.model.apply_monodromy(z, self.fl, [1, -2])
        W0_a = float(jnp.abs(self.model.superpotential_gauge_invariant(m_s, self.tau, f_s)))
        self.assertAlmostEqual(W0_b, W0_a, places=10)


if __name__ == "__main__":
    import unittest
    unittest.main()
