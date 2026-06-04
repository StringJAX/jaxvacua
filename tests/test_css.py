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

"""Tests for complex-structure-sector geometry.

Purpose
-------
Validate the ``css`` class and its Kähler-geometry, period, metric,
monodromy and gauge-kinetic computations.

Main public API
---------------
- ``TestCSSector``: broad numerical checks for periods, prepotential,
  Kähler potential, metrics, derivatives and gauge-kinetic data.
- ``TestKahlerGeometry``: focused checks for Kähler-geometric identities.
- ``TestMonodromy``: checks monodromy and fundamental-domain behaviour.

Design notes
------------
The tests use small Kreuzer-Skarke fixtures so expensive geometry is exercised
without requiring large model data.
"""

import sys, os, warnings
import jax
import pytest
from functools import partial
from scipy.optimize import root
from util import *

sys.path.append("./../")
import jaxvacua


# ---------------------------------------------------------------------------
# Diagnostic autouse fixture (kept for future use).  Writes START/END markers
# around every test directly to fd 2, bypassing pytest's stdout/stderr capture
# (combine with ``-s`` and ``PYTHONUNBUFFERED=1`` on the CI command).
# Flip ``autouse=False`` -> ``autouse=True`` to re-enable when diagnosing a
# hang in this file again.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=False)
def _ci_test_marker(request):
    os.write(2, f">>> START {request.node.nodeid}\n".encode())
    yield
    os.write(2, f">>> END   {request.node.nodeid}\n".encode())


# ==============================================================================
#  TestCSSector
# ==============================================================================

class TestCSSector(TestCase):
    r"""
    **Description:**
    Test suite for the complex structure sector (CSS) class in Type IIB
    orientifold flux compactifications.  The tests probe the mirror volume
    :math:`\mathcal{A}`, the Kähler potential :math:`K`, all first and second
    derivatives of :math:`K`, the Kähler metric :math:`K_{I\bar{J}}` and its
    inverse, the ISD matrix :math:`\mathcal{M}`, the gauge kinetic matrix
    :math:`\mathcal{N}`, the prepotential :math:`F`, and the period vector
    :math:`\Pi`.  The class also covers correctness of class-level aliases,
    the moduli-to-periods round-trip, algebraic properties of
    :math:`\mathcal{N}`, the relation between the metric and second derivatives
    of :math:`K`, and the symplectic inner-product relations satisfied by
    :math:`\Pi`.

    Attributes:
        model (jaxvacua.flux_eft): FluxEFT model with ``h12=2``,
            ``model_ID=1``, ``model_type="KS"`` and ``maximum_degree=2``.
        z (jnp.ndarray): Fixed representative complex-structure moduli of
            shape ``(h12,)``.
        cz (jnp.ndarray): Complex conjugate of ``z``.
        tau (complex): Fixed representative axio-dilaton with positive
            imaginary part.
        ctau (complex): Complex conjugate of ``tau``.
        f (jnp.ndarray): Fixed integer flux vector of shape
            ``(4*(h12+1),)`` cast to float.
        x (jnp.ndarray): Real parameter vector ``[Re(z_1), Im(z_1), ...,
            Re(tau), Im(tau)]`` of shape ``(2*(h12+1),)``.
        tau_fd (complex): Axio-dilaton mapped to fundamental domain.
        f_fd (jnp.ndarray): Flux vector corresponding to ``tau_fd``.
        ctau_fd (complex): Complex conjugate of ``tau_fd``.
        sigma (jnp.ndarray): Symplectic matrix
            :math:`\Sigma` of shape ``(2*(h12+1), 2*(h12+1))``.
        f_solution (jnp.ndarray): Known SUSY-minimum flux vector.
        zsol (jnp.ndarray): Complex structure moduli at the SUSY minimum.
        czsol (jnp.ndarray): Complex conjugate of ``zsol``.
        tausol (complex): Axio-dilaton at the SUSY minimum.
        ctausol (complex): Complex conjugate of ``tausol``.
        solution (jnp.ndarray): Real solution vector at the SUSY minimum.
    """

    # --------------------------------------------------------------------------

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        h12 = 2

        cls.model = jaxvacua.FluxEFT(h12=h12, 
                                     model_ID=1, 
                                     model_type="KS", 
                                     maximum_degree=2, 
                                     prange=20)
        
        cls.model.lcs_tree.a_matrix = jnp.array([[4.5,1.5],[1.5,0.]])
        # Deterministic interior LCS-style fixture; avoids test-order and
        # session-dependent numerical variation in sign-sensitive matrix tests.
        cls.z = jnp.array([0.23 + 3.4j, -0.31 + 2.8j])
        cls.cz = jnp.conj(cls.z)
        cls.tau = 0.37 + 4.6j
        cls.ctau = jnp.conj(cls.tau)
        cls.f = jnp.array([3, -2, 1, 4, -1, 2, -3, 0, 5, -4, 2, -1]).astype(float)

        cls.x = jnp.array(np.append(np.append([cls.z.real], [cls.z.imag], axis=0).T.flatten(), [cls.tau.real, cls.tau.imag]))

        cls.tau_fd, cls.f_fd = cls.model.map_to_fd_tau(cls.tau, cls.f)
        cls.f_fd = jnp.array(cls.f_fd).astype(float)
        cls.ctau_fd = jnp.conj(cls.tau_fd)

        cls.sigma = cls.model.periods.sigma

        # Known SUSY minimum solution
        cls.f_solution = jnp.array([7, 3, -24, 0, -16, 50, 0, 3, -4, 0, 0, 0])
        u1sol = 2.74215479602462524879172086700112955631003945168828832743217138983767 * 1j
        u2sol = 2.05661613496943436323419976712599580262262253939859294519039244649420 * 1j
        tausol = 6.85540179778358427172610564536555609784128313762349971439377181031816 * 1j

        x0 = jnp.array([0., u1sol.imag, 0., u2sol.imag, 0., tausol.imag])
        res = root(cls.model.DW_x, x0=x0, args=(cls.f_solution,), method="hybr", jac=cls.model.dDW_x)

        if not res.success:
            raise ValueError("Unable to find minimum using scipy.optimize.root!")

        x = res.x

        cls.tausol = x[4] + 1j * x[5]
        cls.zsol = jnp.array([x[0] + 1j * x[1], x[2] + 1j * x[3]])
        cls.czsol = jnp.conj(cls.zsol)
        cls.ctausol = jnp.conj(cls.tausol)
        cls.solution = jnp.array(x)


    # ==========================================================================
    #  1. Mirror Volume and Kahler Potential
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_mirror_volume(self):
        r"""**Description:**
        Tests the mirror dual Calabi-Yau volume :math:`\mathcal{A}`.

        The mirror volume is defined as

        .. math::
            \mathcal{A}(z^i,\overline{z}^i)
            = -\mathrm{i}\,\Pi^\dagger(\overline{z}^i)\cdot\Sigma\cdot\Pi(z^i)
            = -\mathrm{i}\int_X \Omega_3\wedge\overline{\Omega}_3 \,,

        where :math:`\Pi` is the period vector and :math:`\Sigma` is the
        symplectic matrix.  Being proportional to the volume of the mirror
        Calabi-Yau :math:`\widetilde{X}` in its Kähler cone, this quantity is
        real and positive.

        The test verifies:

        * ``model.A`` is a scalar of complex type whose imaginary part
          vanishes, confirming that :math:`\mathcal{A}\in\mathbb{R}_{>0}`.
        * The aliases ``A``, ``V_tilde``, and ``mirror_volume`` all return
          the same numerical value.

        .. note::
            The class exposes three aliases for the same function:
            ``model.A``, ``model.V_tilde``, and ``model.mirror_volume``.
        """

        # Evaluate the mirror volume via the primary alias
        Vtilde = self.variant(self.model.A)(self.z, self.cz)

        # Output must be a complex scalar (imaginary unit absorbed internally)
        chex.assert_type(Vtilde, complex)
        chex.assert_shape(Vtilde, ())

        # Mirror volume is real: imaginary part must vanish
        self.assertAllClose(Vtilde.imag, 0.,
                            msg="mirror_volume must be real: Im(A) != 0")


    @chex.variants(with_jit=True, without_jit=True)
    def test_kahler_potential(self):
        r"""**Description:**
        Tests the Kähler potential :math:`K`.

        At tree level in Type IIB string theory the Kähler potential for
        complex structure moduli and the axio-dilaton is

        .. math::
            K(z^i,\overline{z}^i,\tau,\overline{\tau})
            = -\log\bigl[-\mathrm{i}(\tau-\overline{\tau})\bigr]
             -\log\bigl[\mathcal{A}(z^i,\overline{z}^i)\bigr]\,,

        where :math:`\mathcal{A}` is the mirror dual Calabi-Yau volume.
        Since both :math:`-\mathrm{i}(\tau-\overline{\tau})=2\,\mathrm{Im}(\tau)>0`
        and :math:`\mathcal{A}>0`, the logarithms are real, so :math:`K\in\mathbb{R}`.

        The test verifies:

        * ``kahler_potential`` returns a complex scalar.
        * The imaginary part of :math:`K` vanishes numerically.
        """

        # Evaluate the Kähler potential
        KP = self.variant(self.model.kahler_potential)(self.z, self.cz, self.tau, self.ctau)

        # Output must be a complex scalar
        chex.assert_type(KP, complex)
        chex.assert_shape(KP, ())

        # Kähler potential is real-valued
        self.assertAllClose(KP.imag, 0.,
                            msg="kahler_potential must be real: Im(K) != 0")


    # ==========================================================================
    #  2. First Derivatives of the Kahler Potential
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_dK_z(self):
        r"""**Description:**
        Tests the holomorphic derivative :math:`\partial_{z^i}K` and its
        complex conjugate :math:`\partial_{\overline{z}^i}K`.

        Because :math:`K` is real, holomorphic and anti-holomorphic
        derivatives are related by complex conjugation,

        .. math::
            \partial_{\overline{z}^i}K = \overline{\partial_{z^i}K}\,.

        The test verifies shapes, types, and the conjugation relation for
        ``dK_z`` and ``dK_cz``.
        """

        # Holomorphic derivative w.r.t. complex structure moduli
        dK_z = self.variant(self.model.dK_z)(self.z, self.cz, self.tau, self.ctau)

        chex.assert_type(dK_z, complex)
        chex.assert_shape(dK_z, (self.model.h12,))

        # Anti-holomorphic derivative w.r.t. conjugate complex structure moduli
        dK_cz = self.variant(self.model.dK_cz)(self.z, self.cz, self.tau, self.ctau)

        chex.assert_type(dK_cz, complex)
        chex.assert_shape(dK_cz, (self.model.h12,))

        # Conjugation relation: dK_cz = conj(dK_z)
        self.assertAllClose(dK_cz, jnp.conj(dK_z),
                            msg="dK_cz must equal conj(dK_z)")


    @chex.variants(with_jit=True, without_jit=True)
    def test_dK_tau(self):
        r"""**Description:**
        Tests the holomorphic derivative :math:`\partial_{\tau}K` and its
        complex conjugate :math:`\partial_{\overline{\tau}}K`.

        From the contribution :math:`K\supset -\log\bigl[-\mathrm{i}(\tau-\overline{\tau})\bigr]`
        one finds

        .. math::
            \partial_\tau K = \frac{-1}{\tau-\overline{\tau}} = \frac{\mathrm{i}}{2\,\mathrm{Im}(\tau)}\,,

        and, since :math:`K` is real,
        :math:`\partial_{\overline{\tau}}K = \overline{\partial_{\tau}K}`.

        The test verifies shapes, types, and the conjugation relation for
        ``dK_tau`` and ``dK_ctau``.
        """

        # Holomorphic derivative w.r.t. axio-dilaton
        dK_tau = self.variant(self.model.dK_tau)(self.z, self.cz, self.tau, self.ctau)

        chex.assert_type(dK_tau, complex)
        chex.assert_shape(dK_tau, ())

        # Anti-holomorphic derivative w.r.t. conjugate axio-dilaton
        dK_ctau = self.variant(self.model.dK_ctau)(self.z, self.cz, self.tau, self.ctau)

        chex.assert_type(dK_ctau, complex)
        chex.assert_shape(dK_ctau, ())

        # Conjugation relation: dK_ctau = conj(dK_tau)
        self.assertAllClose(dK_ctau, jnp.conj(dK_tau),
                            msg="dK_ctau must equal conj(dK_tau)")


    @chex.variants(with_jit=True, without_jit=True)
    def test_dK(self):
        r"""**Description:**
        Tests the combined holomorphic gradient :math:`\partial_I K` and its
        complex conjugate :math:`\partial_{\overline{I}}K` where
        :math:`I=(i,\tau)` runs over all moduli.

        The combined gradient is assembled as

        .. math::
            \partial_I K = (\partial_{z^i}K,\,\partial_\tau K)\,,\quad
            \partial_{\overline{I}} K = (\partial_{\overline{z}^i}K,\,\partial_{\overline{\tau}}K)\,.

        Since :math:`K` is real, these are related by
        :math:`\partial_{\overline{I}}K = \overline{\partial_I K}`.

        The test verifies shapes, types, and the conjugation relation for
        ``dK`` and ``dK_c``.
        """

        # Full holomorphic gradient
        dK = self.variant(self.model.dK)(self.z, self.cz, self.tau, self.ctau)

        chex.assert_type(dK, complex)
        chex.assert_shape(dK, (self.model.h12 + 1,))

        # Full anti-holomorphic gradient
        dK_c = self.variant(self.model.dK_c)(self.z, self.cz, self.tau, self.ctau)

        chex.assert_type(dK_c, complex)
        chex.assert_shape(dK_c, (self.model.h12 + 1,))

        # Conjugation relation: dK_c = conj(dK)
        self.assertAllClose(dK_c, jnp.conj(dK),
                            msg="dK_c must equal conj(dK)")


    # ==========================================================================
    #  3. Second Derivatives of the Kahler Potential
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_ddK(self):
        r"""**Description:**
        Tests all second (mixed and unmixed) derivatives of the Kähler
        potential :math:`K`.

        The Kähler metric of the moduli space is the Hermitian matrix of
        mixed holomorphic-anti-holomorphic second derivatives,

        .. math::
            K_{i\bar{\jmath}} = \partial_{z^i}\partial_{\overline{z}^j}K\,,\quad
            K_{\tau\overline{\tau}} = \partial_\tau\partial_{\overline{\tau}}K\,.

        Reality of :math:`K` implies the following conjugation relations among
        all second derivatives:

        .. math::
            K_{\bar{\imath}j} &= \overline{K_{i\bar{\jmath}}}\,,\quad
            K_{ij}            = \overline{K_{\bar{\imath}\bar{\jmath}}}\,,\\
            K_{\bar{\imath}\overline{\tau}} &= \overline{K_{i\overline{\tau}}}\,,\quad
            K_{\bar{\imath}\tau} = \overline{K_{i\overline{\tau}}}\,,\\
            K_{\overline{\tau}\overline{\tau}} &= \overline{K_{\tau\tau}}\,.

        Additionally, the mixed metric block :math:`K_{i\bar{\jmath}}` is
        Hermitian as a matrix and its diagonal is real.

        The test verifies shapes, types, and all conjugation relations for
        the nine available second-derivative functions.
        """

        # ---------------------------------------------------------------
        # Mixed holomorphic-antiholomorphic: K_i_cj and K_ci_j
        # ---------------------------------------------------------------

        # K_{i \bar{j}} = partial_z^i partial_cz^j K
        ddK_z_cz = self.variant(self.model.ddK_z_cz)(self.z, self.cz, self.tau, self.ctau)

        chex.assert_type(ddK_z_cz, complex)
        chex.assert_shape(ddK_z_cz, (self.model.h12, self.model.h12))

        # K_{\bar{j} i} = partial_cz^j partial_z^i K (should equal conj of K_{i \bar{j}})
        ddK_cz_z = self.variant(self.model.ddK_cz_z)(self.z, self.cz, self.tau, self.ctau)

        chex.assert_type(ddK_cz_z, complex)
        chex.assert_shape(ddK_cz_z, (self.model.h12, self.model.h12))

        # Conjugation relation for mixed z-cz block
        self.assertAllClose(ddK_cz_z, jnp.conj(ddK_z_cz),
                            msg="ddK_cz_z must equal conj(ddK_z_cz)")

        # ---------------------------------------------------------------
        # Pure holomorphic / anti-holomorphic: K_z_z and K_cz_cz
        # ---------------------------------------------------------------

        # K_{ij} = partial_z^i partial_z^j K
        ddK_z_z = self.variant(self.model.ddK_z_z)(self.z, self.cz, self.tau, self.ctau)

        chex.assert_type(ddK_z_z, complex)
        chex.assert_shape(ddK_z_z, (self.model.h12, self.model.h12))

        # K_{\bar{i}\bar{j}} = partial_cz^i partial_cz^j K
        ddK_cz_cz = self.variant(self.model.ddK_cz_cz)(self.z, self.cz, self.tau, self.ctau)

        chex.assert_type(ddK_cz_cz, complex)
        chex.assert_shape(ddK_cz_cz, (self.model.h12, self.model.h12))

        # Conjugation relation for pure holomorphic / anti-holomorphic block
        self.assertAllClose(ddK_cz_cz, jnp.conj(ddK_z_z),
                            msg="ddK_cz_cz must equal conj(ddK_z_z)")

        # ---------------------------------------------------------------
        # Mixed moduli-axio-dilaton: K_z_ctau and K_cz_tau
        # ---------------------------------------------------------------

        # K_{i \bar{\tau}} = partial_z^i partial_ctau K
        ddK_z_ctau = self.variant(self.model.ddK_z_ctau)(self.z, self.cz, self.tau, self.ctau)

        chex.assert_type(ddK_z_ctau, complex)
        chex.assert_shape(ddK_z_ctau, (self.model.h12,))

        # K_{\bar{i} \tau} = partial_cz^i partial_tau K
        ddK_cz_tau = self.variant(self.model.ddK_cz_tau)(self.z, self.cz, self.tau, self.ctau)

        chex.assert_type(ddK_cz_tau, complex)
        chex.assert_shape(ddK_cz_tau, (self.model.h12,))

        # Conjugation relation: K_{\bar{i}\tau} = conj(K_{i \bar{\tau}})
        self.assertAllClose(ddK_cz_tau, jnp.conj(ddK_z_ctau),
                            msg="ddK_cz_tau must equal conj(ddK_z_ctau)")

        # ---------------------------------------------------------------
        # Mixed moduli-axio-dilaton: K_z_tau and K_cz_ctau
        # ---------------------------------------------------------------

        # K_{i\tau} = partial_z^i partial_tau K
        ddK_z_tau = self.variant(self.model.ddK_z_tau)(self.z, self.cz, self.tau, self.ctau)

        chex.assert_type(ddK_z_tau, complex)
        chex.assert_shape(ddK_z_tau, (self.model.h12,))

        # K_{\bar{i}\bar{\tau}} = partial_cz^i partial_ctau K
        ddK_cz_ctau = self.variant(self.model.ddK_cz_ctau)(self.z, self.cz, self.tau, self.ctau)

        chex.assert_type(ddK_cz_ctau, complex)
        chex.assert_shape(ddK_cz_ctau, (self.model.h12,))

        # Conjugation relation: K_{\bar{i}\bar{\tau}} = conj(K_{i\tau})
        self.assertAllClose(ddK_cz_ctau, jnp.conj(ddK_z_tau),
                            msg="ddK_cz_ctau must equal conj(ddK_z_tau)")

        # ---------------------------------------------------------------
        # Second derivatives w.r.t. axio-dilaton tau
        # ---------------------------------------------------------------

        # K_{\tau\tau} = partial_tau^2 K
        ddK_tau_tau = self.variant(self.model.ddK_tau_tau)(self.z, self.cz, self.tau, self.ctau)

        chex.assert_type(ddK_tau_tau, complex)
        chex.assert_shape(ddK_tau_tau, ())

        # K_{\bar{\tau}\bar{\tau}} = partial_ctau^2 K
        ddK_ctau_ctau = self.variant(self.model.ddK_ctau_ctau)(self.z, self.cz, self.tau, self.ctau)

        chex.assert_type(ddK_ctau_ctau, complex)
        chex.assert_shape(ddK_ctau_ctau, ())

        # Conjugation relation: K_ctau_ctau = conj(K_tau_tau)
        self.assertAllClose(ddK_ctau_ctau, jnp.conj(ddK_tau_tau),
                            msg="ddK_ctau_ctau must equal conj(ddK_tau_tau)")

        # Mixed axio-dilaton: K_{\tau \bar{\tau}}
        ddK_tau_ctau = self.variant(self.model.ddK_tau_ctau)(self.z, self.cz, self.tau, self.ctau)

        chex.assert_type(ddK_tau_ctau, complex)
        chex.assert_shape(ddK_tau_ctau, ())

        # ---------------------------------------------------------------
        # Hermiticity and reality checks on the moduli block
        # ---------------------------------------------------------------

        # Diagonal of K_{i \bar{j}} must be real (Hermitian matrix)
        self.assertAllClose(jnp.diag(ddK_z_cz).imag, 0.,
                            msg="diagonal of ddK_z_cz must be real")

        # K_{i \bar{j}} is Hermitian: K_{i \bar{j}} = conj(K_{j \bar{i}})
        self.assertAllClose(ddK_z_cz, jnp.conj(ddK_z_cz.T),
                            msg="ddK_z_cz must be Hermitian")

        # Mixed dilation-modulus block: K_{i \bar{\tau}} = conj(K_{i \bar{\tau}})^T
        self.assertAllClose(ddK_z_ctau, jnp.conj(ddK_z_ctau.T),
                            msg="ddK_z_ctau must equal its own conjugate-transpose")

        # K_{\tau \bar{\tau}} is real
        self.assertAllClose(ddK_tau_ctau, jnp.conj(ddK_tau_ctau),
                            msg="ddK_tau_ctau must be real")


    # ==========================================================================
    #  4. Kahler Metric and Its Inverse
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_kahler_metric(self):
        r"""**Description:**
        Tests the Kähler metric :math:`K_{I\overline{J}}`, its inverse
        :math:`K^{\overline{I}J}`, and the gradient of the inverse metric.

        The full Kähler metric is

        .. math::
            K_{I\overline{J}} = \begin{pmatrix}
                K_{i\overline{j}} & K_{i\overline{\tau}} \\[4pt]
                K_{\tau\overline{j}} & K_{\tau\overline{\tau}}
            \end{pmatrix}\,,\quad I,J\in\{z^1,\ldots,z^{h^{1,2}},\tau\}\,.

        For the standard tree-level Kähler potential the cross-terms vanish,
        :math:`K_{i\overline{\tau}}=K_{\tau\overline{j}}=0`, yielding a
        block-diagonal structure.  Both the general (``mode=None``) and the
        block-diagonal (``mode="block diagonal"``) evaluations are tested.

        A valid Kähler metric must be:

        * **Hermitian**: :math:`K_{I\overline{J}} = \overline{K_{J\overline{I}}}`.
        * **Positive definite**: all eigenvalues are real and positive.

        The inverse metric is compared against the matrix inverse and checked
        for Hermiticity.  The gradient of the inverse metric is verified for
        the correct output shape.
        """

        # Full metric (general mode - assembles all second derivatives)
        KM = self.variant(
            lambda x, y, z, u: self.model.kahler_metric(x, y, z, u, mode=None)
        )(self.z, self.cz, self.tau, self.ctau)

        # Block-diagonal metric (assumes no mixing between z and tau)
        KM_bd = self.variant(
            lambda x, y, z, u: self.model.kahler_metric(x, y, z, u, mode="block diagonal")
        )(self.z, self.cz, self.tau, self.ctau)

        chex.assert_shape(KM, (self.model.h12 + 1, self.model.h12 + 1))
        chex.assert_shape(KM_bd, (self.model.h12 + 1, self.model.h12 + 1))

        # Both modes must agree for the tree-level Kähler potential
        self.assertAllClose(KM, KM_bd,
                            msg="general and block-diagonal Kähler metrics must agree")

        # Diagonal entries of the Kähler metric must be real
        self.assertAllClose(jnp.diag(KM).imag, 0.,
                            msg="diagonal of kahler_metric must be real")

        # Metric must be Hermitian: K = conj(K.T)
        self.assertAllClose(KM, jnp.conj(KM.T),
                            msg="kahler_metric must be Hermitian")

        # Eigenvalues must be real
        eigvals = jnp.linalg.eigvals(KM)
        self.assertAllClose(eigvals.imag, 0., rtol=1e-11, atol=1e-11,
                            msg="eigenvalues of kahler_metric must be real")

        # Eigenvalues must be positive (positive-definite metric)
        self.assertAllClose(jnp.min(eigvals.real) / jnp.min(jnp.abs(eigvals.real)), 1.,
                            rtol=1e-11, atol=1e-11,
                            msg="kahler_metric must be positive definite")
        self.assertAllClose(jnp.sign(eigvals.real), jnp.ones(len(eigvals)),
                            rtol=1e-11, atol=1e-11,
                            msg="all eigenvalues of kahler_metric must be positive")

        # ------------------------------------------------------------------
        # Inverse Kähler metric
        # ------------------------------------------------------------------

        IKM = self.variant(
            lambda x, y, z, u: self.model.inverse_kahler_metric(x, y, z, u, mode=None)
        )(self.z, self.cz, self.tau, self.ctau)

        chex.assert_shape(IKM, (self.model.h12 + 1, self.model.h12 + 1))

        # Diagonal of the inverse metric must be real
        self.assertAllClose(jnp.diag(IKM).imag, 0.,
                            msg="diagonal of inverse_kahler_metric must be real")

        # Inverse metric must be Hermitian
        self.assertAllClose(IKM, jnp.conj(IKM.T),
                            msg="inverse_kahler_metric must be Hermitian")

        # Inverse metric must equal the numerical matrix inverse
        self.assertAllClose(IKM, jnp.linalg.inv(KM),
                            msg="inverse_kahler_metric must equal matrix inverse of kahler_metric")

        # ------------------------------------------------------------------
        # Gradient of the inverse Kähler metric
        # ------------------------------------------------------------------

        dIKM = self.variant(
            lambda x, y, z, u: self.model.inverse_kahler_metric_grad(x, y, z, u, mode=None)
        )(self.z, self.cz, self.tau, self.ctau)

        # Shape: (h12+1) x (h12+1) matrix for each of h12 moduli derivatives
        chex.assert_shape(dIKM, (self.model.h12 + 1, self.model.h12 + 1, self.model.h12))


    # ==========================================================================
    #  5. ISD Matrix
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    @pytest.mark.slow
    def test_ISD(self):
        r"""**Description:**
        Tests the imaginary self-dual (ISD) matrix :math:`\mathcal{M}` and
        its holomorphic and anti-holomorphic derivatives.

        The ISD matrix is the real, symmetric, symplectic matrix that
        projects three-form fluxes onto their ISD and IASD parts.  It
        satisfies the algebraic relations

        .. math::
            \mathcal{M} = \mathcal{M}^T\,,\quad
            \mathcal{M}^T\,\Sigma\,\mathcal{M} = \Sigma\,,\quad
            \mathcal{M}^{-1} = \Sigma^T\,\mathcal{M}\,\Sigma\,,

        where :math:`\Sigma` is the symplectic matrix.

        The holomorphic and anti-holomorphic derivatives with respect to the
        full set of periods (:math:`dM\_X`, :math:`dM\_cX`) and the
        physical moduli (:math:`dM`, :math:`dM\_c`) are also tested for
        correct shapes and the conjugation relation
        :math:`\partial_{\overline{X}^I}\mathcal{M}=\overline{\partial_{X^I}\mathcal{M}}`.
        """

        # Evaluate the ISD matrix at the test point
        M = self.variant(self.model.ISD_matrix)(self.z, self.cz)

        chex.assert_type(M, complex)
        chex.assert_shape(M, (2 * (self.model.h12 + 1), 2 * (self.model.h12 + 1)))

        # Imaginary part of the ISD matrix must vanish (it is a real matrix)
        self.assertAllClose(M.imag, 0., rtol=1e-11, atol=1e-11,
                            msg="ISD_matrix must be real-valued")

        # Symmetry of the ISD matrix: M = M^T
        self.assertAllClose(M, M.T, rtol=1e-11, atol=1e-11,
                            msg="ISD_matrix must be symmetric: M != M.T")

        # Symplectic property: M^T Sigma M = Sigma
        self.assertAllClose(
            jnp.matmul(M.T, jnp.matmul(self.sigma, M)),
            self.sigma,
            rtol=1e-11, atol=1e-11,
            msg="ISD_matrix must satisfy M^T Sigma M = Sigma",
        )

        # Inverse identity: M^{-1} = Sigma^T M Sigma
        self.assertAllClose(
            jnp.linalg.inv(M),
            jnp.matmul(self.sigma.T, jnp.matmul(M, self.sigma)),
            rtol=1e-11, atol=1e-11,
            msg="M^{-1} must equal Sigma^T M Sigma",
        )

        # ------------------------------------------------------------------
        # Derivatives w.r.t. periods X^I
        # ------------------------------------------------------------------

        dM = self.variant(self.model.dM_X)(self.z, self.cz)
        dM_c = self.variant(self.model.dM_cX)(self.z, self.cz)

        chex.assert_type(dM, complex)
        chex.assert_shape(dM, (2 * (self.model.h12 + 1), 2 * (self.model.h12 + 1), self.model.h12 + 1))
        chex.assert_type(dM_c, complex)
        chex.assert_shape(dM_c, (2 * (self.model.h12 + 1), 2 * (self.model.h12 + 1), self.model.h12 + 1))

        # Anti-holomorphic derivative w.r.t. periods equals complex conjugate of holomorphic derivative
        self.assertAllClose(dM_c, jnp.conj(dM),
                            msg="dM_cX must equal conj(dM_X)")

        # ------------------------------------------------------------------
        # Derivatives w.r.t. moduli z^i
        # ------------------------------------------------------------------

        dM = self.variant(self.model.dM)(self.z, self.cz)
        dM_c = self.variant(self.model.dM_c)(self.z, self.cz)

        chex.assert_type(dM, complex)
        chex.assert_shape(dM, (2 * (self.model.h12 + 1), 2 * (self.model.h12 + 1), self.model.h12))
        chex.assert_type(dM_c, complex)
        chex.assert_shape(dM_c, (2 * (self.model.h12 + 1), 2 * (self.model.h12 + 1), self.model.h12))

        # Conjugation relation for moduli derivatives
        self.assertAllClose(dM_c, jnp.conj(dM),
                            msg="dM_c must equal conj(dM)")


    # ==========================================================================
    #  6. Gauge Kinetic Matrix
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    @pytest.mark.slow
    def test_gauge_kinetic_matrix(self):
        r"""**Description:**
        Tests the gauge kinetic matrix :math:`\mathcal{N}` and its derivatives
        for both the holomorphic (``conj=False``) and anti-holomorphic
        (``conj=True``) branches.

        The gauge kinetic matrix :math:`\mathcal{N}_{IJ}` governs the kinetic
        terms and topological couplings of the gauge fields arising from
        dimensional reduction of the Ramond-Ramond four-form potential.  At
        large complex structure it can be expressed in terms of the
        prepotential through

        .. math::
            \mathcal{N}_{IJ} = \overline{F}_{IJ}
            + 2\mathrm{i}\,\frac{\mathrm{Im}(F_{IK})X^K\,\mathrm{Im}(F_{JL})X^L}
            {\mathrm{Im}(F_{KL})X^K X^L}\,,

        where :math:`F_{IJ}=\partial_{X^I}\partial_{X^J}F`.

        The conjugation relation :math:`\mathcal{N}_c = \overline{\mathcal{N}}`
        and the corresponding relations for all derivatives are verified.
        """

        # ---------------------------------------------------------------
        # conj=False branch
        # ---------------------------------------------------------------

        conj = False
        N = self.variant(lambda x, y: self.model.gauge_kinetic_matrix(x, y, conj=conj))(self.z, self.cz)

        chex.assert_type(N, complex)
        chex.assert_shape(N, (self.model.h12 + 1, self.model.h12 + 1))

        # Holomorphic derivative w.r.t. all periods X^I
        dN_X = self.variant(lambda x, y: self.model.dN_X(x, y, conj=conj))(self.z, self.cz)
        dN_cX = self.variant(lambda x, y: self.model.dN_cX(x, y, conj=conj))(self.z, self.cz)

        chex.assert_type(dN_X, complex)
        chex.assert_shape(dN_X, (self.model.h12 + 1, self.model.h12 + 1, self.model.h12 + 1))
        chex.assert_type(dN_cX, complex)
        chex.assert_shape(dN_cX, (self.model.h12 + 1, self.model.h12 + 1, self.model.h12 + 1))

        # Holomorphic derivative w.r.t. moduli z^i
        dN_z = self.variant(lambda x, y: self.model.dN(x, y, conj=conj))(self.z, self.cz)
        dN_cz = self.variant(lambda x, y: self.model.dN_c(x, y, conj=conj))(self.z, self.cz)

        chex.assert_type(dN_z, complex)
        chex.assert_shape(dN_z, (self.model.h12 + 1, self.model.h12 + 1, self.model.h12))
        chex.assert_type(dN_cz, complex)
        chex.assert_shape(dN_cz, (self.model.h12 + 1, self.model.h12 + 1, self.model.h12))

        # ---------------------------------------------------------------
        # conj=True branch
        # ---------------------------------------------------------------

        conj = True
        N_c = self.variant(lambda x, y: self.model.gauge_kinetic_matrix(x, y, conj=conj))(self.z, self.cz)

        chex.assert_type(N_c, complex)
        chex.assert_shape(N_c, (self.model.h12 + 1, self.model.h12 + 1))

        # Holomorphic derivative w.r.t. all periods (conjugate branch)
        dN_X_c = self.variant(lambda x, y: self.model.dN_X(x, y, conj=conj))(self.z, self.cz)
        dN_cX_c = self.variant(lambda x, y: self.model.dN_cX(x, y, conj=conj))(self.z, self.cz)

        chex.assert_type(dN_X_c, complex)
        chex.assert_shape(dN_X_c, (self.model.h12 + 1, self.model.h12 + 1, self.model.h12 + 1))
        chex.assert_type(dN_cX_c, complex)
        chex.assert_shape(dN_cX_c, (self.model.h12 + 1, self.model.h12 + 1, self.model.h12 + 1))

        # Holomorphic derivative w.r.t. moduli (conjugate branch)
        dN_z_c = self.variant(lambda x, y: self.model.dN(x, y, conj=conj))(self.z, self.cz)
        dN_cz_c = self.variant(lambda x, y: self.model.dN_c(x, y, conj=conj))(self.z, self.cz)

        chex.assert_type(dN_z_c, complex)
        chex.assert_shape(dN_z_c, (self.model.h12 + 1, self.model.h12 + 1, self.model.h12))
        chex.assert_type(dN_cz_c, complex)
        chex.assert_shape(dN_cz_c, (self.model.h12 + 1, self.model.h12 + 1, self.model.h12))

        # ---------------------------------------------------------------
        # Conjugation relations across the two branches
        # ---------------------------------------------------------------

        # N_c must equal the complex conjugate of N
        self.assertAllClose(N_c, jnp.conj(N),
                            msg="gauge_kinetic_matrix(conj=True) must equal conj(gauge_kinetic_matrix(conj=False))")

        # Cross-conjugation: dN_cX(conj=True) = conj(dN_X(conj=False))
        self.assertAllClose(dN_cX_c, jnp.conj(dN_X),
                            msg="dN_cX(conj=True) must equal conj(dN_X(conj=False))")

        # Cross-conjugation: dN(conj=True) = conj(dN_c(conj=False))  (z-index)
        self.assertAllClose(dN_cz_c, jnp.conj(dN_z),
                            msg="dN_c(conj=True) must equal conj(dN(conj=False))")

        # Cross-conjugation: dN_X(conj=True) = conj(dN_cX(conj=False))
        self.assertAllClose(dN_X_c, jnp.conj(dN_cX),
                            msg="dN_X(conj=True) must equal conj(dN_cX(conj=False))")

        # Cross-conjugation: dN_c(conj=True) = conj(dN(conj=False))  (z-index)
        self.assertAllClose(dN_z_c, jnp.conj(dN_cz),
                            msg="dN(conj=True) must equal conj(dN_c(conj=False))")


    # ==========================================================================
    #  7. Prepotential and Period Vector
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_prepotential(self):
        r"""**Description:**
        Tests the prepotential :math:`F`, its LCS polynomial and instanton
        contributions, its holomorphic gradient :math:`\partial_{z^i}F`, and
        the LCS boundary conditions.

        At large complex structure the prepotential splits as

        .. math::
            F_{\mathrm{LCS}}(z^i) = F_{\mathrm{poly}}(z^i) + F_{\mathrm{inst}}(z^i)\,,

        where the polynomial part is

        .. math::
            F_{\mathrm{poly}}(z^i) = -\tfrac{1}{6}\widetilde{\kappa}_{ijk}z^iz^jz^k
            + \tfrac{1}{2}a_{ij}z^iz^j + b_iz^i + \tfrac{\mathrm{i}}{2}\xi\,,

        and the instanton corrections are exponentially small away from the
        boundary.

        The conjugation relations :math:`F_c = \overline{F}` and
        :math:`(\partial_{z^i}F)_c = \overline{\partial_{z^i}F}` are tested.

        The LCS boundary conditions are verified by evaluating
        :math:`F_{\mathrm{poly}}` and its derivatives at :math:`z^i=0`:

        * :math:`F_{\mathrm{poly}}(0) = \xi/2`,
        * :math:`\partial_{z^i}F_{\mathrm{poly}}\big|_0 = b_i`,
        * :math:`\partial_{z^i}\partial_{z^j}F_{\mathrm{poly}}\big|_0 = a_{ij}`,
        * :math:`\partial_{z^i}\partial_{z^j}\partial_{z^k}F_{\mathrm{poly}}\big|_0 = -\widetilde{\kappa}_{ijk}`.

        .. note::
            The class exposes ``F``, ``prepot``, and ``model.prepot`` as
            aliases for the same function.
        """

        conj = False

        # Evaluate prepotential and its components (holomorphic branch)
        F = self.variant(lambda x: self.model.F(x, conj=conj))(self.z)
        F_LCS = self.variant(lambda x: self.model.F_LCS(x, conj=conj))(self.z)
        F_LCS_poly = self.variant(lambda x: self.model.F_LCS_poly(x, conj=conj))(self.z)
        F_inst = self.variant(lambda x: self.model.F_inst(x, conj=conj))(self.z)
        dF = self.variant(lambda x: self.model.dF(x, conj=conj))(self.z)

        conj = True

        # Evaluate complex-conjugate branch
        F_c = self.variant(lambda x: self.model.F(x, conj=conj))(self.cz)
        F_LCS_c = self.variant(lambda x: self.model.F_LCS(x, conj=conj))(self.cz)
        F_LCS_poly_c = self.variant(lambda x: self.model.F_LCS_poly(x, conj=conj))(self.cz)
        F_inst_c = self.variant(lambda x: self.model.F_inst(x, conj=conj))(self.cz)
        dF_c = self.variant(lambda x: self.model.dF(x, conj=conj))(self.cz)

        # Shape checks
        chex.assert_shape(dF, (self.model.h12,))
        chex.assert_shape(dF_c, (self.model.h12,))

        # Conjugation relations
        self.assertAllClose(F_c, jnp.conj(F),
                            msg="F(conj=True)(cz) must equal conj(F(conj=False)(z))")
        self.assertAllClose(F_LCS_c, jnp.conj(F_LCS),
                            msg="F_LCS(conj=True)(cz) must equal conj(F_LCS(z))")
        self.assertAllClose(F_LCS_poly_c, jnp.conj(F_LCS_poly),
                            msg="F_LCS_poly(conj=True)(cz) must equal conj(F_LCS_poly(z))")
        self.assertAllClose(F_inst_c, jnp.conj(F_inst),
                            msg="F_inst(conj=True)(cz) must equal conj(F_inst(z))")
        self.assertAllClose(dF_c, jnp.conj(dF),
                            msg="dF(conj=True)(cz) must equal conj(dF(z))")

        # ------------------------------------------------------------------
        # LCS boundary conditions at z = 0
        # ------------------------------------------------------------------

        conj = False
        zzero = jnp.zeros(self.z.shape[0]) * 1j  # origin of moduli space, z = 0

        # F_poly(0) = xi / 2 = K0 / 2
        F_0 = self.variant(lambda x: self.model.F_LCS_poly(x, conj=conj))(zzero)
        chex.assert_shape(F_0, ())
        self.assertAllClose(F_0, self.model.lcs_tree.K0 / 2.,
                            msg="F_LCS_poly(0) must equal K0/2")

        # dF_poly(0) = b_i (b-vector)
        dF_0 = self.variant(
            lambda x: jax.grad(self.model.F_LCS_poly, holomorphic=True)(x, conj=conj)
        )(zzero)
        chex.assert_shape(dF_0, (self.model.h12,))
        self.assertAllClose(dF_0, self.model.lcs_tree.b_vector,
                            msg="first derivative of F_LCS_poly at z=0 must equal b_vector")

        # ddF_poly(0) = a_ij (a-matrix)
        ddF_0 = self.variant(
            lambda x: jax.jacfwd(jax.grad(self.model.F_LCS_poly, holomorphic=True), holomorphic=True)(x, conj=conj)
        )(zzero)
        chex.assert_shape(ddF_0, (self.model.h12, self.model.h12))
        self.assertAllClose(ddF_0, self.model.lcs_tree.a_matrix,
                            msg="second derivative of F_LCS_poly at z=0 must equal a_matrix")

        # dddF_poly(0) = -kappa_ijk (mirror intersection numbers)
        dddF_0 = self.variant(
            lambda x: jax.jacfwd(
                jax.jacfwd(jax.grad(self.model.F_LCS_poly, holomorphic=True), holomorphic=True),
                holomorphic=True
            )(x, conj=conj)
        )(zzero)
        chex.assert_shape(dddF_0, (self.model.h12, self.model.h12, self.model.h12))
        self.assertAllClose(dddF_0, -self.model.lcs_tree.intnums,
                            msg="third derivative of F_LCS_poly at z=0 must equal minus the mirror intersection numbers")


    @chex.variants(with_jit=True, without_jit=True)
    def test_period_vector(self):
        r"""**Description:**
        Tests the period vector :math:`\Pi` and its complex conjugate
        :math:`\overline{\Pi}`.

        In the gauge :math:`X^0=1` the period vector is

        .. math::
            \Pi(z^i) = \begin{pmatrix}
                2F - z^i F_i \\ F_i \\ 1 \\ z^i
            \end{pmatrix}\,,\quad i = 1,\ldots,h^{1,2}\,,

        and it has length :math:`2(h^{1,2}+1)`.  Since the prepotential is
        holomorphic, the conjugate period vector satisfies
        :math:`\Pi_c = \overline{\Pi}`.

        The test verifies shapes and the conjugation relation for
        ``period_vector`` with ``conj=False`` and ``conj=True``.
        """

        conj = False
        # Holomorphic period vector
        Pi = self.variant(lambda x: self.model.period_vector(x, conj=conj))(self.z)

        conj = True
        # Anti-holomorphic (conjugate) period vector
        Pi_c = self.variant(lambda x: self.model.period_vector(x, conj=conj))(self.cz)

        chex.assert_shape(Pi, (2 * (self.model.h12 + 1),))
        chex.assert_shape(Pi_c, (2 * (self.model.h12 + 1),))

        # Conjugation relation: Pi_c must equal conj(Pi)
        self.assertAllClose(Pi_c, jnp.conj(Pi),
                            msg="period_vector(cz, conj=True) must equal conj(period_vector(z, conj=False))")


    # ==========================================================================
    #  8. Aliases
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_aliases(self):
        r"""**Description:**
        Verifies that the class-level aliases return numerically identical
        results to the primary methods they wrap.

        The following aliases are defined in the ``css`` class:

        .. list-table::
            :header-rows: 1

            * - Alias
              - Primary method
            * - ``F``
              - ``prepot``
            * - ``N``
              - ``gauge_kinetic_matrix``
            * - ``M``
              - ``ISD_matrix``
            * - ``A``
              - ``mirror_volume``
            * - ``V_tilde``
              - ``mirror_volume``
            * - ``K``
              - ``kahler_potential``

        Each alias is evaluated at the test point and compared against the
        primary method to confirm exact numerical equality.
        """

        # ---------------------------------------------------------------
        # Prepotential aliases: F = prepot
        # ---------------------------------------------------------------

        F_alias = self.variant(lambda x: self.model.F(x, conj=False))(self.z)
        F_prepot = self.variant(lambda x: self.model.prepot(x, conj=False))(self.z)

        # Both aliases must return the same prepotential value
        self.assertAllClose(F_alias, F_prepot,
                            msg="F and prepot must return identical values")

        # ---------------------------------------------------------------
        # Gauge kinetic matrix alias: N = gauge_kinetic_matrix
        # ---------------------------------------------------------------

        N_alias = self.variant(lambda x, y: self.model.N(x, y, conj=False))(self.z, self.cz)
        N_primary = self.variant(lambda x, y: self.model.gauge_kinetic_matrix(x, y, conj=False))(self.z, self.cz)

        # Alias N must return the same matrix as gauge_kinetic_matrix
        self.assertAllClose(N_alias, N_primary,
                            msg="N and gauge_kinetic_matrix must return identical values")

        # ---------------------------------------------------------------
        # ISD matrix alias: M = ISD_matrix
        # ---------------------------------------------------------------

        M_alias = self.variant(self.model.M)(self.z, self.cz)
        M_primary = self.variant(self.model.ISD_matrix)(self.z, self.cz)

        # Alias M must return the same matrix as ISD_matrix
        self.assertAllClose(M_alias, M_primary,
                            msg="M and ISD_matrix must return identical values")

        # ---------------------------------------------------------------
        # Mirror volume aliases: A = V_tilde = mirror_volume
        # ---------------------------------------------------------------

        A_alias = self.variant(self.model.A)(self.z, self.cz)
        Vtilde_alias = self.variant(self.model.V_tilde)(self.z, self.cz)
        A_primary = self.variant(self.model.mirror_volume)(self.z, self.cz)

        # Both A and V_tilde must agree with mirror_volume
        self.assertAllClose(A_alias, A_primary,
                            msg="A and mirror_volume must return identical values")
        self.assertAllClose(Vtilde_alias, A_primary,
                            msg="V_tilde and mirror_volume must return identical values")

        # ---------------------------------------------------------------
        # Kähler potential alias: K = kahler_potential
        # ---------------------------------------------------------------

        K_alias = self.variant(self.model.K)(self.z, self.cz, self.tau, self.ctau)
        K_primary = self.variant(self.model.kahler_potential)(self.z, self.cz, self.tau, self.ctau)

        # Alias K must return the same value as kahler_potential
        self.assertAllClose(K_alias, K_primary,
                            msg="K and kahler_potential must return identical values")


    # ==========================================================================
    #  9. Moduli-Periods Round-Trip
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_moduli_periods_roundtrip(self):
        r"""**Description:**
        Tests that the composition ``periods_to_moduli`` ∘ ``moduli_to_periods``
        is the identity on the complex structure moduli.

        Given moduli :math:`z^i`, the function ``moduli_to_periods`` lifts them
        to periods by prepending the gauge choice :math:`X^0=1`,

        .. math::
            (z^1,\ldots,z^{h^{1,2}}) \xmapsto{\mathrm{moduli\_to\_periods}}
            (1, z^1, \ldots, z^{h^{1,2}})\,.

        The inverse ``periods_to_moduli`` recovers the projective moduli by
        dividing the :math:`i>0` components by the zeroth component,

        .. math::
            (X^0, X^1, \ldots, X^{h^{1,2}}) \xmapsto{\mathrm{periods\_to\_moduli}}
            X^i / X^0 = z^i\, .

        The round-trip must recover the original moduli to within machine
        precision.
        """

        # Lift moduli to periods (prepend X^0 = gauge_choice)
        XPer = self.variant(lambda x: self.model.moduli_to_periods(x, conj=False))(self.z)

        chex.assert_shape(XPer, (self.model.h12 + 1,))

        # The zeroth component must equal the gauge choice
        self.assertAllClose(XPer[0], self.model.gauge_choice,
                            msg="moduli_to_periods must prepend the gauge choice as the zeroth period")

        # The remaining components must equal the moduli themselves
        self.assertAllClose(XPer[1:], self.z * self.model.gauge_choice,
                            msg="moduli_to_periods must scale moduli by the gauge choice")

        # Recover moduli from periods
        z_recovered = self.variant(self.model.periods_to_moduli)(XPer)

        chex.assert_shape(z_recovered, (self.model.h12,))

        # Round-trip must recover the original moduli to machine precision
        self.assertAllClose(z_recovered, self.z, rtol=1e-14, atol=1e-14,
                            msg="periods_to_moduli(moduli_to_periods(z)) must recover z")


    # ==========================================================================
    #  10. Gauge Kinetic Matrix Properties
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_gauge_kinetic_matrix_properties(self):
        r"""**Description:**
        Tests the defining algebraic properties of the gauge kinetic matrix
        :math:`\mathcal{N}`.

        The gauge kinetic matrix governs the gauge kinetic function for the
        Ramond-Ramond gauge fields.  It satisfies the following properties:

        (a) **Symmetry**: :math:`\mathcal{N} = \mathcal{N}^T`.
        (b) **Negative-semidefinite imaginary part**:
            all eigenvalues of :math:`\mathrm{Im}(\mathcal{N})` are
            :math:`\leq 0`, reflecting the positivity of the gauge kinetic term
            in the effective action.
        (c) **Conjugation**: :math:`\mathcal{N}_c = \overline{\mathcal{N}}`.

        These conditions are necessary for the consistent four-dimensional
        effective field theory.
        """

        # Holomorphic and anti-holomorphic gauge kinetic matrices
        N = self.variant(lambda x, y: self.model.gauge_kinetic_matrix(x, y, conj=False))(self.z, self.cz)
        N_c = self.variant(lambda x, y: self.model.gauge_kinetic_matrix(x, y, conj=True))(self.z, self.cz)

        chex.assert_shape(N, (self.model.h12 + 1, self.model.h12 + 1))
        chex.assert_shape(N_c, (self.model.h12 + 1, self.model.h12 + 1))

        # (a) Symmetry: N must be symmetric
        self.assertAllClose(N, N.T, rtol=1e-11, atol=1e-11,
                            msg="gauge_kinetic_matrix must be symmetric: N != N.T")

        # (b) Negative-semidefinite Im(N): all eigenvalues of Im(N) must be <= 0
        eigvals_ImN = jnp.linalg.eigvalsh(N.imag)

        # All eigenvalues of Im(N) must be non-positive
        self.assertTrue(
            jnp.all(eigvals_ImN <= 1e-8),
            msg="Im(gauge_kinetic_matrix) must be negative semidefinite: found positive eigenvalue(s)"
        )

        # (c) Conjugation: N_c must equal the complex conjugate of N
        self.assertAllClose(N_c, jnp.conj(N), rtol=1e-11, atol=1e-11,
                            msg="gauge_kinetic_matrix(conj=True) must equal conj(gauge_kinetic_matrix(conj=False))")

    def test_seeded_sample_geometry_properties(self):
        r"""**Description:**
        Check representative Kähler and gauge-kinetic identities on a small
        deterministic ensemble.  This complements the fixed class fixture while
        keeping the sample reproducible and non-JIT.
        """
        rng = np.random.default_rng(20260525)

        for _ in range(4):
            z = jnp.asarray(
                rng.uniform(-0.35, 0.35, self.model.h12)
                + 1j * rng.uniform(2.0, 4.5, self.model.h12)
            )
            cz = jnp.conj(z)
            tau = complex(
                rng.uniform(-0.4, 0.4),
                rng.uniform(2.5, 7.0),
            )
            ctau = jnp.conj(tau)

            X = self.model.moduli_to_periods(z, conj=False)
            self.assertAllClose(
                self.model.periods_to_moduli(X),
                z,
                rtol=1e-12,
                atol=1e-12,
            )

            K = self.model.kahler_potential(z, cz, tau, ctau)
            KM = self.model.kahler_metric(z, cz, tau, ctau)
            self.assertAllClose(jnp.imag(K), 0.0, rtol=1e-11, atol=1e-11)
            self.assertAllClose(KM, jnp.conj(KM.T), rtol=1e-10, atol=1e-10)
            self.assertTrue(
                jnp.all(jnp.linalg.eigvalsh(jnp.real(KM)) > 0.0),
                msg="Seeded sample produced a non-positive Kähler metric eigenvalue",
            )

            N = self.model.gauge_kinetic_matrix(z, cz, conj=False)
            N_c = self.model.gauge_kinetic_matrix(z, cz, conj=True)
            self.assertAllClose(N, N.T, rtol=1e-10, atol=1e-10)
            self.assertAllClose(N_c, jnp.conj(N), rtol=1e-10, atol=1e-10)
            self.assertTrue(
                jnp.all(jnp.linalg.eigvalsh(jnp.imag(N)) <= 1e-8),
                msg="Seeded sample produced a positive Im(N) eigenvalue",
            )


    # ==========================================================================
    #  11. Kahler Metric from Second Derivatives
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_kahler_metric_from_second_derivatives(self):
        r"""**Description:**
        Verifies the assembly of the Kähler metric :math:`K_{I\overline{J}}`
        from the individual blocks of second derivatives of :math:`K`.

        The full Kähler metric is assembled as

        .. math::
            K_{I\overline{J}} = \left(\begin{array}{cc}
                K_{i\overline{j}} & K_{i\overline{\tau}} \\[4pt]
                K_{\tau\overline{j}} & K_{\tau\overline{\tau}}
            \end{array}\right)\,,

        where the upper-left :math:`h^{1,2}\times h^{1,2}` block is
        :math:`\partial_{z^i}\partial_{\overline{z}^j}K` and the lower-right
        entry is :math:`\partial_\tau\partial_{\overline{\tau}}K`.

        The test verifies:

        * The upper-left block of ``kahler_metric(mode=None)`` equals
          ``ddK_z_cz``.
        * The lower-right entry of ``kahler_metric(mode=None)`` equals
          ``ddK_tau_ctau``.
        """

        # Full Kähler metric assembled from all second derivatives
        KM = self.variant(
            lambda x, y, z, u: self.model.kahler_metric(x, y, z, u, mode=None)
        )(self.z, self.cz, self.tau, self.ctau)

        # Individual second derivative blocks
        ddK_z_cz = self.variant(self.model.ddK_z_cz)(self.z, self.cz, self.tau, self.ctau)
        ddK_tau_ctau = self.variant(self.model.ddK_tau_ctau)(self.z, self.cz, self.tau, self.ctau)

        h12 = self.model.h12

        # Upper-left h12 x h12 block of the metric equals K_{i \bar{j}}
        self.assertAllClose(
            KM[:h12, :h12], ddK_z_cz, rtol=1e-11, atol=1e-11,
            msg="kahler_metric[:h12,:h12] must equal ddK_z_cz"
        )

        # Lower-right scalar entry of the metric equals K_{\tau \bar{\tau}}
        self.assertAllClose(
            KM[-1, -1], ddK_tau_ctau, rtol=1e-11, atol=1e-11,
            msg="kahler_metric[-1,-1] must equal ddK_tau_ctau"
        )


    # ==========================================================================
    #  12. Period Vector Symplectic Relations
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_period_vector_symplectic(self):
        r"""**Description:**
        Verifies the symplectic inner-product relations satisfied by the period
        vector :math:`\Pi`.

        The holomorphic three-form :math:`\Omega_3` satisfies
        :math:`\Omega_3\wedge\Omega_3=0` by degree counting, which translates
        into the *isotropicity* condition for the period vector,

        .. math::
            \Pi^T \cdot \Sigma \cdot \Pi = 0\,.

        The mirror dual Calabi-Yau volume :math:`\mathcal{A}` is defined by

        .. math::
            \mathcal{A}(z^i, \overline{z}^i)
            = -\mathrm{i}\,\overline{\Pi}^T\cdot\Sigma\cdot\Pi
            = -\mathrm{i}\,\Pi_c^T\cdot\Sigma\cdot\Pi\,,

        which encodes the positivity condition required for a physical Kähler
        potential.

        The test verifies:

        * Isotropicity: :math:`\Pi^T\Sigma\Pi = 0`.
        * Internal consistency of ``mirror_volume`` with its definition:
          :math:`\mathcal{A} = -\mathrm{i}\,\Pi_c^T\Sigma\Pi`.

        .. note::
            The symplectic matrix :math:`\Sigma` is of the form

            .. math::
                \Sigma = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}

            in :math:`(h^{1,2}+1)\times(h^{1,2}+1)` blocks.
        """

        # Evaluate holomorphic and conjugate period vectors
        Pi = self.variant(lambda x: self.model.period_vector(x, conj=False))(self.z)
        Pi_c = self.variant(lambda x: self.model.period_vector(x, conj=True))(self.cz)

        # Evaluate the mirror volume
        A = self.variant(self.model.A)(self.z, self.cz)

        # Isotropicity: Pi^T Sigma Pi = 0
        # (follows from Omega ^ Omega = 0 by degree counting)
        Pi_sigma_Pi = jnp.matmul(Pi, jnp.matmul(self.sigma, Pi))
        self.assertAllClose(Pi_sigma_Pi, 0., rtol=1e-11, atol=1e-11,
                            msg="period vector must satisfy Pi.T @ sigma @ Pi = 0 (isotropicity)")

        # Definition of mirror volume: A = -i * Pi_c^T @ sigma @ Pi
        A_from_periods = -1.j * jnp.matmul(Pi_c, jnp.matmul(self.sigma, Pi))
        self.assertAllClose(A_from_periods, A, rtol=1e-11, atol=1e-11,
                            msg="mirror_volume must equal -i * Pi_c @ sigma @ Pi")


# ==============================================================================
#  TestKahlerGeometry — Christoffel symbols, Riemann tensor, and related
# ==============================================================================

class TestKahlerGeometry(TestCase):
    r"""
    **Description:**
    Test suite for the Kähler geometry methods on :class:`css`:
    :func:`dddK`, :func:`dddK_c`, :func:`christoffel_symbols`,
    :func:`riemann_tensor`, :func:`dIKM_c`, :func:`ddIKM`, :func:`dGamma`.

    .. admonition:: Background
        :class: dropdown

        On the moduli space :math:`\mathcal{M}_{\rm cs} \times \mathcal{M}_\tau`
        of the IIB compactification, the Kähler metric
        :math:`K_{I\bar J} = \partial_I \partial_{\bar J} K` defines a
        Kähler–Hodge geometry.  The Christoffel symbols

        .. math::
            \Gamma^E_{AC} = K^{E\bar F}\,\partial_A K_{C\bar F}

        and the Riemann curvature tensor

        .. math::
            R_{i\bar jk\bar l} = \partial_k\partial_{\bar l}K_{i\bar j}
            - K^{m\bar n}(\partial_k K_{i\bar n})(\partial_{\bar l}K_{m\bar j})

        govern the mass spectrum and moduli stabilisation.  These tests verify
        shapes, symmetries, and algebraic identities at generic (non-SUSY) points.

    These tests do **not** require finding a SUSY vacuum — they only need
    a model and generic field-space points.

    Attributes:
        model (FluxEFT): Physics model with :math:`h^{1,2}=2`.
        n (int): Dimension :math:`h^{1,2}+1 = 3`.
        test_points (list): Two random ``(z, zc, tau, tauc)`` tuples.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        h12 = 2
        cls.model = jaxvacua.FluxEFT(
            h12=h12, model_ID=1, model_type="KS", maximum_degree=0
        )
        cls.n = h12 + 1

        # Two generic test points with Im(z) ∈ [2.5, 4.5], Im(τ) ∈ [2, 7]
        # Random but reproducible via fixed seed
        rng = np.random.default_rng(42)
        cls.test_points = []
        for _ in range(2):
            z = jnp.array(
                rng.uniform(-0.3, 0.3, h12) + 1j * rng.uniform(2.5, 4.5, h12)
            )
            tau = complex(rng.uniform(-0.4, 0.4) + 1j * rng.uniform(2.0, 7.0))
            cls.test_points.append((z, jnp.conj(z), tau, jnp.conj(tau)))

        cls.z, cls.cz, cls.tau, cls.ctau = cls.test_points[0]

    # ------------------------------------------------------------------
    # dddK / dddK_c
    # ------------------------------------------------------------------

    @chex.variants(with_jit=True, without_jit=True)
    def test_dddK_shape(self):
        r"""
        **Description:**
        Verify that the holomorphic third Kähler derivative
        :math:`\partial_A K_{C\bar{F}}` has shape ``(n, n, n)``.

        This is the building block for the Christoffel symbols:
        :math:`\Gamma^E_{AC} = K^{E\bar F}\,(\texttt{dddK})_{C,\bar F,A}`.
        Indices: ``[C, F̄, A]`` where :math:`C,\bar F,A` each run over
        :math:`0,\ldots,n-1` with :math:`n = h^{1,2}+1`.
        """
        fn = self.variant(self.model.dddK)
        K3h = fn(self.z, self.cz, self.tau, self.ctau)
        # K3h[C, F̄, A] = ∂_A K_{CF̄}
        chex.assert_shape(K3h, (self.n, self.n, self.n))

    @chex.variants(with_jit=True, without_jit=True)
    def test_dddK_c_shape(self):
        r"""
        **Description:**
        Verify that the antiholomorphic third Kähler derivative
        :math:`\partial_{\bar{B}} K_{C\bar{F}}` has shape ``(n, n, n)``.

        This is the conjugate building block, used in :func:`dIKM_c`
        and the Riemann tensor.  Indices: ``[C, F̄, B̄]``.
        """
        fn = self.variant(self.model.dddK_c)
        K3a = fn(self.z, self.cz, self.tau, self.ctau)
        # K3a[C, F̄, B̄] = ∂_{B̄} K_{CF̄}
        chex.assert_shape(K3a, (self.n, self.n, self.n))

    # ------------------------------------------------------------------
    # christoffel_symbols
    # ------------------------------------------------------------------

    @chex.variants(with_jit=True, without_jit=True)
    def test_christoffel_shape(self):
        r"""
        **Description:**
        Verify that the Christoffel symbols :math:`\Gamma^E_{AC}` are a rank-3
        tensor of shape ``(n, n, n)`` where :math:`n = h^{1,2}+1`.
        """
        fn = self.variant(self.model.christoffel_symbols)
        Gamma = fn(self.z, self.cz, self.tau, self.ctau)
        # Γ^E_{AC} has 3 indices: upper E, lower A, lower C — each runs 0..n-1
        chex.assert_shape(Gamma, (self.n, self.n, self.n))

    @chex.variants(with_jit=True, without_jit=True)
    def test_christoffel_torsion_free(self):
        r"""
        **Description:**
        Verify the torsion-free property of the Kähler connection:
        :math:`\Gamma^E_{AC} = \Gamma^E_{CA}`.

        This symmetry in the lower indices follows from the Kähler condition
        (the metric derives from a single real function :math:`K`), which
        implies that the connection has no torsion.  Tested at two random
        field-space points.
        """
        fn = self.variant(self.model.christoffel_symbols)
        for z, cz, tau, ctau in self.test_points:
            Gamma = fn(z, cz, tau, ctau)
            # Γ^E_{AC} vs Γ^E_{CA}: transpose indices (1,2) ↔ swap A and C
            self.assertAllClose(
                Gamma, Gamma.transpose(0, 2, 1), atol=1e-12,
                msg="Christoffel symbols must be symmetric in lower indices",
            )

    @chex.variants(with_jit=True, without_jit=True)
    def test_christoffel_from_metric(self):
        r"""
        **Description:**
        Verify :math:`\Gamma^E_{AC} = K^{E\bar{F}} \partial_A K_{C\bar{F}}`
        by comparing :func:`christoffel_symbols` to an explicit construction
        from :func:`dddK` (the third Kähler derivative) and
        :func:`inverse_kahler_metric`.

        This cross-checks two independent code paths: one computes
        :math:`\Gamma` directly, the other contracts :math:`K^{-1}` with
        :math:`\partial_A K_{C\bar F}` via ``einsum``.
        """
        fn_gamma = self.variant(self.model.christoffel_symbols)
        fn_dddK = self.variant(self.model.dddK)
        fn_ikm = self.variant(self.model.inverse_kahler_metric)

        # Compute Γ from the dedicated method
        Gamma = fn_gamma(self.z, self.cz, self.tau, self.ctau)

        # Compute the same quantity from its definition:
        # K3h[C,F̄,A] = ∂_A K_{CF̄}  (third Kähler derivative)
        # IKM[E,F̄] = K^{EF̄}          (inverse metric)
        K3h = fn_dddK(self.z, self.cz, self.tau, self.ctau)
        IKM = fn_ikm(self.z, self.cz, self.tau, self.ctau)

        # Γ^E_{AC} = K^{EF̄} ∂_A K_{CF̄} = einsum over F̄
        Gamma_ref = jnp.einsum('ef,cfa->eac', IKM, K3h)

        # The two computations must agree to machine precision
        self.assertAllClose(Gamma, Gamma_ref, atol=1e-12)

    # ------------------------------------------------------------------
    # riemann_tensor
    # ------------------------------------------------------------------

    @chex.variants(with_jit=True, without_jit=True)
    def test_riemann_shape(self):
        r"""
        **Description:**
        Verify that the Riemann curvature tensor :math:`R_{i\bar{j}k\bar{l}}`
        is a rank-4 tensor of shape ``(n, n, n, n)`` with the index pattern
        holomorphic–antiholomorphic–holomorphic–antiholomorphic.
        """
        fn = self.variant(self.model.riemann_tensor)
        R = fn(self.z, self.cz, self.tau, self.ctau)
        # R_{ij̄kl̄} has 4 indices: holo i, antiholo j̄, holo k, antiholo l̄
        chex.assert_shape(R, (self.n, self.n, self.n, self.n))

    @chex.variants(with_jit=True, without_jit=True)
    def test_riemann_symmetry(self):
        r"""
        **Description:**
        Verify the Kähler symmetry of the Riemann tensor:
        :math:`R_{i\bar{j}k\bar{l}} = R_{k\bar{j}i\bar{l}}`.

        This symmetry is specific to Kähler manifolds and follows from the
        fact that the Riemann tensor derives from a single Kähler potential.
        It swaps the two holomorphic indices while keeping the antiholomorphic
        ones fixed.  In array index notation: ``R[i,j,k,l] == R[k,j,i,l]``
        (transpose axes 0 and 2).  Tested at two random points.
        """
        fn = self.variant(self.model.riemann_tensor)
        for z, cz, tau, ctau in self.test_points:
            R = fn(z, cz, tau, ctau)
            # Swap holomorphic indices: (i, j̄, k, l̄) → (k, j̄, i, l̄)
            self.assertAllClose(
                R, R.transpose(2, 1, 0, 3), atol=1e-12,
                msg="Riemann tensor must satisfy R_{ijkl} = R_{kjil}",
            )

    # ------------------------------------------------------------------
    # dIKM_c
    # ------------------------------------------------------------------

    @chex.variants(with_jit=True, without_jit=True)
    def test_dIKM_c_shape(self):
        r"""
        **Description:**
        Verify that the antiholomorphic derivative of the inverse Kähler metric
        :math:`\partial_{\bar{B}}K^{I\bar{J}}` has shape ``(n, n, n)``
        with indices ``[I, J̄, B̄]``.
        """
        fn = self.variant(self.model.dIKM_c)
        result = fn(self.z, self.cz, self.tau, self.ctau)
        # ∂_{B̄}(K^{IJ̄}) has indices [I, J̄, B̄], each running 0..n-1
        chex.assert_shape(result, (self.n, self.n, self.n))

    @chex.variants(with_jit=True, without_jit=True)
    def test_dIKM_c_identity(self):
        r"""
        **Description:**
        Verify that :func:`dIKM_c` satisfies the matrix inverse derivative identity

        .. math::
            \partial_{\bar B}K^{I\bar J} = -K^{I\bar F}\,(\partial_{\bar B}K_{C\bar F})\,K^{C\bar J}

        which follows from :math:`\partial_{\bar B}(K \cdot K^{-1}) = 0`.
        Cross-checks :func:`dIKM_c` against an explicit ``einsum`` of
        :func:`inverse_kahler_metric` and :func:`dddK_c`.
        """
        fn_dIKM = self.variant(self.model.dIKM_c)
        fn_dddKc = self.variant(self.model.dddK_c)
        fn_ikm = self.variant(self.model.inverse_kahler_metric)

        # Compute ∂_{B̄}(K^{-1}) from the dedicated method
        dIKM_bar = fn_dIKM(self.z, self.cz, self.tau, self.ctau)

        # Build the same quantity from its definition:
        # K3a[C,F̄,B̄] = ∂_{B̄} K_{CF̄}  (antiholomorphic 3rd Kähler derivative)
        K3a = fn_dddKc(self.z, self.cz, self.tau, self.ctau)
        IKM = fn_ikm(self.z, self.cz, self.tau, self.ctau)

        # ∂_{B̄} K^{IJ̄} = -K^{IF̄} (∂_{B̄} K_{CF̄}) K^{CJ̄}
        # Contracted as: -einsum('if,cfb,cj->ijb', IKM, K3a, IKM)
        # where f sums over F̄, c sums over C
        dIKM_ref = -jnp.einsum('if,cfb,cj->ijb', IKM, K3a, IKM)

        # Must match the dedicated method to machine precision
        self.assertAllClose(dIKM_bar, dIKM_ref, atol=1e-12)

    # ------------------------------------------------------------------
    # ddIKM
    # ------------------------------------------------------------------

    @chex.variants(with_jit=True, without_jit=True)
    def test_ddIKM_shape(self):
        r"""
        **Description:**
        Verify that the mixed second derivative of the inverse Kähler metric
        :math:`\partial_A\partial_{\bar{B}}K^{I\bar{J}}` has shape
        ``(n, n, n, n)`` with indices ``[I, J̄, B̄, A]``.  This tensor
        contains the Riemann curvature contribution to the SUGRA Hessian.
        """
        fn = self.variant(self.model.ddIKM)
        result = fn(self.z, self.cz, self.tau, self.ctau)
        # ∂_A∂_{B̄}(K^{IJ̄}) has indices [I, J̄, B̄, A] — rank-4 tensor
        chex.assert_shape(result, (self.n, self.n, self.n, self.n))

    @chex.variants(with_jit=True, without_jit=True)
    def test_ddIKM_vs_jacrev(self):
        r"""
        **Description:**
        Verify :math:`\partial_A\partial_{\bar{B}}(K^{-1})` from the explicit
        Riemann + Christoffel decomposition against a ``jacrev`` reference.

        The function :func:`ddIKM` uses the analytic formula involving the
        Riemann tensor and Christoffel symbols, while the reference computes
        ``jacrev(dIKM_c)`` numerically.  Agreement to ``atol=1e-10`` validates
        both the Riemann tensor computation and the Christoffel symbol algebra.
        """
        # Compute ∂_A∂_{B̄}(K^{-1}) via the analytic decomposition
        # (uses Riemann tensor + Christoffel symbols internally)
        fn_ddIKM = self.variant(self.model.ddIKM)
        d2IKM = fn_ddIKM(self.z, self.cz, self.tau, self.ctau)

        # Build an independent reference via jacrev of dIKM_c:
        # ∂_A(∂_{B̄} K^{-1}) = jacrev(dIKM_c, argnums=(z, tau))
        def _cat(a, b):
            return jnp.concatenate([a, b[..., None]], axis=-1)

        def dIKM_bar_func(z_, cz_, t_, ct_):
            return self.model.dIKM_c(z_, cz_, t_, ct_)

        # Differentiate dIKM_c w.r.t. holomorphic directions (z, tau)
        d2IKM_ref = _cat(
            jax.jacrev(dIKM_bar_func, argnums=0, holomorphic=True)(
                self.z, self.cz, self.tau, self.ctau),
            jax.jacrev(dIKM_bar_func, argnums=2, holomorphic=True)(
                self.z, self.cz, self.tau, self.ctau),
        )

        # Agreement validates both the Riemann tensor and Christoffel algebra
        self.assertAllClose(d2IKM, d2IKM_ref, atol=1e-10)

    # ------------------------------------------------------------------
    # dGamma
    # ------------------------------------------------------------------

    @chex.variants(with_jit=True, without_jit=True)
    def test_dGamma_shape(self):
        r"""
        **Description:**
        Verify that the holomorphic derivative of the Christoffel symbols
        :math:`\partial_B \Gamma^E_{AC}` has the correct rank-4 shape ``(n, n, n, n)``.

        This tensor appears in the holomorphic Hessian :math:`\partial_A\partial_B(K^{-1})`
        via the formula :math:`\partial_A\partial_B K^{IJ̄} = -(\partial_B\Gamma^I_{AC})K^{CJ̄}
        + \Gamma^I_{AC}\Gamma^C_{BD}K^{DJ̄}`.
        """
        fn = self.variant(self.model.dGamma)
        result = fn(self.z, self.cz, self.tau, self.ctau)
        # ∂_B Γ^E_{AC} has indices [E, A, C, B] — rank-4 tensor
        chex.assert_shape(result, (self.n, self.n, self.n, self.n))


# ==============================================================================
#  TestMonodromy
# ==============================================================================

class TestMonodromy(TestCase):
    r"""
    **Description:**
    Test suite for the monodromy matrix methods on the ``css`` class.
    Verifies the analytical LCS formula, the numerical solver, and their agreement.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = jaxvacua.FluxEFT(
            h12=2, model_ID=1, maximum_degree=2, limit="LCS", model_type="KS"
        )
        cls.h = cls.model.h12
        cls.dim = 2 * cls.h + 2
        rng = np.random.default_rng(99)
        cls.z = jnp.array(rng.uniform(-0.3, 0.3, cls.h) + 1j * rng.uniform(2, 5, cls.h))

    # ------------------------------------------------------------------
    # monodromy_matrix_single
    # ------------------------------------------------------------------

    def test_monodromy_single_shape(self):
        r"""
        **Description:**
        ``monodromy_matrix_single(b)`` returns an integer matrix of shape
        ``(2*h12+2, 2*h12+2)``.
        """
        for b in range(self.h):
            T = self.model.monodromy_matrix_single(b)
            self.assertEqual(T.shape, (self.dim, self.dim))
            self.assertTrue(np.issubdtype(T.dtype, np.integer))

    def test_monodromy_single_is_integer(self):
        r"""
        **Description:**
        The monodromy matrix must have integer entries (it is in Sp(2h+2, Z)).
        """
        for b in range(self.h):
            T = self.model.monodromy_matrix_single(b)
            self.assertTrue(np.all(T == T.astype(int)))

    def test_monodromy_single_verify_period_vector(self):
        r"""
        **Description:**
        Verify T_b . Pi(z) = Pi(z + e_b) for each direction b.
        """
        for b in range(self.h):
            result = self.model.verify_monodromy(b, z=self.z, tol=1e-8)
            self.assertTrue(result['passed'],
                            msg=f"Monodromy T_{b} failed: max_error = {result['max_error']:.2e}")

    def test_monodromy_single_LCS_equals_numerical(self):
        r"""
        **Description:**
        The analytical LCS formula must produce the same integer matrix
        as the numerical solver.
        """
        for b in range(self.h):
            T_lcs = self.model._monodromy_matrix_LCS(b)
            T_num = self.model._monodromy_matrix_numerical(b)
            np.testing.assert_array_equal(
                T_lcs, T_num,
                err_msg=f"LCS and numerical monodromy disagree for b={b}"
            )

    # ------------------------------------------------------------------
    # monodromy_matrix (general shift)
    # ------------------------------------------------------------------

    def test_monodromy_general_identity(self):
        r"""
        **Description:**
        A zero shift n = (0, ..., 0) gives the identity matrix.
        """
        T = self.model.monodromy_matrix(np.zeros(self.h, dtype=int))
        np.testing.assert_array_equal(T, np.eye(self.dim, dtype=int))

    def test_monodromy_general_single_direction(self):
        r"""
        **Description:**
        ``monodromy_matrix(e_b)`` must equal ``monodromy_matrix_single(b)``.
        """
        for b in range(self.h):
            e_b = np.zeros(self.h, dtype=int)
            e_b[b] = 1
            T_gen = self.model.monodromy_matrix(e_b)
            T_single = self.model.monodromy_matrix_single(b)
            np.testing.assert_array_equal(T_gen, T_single)

    def test_monodromy_general_verify_period_vector(self):
        r"""
        **Description:**
        Verify T(n) . Pi(z) = Pi(z + n) for several shift vectors.
        """
        for n_vec in [[1, 1], [2, -1], [3, 2], [-1, 3]]:
            n_arr = np.array(n_vec, dtype=int)
            T = self.model.monodromy_matrix(n_arr)

            Pi = np.array(self.model.period_vector(self.z))
            z_shifted = self.z + jnp.array(n_arr, dtype=self.z.dtype)
            Pi_shifted = np.array(self.model.period_vector(z_shifted))

            T_Pi = T @ Pi
            error = np.max(np.abs(T_Pi - Pi_shifted))
            self.assertLess(error, 1e-8,
                            msg=f"General monodromy T(n={n_vec}) failed: error = {error:.2e}")

    def test_monodromy_composition(self):
        r"""
        **Description:**
        T(n1 + n2) = T(n1) @ T(n2) — monodromy matrices compose additively.
        """
        n1 = np.array([2, 1], dtype=int)
        n2 = np.array([-1, 3], dtype=int)
        T1 = self.model.monodromy_matrix(n1)
        T2 = self.model.monodromy_matrix(n2)
        T_sum = self.model.monodromy_matrix(n1 + n2)
        np.testing.assert_array_equal(T1 @ T2, T_sum)

    def test_monodromy_inverse(self):
        r"""
        **Description:**
        T(n) @ T(-n) = I — the inverse monodromy is T(-n).
        """
        for n_vec in [[1, 0], [0, 1], [2, -1]]:
            n_arr = np.array(n_vec, dtype=int)
            T = self.model.monodromy_matrix(n_arr)
            T_inv = self.model.monodromy_matrix(-n_arr)
            product = T @ T_inv
            np.testing.assert_array_equal(
                product, np.eye(self.dim, dtype=int),
                err_msg=f"T(n={n_vec}) @ T(-n) != I"
            )

    def test_conifold_monodromy_matrix_from_index(self):
        r"""
        **Description:**
        The Picard-Lefschetz conifold monodromy for ``conifold_index=b`` has
        the expected rank-one ``F_a <- F_a + c_a c_b z^b`` block.
        """

        T = self.model.conifold_monodromy_matrix(conifold_index=1)
        expected = np.eye(self.dim, dtype=int)
        expected[1 + 1, self.h + 2 + 1] += 1

        np.testing.assert_array_equal(T, expected)

    def test_conifold_monodromy_matrix_from_curve(self):
        r"""
        **Description:**
        A general conifold curve ``c`` should insert the outer product
        ``c_a c_b`` in the magnetic/electric off-diagonal block.
        """

        c = np.array([2, -1], dtype=int)
        T = self.model.conifold_monodromy_matrix(conifold_curve=c)
        expected = np.eye(self.dim, dtype=int)
        expected[1:self.h + 1, self.h + 2:self.dim] += np.outer(c, c)

        np.testing.assert_array_equal(T, expected)

    def test_conifold_monodromy_matrix_rejects_ambiguous_input(self):
        r"""
        **Description:**
        Supplying both ``conifold_curve`` and ``conifold_index`` is ambiguous
        and should be rejected before any basis-dependent logic is used.
        """

        with self.assertRaises(ValueError):
            self.model.conifold_monodromy_matrix(
                conifold_curve=np.array([1, 0]), conifold_index=0,
            )

    def test_conifold_monodromy_matrix_rejects_wrong_curve_length(self):
        r"""The conifold curve length must match ``h12``."""

        with self.assertRaises(ValueError):
            self.model.conifold_monodromy_matrix(conifold_curve=np.array([1, 0, 0]))
            
    import atexit
    atexit.register(lambda: print(">>> test_css.py module exit", flush=True))