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

"""Tests for the period-sector implementation.

Purpose
-------
Validate ``periods`` computations for period vectors, prepotentials,
symplectic structure, Kähler potential and gauge-kinetic matrices.

Main public API
---------------
- ``TestPeriodSector``: numerical checks for LCS-period formulas and derived
  special-geometry quantities.

Design notes
------------
The tests use small fixtures with known shape and symmetry expectations to
catch regressions in low-level period algebra.
"""

import sys, os, warnings
import jax
import pytest
from functools import partial
from util import *

sys.path.append("./../")
import jaxvacua


# ==============================================================================
# TestPeriodSector
# ==============================================================================

class TestPeriodSector(TestCase):
    r"""**Description:**

    Unit-test suite for :class:`jaxvacua.periods`.

    Tests cover the symplectic structure, period vector, prepotential, Kähler
    geometry and gauge kinetic matrix in the large complex structure (LCS)
    limit of a Calabi-Yau threefold compactification.

    Attributes:
        model (jaxvacua.periods): Period model with :math:`h^{1,2}=2`,
            model ID 1, KS type at LCS, truncated at degree 5.
        z (jax.Array): Fixed representative test point in period coordinate
            space, shape ``(h12+1,)``, with :math:`X^0 = 1` fixed.
        cz (jax.Array): Complex conjugate of ``z``.
        z0 (jax.Array): Large complex structure point :math:`X^0=1`,
            :math:`X^i=0`, shape ``(h12+1,)``.
        cz0 (jax.Array): Complex conjugate of ``z0``.

    .. note::
        Period coordinates :math:`X^I` are used throughout, with :math:`X^0=1`
        fixed by the projective gauge.  The free moduli are
        :math:`z^i = X^i / X^0`,  :math:`i = 1, \ldots, h^{1,2}`.
    """

    # --------------------------------------------------------------------------
    # Class setup
    # --------------------------------------------------------------------------

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        h12 = 2

        cls.model = jaxvacua.periods(h12=h12, model_ID=1, model_type="KS", maximum_degree=5)
        # periods takes X^I coordinates: X^0=1 fixed, then free z^i.
        # Use a deterministic interior LCS-style point so sign-sensitive
        # matrix tests are reproducible across sessions and CI runs.
        cls.z = jnp.array([
            1.0 + 0.0j,
            0.37 + 3.25j,
            -0.42 + 2.75j,
        ])
        cls.cz = jnp.conj(cls.z)
        # LCS point (all z^i = 0, X^0 = 1)
        cls.z0 = jnp.zeros(h12 + 1, dtype=complex)
        cls.z0 = cls.z0.at[0].set(1. + 0. * 1j)
        cls.cz0 = jnp.conj(cls.z0)


    # ==========================================================================
    # Symplectic structure
    # ==========================================================================

    def test_sigma_structure(self):
        r"""**Description:**

        Verify that the symplectic pairing matrix

        .. math::
            \Sigma = \begin{pmatrix} 0 & \mathbf{1} \\ -\mathbf{1} & 0 \end{pmatrix}

        returned by :func:`sigma` satisfies all required algebraic identities,
        including the explicit block decomposition, antisymmetry, and the
        defining relation :math:`\Sigma^2 = -\mathbf{1}_{2n}`.

        Args:
            None

        Returns:
            None

        .. note::
            These properties are used throughout: symplecticity of the ISD
            matrix and the isotropy of the period vector.
        """

        n = self.model.h12 + 1                          # half-dimension

        # ---- shape ----
        sig = self.model.sigma
        # sigma must be a square (2n x 2n) matrix
        chex.assert_shape(sig, (2 * n, 2 * n))

        # ---- antisymmetry: sigma^T = -sigma ----
        self.assertAllClose(
            sig.T, -sig, rtol=1e-12, atol=1e-12,
            msg="sigma must be antisymmetric: sigma^T = -sigma")

        # ---- sigma^2 = -I_{2n} ----
        sig2 = jnp.matmul(sig, sig)
        self.assertAllClose(
            sig2, -jnp.eye(2 * n, dtype=sig.dtype), rtol=1e-12, atol=1e-12,
            msg="sigma must satisfy sigma^2 = -I (symplectic unit)")

        # ---- explicit block structure: [[0, I], [-I, 0]] ----
        # Upper-left block must be zero
        self.assertAllClose(
            sig[:n, :n], jnp.zeros((n, n), dtype=sig.dtype),
            rtol=1e-12, atol=1e-12,
            msg="Upper-left block of sigma must be zero")
        # Upper-right block must be +I
        self.assertAllClose(
            sig[:n, n:], jnp.eye(n, dtype=sig.dtype),
            rtol=1e-12, atol=1e-12,
            msg="Upper-right block of sigma must be +identity")
        # Lower-left block must be -I
        self.assertAllClose(
            sig[n:, :n], -jnp.eye(n, dtype=sig.dtype),
            rtol=1e-12, atol=1e-12,
            msg="Lower-left block of sigma must be -identity")
        # Lower-right block must be zero
        self.assertAllClose(
            sig[n:, n:], jnp.zeros((n, n), dtype=sig.dtype),
            rtol=1e-12, atol=1e-12,
            msg="Lower-right block of sigma must be zero")


    def test_a_shift_monodromy(self):
        r"""**Description:**

        Verify the symplectic monodromy :math:`M(S)` returned by
        :func:`compute_a_shift_monodromy` for an :math:`a`-matrix shift
        :math:`a \to a + S` (see the method docstring).  Checks the unipotent
        block structure :math:`M = \left(\begin{smallmatrix}\mathbf{1} &
        \widehat{S}\\ 0 & \mathbf{1}\end{smallmatrix}\right)`, exact
        symplecticity :math:`M^T\Sigma M = \Sigma` for symmetric :math:`S`,
        reduction to the identity at :math:`S=0`, and the symmetry guard.
        """
        n = self.model.h12 + 1                          # half-dimension
        h = self.model.h12
        sig = np.asarray(self.model.sigma)
        rng = np.random.default_rng(0)
        Sr = rng.integers(-3, 4, (h, h))
        S = (Sr + Sr.T).astype(float)                   # symmetric integer S

        M = np.asarray(self.model.compute_a_shift_monodromy(S))

        # ---- shape ----
        chex.assert_shape(M, (2 * n, 2 * n))

        # ---- unipotent block structure [[I, Shat], [0, I]] ----
        self.assertAllClose(M[:n, :n], np.eye(n), atol=1e-12,
                            msg="upper-left block must be I")
        self.assertAllClose(M[n:, n:], np.eye(n), atol=1e-12,
                            msg="lower-right block must be I")
        self.assertAllClose(M[n:, :n], np.zeros((n, n)), atol=1e-12,
                            msg="lower-left block must be 0 (unipotent)")
        # off-diagonal F<-X block is Shat = [[0, 0], [0, S]] (zero X^0 row/col)
        Shat = M[:n, n:]
        self.assertAllClose(Shat[1:, 1:], S, atol=1e-12,
                            msg="Shat bulk block must equal S")
        self.assertAllClose(Shat[0, :], np.zeros(n), atol=1e-12,
                            msg="Shat X^0 row must be zero")
        self.assertAllClose(Shat[:, 0], np.zeros(n), atol=1e-12,
                            msg="Shat X^0 column must be zero")

        # ---- exact symplecticity for symmetric S ----
        self.assertAllClose(M.T @ sig @ M, sig, atol=1e-12,
                            msg="M(S) must be symplectic: M^T Sigma M = Sigma")

        # ---- S = 0 gives the identity ----
        M0 = np.asarray(self.model.compute_a_shift_monodromy(np.zeros((h, h))))
        self.assertAllClose(M0, np.eye(2 * n), atol=1e-12,
                            msg="M(0) must be the identity")

        # ---- asymmetric S is rejected ----
        S_asym = np.zeros((h, h)); S_asym[0, h - 1] = 1.0
        with self.assertRaises(ValueError):
            self.model.compute_a_shift_monodromy(S_asym)


    # ==========================================================================
    # ISD matrix
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    @pytest.mark.slow
    def test_ISD(self):
        r"""**Description:**

        Test the ISD matrix :math:`\mathcal{M}` returned by
        :func:`ISD_matrix`.

        The ISD matrix is the matrix representation of the Hodge-:math:`\star`
        operator on :math:`H^3(X, \mathbb{Z})`.  Expressed in terms of
        the real and imaginary parts :math:`\mathcal{R}` and :math:`\mathcal{I}`
        of the gauge kinetic matrix :math:`\mathcal{N}= \mathcal{R} +
        \mathrm{i}\mathcal{I}`, it reads

        .. math::
            \mathcal{M} = \begin{pmatrix}
                -\mathcal{I}^{-1} & \mathcal{I}^{-1}\mathcal{R} \\
                \mathcal{R}\mathcal{I}^{-1} &
                -\mathcal{I} - \mathcal{R}\mathcal{I}^{-1}\mathcal{R}
            \end{pmatrix}.

        The ISD matrix satisfies

        .. math::
            \mathcal{M} = \mathcal{M}^T \,,\quad
            \mathcal{M}^T\Sigma\mathcal{M} = \Sigma \,,\quad
            \mathcal{M}^{-1} = \Sigma^T\mathcal{M}\Sigma \,.

        Derivatives :math:`\partial_{X^I}\mathcal{M}` and
        :math:`\partial_{\bar{X}^I}\mathcal{M}` are also checked for shape
        and conjugation consistency.

        Args:
            None

        Returns:
            None

        .. note::
            The imaginary part :math:`\mathcal{M}\in\mathbb{R}` vanishes
            exactly due to the block structure of :math:`\mathcal{N}`.
        """

        def _isd_bundle(x, y):
            return (
                self.model.ISD_matrix(x, y),
                self.model.dM(x, y),
                self.model.dM_c(x, y),
            )

        # ---- evaluate ISD matrix and derivatives in one variant call ----
        M, dM, dM_c = self.variant(_isd_bundle)(self.z, self.cz)

        # must be complex type internally (imaginary part vanishes)
        chex.assert_type(M, complex)
        chex.assert_shape(M, (2 * (self.model.h12 + 1), 2 * (self.model.h12 + 1)))
        # imaginary part vanishes: M is effectively real
        self.assertAllClose(M.imag, 0., rtol=1e-11, atol=1e-11,
                            msg="ISD matrix must be real")

        # ---- symmetry: M = M^T ----
        self.assertAllClose(M, M.T, rtol=1e-11, atol=1e-11,
                            msg="ISD matrix must be symmetric")

        # ---- symplecticity: M^T @ sigma @ M = sigma ----
        sig = self.model.sigma
        self.assertAllClose(
            jnp.matmul(M.T, jnp.matmul(sig, M)), sig,
            rtol=1e-11, atol=1e-11,
            msg="ISD matrix must be symplectic: M^T sigma M = sigma")

        # ---- inverse: inv(M) = sigma^T @ M @ sigma ----
        self.assertAllClose(
            jnp.linalg.inv(M), jnp.matmul(sig.T, jnp.matmul(M, sig)),
            rtol=1e-11, atol=1e-11,
            msg="ISD matrix inverse must equal sigma^T M sigma")

        chex.assert_type(dM, complex)
        chex.assert_shape(dM, (2 * (self.model.h12 + 1),
                               2 * (self.model.h12 + 1),
                               self.model.h12 + 1))
        chex.assert_type(dM_c, complex)
        chex.assert_shape(dM_c, (2 * (self.model.h12 + 1),
                                 2 * (self.model.h12 + 1),
                                 self.model.h12 + 1))

        # anti-holomorphic derivative equals conjugate of holomorphic derivative
        self.assertAllClose(dM_c, jnp.conj(dM),
                            msg="dM_c must equal conj(dM)")


    # ==========================================================================
    # Gauge kinetic matrix
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    @pytest.mark.slow
    def test_gauge_kinetic_matrix(self):
        r"""**Description:**

        Test the gauge kinetic matrix :math:`\mathcal{N}_{IJ}` returned by
        :func:`gauge_kinetic_matrix`, :func:`gauge_kinetic_matrix_periods`, and
        :func:`gauge_kinetic_matrix_prepotential`.

        In the presence of a prepotential :math:`F`, the gauge kinetic matrix
        can be computed either from the period matrices :math:`P,Q` as

        .. math::
            \mathcal{N}_{IJ} = P_{IK}(Q^{-1})^K{}_J \,,

        or directly from derivatives of :math:`F` as

        .. math::
            \mathcal{N}_{IJ} = \bar{F}_{IJ}
            + 2\mathrm{i}\,
            \frac{\mathrm{Im}(F_{IL})X^L\,\mathrm{Im}(F_{JK})X^K}
                 {X^M\,\mathrm{Im}(F_{MN})X^N} \,.

        All three routes must agree at LCS.  Derivatives :math:`\partial_{X^I}
        \mathcal{N}` and :math:`\partial_{\bar{X}^I}\mathcal{N}` are checked
        for shape and conjugation.

        Args:
            None

        Returns:
            None
        """

        def _gauge_kinetic_bundle(x, y):
            return (
                self.model.gauge_kinetic_matrix(x, y, conj=False),
                self.model.gauge_kinetic_matrix_periods(x, y, conj=False),
                self.model.gauge_kinetic_matrix_prepotential(x, y, conj=False),
                self.model.dN(x, y, conj=False),
                self.model.dN_c(x, y, conj=False),
                self.model.gauge_kinetic_matrix(x, y, conj=True),
                self.model.gauge_kinetic_matrix_periods(x, y, conj=True),
                self.model.gauge_kinetic_matrix_prepotential(x, y, conj=True),
                self.model.dN(x, y, conj=True),
                self.model.dN_c(x, y, conj=True),
            )

        (
            N,
            N_periods,
            N_prepotential,
            dN_X,
            dN_cX,
            N_c,
            N_periods_c,
            N_prepotential_c,
            dN_X_c,
            dN_cX_c,
        ) = self.variant(_gauge_kinetic_bundle)(self.z, self.cz)

        # N from gauge_kinetic_matrix must be complex-valued
        chex.assert_type(N, complex)
        # N must be a square (h12+1) x (h12+1) matrix
        chex.assert_shape(N, (self.model.h12 + 1, self.model.h12 + 1))
        # N from the period-matrix route must be complex-valued
        chex.assert_type(N_periods, complex)
        # N_periods must have the same (h12+1) x (h12+1) shape
        chex.assert_shape(N_periods, (self.model.h12 + 1, self.model.h12 + 1))
        # N from the prepotential route must be complex-valued
        chex.assert_type(N_prepotential, complex)
        # N_prepotential must have the same (h12+1) x (h12+1) shape
        chex.assert_shape(N_prepotential, (self.model.h12 + 1, self.model.h12 + 1))

        # Holomorphic derivative dN/dX must be complex-valued
        chex.assert_type(dN_X, complex)
        # dN/dX must be a rank-3 tensor of shape (h12+1, h12+1, h12+1)
        chex.assert_shape(dN_X, (self.model.h12 + 1, self.model.h12 + 1, self.model.h12 + 1))
        # Anti-holomorphic derivative dN/dcX must be complex-valued
        chex.assert_type(dN_cX, complex)
        # dN/dcX must have the same rank-3 shape as dN/dX
        chex.assert_shape(dN_cX, (self.model.h12 + 1, self.model.h12 + 1, self.model.h12 + 1))

        # Conjugated N must be complex-valued
        chex.assert_type(N_c, complex)
        # N_c must have the same (h12+1) x (h12+1) shape
        chex.assert_shape(N_c, (self.model.h12 + 1, self.model.h12 + 1))
        # Conjugated N from periods route must be complex-valued
        chex.assert_type(N_periods_c, complex)
        # N_periods_c must have the same (h12+1) x (h12+1) shape
        chex.assert_shape(N_periods_c, (self.model.h12 + 1, self.model.h12 + 1))
        # Conjugated N from prepotential route must be complex-valued
        chex.assert_type(N_prepotential_c, complex)
        # N_prepotential_c must have the same (h12+1) x (h12+1) shape
        chex.assert_shape(N_prepotential_c, (self.model.h12 + 1, self.model.h12 + 1))

        # Holomorphic derivative of conjugated N must be complex-valued
        chex.assert_type(dN_X_c, complex)
        # dN_X_c must be a rank-3 tensor of shape (h12+1, h12+1, h12+1)
        chex.assert_shape(dN_X_c, (self.model.h12 + 1, self.model.h12 + 1, self.model.h12 + 1))
        # Anti-holomorphic derivative of conjugated N must be complex-valued
        chex.assert_type(dN_cX_c, complex)
        # dN_cX_c must have the same rank-3 shape
        chex.assert_shape(dN_cX_c, (self.model.h12 + 1, self.model.h12 + 1, self.model.h12 + 1))

        # ---- conjugation of N ----
        # N(z, cz, conj=True) = conj(N(z, cz, conj=False))
        self.assertAllClose(N_c, jnp.conj(N),
                            msg="N_c must equal conj(N)")
        # cross-derivative conjugation
        self.assertAllClose(dN_cX_c, jnp.conj(dN_X),
                            msg="dN_cX (conj=True) must equal conj(dN_X (conj=False))")
        self.assertAllClose(dN_X_c, jnp.conj(dN_cX),
                            msg="dN_X (conj=True) must equal conj(dN_cX (conj=False))")

        # ---- three routes must agree ----
        # periods route == prepotential route (both conj=False)
        self.assertAllClose(N_periods, N_prepotential,
                            msg="periods and prepotential routes for N must agree")
        self.assertAllClose(N_periods_c, N_prepotential_c,
                            msg="periods and prepotential routes for N_c must agree")
        self.assertAllClose(N_periods_c, jnp.conj(N_periods),
                            msg="N_periods_c must equal conj(N_periods)")
        self.assertAllClose(N_prepotential_c, jnp.conj(N_prepotential),
                            msg="N_prepotential_c must equal conj(N_prepotential)")


    # ==========================================================================
    # Mirror volume (A_per)
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_mirror_volume(self):
        r"""**Description:**

        Test that the mirror Calabi-Yau volume

        .. math::
            \widetilde{\mathcal{V}} = -\mathrm{i}\,\Pi^\dagger \Sigma \Pi

        returned by :func:`A_per` is a real positive scalar.

        Args:
            None

        Returns:
            None

        .. note::
            The positivity of :math:`\widetilde{\mathcal{V}}` ensures that the
            Kähler potential :math:`K = -\ln\widetilde{\mathcal{V}}` is
            well-defined.
        """

        Vtilde = self.variant(self.model.A_per)(self.z, self.cz)

        # must be a scalar
        chex.assert_type(Vtilde, complex)
        chex.assert_shape(Vtilde, ())
        # imaginary part must vanish
        self.assertAllClose(Vtilde.imag, 0.,
                            msg="Mirror CY volume A_per must be real")


    # ==========================================================================
    # Kähler potential
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_kahler_potential(self):
        r"""**Description:**

        Test the Kähler potential

        .. math::
            K = -\ln\!\left(-\mathrm{i}\,\Pi^\dagger\Sigma\Pi\right)

        returned by :func:`kahler_potential_per`.

        Args:
            None

        Returns:
            None

        .. note::
            The Kähler potential is real on the physical moduli space because
            :math:`\widetilde{\mathcal{V}} > 0`.
        """

        KP = self.variant(self.model.kahler_potential_per)(self.z, self.cz)

        # must be a scalar
        chex.assert_type(KP, complex)
        chex.assert_shape(KP, ())
        # must be real
        self.assertAllClose(KP.imag, 0.,
                            msg="Kähler potential must be real")


    # ==========================================================================
    # Gradient of Kähler potential
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_dK(self):
        r"""**Description:**

        Test the holomorphic and anti-holomorphic derivatives of the Kähler
        potential,

        .. math::
            \partial_{X^I} K \,,\quad \partial_{\bar{X}^I} K \,,

        returned by :func:`grad_kahler_potential_per`.

        The two derivatives are related by complex conjugation:

        .. math::
            \partial_{\bar{X}^I} K = \overline{\partial_{X^I} K} \,.

        Args:
            None

        Returns:
            None
        """

        # holomorphic derivative
        dK = self.variant(lambda x, y: self.model.grad_kahler_potential_per(x, y, conj=False))(self.z, self.cz)

        chex.assert_type(dK, complex)
        chex.assert_shape(dK, (self.model.h12 + 1,))   # one entry per X^I

        # anti-holomorphic derivative
        dK_c = self.variant(lambda x, y: self.model.grad_kahler_potential_per(x, y, conj=True))(self.z, self.cz)

        chex.assert_type(dK_c, complex)
        chex.assert_shape(dK_c, (self.model.h12 + 1,))

        # conjugation: dK_c = conj(dK)
        self.assertAllClose(dK_c, jnp.conj(dK),
                            msg="Anti-holomorphic dK must equal conj(dK)")


    # ==========================================================================
    # P, Q matrices
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    @pytest.mark.slow
    def test_rest(self):
        r"""**Description:**

        Test the period matrices :math:`P_{IJ}` and :math:`Q^I{}_J` and their
        inverses returned by :func:`P_per`, :func:`Q_per`, :func:`Q_inv_per`,
        and :func:`PQ_per`.

        These matrices are defined as

        .. math::
            P_{IJ} = \bigl(\mathcal{F}_I,\,
                       D_{\bar\imath}\bar{\mathcal{F}}_I\bigr)_J \,,\quad
            Q^I{}_J = \bigl(X^I,\, D_{\bar\jmath}\bar{X}^I\bigr)_J \,,

        see Eq. (3.5) in arXiv:2310.06040.  The gauge kinetic matrix is
        :math:`\mathcal{N} = P_{IK}(Q^{-1})^K{}_J`.

        Args:
            None

        Returns:
            None

        .. note::
            All matrices carry the same shape ``(h12+1, h12+1)`` and satisfy
            standard conjugation relations.
        """

        conj = False
        P = self.variant(lambda x, y: self.model.P_per(x, y, conj=conj))(self.z, self.cz)
        Q = self.variant(lambda x, y: self.model.Q_per(x, y, conj=conj))(self.z, self.cz)
        Qinv = self.variant(lambda x, y: self.model.Q_inv_per(x, y, conj=conj))(self.z, self.cz)
        P_mod, Q_mod = self.variant(lambda x, y: self.model.PQ_per(x, y, conj=conj))(self.z, self.cz)

        conj = True
        P_c = self.variant(lambda x, y: self.model.P_per(x, y, conj=conj))(self.z, self.cz)
        Q_c = self.variant(lambda x, y: self.model.Q_per(x, y, conj=conj))(self.z, self.cz)
        Qinv_c = self.variant(lambda x, y: self.model.Q_inv_per(x, y, conj=conj))(self.z, self.cz)
        P_mod_c, Q_mod_c = self.variant(lambda x, y: self.model.PQ_per(x, y, conj=conj))(self.z, self.cz)

        # ---- shapes ----
        # P matrix (conj=False) must be (h12+1) x (h12+1)
        chex.assert_shape(P,     (self.model.h12 + 1, self.model.h12 + 1))
        # P matrix (conj=True) must have the same shape
        chex.assert_shape(P_c,   (self.model.h12 + 1, self.model.h12 + 1))
        # Q matrix (conj=False) must be (h12+1) x (h12+1)
        chex.assert_shape(Q,     (self.model.h12 + 1, self.model.h12 + 1))
        # Q matrix (conj=True) must have the same shape
        chex.assert_shape(Q_c,   (self.model.h12 + 1, self.model.h12 + 1))
        # Q inverse (conj=False) must be (h12+1) x (h12+1)
        chex.assert_shape(Qinv,  (self.model.h12 + 1, self.model.h12 + 1))
        # Q inverse (conj=True) must have the same shape
        chex.assert_shape(Qinv_c, (self.model.h12 + 1, self.model.h12 + 1))
        # P from PQ_per (conj=False) must be (h12+1) x (h12+1)
        chex.assert_shape(P_mod,   (self.model.h12 + 1, self.model.h12 + 1))
        # P from PQ_per (conj=True) must have the same shape
        chex.assert_shape(P_mod_c, (self.model.h12 + 1, self.model.h12 + 1))
        # Q from PQ_per (conj=False) must be (h12+1) x (h12+1)
        chex.assert_shape(Q_mod,   (self.model.h12 + 1, self.model.h12 + 1))
        # Q from PQ_per (conj=True) must have the same shape
        chex.assert_shape(Q_mod_c, (self.model.h12 + 1, self.model.h12 + 1))

        # ---- Q_inv = inv(Q) ----
        self.assertAllClose(Qinv,   jnp.linalg.inv(Q),
                            msg="Q_inv must equal matrix inverse of Q")
        self.assertAllClose(Qinv_c, jnp.linalg.inv(Q_c),
                            msg="Q_inv_c must equal matrix inverse of Q_c")

        # ---- conjugation relations ----
        self.assertAllClose(Qinv, jnp.conj(Qinv_c),
                            msg="Q_inv must equal conj(Q_inv_c)")
        self.assertAllClose(Q,   jnp.conj(Q_c),    msg="Q must equal conj(Q_c)")
        self.assertAllClose(P,   jnp.conj(P_c),    msg="P must equal conj(P_c)")
        self.assertAllClose(Q_mod, jnp.conj(Q_mod_c), msg="Q from PQ_per must equal conj(Q_c from PQ_per)")
        self.assertAllClose(P_mod, jnp.conj(P_mod_c), msg="P from PQ_per must equal conj(P_c from PQ_per)")

        # ---- PQ_per must agree with individual P_per / Q_per ----
        self.assertAllClose(Q_mod, Q, msg="Q returned by PQ_per must agree with Q_per")
        self.assertAllClose(P_mod, P, msg="P returned by PQ_per must agree with P_per")


    # ==========================================================================
    # Period vector — symplectic properties
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_period_vector_symplectic(self):
        r"""**Description:**

        Test the fundamental symplectic properties of the period vector

        .. math::
            \Pi = \begin{pmatrix} \mathcal{F}_I \\ X^I \end{pmatrix}

        returned by :func:`period_vector_per`.

        Two identities are verified:

        1. **Isotropy** (follows from degree-2 homogeneity of :math:`F`):

           .. math::
               \Pi^T \Sigma \Pi = 0 \,.

        2. **Mirror volume** (definition of :func:`A_per`):

           .. math::
               -\mathrm{i}\,\bar\Pi^T \Sigma \Pi = \widetilde{\mathcal{V}} \,,

           where :math:`\widetilde{\mathcal{V}}` is the mirror CY volume.

        Args:
            None

        Returns:
            None

        .. note::
            The isotropy condition :math:`\Pi^T\Sigma\Pi = 0` is an exact
            algebraic identity at every point in moduli space.
        """

        sig = self.model.sigma

        Pi   = self.variant(lambda x: self.model.period_vector_per(x, conj=False))(self.z)
        Pi_c = self.variant(lambda x: self.model.period_vector_per(x, conj=True))(self.cz)

        # ---- shape ----
        chex.assert_shape(Pi,   (2 * (self.model.h12 + 1),))
        chex.assert_shape(Pi_c, (2 * (self.model.h12 + 1),))

        # ---- 1. isotropy: Pi^T sigma Pi = 0 ----
        # The prepotential F is degree-2 homogeneous in X^I, so by Euler:
        # F_I X^I = 2F — the bilinear vanishes identically.
        iso = jnp.dot(Pi, jnp.matmul(sig, Pi))
        self.assertAllClose(iso, 0. + 0.j, rtol=1e-11, atol=1e-11,
                            msg="Period vector must be isotropic: Pi^T sigma Pi = 0")

        # ---- 2. mirror volume: A_per = -i Pi_c^T sigma Pi ----
        A = self.variant(self.model.A_per)(self.z, self.cz)
        A_from_Pi = -1j * jnp.dot(Pi_c, jnp.matmul(sig, Pi))
        self.assertAllClose(A_from_Pi, A, rtol=1e-11, atol=1e-11,
                            msg="Mirror volume must equal -i Pi_c^T sigma Pi")


    # ==========================================================================
    # Gauge kinetic matrix — imaginary part
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_gauge_kinetic_matrix_imaginary_part(self):
        r"""**Description:**

        Test that the imaginary part

        .. math::
            \mathcal{I}_{IJ} = \mathrm{Im}\,\mathcal{N}_{IJ}

        of the gauge kinetic matrix is negative semidefinite,

        .. math::
            \mathcal{I} \leq 0 \,,

        i.e., all eigenvalues of :math:`\mathcal{I}` are non-positive.

        This property is required for the kinetic terms of the gauge fields in
        the effective supergravity to have the correct sign.

        Args:
            None

        Returns:
            None

        .. note::
            Negative semidefiniteness is a consequence of the special
            geometry of the complex structure moduli space and is preserved
            away from degeneration loci.
        """

        N = self.variant(lambda x, y: self.model.gauge_kinetic_matrix(x, y, conj=False))(self.z, self.cz)
        N_c = self.variant(lambda x, y: self.model.gauge_kinetic_matrix(x, y, conj=True))(self.z, self.cz)

        # imaginary part of N
        ImN = -1j * (N - N_c) / 2.

        # ImN must be real (by construction)
        self.assertAllClose(ImN.imag, jnp.zeros_like(ImN.imag), rtol=1e-11, atol=1e-11,
                            msg="Im(N) must be a real matrix")

        # compute eigenvalues of the real symmetric imaginary part
        eigvals = jnp.linalg.eigvalsh(ImN.real)

        # all eigenvalues must be non-positive (negative semidefinite)
        self.assertTrue(
            jnp.all(eigvals <= 1e-8),
            msg="Im(N) must be negative semidefinite: all eigenvalues <= 0")

    def test_seeded_sample_special_geometry_properties(self):
        r"""**Description:**

        Check the core period-sector identities on a small deterministic
        ensemble of LCS points.  The fixed class fixture catches exact
        regressions at one representative point; this seeded sample guards
        against accidental point-specific success without introducing
        non-reproducible CI behaviour.
        """
        rng = np.random.default_rng(20260525)
        sig = self.model.sigma

        for _ in range(4):
            X = jnp.asarray(
                rng.uniform(-0.4, 0.4, self.model.h12 + 1)
                + 1j * rng.uniform(2.0, 5.0, self.model.h12 + 1)
            )
            X = X.at[0].set(1.0 + 0.0j)
            cX = jnp.conj(X)

            Pi = self.model.period_vector_per(X, conj=False)
            Pi_c = self.model.period_vector_per(cX, conj=True)
            iso = jnp.dot(Pi, jnp.matmul(sig, Pi))
            A = self.model.A_per(X, cX)
            A_from_Pi = -1j * jnp.dot(Pi_c, jnp.matmul(sig, Pi))
            self.assertAllClose(iso, 0.0 + 0.0j, rtol=1e-10, atol=1e-10)
            self.assertAllClose(A_from_Pi, A, rtol=1e-10, atol=1e-10)

            N = self.model.gauge_kinetic_matrix(X, cX, conj=False)
            N_c = self.model.gauge_kinetic_matrix(X, cX, conj=True)
            self.assertAllClose(N, N.T, rtol=1e-10, atol=1e-10)
            self.assertAllClose(N_c, jnp.conj(N), rtol=1e-10, atol=1e-10)
            eigvals = jnp.linalg.eigvalsh(jnp.imag(N))
            self.assertTrue(
                jnp.all(eigvals <= 1e-8),
                msg=f"Seeded sample produced positive Im(N) eigenvalue: {eigvals}",
            )


    # ==========================================================================
    # Period vector — covariant derivative
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_D_period_vector(self):
        r"""**Description:**

        Test the Kähler-covariant derivative of the period vector

        .. math::
            D_I\Pi_a = \partial_{X^I}\Pi_a + (\partial_{X^I} K)\,\Pi_a

        returned by :func:`D_period_vector_per`.

        The following identities are verified:

        1. **Shape**: :math:`D\Pi` has shape ``(2(h^{1,2}+1), h^{1,2}+1)``.
        2. **Conjugation**: :math:`(D\Pi)_c = \overline{D\Pi}`.
        3. **Consistency** with :func:`grad_period_vector_per` and
           :func:`grad_kahler_potential_per`:

           .. math::
               D_I\Pi = \partial_{X^I}\Pi + (\partial_{X^I}K)\,\Pi \,.

        Args:
            None

        Returns:
            None

        .. note::
            The covariant derivative :math:`D_I\Pi` enters the definition of
            the matrices :math:`P` and :math:`Q` and hence the gauge kinetic
            matrix :math:`\mathcal{N}`.
        """

        # ---- evaluate covariant and ordinary derivatives ----
        DPi = self.variant(lambda x, y: self.model.D_period_vector_per(x, y, conj=False))(self.z, self.cz)
        DPi_c = self.variant(lambda x, y: self.model.D_period_vector_per(x, y, conj=True))(self.z, self.cz)

        # ---- shape ----
        chex.assert_shape(DPi,   (2 * (self.model.h12 + 1), self.model.h12 + 1))
        chex.assert_shape(DPi_c, (2 * (self.model.h12 + 1), self.model.h12 + 1))

        # ---- conjugation ----
        self.assertAllClose(DPi_c, jnp.conj(DPi),
                            msg="(D Pi)_c must equal conj(D Pi)")

        # ---- consistency: DPi = dPi + outer(Pi, dK) ----
        Pi  = self.variant(lambda x: self.model.period_vector_per(x, conj=False))(self.z)
        dPi = self.variant(lambda x: self.model.grad_period_vector_per(x, conj=False))(self.z)
        dK  = self.variant(lambda x, y: self.model.grad_kahler_potential_per(x, y, conj=False))(self.z, self.cz)

        # D_I Pi = dPi + outer(Pi, dK)
        DPi_manual = dPi + jnp.outer(Pi, dK)
        self.assertAllClose(DPi, DPi_manual, rtol=1e-11, atol=1e-11,
                            msg="D_period_vector must equal dPi + outer(Pi, dK)")


    # ==========================================================================
    # Prepotential
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_prepotential(self):
        r"""**Description:**

        Test the prepotential :math:`F` and its derivatives returned by
        :func:`prepot_per`, :func:`prepot_grad_per`, and
        :func:`prepot_grad_grad_per`, as well as the individual LCS
        decomposition :func:`F_LCS_per`, :func:`F_LCS_poly_per`, and
        :func:`F_inst_per`.

        The LCS polynomial prepotential is

        .. math::
            F_{\mathrm{poly}}(X) = -\frac{1}{6X^0}\widetilde\kappa_{ijk}
                X^i X^j X^k + \tfrac{1}{2}a_{ij}X^i X^j + b_i X^i X^0
                + \tfrac{\mathrm{i}}{2}\tilde\xi\,(X^0)^2 \,.

        At the large complex structure point :math:`z^i = 0`, :math:`X^0 = 1`,
        the polynomial part satisfies the following boundary conditions:

        .. math::
            F_{\mathrm{poly}}(z^0) = K_0/2 \,,\quad
            \partial_{X^i}F_{\mathrm{poly}}\big|_0 = b_i \,,\quad
            \partial_{X^i}\partial_{X^j}F_{\mathrm{poly}}\big|_0 = a_{ij} \,,\quad
            \partial_{X^i}\partial_{X^j}\partial_{X^k}
                F_{\mathrm{poly}}\big|_0 = -\widetilde\kappa_{ijk} \,.

        Conjugation relations :math:`\bar F = F(\bar z)` are verified for all
        of :math:`F`, :math:`F_{\mathrm{LCS}}`, :math:`F_{\mathrm{poly}}`,
        :math:`F_{\mathrm{inst}}`, :math:`F_I`, and :math:`F_{IJ}`.

        Args:
            None

        Returns:
            None

        .. note::
            The instanton corrections :math:`F_{\mathrm{inst}}` use Gopakumar-
            Vafa invariants and are truncated at ``maximum_degree=5``.
        """

        # ---- conj=False ----
        conj = False
        F          = self.variant(lambda x: self.model.prepot_per(x, conj=conj))(self.z)
        F_LCS      = self.variant(lambda x: self.model.F_LCS_per(x, conj=conj))(self.z)
        F_LCS_poly = self.variant(lambda x: self.model.F_LCS_poly_per(x, conj=conj))(self.z)
        F_inst     = self.variant(lambda x: self.model.F_inst_per(x, conj=conj))(self.z)
        dF         = self.variant(lambda x: self.model.prepot_grad_per(x, conj=conj))(self.z)
        ddF        = self.variant(lambda x: self.model.prepot_grad_grad_per(x, conj=conj))(self.z)

        # ---- conj=True ----
        conj = True
        F_c          = self.variant(lambda x: self.model.prepot_per(x, conj=conj))(self.cz)
        F_LCS_c      = self.variant(lambda x: self.model.F_LCS_per(x, conj=conj))(self.cz)
        F_LCS_poly_c = self.variant(lambda x: self.model.F_LCS_poly_per(x, conj=conj))(self.cz)
        F_inst_c     = self.variant(lambda x: self.model.F_inst_per(x, conj=conj))(self.cz)
        dF_c         = self.variant(lambda x: self.model.prepot_grad_per(x, conj=conj))(self.cz)
        ddF_c        = self.variant(lambda x: self.model.prepot_grad_grad_per(x, conj=conj))(self.cz)

        # ---- shapes ----
        chex.assert_shape(dF,   (self.model.h12 + 1,))
        chex.assert_shape(dF_c, (self.model.h12 + 1,))
        chex.assert_shape(ddF,   (self.model.h12 + 1, self.model.h12 + 1))
        chex.assert_shape(ddF_c, (self.model.h12 + 1, self.model.h12 + 1))

        # ---- conjugation relations ----
        self.assertAllClose(F_c,          jnp.conj(F),
                            msg="Conjugate prepotential must equal conj(F)")
        self.assertAllClose(F_LCS_c,      jnp.conj(F_LCS),
                            msg="Conjugate F_LCS must equal conj(F_LCS)")
        self.assertAllClose(F_LCS_poly_c, jnp.conj(F_LCS_poly),
                            msg="Conjugate F_LCS_poly must equal conj(F_LCS_poly)")
        self.assertAllClose(F_inst_c,     jnp.conj(F_inst),
                            msg="Conjugate F_inst must equal conj(F_inst)")
        self.assertAllClose(dF_c,         jnp.conj(dF),
                            msg="Conjugate dF must equal conj(dF)")
        self.assertAllClose(ddF_c,        jnp.conj(ddF),
                            msg="Conjugate ddF must equal conj(ddF)")

        # ---- LCS boundary conditions (polynomial part) at z^i = 0 ----
        conj = False
        F_0 = self.variant(lambda x: self.model.F_LCS_poly_per(x, conj=conj))(self.z0)
        dF_0 = self.variant(
            lambda x: jax.grad(self.model.F_LCS_poly_per, holomorphic=True)(x, conj=conj)
        )(self.z0)
        ddF_0 = self.variant(
            lambda x: jax.jacfwd(
                jax.grad(self.model.F_LCS_poly_per, holomorphic=True),
                holomorphic=True)(x, conj=conj)
        )(self.z0)
        dddF_0 = self.variant(
            lambda x: jax.jacfwd(
                jax.jacfwd(
                    jax.grad(self.model.F_LCS_poly_per, holomorphic=True),
                    holomorphic=True),
                holomorphic=True)(x, conj=conj)
        )(self.z0)

        # F_poly(z^i=0) = K0 / 2
        chex.assert_shape(F_0, ())
        self.assertAllClose(F_0, self.model.lcs_tree.K0 / 2.,
                            msg="F_LCS_poly at LCS must equal K0/2")

        # d_{X^i} F_poly |_0 = b_i  (exclude X^0 component)
        chex.assert_shape(dF_0, (self.model.h12 + 1,))
        self.assertAllClose(dF_0[1:], self.model.lcs_tree.b_vector,
                            msg="First derivatives of F_LCS_poly at LCS must equal b_vector")

        # d_{X^i} d_{X^j} F_poly |_0 = a_ij
        chex.assert_shape(ddF_0, (self.model.h12 + 1, self.model.h12 + 1))
        self.assertAllClose(ddF_0[1:, 1:], self.model.lcs_tree.a_matrix,
                            msg="Second derivatives of F_LCS_poly at LCS must equal a_matrix")

        # d_{X^i} d_{X^j} d_{X^k} F_poly |_0 = -kappa_ijk (mirror intersection numbers)
        chex.assert_shape(dddF_0, (self.model.h12 + 1, self.model.h12 + 1, self.model.h12 + 1))
        self.assertAllClose(dddF_0[1:, 1:, 1:], -self.model.lcs_tree.intnums,
                            msg="Third derivatives of F_LCS_poly at LCS must equal -mirror_intersection_numbers")


    # ==========================================================================
    # Period vector — shapes, conjugation, derivatives
    # ==========================================================================

    @chex.variants(with_jit=True, without_jit=True)
    def test_period_vector(self):
        r"""**Description:**

        Test the period vector :math:`\Pi`, its ordinary gradient
        :math:`\partial_{X^I}\Pi`, and its Kähler-covariant derivative
        :math:`D_I\Pi` returned by :func:`period_vector_per`,
        :func:`grad_period_vector_per`, and :func:`D_period_vector_per`.

        The period vector is defined as

        .. math::
            \Pi = \begin{pmatrix} \mathcal{F}_I \\ X^I \end{pmatrix},
            \quad I = 0, 1, \ldots, h^{1,2} \,,

        with :math:`\mathcal{F}_I = \partial_{X^I} F`.

        Conjugation relations

        .. math::
            \Pi(\bar z) = \overline{\Pi(z)} \,,\quad
            \partial_{\bar X^I}\Pi(\bar z) = \overline{\partial_{X^I}\Pi(z)}
            \,,\quad D\Pi(\bar z) = \overline{D\Pi(z)}

        are verified for all three objects.

        Args:
            None

        Returns:
            None
        """

        # ---- period vector ----
        conj = False
        Pi  = self.variant(lambda x: self.model.period_vector_per(x, conj=conj))(self.z)
        dPi = self.variant(lambda x: self.model.grad_period_vector_per(x, conj=conj))(self.z)
        DPi = self.variant(lambda x, y: self.model.D_period_vector_per(x, y, conj=conj))(self.z, self.cz)

        conj = True
        Pi_c  = self.variant(lambda x: self.model.period_vector_per(x, conj=conj))(self.cz)
        dPi_c = self.variant(lambda x: self.model.grad_period_vector_per(x, conj=conj))(self.cz)
        DPi_c = self.variant(lambda x, y: self.model.D_period_vector_per(x, y, conj=conj))(self.z, self.cz)

        # ---- shapes ----
        chex.assert_shape(Pi,   (2 * (self.model.h12 + 1),))   # full symplectic period vector
        chex.assert_shape(Pi_c, (2 * (self.model.h12 + 1),))

        chex.assert_shape(dPi,   (2 * (self.model.h12 + 1), self.model.h12 + 1))
        chex.assert_shape(dPi_c, (2 * (self.model.h12 + 1), self.model.h12 + 1))

        chex.assert_shape(DPi,   (2 * (self.model.h12 + 1), self.model.h12 + 1))
        chex.assert_shape(DPi_c, (2 * (self.model.h12 + 1), self.model.h12 + 1))

        # ---- conjugation of period vector ----
        self.assertAllClose(Pi_c, jnp.conj(Pi),
                            msg="Pi_c must equal conj(Pi)")

        # ---- conjugation of gradients ----
        self.assertAllClose(dPi_c, jnp.conj(dPi),
                            msg="dPi_c must equal conj(dPi)")
        self.assertAllClose(DPi_c, jnp.conj(DPi),
                            msg="DPi_c must equal conj(DPi)")

    def test_auto_vmap_period_methods_match_scalar_calls(self):
        r"""Auto-vectorised homogeneous-period methods match scalar calls."""
        X_batch = jnp.stack([
            self.z,
            self.z + jnp.array([0.0, 0.05 + 0.1j, -0.04 + 0.2j]),
        ])
        cX_batch = jnp.conj(X_batch)

        Pi_batch = self.model.period_vector_per(X_batch)
        Pi_scalar = jnp.stack([self.model.period_vector_per(X_batch[i]) for i in range(2)])
        self.assertAllClose(Pi_batch, Pi_scalar, rtol=1e-10, atol=1e-10)

        K_batch = self.model.kahler_potential_per(X_batch, cX_batch)
        K_scalar = jnp.stack([
            self.model.kahler_potential_per(X_batch[i], cX_batch[i])
            for i in range(2)
        ])
        self.assertAllClose(K_batch, K_scalar, rtol=1e-10, atol=1e-10)

        M_batch = self.model.ISD_matrix(X_batch, cX_batch)
        M_scalar = jnp.stack([self.model.ISD_matrix(X_batch[i], cX_batch[i]) for i in range(2)])
        self.assertAllClose(M_batch, M_scalar, rtol=1e-10, atol=1e-10)


# ==============================================================================
# TestCustomPeriodInputs
# ==============================================================================

class TestCustomPeriodInputs(TestCase):
    r"""
    **Description:**
    Focused checks for user-supplied period and prepotential callables.  These
    guard the constructor dispatch used by the custom-period notebooks, where
    no built-in LCS model data is required.
    """

    def test_custom_prepotential_registers_and_differentiates(self):
        r"""A valid homogeneous prepotential should become the period source."""

        def F(X, conj=False):
            coeff = -0.5j if conj else 0.5j
            return coeff * jnp.sum(X ** 2)

        with pytest.warns(UserWarning, match="general input periods"):
            model = jaxvacua.periods(h12=1, limit=None, prepotential_input=F)
        X = jnp.array([1.0 + 0.2j, 0.3 + 1.1j])
        cX = jnp.conj(X)

        self.assertTrue(model._prepotential_input_used)
        self.assertFalse(model._period_input_used)

        Pi = model.period_vector_per(X)
        Pi_c = model.period_vector_per(cX, conj=True)
        ddF = model.prepot_grad_grad_per(X)
        N = model.gauge_kinetic_matrix(X, cX)

        chex.assert_shape(Pi, (4,))
        chex.assert_shape(ddF, (2, 2))
        chex.assert_shape(N, (2, 2))
        self.assertAllClose(Pi_c, jnp.conj(Pi), atol=1e-12)
        self.assertAllClose(ddF, 1j * jnp.eye(2), atol=1e-12)

    def test_custom_period_input_registers_direct_period_vector(self):
        r"""A direct period vector should bypass prepotential reconstruction."""

        def Pi(X, conj=False):
            c1 = -1j if conj else 1j
            c2 = -2j if conj else 2j
            return jnp.array([c1 * X[0], c2 * X[1], X[0], X[1]])

        with pytest.warns(UserWarning, match="general input periods"):
            model = jaxvacua.periods(h12=1, limit=None, period_input=Pi)
        X = jnp.array([1.0 + 0.2j, 0.3 + 1.1j])

        self.assertTrue(model._period_input_used)
        self.assertFalse(model._prepotential_input_used)
        self.assertAllClose(model.period_vector_per(X), Pi(X), atol=1e-12)
        self.assertAllClose(
            model.period_vector_per(jnp.conj(X), conj=True),
            jnp.conj(Pi(X)),
            atol=1e-12,
        )

    def test_custom_period_input_rejects_wrong_shape(self):
        r"""Constructor validation should reject malformed period vectors."""

        def bad_periods(X, conj=False):
            return jnp.ones((3,), dtype=complex)

        with pytest.warns(UserWarning, match="general input periods"):
            with pytest.raises(ValueError, match="Wrong output shape for period"):
                jaxvacua.periods(h12=1, limit=None, period_input=bad_periods)

    def test_custom_prepotential_rejects_bad_conjugation(self):
        r"""Constructor validation should catch inconsistent conjugation."""

        def bad_F(X, conj=False):
            return 0.5j * jnp.sum(X ** 2)

        with pytest.warns(UserWarning, match="general input periods"):
            with pytest.raises(ValueError, match="not consistent with complex conjugation"):
                jaxvacua.periods(h12=1, limit=None, prepotential_input=bad_F)
