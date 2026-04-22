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
from functools import partial
from util import *

jax.config.update("jax_enable_x64", True)

sys.path.append("./../")
from jaxvacua.one_modulus_models import get_model, list_models, OneModulusModel

# Suppress warnings
warnings.filterwarnings("ignore")


# ==============================================================================
#  TestOneModulusModels
# ==============================================================================

class TestOneModulusModels(TestCase):
    r"""
    **Description:**
    Test suite for :mod:`one_modulus_models`. Verifies model loading, prepotential
    properties, period-vector autodiff, and homogeneity.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Load a representative model at each singularity type
        cls.model_lcs = get_model("X33", "LCS")
        cls.model_kpt = get_model("X33", "Kpoint")
        cls.model_cpt = get_model("X42", "Cpoint")  # X33 has no C-point

        # Test point near the LCS singularity (small |z|)
        cls.X_lcs = jnp.array([1.0 + 0j, 0.01 + 0.05j])
        # Test point near the K-point (generic)
        cls.X_kpt = jnp.array([1.0 + 0j, 0.3 + 0.4j])
        # Test point near the conifold
        cls.X_cpt = jnp.array([1.0 + 0j, 0.01 + 0.02j])

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def test_get_model_returns_OneModulusModel(self):
        r"""
        **Description:**
        Verifies that ``get_model`` returns a proper :class:`OneModulusModel` instance
        for both LCS and K-point singularity types, ensuring the factory function
        correctly constructs period data objects.
        """
        # LCS model must be a OneModulusModel
        self.assertIsInstance(self.model_lcs, OneModulusModel)
        # K-point model must also be a OneModulusModel
        self.assertIsInstance(self.model_kpt, OneModulusModel)

    def test_get_model_invalid_label(self):
        r"""
        **Description:**
        Checks that requesting a non-existent model label raises ``ValueError``,
        guarding against silent failures when users mistype a model name.
        """
        # Requesting an unknown model label must raise ValueError
        with self.assertRaises(ValueError):
            get_model("NONEXISTENT", "LCS")

    def test_get_model_invalid_singularity(self):
        r"""
        **Description:**
        Checks that requesting an unsupported singularity type raises an error,
        since each model only has period data near specific singularities (LCS, Kpoint, Cpoint).
        """
        # Invalid singularity type must raise KeyError or ValueError
        with self.assertRaises((KeyError, ValueError)):
            get_model("X33", "invalid")

    def test_list_models_no_error(self):
        r"""
        **Description:**
        Verifies that ``list_models`` executes without error and that its printed
        output includes at least the well-known :math:`X_{3,3}` model, confirming
        the model database is populated.
        """
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            list_models()
        output = f.getvalue()
        # Output must mention the X33 model (mirror quintic analogue)
        self.assertIn("X33", output)

    def test_model_attributes(self):
        r"""
        **Description:**
        Verifies that a loaded model exposes the expected attributes: label,
        singularity type, LCS data dictionary (containing topological data like
        the triple intersection number kappa), and a callable prepotential.
        """
        m = self.model_lcs
        # Label must match the requested model
        self.assertEqual(m.label, "X33")
        # Singularity type must match
        self.assertEqual(m.singularity, "LCS")
        # LCS data must be a dictionary containing topological constants
        self.assertIsInstance(m.lcs_data, dict)
        # kappa (triple intersection number) must be present in LCS data
        self.assertIn("kappa", m.lcs_data)
        # Prepotential must be a callable function F(X)
        self.assertTrue(callable(m.prepotential))

    # ------------------------------------------------------------------
    # Prepotential
    # ------------------------------------------------------------------

    @chex.variants(with_jit=True, without_jit=True)
    def test_prepotential_returns_scalar(self):
        r"""
        **Description:**
        Verifies that the prepotential :math:`F(X)` returns a rank-0 complex tensor,
        since it is a holomorphic function on the projective coordinate space
        and must yield a single complex number for any input.
        """
        for mdl, X in [(self.model_lcs, self.X_lcs),
                       (self.model_kpt, self.X_kpt)]:
            F = self.variant(lambda x: mdl.prepotential(x, conj=False))(X)
            # F(X) must be a scalar (rank-0 array)
            self.assertEqual(F.shape, ())
            # F(X) must be complex-valued
            chex.assert_type(F, complex)

    @chex.variants(with_jit=True, without_jit=True)
    def test_prepotential_differentiable(self):
        r"""
        **Description:**
        Verifies that the prepotential is compatible with JAX holomorphic autodiff,
        producing finite :math:`\partial F / \partial X^I`. These derivatives form
        the upper half of the period vector :math:`\Pi = (\partial_I F, X^I)`.
        """
        for mdl, X in [(self.model_lcs, self.X_lcs),
                       (self.model_kpt, self.X_kpt)]:
            grad_F = self.variant(
                jax.grad(lambda x: mdl.prepotential(x, conj=False), holomorphic=True)
            )(X)
            # Gradient dF/dX^I must have shape (2,) for h^{2,1}=1 (two projective coords)
            chex.assert_shape(grad_F, (2,))
            # All gradient components must be finite (no NaN or Inf)
            self.assertTrue(jnp.all(jnp.isfinite(grad_F)))

    # ------------------------------------------------------------------
    # Period vector
    # ------------------------------------------------------------------

    @chex.variants(with_jit=True, without_jit=True)
    def test_period_vector_shape(self):
        r"""
        **Description:**
        Verifies that the period vector :math:`\Pi = (\partial_I F, X^I)` has shape
        ``(4,)`` for all singularity types (LCS, Kpoint, Cpoint). For :math:`h^{2,1}=1`
        the symplectic period vector lives in :math:`\mathbb{C}^{2(h^{2,1}+1)} = \mathbb{C}^4`.
        """
        for mdl, X in [(self.model_lcs, self.X_lcs),
                       (self.model_kpt, self.X_kpt),
                       (self.model_cpt, self.X_cpt)]:
            Pi = self.variant(lambda x: mdl.period_vector(x, conj=False))(X)
            # Period vector must have 2*(h^{2,1}+1) = 4 components
            chex.assert_shape(Pi, (4,))

    @chex.variants(with_jit=True, without_jit=True)
    def test_period_vector_lower_half(self):
        r"""
        **Description:**
        Checks that the lower half of the period vector equals the projective
        coordinates :math:`X^I`, i.e. :math:`\Pi = (\partial_I F, X^I)`. This
        is a defining structural property of the special-geometry period vector.
        """
        for mdl, X in [(self.model_lcs, self.X_lcs),
                       (self.model_kpt, self.X_kpt)]:
            Pi = self.variant(lambda x: mdl.period_vector(x, conj=False))(X)
            # Lower two entries Pi[2:] must equal the input coordinates X^I
            self.assertAllClose(Pi[2:], X, atol=1e-14)

    @chex.variants(with_jit=True, without_jit=True)
    def test_period_vector_isotropicity(self):
        r"""
        **Description:**
        Tests the fundamental isotropicity condition :math:`\Pi^T \Sigma \Pi = 0`
        of special geometry. This identity is a consequence of the prepotential
        being a degree-2 homogeneous function and is required for a consistent
        Kahler potential on the complex-structure moduli space.

        The symplectic matrix for ``h^{2,1}=1``: :math:`\Sigma = \begin{pmatrix} 0 & I_2 \\ -I_2 & 0 \end{pmatrix}`.
        """
        # Symplectic pairing matrix Sigma for h^{2,1}=1
        sigma = jnp.array([[0, 0, 1, 0], [0, 0, 0, 1],
                           [-1, 0, 0, 0], [0, -1, 0, 0]], dtype=complex)
        for mdl, X in [(self.model_lcs, self.X_lcs),
                       (self.model_kpt, self.X_kpt)]:
            Pi = self.variant(lambda x: mdl.period_vector(x, conj=False))(X)
            PiSigmaPi = Pi @ sigma @ Pi
            # Isotropicity: Pi^T Sigma Pi must vanish identically
            self.assertAllClose(PiSigmaPi, 0.0 + 0j, atol=1e-11,
                                msg="Period vector must satisfy Pi.T @ sigma @ Pi = 0")

    @chex.variants(with_jit=True, without_jit=True)
    def test_period_vector_conj(self):
        r"""
        **Description:**
        Verifies that the conjugate period vector :math:`\bar\Pi(\bar X)` equals
        :math:`\overline{\Pi(X)}`, confirming the reality condition
        :math:`\bar F(\bar X) = \overline{F(X)}` required for a real Kahler potential.
        """
        for mdl, X in [(self.model_lcs, self.X_lcs),
                       (self.model_kpt, self.X_kpt)]:
            Pi_c = self.variant(lambda x: mdl.period_vector(x, conj=True))(jnp.conj(X))
            Pi   = self.variant(lambda x: mdl.period_vector(x, conj=False))(X)
            # conj(Pi(X)) must equal Pi_c(conj(X)) for a real Kahler potential
            self.assertAllClose(Pi_c, jnp.conj(Pi), atol=1e-11)

    # ------------------------------------------------------------------
    # Multiple models
    # ------------------------------------------------------------------

    def test_multiple_models_load(self):
        r"""
        **Description:**
        Loads all four hypergeometric one-parameter Calabi-Yau models
        (:math:`X_{3,3}`, :math:`X_{4,4}`, :math:`X_{4,2}`, :math:`X_5`) at the
        LCS point, verifying that the full model database from arXiv:2306.01059 is
        accessible and each model is correctly labelled.
        """
        for label in ["X33", "X44", "X42", "X5"]:
            m = get_model(label, "LCS")
            # Each loaded object must be a OneModulusModel
            self.assertIsInstance(m, OneModulusModel)
            # The label attribute must match the requested model
            self.assertEqual(m.label, label)


if __name__ == "__main__":
    import unittest
    unittest.main()
