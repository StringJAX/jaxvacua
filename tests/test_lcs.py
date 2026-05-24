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

"""Tests for ``lcs_tree`` model-data containers.

Purpose
-------
Validate construction, pytree behaviour and data normalisation for
large-complex-structure model data.

Main public API
---------------
- ``_make_h12_2_tree``: helper fixture for a minimal two-modulus tree.
- ``TestLcsTree``: checks stored topology, invariants, basis data and pytree
  round-tripping.

Design notes
------------
Synthetic fixtures keep these tests independent of CYTools and external model
files.
"""

import sys, os, warnings
import jax
import jax.numpy as jnp
import numpy as np
import chex
from functools import partial
from util import *

jax.config.update("jax_enable_x64", True)

sys.path.append("./../")
from jaxvacua.lcs import lcs_tree

# Suppress warnings
warnings.filterwarnings("ignore")


# ==============================================================================
#  Helper: build a minimal lcs_tree for h12=2
# ==============================================================================

def _make_h12_2_tree(**overrides):
    r"""Create an lcs_tree with minimal valid data for h12=2.

    The intersection numbers are taken to be kappa_{111} = 9, kappa_{112} = 3,
    kappa_{122} = 1, which is the topology of the CP_{11169} mirror.
    """
    h12 = 2
    intnums = np.zeros((h12, h12, h12), dtype=np.int32)
    # kappa_{111} = 9
    intnums[0, 0, 0] = 9
    # kappa_{112} = kappa_{121} = kappa_{211} = 3
    intnums[0, 0, 1] = 3
    intnums[0, 1, 0] = 3
    intnums[1, 0, 0] = 3
    # kappa_{122} = kappa_{212} = kappa_{221} = 1
    intnums[0, 1, 1] = 1
    intnums[1, 0, 1] = 1
    intnums[1, 1, 0] = 1

    c2 = jnp.array([36, 12])  # second Chern class

    defaults = dict(
        h12=h12,
        h11=2,
        intnums=intnums,
        c2=c2,
        model_type="KS",
        model_ID=1,
        maximum_degree=0,
    )
    defaults.update(overrides)
    return lcs_tree(**defaults)


# ==============================================================================
#  TestLcsTree
# ==============================================================================

class TestLcsTree(TestCase):
    r"""
    **Description:**
    Test suite for :class:`lcs_tree`.  Verifies construction, attribute shapes
    and symmetries, file I/O, dictionary round-trips, and the ``update`` method.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tree = _make_h12_2_tree()

        # Path to the shipped model file for from_file tests
        cls.model_file = os.path.join(
            os.path.dirname(__file__), "..", "jaxvacua", "models", "KS", "h12_2", "model_1.p"
        )

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def test_constructor_creates_object(self):
        r"""
        **Description:**
        ``lcs_tree(...)`` returns a valid ``lcs_tree`` instance with the
        correct ``h12`` and ``model_type`` attributes.
        """
        tree = _make_h12_2_tree()
        # Must return an lcs_tree instance, not a plain dict or None
        self.assertIsInstance(tree, lcs_tree)
        # h12 attribute must match the input value of 2
        self.assertEqual(tree.h12, 2)
        # model_type attribute must match the input string "KS"
        self.assertEqual(tree.model_type, "KS")
        # model_ID attribute must match the input value of 1
        self.assertEqual(tree.model_ID, 1)

    def test_constructor_missing_intnums_raises(self):
        r"""
        **Description:**
        Omitting both ``intnums`` and ``c2`` raises ``ValueError``.
        """
        with self.assertRaises(ValueError):
            lcs_tree(h12=2, h11=2, model_type="KS", model_ID=1)

    def test_constructor_invalid_limit_raises(self):
        r"""
        **Description:**
        Passing an invalid ``limit`` string raises ``ValueError``.
        """
        with self.assertRaises(ValueError):
            _make_h12_2_tree(limit="INVALID")

    # ------------------------------------------------------------------
    # from_file
    # ------------------------------------------------------------------

    def test_from_file_returns_lcs_tree(self):
        r"""
        **Description:**
        ``lcs_tree.from_file(...)`` loads a model file and returns a valid
        ``lcs_tree`` instance with the expected ``h12`` attribute.
        """
        if not os.path.isfile(self.model_file):
            self.skipTest("Model file not found")
        tree = lcs_tree.from_file(self.model_file, maximum_degree=0)
        # from_file must return an lcs_tree instance, not a raw dict
        self.assertIsInstance(tree, lcs_tree)
        # Loaded model must have h12=2 matching the h12_2 model file
        self.assertEqual(tree.h12, 2)

    def test_from_file_has_key_attributes(self):
        r"""
        **Description:**
        A model loaded from file has ``intnums``, ``a_matrix``,
        ``b_vector``, ``K0``, and ``chi`` attributes.
        """
        if not os.path.isfile(self.model_file):
            self.skipTest("Model file not found")
        tree = lcs_tree.from_file(self.model_file, maximum_degree=0)
        for attr in ("intnums", "a_matrix", "b_vector", "K0", "chi"):
            self.assertTrue(hasattr(tree, attr), msg=f"Missing attribute: {attr}")

    # ------------------------------------------------------------------
    # Attribute shapes and types
    # ------------------------------------------------------------------

    def test_intnums_shape(self):
        r"""
        **Description:**
        Intersection numbers ``intnums`` have shape ``(h12, h12, h12)``.

        The triple intersection numbers kappa_{ijk} = \int J_i \wedge J_j \wedge J_k
        are the triple intersection numbers of the mirror Calabi-Yau threefold.
        They form a fully symmetric rank-3 tensor indexed by the h^{1,2}
        complex-structure moduli (equivalently h^{1,1} of the mirror).
        """
        self.assertEqual(self.tree.intnums.shape, (2, 2, 2))

    def test_a_matrix_shape(self):
        r"""
        **Description:**
        The ``a_matrix`` has shape ``(h12, h12)``.

        It encodes the quadratic piece of the prepotential:
        F \supset a_{ij} t^i t^j / 2.
        """
        self.assertEqual(self.tree.a_matrix.shape, (2, 2))

    def test_b_vector_shape(self):
        r"""
        **Description:**
        The ``b_vector`` has shape ``(h12,)``.

        It defaults to c_2 / 24 and enters the linear term of the
        prepotential: F \supset b_i t^i.
        """
        self.assertEqual(self.tree.b_vector.shape, (2,))

    def test_K0_is_complex(self):
        r"""
        **Description:**
        ``K0`` is a complex scalar derived from zeta(3) and the Euler
        characteristic chi: K0 = zeta(3) * chi / (2 pi i)^3.
        """
        # K0 must have a complex dtype since it involves zeta(3)/(2*pi*i)^3
        self.assertTrue(jnp.issubdtype(self.tree.K0.dtype, jnp.complexfloating))
        # K0 must be a scalar (0-dimensional array)
        self.assertEqual(self.tree.K0.shape, ())

    def test_chi_is_integer(self):
        r"""
        **Description:**
        ``chi`` is an integer (the Euler characteristic of the mirror CY).

        For the mirror, chi = -2 * (h11 - h12).
        """
        self.assertTrue(jnp.issubdtype(self.tree.chi.dtype, jnp.integer))
        self.assertEqual(int(self.tree.chi), -2 * (self.tree.h11 - self.tree.h12))

    # ------------------------------------------------------------------
    # Symmetries
    # ------------------------------------------------------------------

    def test_a_matrix_symmetry(self):
        r"""
        **Description:**
        The ``a_matrix`` should be symmetric: a_{ij} = a_{ji}.

        Symmetry follows from the cubic prepotential
        F = kappa_{ijk} t^i t^j t^k / 6, whose second derivatives
        a_{ij} = kappa_{iij}/2 (for i >= j) are symmetric.
        """
        A = self.tree.a_matrix
        self.assertAllClose(A, A.T, atol=1e-14,
                            msg="a_matrix must be symmetric: a_{ij} = a_{ji}")

    def test_intnums_symmetry(self):
        r"""
        **Description:**
        The intersection numbers kappa_{ijk} are fully symmetric under
        permutation of indices.

        The triple intersection numbers kappa_{ijk} = \int J_i \wedge J_j \wedge J_k
        are the triple intersection numbers of the mirror Calabi-Yau threefold.
        Because the wedge product of closed (1,1)-forms on a Calabi-Yau threefold
        is commutative up to exact forms, the integral is invariant under all
        permutations of (i, j, k):
            kappa_{ijk} = kappa_{ikj} = kappa_{jik} = kappa_{jki}
                        = kappa_{kij} = kappa_{kji}.
        """
        K = np.asarray(self.tree.intnums)
        h = K.shape[0]
        for i in range(h):
            for j in range(h):
                for k in range(h):
                    val = K[i, j, k]
                    self.assertEqual(val, K[i, k, j],
                                     msg=f"kappa_{{{i}{j}{k}}} != kappa_{{{i}{k}{j}}}")
                    self.assertEqual(val, K[j, i, k],
                                     msg=f"kappa_{{{i}{j}{k}}} != kappa_{{{j}{i}{k}}}")
                    self.assertEqual(val, K[j, k, i],
                                     msg=f"kappa_{{{i}{j}{k}}} != kappa_{{{j}{k}{i}}}")
                    self.assertEqual(val, K[k, i, j],
                                     msg=f"kappa_{{{i}{j}{k}}} != kappa_{{{k}{i}{j}}}")
                    self.assertEqual(val, K[k, j, i],
                                     msg=f"kappa_{{{i}{j}{k}}} != kappa_{{{k}{j}{i}}}")

    # ------------------------------------------------------------------
    # from_dict / filter_dict round-trip
    # ------------------------------------------------------------------

    def test_from_dict_round_trip(self):
        r"""
        **Description:**
        Creating an ``lcs_tree`` via ``from_dict`` with the same keyword
        arguments reproduces the original object's key attributes.

        This verifies that ``filter_dict`` correctly casts numpy arrays to
        JAX arrays and that ``from_dict`` passes them through to the
        constructor.
        """
        d = dict(
            h12=2,
            h11=2,
            intnums=np.array(self.tree.intnums),
            c2=np.array(self.tree.c2),
            model_type="KS",
            model_ID=1,
            maximum_degree=0,
        )
        tree2 = lcs_tree.from_dict(d)
        # from_dict must return an lcs_tree instance
        self.assertIsInstance(tree2, lcs_tree)
        # h12 must survive the dict round-trip unchanged
        self.assertEqual(tree2.h12, 2)
        # Intersection numbers must match the original after round-trip
        self.assertAllClose(tree2.intnums, self.tree.intnums, atol=1e-14)
        # Derived a_matrix must be recomputed identically from the same intnums
        self.assertAllClose(tree2.a_matrix, self.tree.a_matrix, atol=1e-14)
        # Derived b_vector must be recomputed identically from the same c2
        self.assertAllClose(tree2.b_vector, self.tree.b_vector, atol=1e-14)

    def test_filter_dict_converts_arrays(self):
        r"""
        **Description:**
        ``filter_dict`` converts numpy arrays in a dict to JAX arrays.
        """
        d = dict(
            intnums=np.ones((2, 2, 2)),
            c2=np.array([1.0, 2.0]),
            some_scalar=42,
        )
        d2 = lcs_tree.filter_dict(d)
        self.assertIsInstance(d2["intnums"], jnp.ndarray)
        self.assertIsInstance(d2["c2"], jnp.ndarray)
        # Scalars are left alone
        self.assertEqual(d2["some_scalar"], 42)

    # ------------------------------------------------------------------
    # update method
    # ------------------------------------------------------------------

    def test_update_changes_attribute(self):
        r"""
        **Description:**
        ``update(name=...)`` sets/changes the attribute on the object.
        """
        tree = _make_h12_2_tree()
        # name attribute must default to None before any update
        self.assertIsNone(tree.name)
        tree.update(name="test_model")
        # After update, name attribute must reflect the new value
        self.assertEqual(tree.name, "test_model")

    def test_update_adds_new_attribute(self):
        r"""
        **Description:**
        ``update`` can add attributes that were not set at construction.
        """
        tree = _make_h12_2_tree()
        self.assertFalse(hasattr(tree, "custom_attr"))
        tree.update(custom_attr=jnp.array([1.0, 2.0]))
        self.assertTrue(hasattr(tree, "custom_attr"))
        self.assertAllClose(tree.custom_attr, jnp.array([1.0, 2.0]), atol=1e-14)

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def test_repr_contains_hodge_numbers(self):
        r"""
        **Description:**
        String representation includes h11 and h12.
        """
        s = repr(self.tree)
        # String representation must mention h11
        self.assertIn("h11", s)
        # String representation must mention h12
        self.assertIn("h12", s)


if __name__ == "__main__":
    import unittest
    unittest.main()
