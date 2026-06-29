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

"""Tests for integer and conifold lattice utilities.

Purpose
-------
Validate the lattice algebra used by conifold basis changes and structural
helpers.

Main public API
---------------
- ``TestExtendedEuclidean`` and ``TestOrthogonalLattice``: pure-integer
  lattice helper tests.
- ``TestGetBasisChange`` and ``TestGetAMatrix``: conifold basis and
  prepotential-matrix checks.

Design notes
------------
Pure-integer routines are tested with synthetic data.  CYTools-dependent
conifold discovery paths are skipped when CYTools is unavailable.
"""

import sys, os, warnings
import numpy as np
import jax.numpy as jnp
import pytest
from itertools import permutations
from math import gcd
from functools import reduce

from util import *

sys.path.append("./../")
from jaxvacua.conifold import (
    extended_euclidean,
    orthogonal_lattice,
    get_basis_change,
    get_bulk_embedding,
    get_bulk_projection,
)
from jaxvacua.conifold.conifold_utils import get_embedding

# Suppress warnings
warnings.filterwarnings("ignore")


# ==============================================================================
#  TestExtendedEuclidean
# ==============================================================================

class TestExtendedEuclidean(TestCase):
    r"""
    **Description:**
    Test suite for :func:`extended_euclidean`, which computes Bézout coefficients
    and a unimodular basis transformation for an integer array :math:`w`.

    .. admonition:: Background
        :class: dropdown

        Given :math:`w = (w_1, \ldots, w_n)`, the function returns:

        - Bézout coefficients :math:`b_i` with :math:`\sum b_i w_i = \gcd(w)`
        - A unimodular matrix :math:`\Lambda \in \mathrm{GL}(n, \mathbb{Z})`
          with :math:`\Lambda\,w = (\gcd(w), 0, \ldots, 0)^T`

        This is the higher-dimensional generalisation of the extended Euclidean
        algorithm and is used to construct the conifold basis change.
    """

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    def test_bezout_identity(self):
        r"""
        **Description:**
        Verify the Bézout identity :math:`\sum_i b_i w_i = \gcd(w)` for
        several test vectors.
        """
        test_vectors = [
            [6, 10, 15],
            [12, 18],
            [7, 11, 13, 17],
            [2, 3, 5],
            [4, 6, 8, 10],
            [1, 0, 0],
            [0, 5, 0],
        ]
        for w in test_vectors:
            w_arr = np.array(w)
            bezout, g, Lambda = extended_euclidean(w)
            # Bézout identity: b · w = gcd(w)
            self.assertEqual(
                int(np.dot(bezout, w_arr)), int(g),
                msg=f"Bézout identity failed for w={w}: "
                    f"b·w={int(np.dot(bezout, w_arr))} != gcd={int(g)}"
            )

    def test_gcd_value(self):
        r"""
        **Description:**
        Verify that the returned GCD matches Python's ``math.gcd`` for
        several test vectors.
        """
        test_vectors = [
            ([6, 10, 15], 1),
            ([12, 18], 6),
            ([4, 6, 8, 10], 2),
            ([7, 14, 21], 7),
            ([0, 5, 0], 5),
        ]
        for w, expected_gcd in test_vectors:
            _, g, _ = extended_euclidean(w)
            # gcd must match the expected value
            self.assertEqual(int(g), expected_gcd,
                             msg=f"GCD({w}) = {int(g)}, expected {expected_gcd}")

    def test_unimodular_property(self):
        r"""
        **Description:**
        Verify that :math:`\Lambda` is unimodular: :math:`\det(\Lambda) = \pm 1`.

        A unimodular matrix has integer entries and integer inverse, which is
        equivalent to :math:`|\det(\Lambda)| = 1`.
        """
        test_vectors = [
            [6, 10, 15],
            [2, 3, 5],
            [7, 11, 13, 17],
            [1, 0, 0, 0],
        ]
        for w in test_vectors:
            _, _, Lambda = extended_euclidean(w)
            det = int(np.round(np.linalg.det(Lambda)))
            # |det(Λ)| must be 1
            self.assertEqual(abs(det), 1,
                             msg=f"|det(Λ)| = {abs(det)} != 1 for w={w}")

    def test_lambda_maps_w_to_canonical(self):
        r"""
        **Description:**
        Verify that :math:`\Lambda\,w = (\gcd(w), 0, \ldots, 0)^T`.

        This is the defining property of the unimodular basis change:
        in the new basis, the input vector has all its content in the first
        component.
        """
        test_vectors = [
            [6, 10, 15],
            [2, 3, 5],
            [12, 18],
            [4, 6, 8, 10],
        ]
        for w in test_vectors:
            w_arr = np.array(w)
            _, g, Lambda = extended_euclidean(w)
            # Λ w must equal (gcd, 0, ..., 0)
            result = Lambda @ w_arr
            expected = np.zeros(len(w), dtype=int)
            expected[0] = int(g)
            self.assertAllEqual(result, expected)

    def test_integer_entries(self):
        r"""
        **Description:**
        Verify that all entries of :math:`\Lambda` and the Bézout vector are
        integers (no floating-point rounding artifacts).
        """
        bezout, g, Lambda = extended_euclidean([6, 10, 15])
        # Bézout coefficients must be exact integers
        self.assertTrue(np.all(bezout == np.round(bezout)),
                        msg="Bézout coefficients are not integers")
        # Lambda entries must be exact integers
        self.assertTrue(np.all(Lambda == np.round(Lambda)),
                        msg="Lambda entries are not integers")

    def test_single_nonzero(self):
        r"""
        **Description:**
        Edge case: input with a single nonzero entry.
        :math:`\gcd(0, 5, 0) = 5` and :math:`\Lambda` should permute the
        nonzero entry to the first position.
        """
        bezout, g, Lambda = extended_euclidean([0, 5, 0])
        # GCD is the single nonzero entry
        self.assertEqual(int(g), 5)
        # Λ maps [0,5,0] → [5,0,0]
        result = Lambda @ np.array([0, 5, 0])
        self.assertEqual(result[0], 5)
        self.assertEqual(result[1], 0)
        self.assertEqual(result[2], 0)

    def test_two_elements(self):
        r"""
        **Description:**
        Classical 2D case: :math:`\gcd(12, 18) = 6` with Bézout coefficients
        satisfying :math:`b_1 \cdot 12 + b_2 \cdot 18 = 6`.
        """
        bezout, g, Lambda = extended_euclidean([12, 18])
        self.assertEqual(int(g), 6)
        self.assertEqual(int(np.dot(bezout, [12, 18])), 6)
        self.assertEqual(abs(int(np.round(np.linalg.det(Lambda)))), 1)


# ==============================================================================
#  TestOrthogonalLattice
# ==============================================================================

class TestOrthogonalLattice(TestCase):
    r"""
    **Description:**
    Test suite for :func:`orthogonal_lattice`, which computes generators of the
    integer lattice orthogonal to given generators.

    .. admonition:: Background
        :class: dropdown

        Given generators :math:`g_1, \ldots, g_d \in \mathbb{Z}^n`, the
        orthogonal lattice is :math:`L^\perp = \{v \in \mathbb{Z}^n : v \cdot g_i = 0 \;\forall i\}`.
        This has rank :math:`n - d` and is computed via LLL reduction using
        ``flint.fmpz_mat``.
    """

    def test_orthogonality(self):
        r"""
        **Description:**
        Verify that all returned generators are orthogonal to the input
        generators: :math:`v \cdot g = 0` for all :math:`v \in L^\perp`.
        """
        test_cases = [
            [[1, 2, 3]],
            [[1, 0, 0, 1]],
            [[2, 3, 5]],
        ]
        for gens in test_cases:
            orth = orthogonal_lattice(gens)
            for v in orth:
                for g in gens:
                    # Each orthogonal generator must be perpendicular to each input generator
                    dot = int(np.dot(v, g))
                    self.assertEqual(dot, 0,
                                     msg=f"v·g = {dot} != 0 for v={v}, g={g}")

    def test_correct_rank(self):
        r"""
        **Description:**
        Verify that the number of orthogonal generators equals :math:`n - d`
        where :math:`n` is the ambient dimension and :math:`d` is the number
        of input generators.
        """
        # 1 generator in R^3 → 2 orthogonal generators
        orth = orthogonal_lattice([[1, 2, 3]])
        self.assertEqual(len(orth), 2)

        # 1 generator in R^4 → 3 orthogonal generators
        orth = orthogonal_lattice([[1, 0, 0, 1]])
        self.assertEqual(len(orth), 3)

    def test_integer_entries(self):
        r"""
        **Description:**
        Verify that all returned generators have integer entries.
        """
        orth = orthogonal_lattice([[2, 3, 5]])
        for v in orth:
            for entry in v:
                # Each entry must be an exact integer
                self.assertEqual(int(entry), entry,
                                 msg=f"Non-integer entry {entry} in orthogonal generator {v}")

    def test_linear_independence(self):
        r"""
        **Description:**
        Verify that the returned generators are linearly independent by
        checking that their matrix has full rank.
        """
        orth = orthogonal_lattice([[1, 2, 3, 4]])
        mat = np.array(orth)
        # Rank must equal the number of generators (= n - d = 3)
        self.assertEqual(np.linalg.matrix_rank(mat), len(orth))


# ==============================================================================
#  TestGetBasisChange
# ==============================================================================

class TestGetBasisChange(TestCase):
    r"""
    **Description:**
    Test suite for :func:`get_basis_change`, which constructs the unimodular
    transformation :math:`\Lambda` mapping a conifold charge vector
    :math:`q` to :math:`(1, 0, \ldots, 0)^T`.

    .. admonition:: Background
        :class: dropdown

        The conifold charge vector :math:`q \in \mathbb{Z}^{h^{1,1}}` defines
        the shrinking curve.  The basis change :math:`\Lambda` rotates the
        moduli basis so that :math:`z^1` directly parameterises the approach
        to the conifold singularity, while :math:`z^2, \ldots, z^{h^{1,1}}`
        span the bulk directions.
    """

    def test_maps_to_canonical(self):
        r"""
        **Description:**
        Verify :math:`\Lambda\,q = (1, 0, \ldots, 0)^T` for several
        charge vectors with :math:`\gcd(q) = 1`.
        """
        test_charges = [
            [1, 0, 0],
            [2, 3, 5],
            [1, 1, 1, 1],
            [3, 7],
        ]
        for q in test_charges:
            q_arr = np.array(q)
            Lambda = get_basis_change(q)
            result = Lambda @ q_arr
            # First entry must be 1 (since gcd(q)=1)
            self.assertEqual(int(result[0]), 1,
                             msg=f"First entry of Λq is {int(result[0])}, expected 1")
            # Remaining entries must be 0
            for i in range(1, len(q)):
                self.assertEqual(int(result[i]), 0,
                                 msg=f"Entry {i} of Λq is {int(result[i])}, expected 0")

    def test_unimodular(self):
        r"""
        **Description:**
        Verify that :math:`|\det(\Lambda)| = 1` (unimodularity).
        """
        for q in [[2, 3, 5], [1, 1, 1, 1], [3, 7]]:
            Lambda = get_basis_change(q)
            det = abs(int(np.round(np.linalg.det(Lambda))))
            self.assertEqual(det, 1,
                             msg=f"|det(Λ)| = {det} != 1 for q={q}")

    def test_shape(self):
        r"""
        **Description:**
        Verify that :math:`\Lambda` has shape :math:`(n, n)` where :math:`n = \text{len}(q)`.
        """
        for n in [2, 3, 4, 5]:
            q = list(range(1, n + 1))  # [1, 2, ..., n]
            Lambda = get_basis_change(q)
            # Λ must be square with dimension n
            self.assertEqual(Lambda.shape, (n, n))

    def test_integer_matrix(self):
        r"""
        **Description:**
        Verify that :math:`\Lambda` has integer entries and integer inverse.
        """
        Lambda = get_basis_change([2, 3, 5])
        # All entries must be integers
        self.assertTrue(np.all(Lambda == np.round(Lambda)))
        # Inverse must also have integer entries (since det = ±1)
        Lambda_inv = np.round(np.linalg.inv(Lambda)).astype(int)
        # Λ Λ^{-1} = I
        self.assertAllEqual(Lambda @ Lambda_inv, np.eye(3, dtype=int))

    def test_first_row_is_bezout(self):
        r"""
        **Description:**
        Verify that the first row of :math:`\Lambda` consists of the Bézout
        coefficients: :math:`\Lambda_{1,:} \cdot q = 1`.
        """
        q = [2, 3, 5]
        Lambda = get_basis_change(q)
        # First row dotted with q must give 1 (= gcd)
        self.assertEqual(int(Lambda[0] @ np.array(q)), 1)

    def test_bulk_embedding_projection_axis_aligned(self):
        r"""
        **Description:**
        In the conifold-aligned basis, the conifold embedding and bulk
        embedding/projection reduce to the obvious coordinate slices.
        """
        q = np.array([1, 0, 0])
        e_q = np.asarray(get_embedding(q))
        bulk_embedding = np.asarray(get_bulk_embedding(q))
        bulk_projection = np.asarray(get_bulk_projection(q))

        expected_bulk = np.array([[0, 0], [1, 0], [0, 1]])
        self.assertAllEqual(e_q, np.array([1, 0, 0]))
        self.assertAllEqual(bulk_embedding, expected_bulk)
        self.assertAllEqual(bulk_projection, expected_bulk)

    def test_bulk_embedding_projection_general_basis_roundtrip(self):
        r"""
        **Description:**
        In a general basis, bulk embedding and bulk projection differ but must
        still give an exact decomposition

        ``v = (q·v) e_q + bulk_embedding @ (v @ bulk_projection)``.

        This pure-lattice regression catches projection/embedding swaps without
        requiring CYTools or coniLCS model construction.
        """
        q = np.array([-1, 1, 0])
        e_q = np.asarray(get_embedding(q), dtype=float)
        bulk_embedding = np.asarray(get_bulk_embedding(q), dtype=float)
        bulk_projection = np.asarray(get_bulk_projection(q), dtype=float)

        self.assertAlmostEqual(float(q @ e_q), 1.0)
        self.assertAllClose(q @ bulk_embedding, np.zeros(2), atol=1e-12)
        self.assertAllClose(
            bulk_embedding.T @ bulk_projection,
            np.eye(2),
            atol=1e-12,
        )
        self.assertGreater(
            float(np.max(np.abs(bulk_embedding - bulk_projection))),
            0.0,
            msg="embedding and projection should differ in this general basis",
        )

        test_vectors = [
            np.array([2.5, -0.75, 4.0]),
            np.array([-1.0, 3.0, 0.5]),
            np.array([0.0, 1.25, -2.0]),
        ]
        for v in test_vectors:
            z_cf = float(q @ v)
            z_bulk = v @ bulk_projection
            reconstructed = z_cf * e_q + bulk_embedding @ z_bulk
            self.assertAllClose(reconstructed, v, atol=1e-12)


# NOTE: ``TestGetAMatrix`` was removed on 2026-06-17.  It exercised the
# deprecated ``jaxvacua.conifold.conifold_utils.compute_a_matrix`` helper
# (lower-triangular asymmetric :math:`a`-matrix from the OLD jaxvacua
# convention).  The HKTY-canonical symmetric :math:`a`-matrix is now
# constructed in-line in :class:`jaxvacua.lcs.lcs_tree._prepare_prepot`;
# see :class:`tests.test_css.TestMonodromyAsymmetric` for the regression
# net that exercises the new convention.


if __name__ == "__main__":
    import unittest
    unittest.main()
