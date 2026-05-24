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
    compute_a_matrix,
)

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


# ==============================================================================
#  TestGetAMatrix
# ==============================================================================

class TestGetAMatrix(TestCase):
    r"""
    **Description:**
    Test suite for :func:`compute_a_matrix`, which computes the :math:`a`-matrix from
    the triple intersection number tensor :math:`\kappa_{ijk}`.

    .. admonition:: Background
        :class: dropdown

        The :math:`a`-matrix enters the quadratic term of the LCS prepotential:

        .. math::
            F_{\rm poly}(z) \supset \tfrac{1}{2}\,a_{ij}\,z^i z^j

        Its entries are

        .. math::
            a_{ij} = \begin{cases}
                \kappa_{iij}/2 & i \geq j \\
                \kappa_{ijj}/2 & i < j
            \end{cases}
    """

    def test_shape(self):
        r"""
        **Description:**
        Verify that ``compute_a_matrix`` returns shape :math:`(n, n)` for an
        :math:`(n, n, n)` intersection tensor.
        """
        for n in [2, 3, 4]:
            kappa = np.zeros((n, n, n))
            A = compute_a_matrix(kappa)
            # a-matrix is square with same dimension as each axis of kappa
            self.assertEqual(A.shape, (n, n))

    def test_known_values(self):
        r"""
        **Description:**
        Verify the explicit formula :math:`a_{ij} = \kappa_{iij}/2` for
        :math:`i \geq j` and :math:`a_{ij} = \kappa_{ijj}/2` for :math:`i < j`
        on a hand-constructed example.

        Example: :math:`h^{1,2} = 2` with :math:`\kappa_{000} = 6`,
        :math:`\kappa_{001} = 3`, :math:`\kappa_{011} = 1`, :math:`\kappa_{111} = 2`.
        """
        kappa = np.zeros((2, 2, 2))
        kappa[0, 0, 0] = 6
        kappa[0, 0, 1] = kappa[0, 1, 0] = kappa[1, 0, 0] = 3
        kappa[0, 1, 1] = kappa[1, 0, 1] = kappa[1, 1, 0] = 1
        kappa[1, 1, 1] = 2

        A = compute_a_matrix(kappa)

        # a[0,0] = kappa[0,0,0]/2 = 3 (i >= j, so kappa_{iij}/2)
        self.assertAllClose(A[0, 0], 3.0)
        # a[0,1] = kappa[0,1,1]/2 = 0.5 (i < j, so kappa_{ijj}/2)
        self.assertAllClose(A[0, 1], 0.5)
        # a[1,0] = kappa[1,1,0]/2 = 0.5 (i >= j, so kappa_{iij}/2)
        self.assertAllClose(A[1, 0], 0.5)
        # a[1,1] = kappa[1,1,1]/2 = 1 (i >= j, so kappa_{iij}/2)
        self.assertAllClose(A[1, 1], 1.0)

    def test_symmetric_for_symmetric_kappa(self):
        r"""
        **Description:**
        For fully symmetric :math:`\kappa_{ijk}`, the :math:`a`-matrix is
        symmetric: :math:`a_{ij} = a_{ji}`.

        When :math:`\kappa` is fully symmetric (physical case), both branches
        of the formula give the same result since
        :math:`\kappa_{iij} = \kappa_{iji} = \kappa_{jii}`.
        """
        kappa = np.zeros((3, 3, 3))
        # Fill symmetrically
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    key = tuple(sorted([i, j, k]))
                    kappa[i, j, k] = hash(key) % 10

        A = compute_a_matrix(kappa)
        # a-matrix must be symmetric for symmetric kappa
        self.assertAllClose(A, A.T, atol=1e-14)

    def test_zero_kappa(self):
        r"""
        **Description:**
        Verify that zero intersection numbers give a zero :math:`a`-matrix.
        """
        A = compute_a_matrix(np.zeros((3, 3, 3)))
        # All entries must be zero
        self.assertAllClose(A, np.zeros((3, 3)))


if __name__ == "__main__":
    import unittest
    unittest.main()
