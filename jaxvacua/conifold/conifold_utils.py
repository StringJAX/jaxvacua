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

"""Structural helpers for the conifold subsystem.

Purpose
-------
Collect conifold-specific lattice, basis-change, projection, flux-splitting
and index-manipulation helpers that do not involve the dynamics of the
conifold modulus.

Main public API
---------------
- Lattice and basis algebra: ``get_basis_change``, ``compute_a_matrix``,
  ``get_embedding``, ``get_bulk_embedding`` and ``get_bulk_projection``.
- Flux and index helpers attached to ``periods`` or ``FluxEFT``:
  ``conifold_fluxes`` and ``delete_coni_index``.
- Compatibility re-exports of ``extended_euclidean`` and
  ``orthogonal_lattice`` from ``jaxvacua.util``.

Design notes
------------
This module is intentionally limited to structural operations.  The
``z_cf`` F-term, log-coefficient and bulk-EFT routines live in
``jaxvacua.conifold.zcf_solver``.
"""

from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import jit

# Re-export the general-purpose lattice helpers so callers that historically
# pulled them from ``jaxvacua.conifold_utils`` keep working when they switch
# to ``jaxvacua.conifold``.  The implementations live in :mod:`jaxvacua.util`.
from jaxvacua.util import extended_euclidean, orthogonal_lattice  # noqa: F401


# ---------------------------------------------------------------------------
# Conifold-specific lattice / basis algebra
# ---------------------------------------------------------------------------

def get_basis_change(coninop):
    r"""
    **Description:**
    Constructs the unimodular integer basis transformation that maps a
    conifold curve charge vector to the canonical form
    :math:`(1, 0, \ldots, 0)`.

    .. admonition:: Details
        :class: dropdown

        Given the integer charge vector :math:`q \in \mathbb{Z}^{h^{1,1}}`
        of the conifold curve (with :math:`\gcd(q) = 1`), the function
        builds the :math:`h^{1,1}\times h^{1,1}` unimodular matrix
        :math:`\Lambda \in \mathrm{GL}(h^{1,1}, \mathbb{Z})` satisfying

        .. math::
            \Lambda \, q = (1,\, 0,\,\ldots,\, 0)^T \,.

        :math:`\Lambda` is assembled from two complementary pieces:

        1. **First row** — the Bézout coefficients :math:`b` computed by
           :func:`extended_euclidean`, satisfying :math:`b \cdot q = 1`.
        2. **Remaining rows** — generators of the lattice
           :math:`\{v : v \cdot q = 0\}`, computed by
           :func:`orthogonal_lattice`.

        The resulting :math:`\Lambda` is passed to :class:`lcs_tree` to
        change the complex-structure-moduli basis so that :math:`z^1`
        directly parameterises the approach to the conifold singularity.

    Args:
        coninop (Array): Integer charge vector :math:`q \in \mathbb{Z}^n`
            of the conifold curve, satisfying :math:`\gcd(q) = 1`.

    Returns:
        np.ndarray: Unimodular integer matrix :math:`\Lambda` of shape
            :math:`(n, n)` satisfying :math:`\Lambda\, q = (1, 0,\ldots,0)^T`.

    See also: :func:`extended_euclidean`, :func:`orthogonal_lattice`
    """
    matrix = [extended_euclidean(coninop)[0].tolist()]

    matrix = matrix + orthogonal_lattice([coninop])

    matrix = np.array(matrix)

    # Guard the orientation: ensure Λ q = (+1, 0, …, 0), not (−1, 0, …, 0).
    # extended_euclidean uses a positive-gcd convention so this is normally a
    # no-op, but flip the Bézout row if a future change yields the −1 sign,
    # which would otherwise sign-flip the conifold modulus / embedding.
    if int(np.asarray(coninop) @ matrix[0]) == -1:
        matrix[0] = -matrix[0]

    return matrix


def compute_a_matrix(intnumstensor):
    r"""
    **Description:**
    Computes the :math:`a`-matrix from the triple intersection number tensor.

    .. admonition:: Details
        :class: dropdown

        The :math:`a`-matrix appears in the polynomial part of the LCS
        prepotential

        .. math::
            F_{\rm poly}(z) = -\frac{1}{6}\,\kappa_{ijk}\,z^i z^j z^k
                + \frac{1}{2}\,a_{ij}\,z^i z^j + b_i\,z^i
                + \frac{i}{2}\,\tilde{\xi} \,,

        where :math:`\kappa_{ijk}` are the triple intersection numbers of
        the mirror Calabi-Yau.  The entries of :math:`a` are

        .. math::
            a_{ij} = \begin{cases}
                \kappa_{iij}/2 & i \geq j \\
                \kappa_{ijj}/2 & i < j
            \end{cases} \,.

        This convention is used for the coniLCS, coniLCS_series, and
        coniLCS_bulk limits in :meth:`jaxvacua.lcs.lcs_tree._prepare_prepot`.

    Args:
        intnumstensor (Array): Triple intersection number tensor
            :math:`\kappa_{ijk}` of shape
            :math:`(h^{1,2}, h^{1,2}, h^{1,2})`.

    Returns:
        np.ndarray: The :math:`a`-matrix of shape
        :math:`(h^{1,2}, h^{1,2})`.

    See also: :class:`jaxvacua.lcs.lcs_tree`
    """

    h11 = len(intnumstensor)
    return np.array([[intnumstensor[i][j][j]/2 if i<j else intnumstensor[i][i][j]/2 for j in range(h11)] for i in range(h11)])


def get_bulk_embedding(q):
    r"""
    **Description:**
    Bulk **embedding** matrix for the conifold charge :math:`q`: the bulk rows
    :math:`\Lambda_{1:}` of the basis change :math:`\Lambda` (from
    :func:`get_basis_change`), transposed to shape
    :math:`(h^{1,2}, h^{1,2}-1)`.

    .. admonition:: Details
        :class: dropdown

        Used to build a full :math:`h^{1,2}`-vector from its
        :math:`(h^{1,2}-1)` bulk components.  It serves three roles, all with
        the *same* matrix because charges are covectors and moduli are vectors
        of the same lattice:

        1. embed bulk moduli/periods: ``X_bulk -> bulk_embedding @ X_bulk``,
           which carries zero conifold component (:math:`q\cdot\Lambda_{1:}^T = 0`);
        2. contract the intersection/``a``/``b`` tensors along bulk directions;
        3. project a charge/flux covector onto the bulk:
           ``charge @ bulk_embedding`` :math:`= \Lambda_{1:}\, \tilde q`.

        Together with the conifold embedding :func:`get_embedding`
        (:math:`e_q = \Lambda_0`, the conifold direction of a vector) and the
        conifold charge :math:`q` it satisfies the resolution of identity
        :math:`q\, e_q^T + \text{bulk\_proj}\cdot\text{bulk\_embedding}^T = \mathbb{1}`,
        where the matching bulk **projection** (the left inverse,
        :math:`\text{bulk\_embedding}^T\,\text{bulk\_proj} = \mathbb{1}_{h^{1,2}-1}`)
        is :func:`get_bulk_projection`.  In the conifold-aligned basis
        (:math:`q=(1,0,\ldots,0)`) it reduces to :math:`[0;\mathbb{1}]`.

    Args:
        q (Array): Integer conifold charge vector of length :math:`h^{1,2}`.

    Returns:
        Array: Bulk embedding matrix of shape :math:`(h^{1,2}, h^{1,2}-1)`.

    See also: :func:`get_bulk_projection`, :func:`get_embedding`, :func:`get_basis_change`
    """
    return jnp.asarray(np.asarray(get_basis_change(q))[1:]).T


def get_bulk_projection(q):
    r"""
    **Description:**
    Bulk **projection** matrix for the conifold charge :math:`q`: the bulk
    columns :math:`(\Lambda^{-1})_{:,1:}` of the inverse basis change, shape
    :math:`(h^{1,2}, h^{1,2}-1)`.

    .. admonition:: Details
        :class: dropdown

        Extracts the :math:`(h^{1,2}-1)` bulk components of a modulus/period
        **vector** ``v`` via ``v @ bulk_projection``
        (:math:`= (\Lambda^{-T} v)_{1:}`).  It is the left inverse of the bulk
        embedding (:func:`get_bulk_embedding`),
        :math:`\text{bulk\_embedding}^T\,\text{bulk\_projection} = \mathbb{1}_{h^{1,2}-1}`,
        so the "extract bulk then re-embed" round-trip is exact.  This differs
        from :func:`get_bulk_embedding` in a general basis (projection vs
        embedding of a vector); the two coincide only in the conifold-aligned
        basis, where both reduce to :math:`[0;\mathbb{1}]`.  :math:`\Lambda` is
        unimodular, so :math:`\Lambda^{-1}` is integer.

    Args:
        q (Array): Integer conifold charge vector of length :math:`h^{1,2}`.

    Returns:
        Array: Bulk projection matrix of shape :math:`(h^{1,2}, h^{1,2}-1)`.

    See also: :func:`get_bulk_embedding`, :func:`get_embedding`, :func:`get_basis_change`
    """
    basis_change = np.asarray(get_basis_change(q))
    basis_change_inv = np.rint(np.linalg.inv(basis_change)).astype(int)
    return jnp.asarray(basis_change_inv[:, 1:])


def get_embedding(q):
    r"""
    **Description:**
    Conifold embedding direction :math:`e_q` for the charge :math:`q`: the
    Bézout vector (first row of :math:`\Lambda` from
    :func:`extended_euclidean`), satisfying :math:`q \cdot e_q = 1`.

    Gives the conifold direction of a **vector** (modulus/period): a full vector
    is reconstructed as
    :math:`z_{\rm full} = z_{\rm cf}\, e_q + \text{bulk\_embedding}\, z_{\rm bulk}`,
    which equals :math:`\Lambda^T (z_{\rm cf}, z_{\rm bulk})` and so inverts the
    basis change consistently with :func:`get_bulk_embedding` /
    :func:`get_bulk_projection`.  In the conifold-aligned basis
    (:math:`q = (1,0,\ldots,0)`) this reduces to :math:`(1,0,\ldots,0)`.

    Args:
        q (Array): Integer conifold charge vector of length :math:`h^{1,2}`.

    Returns:
        Array: Embedding vector of length :math:`h^{1,2}` with :math:`q\cdot e_q = 1`.

    See also: :func:`get_bulk_embedding`, :func:`get_bulk_projection`, :func:`get_basis_change`
    """
    return jnp.asarray(extended_euclidean(q)[0])


# ---------------------------------------------------------------------------
# Flux splitting + index manipulation (attached to ``periods`` / ``FluxEFT``)
# ---------------------------------------------------------------------------

#@jit
def conifold_fluxes(self, flux):
    f1, f2, h1, h2 = self._split_fluxes(flux)

    M0 = f2[0]
    H0 = h2[0]
    P0 = f1[0]
    K0 = h1[0]

    if self.lcs_tree.conifold_basis:

        M1 = f2[1]
        H1 = h2[1]
        P1 = f1[1]
        K1 = h1[1]
        Malpha = f2[2:]
        Halpha = h2[2:]
        Palpha = f1[2:]
        Kalpha = h1[2:]

    else:
        # The flux blocks split by how they pair in W = f1·X − f2·𝓕:
        #   • f1, h1 pair with the electric periods X^I ⇒ COVARIANT, like
        #     charges (transform with Λ): conifold component = e_q·f = (Λf)[0]
        #     (e_q = Λ[0], q·e_q = 1); bulk = f @ bulk_embedding (= Λ[1:]ᵀ).
        #   • f2, h2 pair with the magnetic periods 𝓕_I ⇒ CONTRAVARIANT, like
        #     moduli (transform with Λ⁻ᵀ): conifold component = q·f = (Λ⁻ᵀf)[0]
        #     (q = conifold_curve = Λ⁻¹[:,0]); bulk = f @ bulk_projection
        #     (= Λ⁻¹[:,1:]).
        # All four reduce to the aligned index-0 / [1:] slices when q=e_q=(1,0,…)
        # and bulk_embedding=bulk_projection=[0;I].
        e_q             = self.lcs_tree.conifold.embedding
        q               = self.lcs_tree.conifold.conifold_curve
        bulk_embedding  = self.lcs_tree.conifold.bulk_embedding
        bulk_projection = self.lcs_tree.conifold.bulk_projection

        # f2, h2: contravariant (magnetic) ⇒ q / bulk_projection
        M1 = f2[1:] @ q
        H1 = h2[1:] @ q
        Malpha = f2[1:] @ bulk_projection
        Halpha = h2[1:] @ bulk_projection

        # f1, h1: covariant (electric) ⇒ e_q / bulk_embedding
        P1 = f1[1:] @ e_q
        K1 = h1[1:] @ e_q
        Palpha = f1[1:] @ bulk_embedding
        Kalpha = h1[1:] @ bulk_embedding

    return M0, H0, M1, H1, Malpha, Halpha, P1, K1, Palpha, Kalpha, P0, K0


@jit
def delete_coni_index(self, x, indx):
    y = jnp.arange(len(x))
    flags = jnp.where(jnp.arange(len(x) - 1) < indx, y[:-1], y[1:])
    return x[flags]
