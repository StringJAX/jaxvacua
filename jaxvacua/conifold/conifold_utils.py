# ==============================================================================
# This code is written by Andreas Schachner.
#
# If any questions arise, please feel free to reach out to me (Andreas) either at
# andreas.schachner@gmx.net or at as3475@cornell.edu .
# ==============================================================================
"""``jaxvacua.conifold.conifold_utils`` — structural helpers for the conifold subsystem.

Contents:
    - lattice / basis algebra: ``get_basis_change``, ``getAMatrix``, ``get_projection``
    - flux & index helpers (attached to ``periods`` / ``FluxEFT``):
      ``conifold_fluxes``, ``delete_coni_index``
    - re-exports of general-purpose number theory: ``extended_euclidean``,
      ``orthogonal_lattice`` (live in :mod:`jaxvacua.util`).

These are the *structural* helpers of the conifold subsystem — anything that
operates on basis transformations, flux index splitting or topological
projection without invoking z_cf physics.  The flux-EFT solvers themselves
live in :mod:`jaxvacua.conifold.zcf_solver`.
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

    return np.array(matrix)


def getAMatrix(intnumstensor):
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


def get_projection(q):
    return jnp.asarray(extended_euclidean(q)[2][1:len(q)]).T


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
        q = self.lcs_tree.conifold.conifold_curve
        # Ensure projection is cached before the JIT-compiled solver tries to access it.
        w_proj = self.lcs_tree.conifold.projection

        M1 = f2[1:] @ q
        H1 = h2[1:] @ q
        P1 = f1[1:] @ q
        K1 = h1[1:] @ q

        Malpha = f2[1:] @ w_proj
        Halpha = h2[1:] @ w_proj
        Palpha = f1[1:] @ w_proj
        Kalpha = h1[1:] @ w_proj

    return M0, H0, M1, H1, Malpha, Halpha, P1, K1, Palpha, Kalpha, P0, K0


@jit
def delete_coni_index(self, x, indx):
    y = jnp.arange(len(x))
    flags = jnp.where(jnp.arange(len(x) - 1) < indx, y[:-1], y[1:])
    return x[flags]
