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

"""coniLCS prepotential functions.

Purpose
-------
Provide the prepotential family used near conifold large-complex-structure
limits, split into period-coordinate and affine-modulus-coordinate layers.

Main public API
---------------
- Per-period functions attached to ``periods``: ``F_coniLCS_bulk_per``,
  ``F_coniLCS_poly_split_per``, ``F_coni_per``, ``F_inst_per_coni``,
  ``F_coniLCS_exp_per`` and ``F_coniLCS_series_per``.
- First through fourth derivatives of the per-period polynomial part.
- Per-modulus functions attached to ``css``: ``F_coniLCS_bulk``,
  ``F_coniLCS_exp``, ``dF_coniLCS_exp`` and ``F_coniLCS_series``.

Design notes
------------
The functions are pure prepotential calculations and have no flux dependence.
They are defined as plain functions taking ``self`` first so consumer modules
can attach them to the relevant classes.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from jax import Array
from jax.numpy import pi as Pi

from jaxpolylog import jax_polylog_vmap


# ===========================================================================
# §2 — per-period (X-coordinate) family attached to ``periods``
# ===========================================================================

def F_coniLCS_bulk_per(self, XPer: Array, conj: bool = False) -> complex:
    r"""
    **Description:**
    Computes the *bulk approximation* to the coniLCS prepotential.

    .. admonition:: Details
        :class: dropdown

        In the ``coniLCS_bulk`` limit the conifold modulus :math:`X^{\mathrm{cf}}` is
        integrated out at leading order, leaving an effective correction to the polynomial
        part of the prepotential.  The leading effect is a shift of the linear coefficient
        :math:`b_0 \to b_0 + n_{\mathrm{cf}}/24`, which translates to an additive term

        .. math::

            F_{\mathrm{bulk}}(X) = F_{\mathrm{LCS}}(X)
                + \frac{n_{\mathrm{cf}}}{24}\,(q^{\mathrm{cf}}_i\, X^i)\, X^0 \,,

        where :math:`q^{\mathrm{cf}}_i` is the charge vector of the conifold curve
        (stored in ``lcs_tree.conifold.conifold_curve0``) and :math:`n_{\mathrm{cf}}` is its GV invariant.
        This is the same shift applied to ``b_vector[0]`` in :func:`jaxvacua.flux_utils.pfv_to_flux`
        for the ``coniLCS`` limit.

    Args:
        XPer (Array): Period vector :math:`(X^0, X^1, \ldots, X^{h^{1,2}})`.
        conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

    Returns:
        complex: Value of the bulk-approximated coniLCS prepotential.

    See also: :func:`F_LCS_per`, :func:`F_coniLCS_series_per`
    """

    return self.F_LCS_per(XPer,conj=conj)+self.lcs_tree.conifold.ncf*(self.lcs_tree.conifold.conifold_curve0@XPer[1:])*XPer[0]/24.



@partial(jit, static_argnums = (4,))
def F_coniLCS_poly_split_per(self, X0: Array, Xcf: Array, Xbulk: Array, conj: bool = False) -> complex:
    r"""
    **Description:**
    Dummy function to compute the polynomial part :math:`F_{\mathrm{poly}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
    in terms of the periods, where the periods are split into conifold and bulk parts.

    Args:
        X0 (Array): Value of period :math:`X^0`.
        Xcf (Array): Value of conifold period :math:`X^{\mathrm{cf}}`.
        Xbulk (Array): Values of bulk periods :math:`X^i`.
        conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

    Returns:
        complex: Value of the polynomial contribution :math:`F_{\mathrm{poly}}` to the LCS prepotential :math:`F_{\mathrm{LCS}}`.

    """
    # Reconstruct the full period vector from (X^0, X^cf, X^bulk).  Aligned basis:
    # conifold at index 1 -> [X0, Xcf, *Xbulk].  General basis: embed the bulk
    # via X_mod = Xcf·e_q + bulk_embedding·Xbulk (conifold direction e_q = Λ[0],
    # bulk embedding = Λ[1:]ᵀ), so ∂/∂Xcf correctly picks up the conifold
    # direction for the derivatives and the round-trip with the bulk projection
    # (Λ⁻¹[:,1:]) is exact.
    if self.lcs_tree.conifold_basis:
        XPer = jnp.append(jnp.array([X0, Xcf]), Xbulk)
    else:
        e_q            = jnp.asarray(self.lcs_tree.conifold.embedding,      dtype=Xbulk.dtype)
        bulk_embedding = jnp.asarray(self.lcs_tree.conifold.bulk_embedding, dtype=Xbulk.dtype)
        X_mod  = Xcf * e_q + bulk_embedding @ Xbulk
        XPer   = jnp.append(jnp.array([X0]), X_mod)

    return self.F_LCS_poly_per(XPer,conj=conj)


@partial(jit, static_argnums = (4,))
def dF_coniLCS_poly_per(self, X0: Array, Xcf: Array, Xbulk: Array, conj: bool = False) -> complex:
    r"""
    **Description:**
    Computes the first derivative of the polynomial part :math:`F_{\mathrm{poly}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
    in terms of the periods.

    Args:
        X0 (Array): Value of period :math:`X^0`.
        Xcf (Array): Value of conifold period :math:`X^{\mathrm{cf}}`.
        Xbulk (Array): Values of bulk periods :math:`X^i`.
        conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

    Returns:
        complex: Value of the first derivative of the polynomial contribution :math:`F_{\mathrm{poly}}` to the LCS prepotential :math:`F_{\mathrm{LCS}}`.

    See also: :func:`F_coniLCS_series_per`
    """

    return jax.grad(self.F_coniLCS_poly_split_per,holomorphic=True,argnums=1)(X0,Xcf,Xbulk,conj=conj)

@partial(jit, static_argnums = (4,))
def ddF_coniLCS_poly_per(self, X0: Array, Xcf: Array, Xbulk: Array, conj: bool = False) -> complex:
    r"""
    **Description:**
    Computes the second derivative of the polynomial part :math:`F_{\mathrm{poly}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
    in terms of the periods.

    Args:
        X0 (Array): Value of period :math:`X^0`.
        Xcf (Array): Value of conifold period :math:`X^{\mathrm{cf}}`.
        Xbulk (Array): Values of bulk periods :math:`X^i`.
        conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

    Returns:
        complex: Value of the second derivative of the polynomial contribution :math:`F_{\mathrm{poly}}` to the LCS prepotential :math:`F_{\mathrm{LCS}}`.

    See also: :func:`F_coniLCS_series_per`
    """

    return jax.grad(self.dF_coniLCS_poly_per,holomorphic=True,argnums=1)(X0,Xcf,Xbulk,conj=conj)

@partial(jit, static_argnums = (4,))
def dddF_coniLCS_poly_per(self, X0: Array, Xcf: Array, Xbulk: Array, conj: bool = False) -> complex:
    r"""
    **Description:**
    Computes the third derivative of the polynomial part :math:`F_{\mathrm{poly}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
    in terms of the periods.

    Args:
        X0 (Array): Value of period :math:`X^0`.
        Xcf (Array): Value of conifold period :math:`X^{\mathrm{cf}}`.
        Xbulk (Array): Values of bulk periods :math:`X^i`.
        conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

    Returns:
        complex: Value of the third derivative of the polynomial contribution :math:`F_{\mathrm{poly}}` to the LCS prepotential :math:`F_{\mathrm{LCS}}`.

    See also: :func:`F_coniLCS_series_per`
    """

    return jax.grad(self.ddF_coniLCS_poly_per,holomorphic=True,argnums=1)(X0,Xcf,Xbulk,conj=conj)


@partial(jit, static_argnums = (4,))
def ddddF_coniLCS_poly_per(self, X0: Array, Xcf: Array, Xbulk: Array, conj: bool = False) -> complex:
    r"""
    **Description:**
    Computes the fourth derivative of the polynomial part :math:`F_{\mathrm{poly}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
    in terms of the periods.

    Args:
        X0 (Array): Value of period :math:`X^0`.
        Xcf (Array): Value of conifold period :math:`X^{\mathrm{cf}}`.
        Xbulk (Array): Values of bulk periods :math:`X^i`.
        conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

    Returns:
        complex: Value of the fourth derivative of the polynomial contribution :math:`F_{\mathrm{poly}}` to the LCS prepotential :math:`F_{\mathrm{LCS}}`.

    See also: :func:`F_coniLCS_series_per`
    """

    # This function should vanish!
    return jax.grad(self.dddF_coniLCS_poly_per,holomorphic=True,argnums=1)(X0,Xcf,Xbulk,conj=conj)


@partial(jit, static_argnums = (3,4,))
def F_inst_per_coni(self, X0: complex, XPerBulk: Array, conj: bool = False, n: int = 0) -> complex:
    r"""
    **Description:**
    Computes the instanton part :math:`F_{\mathrm{inst}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
    in terms of the periods.

    .. admonition:: Details
        :class: dropdown

        The instanton part :math:`F_{\mathrm{inst}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
        can be expressed in terms of the periods :math:`X^I=(X^0,X^i)` as

        .. math::
            F_{\mathrm{inst}}(X) = -\frac{(X^0)^2}{(2\pi\mathrm{i})^3}\, \sum_{q\in\mathcal{M}(\widetilde{X})}\,
            n_q^{0}\, \text{Li}_3\left (\text{e}^{2\pi \text{i} q_i X^i / X^0}\right )\; , \quad
            \text{Li}_3\left (x\right )=\sum_{m=1}^{\infty}\, \dfrac{x^{m}}{m^{3}}\, .

        Here the sum is performed over all effective curve classes :math:`q\in\mathcal{M}(\widetilde{X})`
        in the Mori cone :math:`\mathcal{M}(\widetilde{X})` of the mirror dual manifold :math:`\widetilde{X}`.
        Here, the :math:`n_q^{0}` are the genus-0 Gopakumar-Vafa (GV) invariants which can be computed
        systematically using methods described in `hep-th/9308122 <https://arxiv.org/pdf/hep-th/9308122.pdf>`_.

        The infinite sum appearing in the poly-logarithm :math:`\text{Li}_3` can be rewritten to arrive at

        .. math::
            \sum_{q\in\mathcal{M}(\widetilde{X})}\, n_q^{0}\, \text{Li}_3\left (\text{e}^{2\pi \text{i} q_i X^i / X^0}\right )
            = \sum_{q\in\mathcal{M}(\widetilde{X})}\, N_q\, \text{e}^{2\pi \text{i} q_i X^i / X^0}

        in terms of genus-0 Gromov-Witten (GW) invariants :math:`N_q`. We typically work with the latter to simplify the calculation.
        The relation between the two types of invariants is more explicitly given by

        .. math::
            N_q = \sum_{d|q}\, \dfrac{1}{d^3}\, n_{q/d}^{0}\, .

        Here, the sum runs over all divisors :math:`d` of the curve class :math:`q`.
        The :math:`N_q` are typically rational numbers, while the :math:`n_q^{0}` are integers.
        The curve classes :math:`q` are specified in a basis of the Mori cone :math:`\mathcal{M}(\widetilde{X})`
        of the mirror dual Calabi-Yau threefold :math:`\widetilde{X}`.
        The Mori cone is dual to the Kähler cone of :math:`\widetilde{X}`.
        The curve classes :math:`q` can be expressed in terms of the generators of the Mori cone.
        The curve classes are also referred to as curve charges in the following.
        The curve charges are stored in the attribute ``GW_charges`` (``GV_charges``) for the GW (GV) invariants.
        The corresponding invariants are stored in the attributes ``GW_inv`` (``GV_inv``).
        The curve charges and invariants are limited to a certain maximum degree :math:`d=\text{max_deg}` in the attributes
        ``GW_charges_lim``, ``GV_charges_lim``, ``GW_inv_lim``, ``GV_inv_lim``.
        The maximum degree can be specified when initialising the class.
        The maximum degree is defined with respect to a grading vector which can be specified when initialising
        the class. If no grading vector is provided, the default grading vector :math:`(1,1,\ldots,1)` is used.
        The maximum available degree for the instanton data is stored in the attribute ``max_available_deg``.
        If the specified maximum degree exceeds the maximum available degree, a warning is raised and
        the maximum available degree is used instead.

    .. admonition:: Note
        :class: dropdown

        The sum over curve classes is truncated at a certain maximum degree :math:`d=\text{max_deg}` in our implementation.
        This is justified since the instanton contributions are exponentially suppressed at large complex structure.
        The maximum degree can be specified when initialising the class.


    Args:
        X0 (complex): Value of the period :math:`X^0`.
        XPerBulk (Array): Values of bulk periods :math:`X^i` (excluding the conifold period).
        conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        n (int, optional): Derivative order with respect to the conifold period. Defaults to ``0``.

    Returns:
        complex: Value of the instanton part :math:`F_{\mathrm{inst}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`.

    See also: :func:`F_coniLCS_series_per`

    """

    if self.include_mirror_wsi:
        if self.limit in ["coniLCS_series","coniLCS_bulk"]:
            bulk_invs = self.lcs_tree.gv_invariants
            bulk_charges = self.lcs_tree.gv_charges
        else:
            bulk_invs = self.delete_coni_index(self.lcs_tree.gv_invariants, self.lcs_tree.coni_index)
            bulk_charges = self.delete_coni_index(self.lcs_tree.gv_charges, self.lcs_tree.coni_index)

    # Reconstruct the full period at the conifold locus (X^cf = 0).  Aligned:
    # [X0, 0, *XPerBulk].  General: embed the bulk periods via bulk_embedding
    # (= Λ[1:]ᵀ; conifold component is zero), and identify the conifold-direction
    # charge (q̃₁) as bulk_charges·e_q (the embedding, q·e_q=1) instead of the
    # index-0 column.
    if self.lcs_tree.conifold_basis:
        XPer  = jnp.append(X0, jnp.append(0.+1j*0., XPerBulk))
        beta1 = bulk_charges[:, 0]
    else:
        bulk_embedding = jnp.asarray(self.lcs_tree.conifold.bulk_embedding, dtype=XPerBulk.dtype)
        e_q            = jnp.asarray(self.lcs_tree.conifold.embedding,       dtype=bulk_charges.dtype)
        XPer   = jnp.append(X0, bulk_embedding @ XPerBulk)
        beta1  = bulk_charges @ e_q

    coeff = 2.*Pi*1j

    if conj:
        coeff = -coeff

    # Compute exponentiated argument of polylog
    t = jnp.matmul(bulk_charges,XPer[1:])
    t = t/XPer[0]
    z = jnp.exp(coeff*t)

    # Compute polylog
    Li_val = jax_polylog_vmap(z,3-n,self.prange,approx="patch")

    # Compute sum over curves
    res = jnp.matmul(bulk_invs*(beta1**n),Li_val)

    # Add numerical prefactor
    return -res/(coeff)**(3-n)*XPer[0]**(2-n)


@partial(jit, static_argnums = (2,))
def F_coni_per(self, X: Array, conj: bool = False) -> complex:
    r"""
    **Description:**
    Computes the conifold part :math:`F_{\mathrm{conifold}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
    in terms of the periods.

    .. admonition:: Details
        :class: dropdown

        The conifold part :math:`F_{\mathrm{conifold}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
        can be expressed in terms of the periods :math:`X^I=(X^0,X^i)` as

        .. math::
            F_{\mathrm{conifold}}(X) = \frac{n_{\mathrm{cf}}}{(2\pi\mathrm{i})^2}\, (X^{\mathrm{cf}})^2\, \log\left (\frac{X^{\mathrm{cf}}}{X^0}\right )\, .

        Here, :math:`X^{\mathrm{cf}}` is the period associated to the conifold curve.
        The conifold period can be identified as the period which vanishes at the conifold locus.
        The conifold period can be obtained from the other periods by applying a suitable basis transformation.
        The number of conifolds is denoted by :math:`n_{\mathrm{cf}}`.
        The conifold curve can be specified when initialising the class.
        The conifold curve is stored in the attribute ``conifold_curve``.
        The conifold curve is expressed in the basis of the Mori cone generators of the mirror dual Calabi-Yau threefold :math:`\widetilde{X}`.
        The conifold curve can also be specified indirectly by providing a basis transformation.
        The basis transformation is applied to the periods such that the first period corresponds to the conifold period: :math:`X^1 = X^{\mathrm{cf}}`.
        The basis transformation can be specified when initialising the class.
        If no basis transformation is provided, the identity is assumed.
        The basis transformation is stored in the attribute ``basis_change``.

    Args:
        X (Array): Values of periods.
        conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

    Returns:
        complex: Value of the conifold part :math:`F_{\mathrm{conifold}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`.

    """
    # Conifold period X^cf: the period selected by the conifold charge.  In the
    # aligned basis (conifold_curve0 = (1,0,…,0)) this is X[1]; in a general basis
    # it is the contraction conifold_curve · X[1:].  Both reduce identically when
    # qcf = (1,0,…,0).
    qcf = (self.lcs_tree.conifold.conifold_curve0 if self.lcs_tree.conifold_basis
           else self.lcs_tree.conifold.conifold_curve)
    Xcf = jnp.asarray(qcf, dtype=X.dtype) @ X[1:]
    if conj:
        return self.lcs_tree.conifold.ncf*(Xcf)**2/(-4*jnp.pi*1j)*jnp.log(2*jnp.pi*1j*Xcf/X[0])
    else:
        return self.lcs_tree.conifold.ncf*(Xcf)**2/(4*jnp.pi*1j)*jnp.log(-2*jnp.pi*1j*Xcf/X[0])


@partial(jit, static_argnums = (4,5,))
def F_coniLCS_exp_per(self,
                    X0: complex,
                    XConi: complex,
                    XPerBulk: Array,
                    conj: bool = False,
                    n: int = 0
                    ) -> complex:
    r"""
    **Description:**
    Computes the expansion of the prepotential :math:`F` around the conifold point in terms of the periods.

    .. admonition:: Details
        :class: dropdown

        The prepotential :math:`F` can be expanded around the conifold point as

        .. math::
            F(X) = \sum_{n=0}^{\infty}\, \dfrac{1}{n!}\, \dfrac{\partial^n F}{\partial (X^{\mathrm{cf}})^n}\Bigg |_{X^{\mathrm{cf}}=0}\, (X^{\mathrm{cf}})^n\, .

        Here, :math:`X^{\mathrm{cf}}` is the period associated to the conifold curve.

        This function returns the **coefficient** of :math:`(X^{\mathrm{cf}})^n` in this
        expansion, i.e. it computes
        :math:`\tfrac{1}{n!}\,\partial^n_{\!X^{\mathrm{cf}}} F\big|_{X^{\mathrm{cf}}=0}`.
        The full coniLCS prepotential is reconstructed in :func:`F_coniLCS_series_per`
        by summing these coefficients up to order ``nmax``.

        The individual contributions are:

        * **Polynomial part** — obtained from :func:`F_coniLCS_poly_split_per` and
            its derivatives :func:`dF_coniLCS_poly_per`, :func:`ddF_coniLCS_poly_per`,
            :func:`dddF_coniLCS_poly_per` evaluated at :math:`X^{\mathrm{cf}}=0`.
        * **Conifold part** — the dilogarithm :math:`F_{\mathrm{coni}}` contributes
            Riemann zeta values at integer arguments:
            :math:`\zeta(3)` at :math:`n=0`, :math:`\zeta(2)=\pi^2/6` at :math:`n=1`,
            :math:`3/2` at :math:`n=2`, and the Bernoulli numbers :math:`B_{n-2}` for
            :math:`n \ge 3`.
        * **Worldsheet instantons** — added via :func:`F_inst_per_coni` when
            ``maximum_degree > 0``.

    Args:
        X0 (complex): Value of period :math:`X^0`.
        XConi (complex): Value of conifold period :math:`X^{\mathrm{cf}}`.
        XPerBulk (Array): Values of bulk periods :math:`X^i`.
        conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        n (int, optional): Order of derivative. Defaults to ``0``.

    Returns:
        complex: Value of the :math:`n`-th derivative of the prepotential :math:`F` around the conifold point.
    """


    # ----------------------------------------
    # Determine normalization for zeta contributions
    # ----------------------------------------
    # The conifold contribution involves powers of (2πi).
    # Complex conjugation flips the sign of i → -i.
    zeta_denom = 2 * jnp.pi * 1j
    if conj:
        zeta_denom = (-zeta_denom)

    # Raise to correct power depending on derivative order
    # Exception: for n = 3, the normalization simplifies to 1
    if n != 3:
        zeta_denom = zeta_denom ** (3 - n)
    else:
        zeta_denom = 1.

    # ----------------------------------------
    # Overall conifold coefficient
    # ----------------------------------------
    # ncf = number of conifold divisors
    # X0^(2-n) reflects homogeneity of the prepotential
    coeff_cf = self.lcs_tree.conifold.ncf / zeta_denom * X0 ** (2 - n)

    # We expand around the conifold point Xcf = 0
    Xcf = 0. + 1j * 0.

    # ----------------------------------------
    # Polynomial + conifold contributions
    # ----------------------------------------
    # Each n corresponds to a derivative of the prepotential:
    #
    # n = 0 → F
    # n = 1 → dF
    # n = 2 → ddF
    # n = 3 → dddF
    #
    # The conifold piece contributes special constants:
    #   ζ(3), ζ(2), 3/2, Bernoulli numbers

    if n == 0:
        # Zeroth-order term (value of F at conifold point)
        val = (
            self.F_coniLCS_poly_split_per(X0, Xcf, XPerBulk, conj=conj)
            - coeff_cf * jax.scipy.special.zeta(3, q=1)
        )

    elif n == 1:
        # First derivative
        val = (
            self.dF_coniLCS_poly_per(X0, Xcf, XPerBulk, conj=conj)
            - coeff_cf * jax.scipy.special.zeta(2, q=1)
        )

    elif n == 2:
        # Second derivative
        val = (
            self.ddF_coniLCS_poly_per(X0, Xcf, XPerBulk, conj=conj)
            - coeff_cf * 3 / 2
        )

    elif n == 3:
        # Third derivative
        # Uses ζ(0) = -1/2 and B₁ = +1/2
        val = (
            self.dddF_coniLCS_poly_per(X0, Xcf, XPerBulk, conj=conj)
            + coeff_cf / 2
        )

    elif n >= 4:
        # Higher derivatives:
        # Controlled by Bernoulli numbers B_{n-2}
        val = (
            coeff_cf *
            jax.scipy.special.bernoulli(n - 2)[-1] /
            (n - 2)
        )



    if self.include_mirror_wsi:
        val = val + self.F_inst_per_coni(X0,XPerBulk,conj=conj,n=n)

    if n>0:
        val = val*(XConi)**(n)

    return val



@partial(jit, static_argnums = (2,))
def F_coniLCS_series_per(self, XPer: Array, conj: bool = False) -> complex:
    r"""
    **Description:**
    Computes the coniLCS prepotential as a Taylor series in the conifold period
    :math:`X^{\mathrm{cf}}` up to order ``nmax``.

    .. admonition:: Details
        :class: dropdown

        Near the conifold locus :math:`X^{\mathrm{cf}} \to 0`, the prepotential takes
        the form

        .. math::
            F_{\mathrm{coniLCS}}(X) = F_{\mathrm{coni}}(X^0, X^{\mathrm{cf}})
                + \sum_{n=0}^{n_{\mathrm{max}}} c_n\, (X^{\mathrm{cf}})^n \,,

        where :math:`F_{\mathrm{coni}}` is the logarithmic conifold contribution

        .. math::
            F_{\mathrm{coni}}(X^0, X^{\mathrm{cf}}) = \frac{n_{\mathrm{cf}}}{4\pi\mathrm{i}}\,
                (X^{\mathrm{cf}})^2\, \log\!\left(-\frac{2\pi\mathrm{i}\, X^{\mathrm{cf}}}{X^0}\right)

        and the coefficients :math:`c_n` are the Taylor coefficients of the remaining
        (polynomial + instanton) part of the prepotential around :math:`X^{\mathrm{cf}}=0`,
        computed by :func:`F_coniLCS_exp_per`.

        Worldsheet instanton contributions are included in the coefficients :math:`c_n`
        when ``maximum_degree > 0`` (via :func:`F_inst_per_coni`).

        This expansion replaces the full dilogarithm resummation of the standard
        ``coniLCS`` prepotential with a finite polynomial series, providing a simpler
        analytic handle on the vacuum structure at small :math:`|X^{\mathrm{cf}}|`.
        The accuracy improves as ``nmax`` increases.

    Args:
        XPer (Array): Period vector :math:`(X^0, X^{\mathrm{cf}}, X^2, \ldots, X^{h^{1,2}})`.
        conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

    Returns:
        complex: Value of the series-expanded coniLCS prepotential.

    See also: :func:`F_coni_per`, :func:`F_coniLCS_exp_per`, :func:`F_inst_per_coni`
    """

    # ----------------------------------------
    # Start with the exact conifold contribution
    # ----------------------------------------
    # This includes the logarithmic term:
    #   F_coni ~ (X_cf)^2 log(X_cf / X0)
    # and captures the singular behavior at the conifold point
    # F_coni_per is basis-aware (identifies X^cf via the conifold charge), so
    # pass the full period vector rather than the index-0/1 slice.
    val = self.F_coni_per(XPer, conj=conj)

    # ----------------------------------------
    # Extract periods (basis-aware)
    # ----------------------------------------
    # Aligned basis: conifold at index 1 -> plain slices (bit-identical).
    # General basis: identify the conifold period X^cf = q·X[1:] (conifold curve
    # q) and extract the bulk periods with the bulk PROJECTION (= Λ⁻¹[:,1:]),
    # the left inverse of the bulk embedding so the reconstruction round-trip in
    # F_coniLCS_exp_per is exact.  Both reduce to the slices when q=(1,0,…),
    # bulk_projection=[0;I].
    X0 = XPer[0]    # Fundamental period X^0
    if self.lcs_tree.conifold_basis:
        XConi    = XPer[1]    # Conifold period X_cf
        XPerBulk = XPer[2:]   # Bulk (non-conifold) periods
    else:
        qcf            = jnp.asarray(self.lcs_tree.conifold.conifold_curve,  dtype=XPer.dtype)
        bulk_projection = jnp.asarray(self.lcs_tree.conifold.bulk_projection, dtype=XPer.dtype)
        XConi    = qcf @ XPer[1:]
        XPerBulk = XPer[1:] @ bulk_projection


    # ----------------------------------------
    # Add Taylor expansion around X_cf = 0
    # ----------------------------------------
    # We sum:
    #   sum_{n=0}^{nmax} c_n (X_cf)^n
    #
    # where c_n = (1/n!) ∂^n F / ∂(X_cf)^n |_{X_cf=0}
    # computed by F_coniLCS_exp_per
    for n_exp in range(self.nmax + 1):

        # Add nth-order term in expansion
        tmp = self.F_coniLCS_exp_per(
            X0,           # X^0 (overall scaling period)
            XConi,        # NOTE: passed as Xcf-like argument (implementation detail)
            XPerBulk,     # Bulk periods
            conj=conj,
            n=n_exp       # Order of expansion
        )

        # ----------------------------------------
        # Divide by n! (Taylor expansion normalization)
        # ----------------------------------------
        # This converts derivatives into expansion coefficients
        if n_exp > 1:
            tmp = tmp / jax.scipy.special.gamma(n_exp + 1)

        val += tmp

    # ----------------------------------------
    # Return full prepotential
    # ----------------------------------------
    return val


# ===========================================================================
# §3 — per-modulus (z-coordinate) family attached to ``css``
# ===========================================================================

@partial(jit, static_argnums = (2,))
def F_coniLCS_bulk(self, moduli: Array, conj: bool = False) -> complex:
    r"""

    **Description:**
    Calculates the value of the LCS prepotential in terms of the complex structure moduli :math:`z^{i}`.


    .. admonition:: Details
        :class: dropdown

        We want to compute the effective bulk superpotential for coni-LCS limits.

        .. math::
            F_{\text{coni-LCS-bulk}}(z^1,\ldots , z^{h^{1,2}}) = F_{\text{LCS}}(z^1,\ldots , z^{h^{1,2}}) + \dfrac{n_{\text{cf}}\, z^1}{24}\, ,

        where we always work in a basis in which :math:`z^1 = z_{\text{cf}}`.
        At LCS, we can write the prepotential as

        .. math::
            F_{\text{LCS}}(z^1,\ldots , z^{h^{1,2}})=F_{\text{poly}}(z^1,\ldots , z^{h^{1,2}}) + F_{\text{inst}}(z^1,\ldots , z^{h^{1,2}})

        Here, the polynomial piece is given by

        .. math::
            F_{\text{poly}}(z^1,\ldots , z^{h^{1,2}})=\dfrac{1}{6}\kappa_{ijk}\, z^i z^j z^k+\dfrac{1}{2} a_{ij}z^i z^j+b_i\, z^i+\dfrac{\text{i}}{2}\tilde{\xi}\, ,

        Here, :math:`\widetilde{\kappa}_{ijk}` are the triple intersection numbers of
        the mirror dual Calabi-Yau threefold :math:`\widetilde{X}`.
        Further, we defined

        .. math::
            a_{ij} = \dfrac{1}{2}\begin{cases}
                                    \widetilde{\kappa}_{iij} & i\geq j\\[0.3em]
                                    \widetilde{\kappa}_{ijj} & i<j
                                \end{cases} \, , \quad
            b_i = \dfrac{1}{24} \int_{\tilde{D}^i}\, c_2(\widetilde{X})\, , \quad
            \tilde{\xi}=\frac{\zeta(3)\, \chi(\widetilde{X})}{(2\pi)^3}\, .

        The instanton contributions read

        .. math::
            F_{\mathrm{inst}}(z) = -\frac{1}{(2\pi\mathrm{i})^3}\, \sum_{q\in\mathcal{M}(\widetilde{X})}\,
            n_q^{0}\, \text{Li}_3\left (\text{e}^{2\pi \text{i} q_i z^i}\right )\; , \quad
            \text{Li}_3\left (x\right )=\sum_{m=1}^{\infty}\, \dfrac{x^{m}}{m^{3}}\, .

        in terms of genus-0 Gopakumar-Vafa (GV) invariants :math:`n_q^0` and the 3rd polylogarithm :math:`\text{Li}_3(x)`.
        Both pieces are computed in separate functions, see :func:`F_LCS_poly`
        and :func:`F_inst` for details.


    .. warning::
        The effective description for the bulk theory holds only provided that :math:`z^1 = z_{\text{cf}}\ll 1`.


    Args:
        moduli (Array): Complex structure moduli values.
        conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

    Returns:
        complex: Value of the LCS prepotential :math:`F_{\text{LCS}}`.

    See also: :func:`F_LCS_poly`, :func:`F_inst`

    """

    # The added term is ncf · z_cf / 24, with z_cf the conifold modulus = the
    # conifold-charge contraction of the moduli (moduli[0] in the aligned basis).
    qcf = (self.lcs_tree.conifold.conifold_curve0 if self.lcs_tree.conifold_basis
           else self.lcs_tree.conifold.conifold_curve)
    return self.F_LCS(moduli,conj=conj)+self.lcs_tree.conifold.ncf*jnp.asarray(qcf,dtype=moduli.dtype)@moduli/24.


@partial(jit, static_argnums = (2,3,))
def F_coniLCS_exp(self,
                  bulk_moduli: Array,
                  conj: bool = False,
                  n: int = 1
                  ) -> complex:
    r"""
    **Description:**
    Calculates the expansion coefficients of the prepotential :math:`F` around the conifold point in terms of the complex structure moduli :math:`z^i`.

    .. admonition:: Details
        :class: dropdown

        The prepotential :math:`F` can be expanded around the conifold point as
        :math:`F(z) = \sum_{n=0}^{\infty}\, \dfrac{1}{n!}\, \dfrac{\partial^n F}{\partial (z_{\mathrm{cf}})^n}\Bigg |_{z_{\mathrm{cf}}=0}\, (z_{\mathrm{cf}})^n\, .`
        This function computes the **coefficient** of :math:`(z_{\mathrm{cf}})^n` in this expansion, i.e. it computes
        :math:`\tfrac{1}{n!}\,\partial^n_{\!z_{\mathrm{cf}}} F\big|_{z_{\mathrm{cf}}=0}`.
        The full coniLCS prepotential is reconstructed in :func:`F_coniLCS_series` by summing these coefficients up to order ``nmax``.

    Args:
        bulk_moduli (Array): Bulk structure moduli values.
        conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        n_exp (int, optional): Order of expansion in the conifold modulus. Defaults to ``1``.

    Returns:
        complex: Value of the :math:`n`-th derivative of the prepotential :math:`F` around the conifold point.

    See also: :func:`F_coniLCS_series`
    """
    # ----------------------------------------
    # Extract periods (basis-aware)
    # ----------------------------------------
    # bulk_moduli are the (h12-1) bulk coordinates; embed into full moduli with
    # zero conifold component.  Aligned: prepend a zero (conifold at index 0) and
    # take XPer[2:] for the bulk.  General: EMBED the bulk moduli with
    # bulk_embedding (= Λ[1:]ᵀ) and EXTRACT the bulk periods with bulk_projection
    # (= Λ⁻¹[:,1:]) — distinct matrices in a general basis (their composition
    # bulk_embeddingᵀ·bulk_projection = I makes the round-trip exact).  Both
    # reduce to the slices when bulk_embedding=bulk_projection=[0;I].
    if self.lcs_tree.conifold_basis:
        moduli = jnp.append(jnp.zeros(1), bulk_moduli)
        XPer = self.moduli_to_periods(moduli,conj=conj)
        XPerBulk = XPer[2:]   # Bulk (non-conifold) periods
    else:
        bulk_embedding  = jnp.asarray(self.lcs_tree.conifold.bulk_embedding,  dtype=bulk_moduli.dtype)
        bulk_projection = jnp.asarray(self.lcs_tree.conifold.bulk_projection, dtype=bulk_moduli.dtype)
        moduli = bulk_embedding @ bulk_moduli
        XPer = self.moduli_to_periods(moduli,conj=conj)
        XPerBulk = XPer[1:] @ bulk_projection
    XConi    = 1. + 0*1j    # Conifold period X_cf (normalisation marker)
    X0       = XPer[0]    # Fundamental period X^0

    return self.periods.F_coniLCS_exp_per(X0, XConi, XPerBulk, conj=conj, n=n)


@partial(jit, static_argnums = (2,3,))
def dF_coniLCS_exp(self,
                  bulk_moduli: Array,
                  conj: bool = False,
                  n: int = 1
                  ) -> complex:
    r"""
    **Description:**
    Calculates the derivative of the conifold-LCS prepotential :math:`F_{\mathrm{coniLCS}}^{(n)}` with respect to the bulk moduli.

    Args:
        bulk_moduli (Array): Bulk structure moduli values.
        conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        n_exp (int, optional): Order of expansion in the conifold modulus. Defaults to ``1``.

    Returns:
        complex: Value of the derivative of the conifold-LCS prepotential :math:`F_{\mathrm{coniLCS}}^{(n)}` with respect to the bulk moduli.
    """

    return jax.grad(self.F_coniLCS_exp,holomorphic=True,argnums=0)(bulk_moduli,conj=conj,n=n)

@partial(jit, static_argnums = (2,))
def F_coniLCS_series(self,
                     moduli: Array,
                     conj: bool = False
                     ) -> complex:
    r"""
    **Description:**
    Calculates the full conifold-LCS prepotential :math:`F_{\mathrm{coniLCS}}` in terms of the complex structure moduli :math:`z^i`.


    .. admonition:: Details
        :class: dropdown

        At the conifold-LCS (coni-LCS) limit, the prepotential can be decomposed as

        .. math::
            F_{\mathrm{coni-LCS}}(z_{\mathrm{cf}}, z^a) = F_{\mathrm{coni}}(z_{\mathrm{cf}}) + \sum_{n=0}^{n_{\mathrm{max}}} F_{\mathrm{LCS}}^{(n)}(z^a) z_{\mathrm{cf}}^n

        where :math:`z_{\mathrm{cf}}` is the conifold modulus and :math:`z^a` are the bulk complex structure moduli.

        The conifold part :math:`F_{\mathrm{coni}}` encodes the singular behavior near the conifold point:

        .. math::
            F_{\mathrm{coni}}(z_{\mathrm{cf}}) = \dfrac{1}{2}\, n_{\mathrm{cf}}\, \left ( \dfrac{z_{\mathrm{cf}}^2}{2\pi \mathrm{i}}\, \log \left ( -2\pi \mathrm{i} z_{\mathrm{cf}} \right ) \right )

        The corrections :math:`F_{\mathrm{LCS}}^{(n)}` involve polynomials and polylogarithms and are summed over powers of :math:`z_{\mathrm{cf}}`.


    Args:
        moduli (Array): Complex structure moduli values, where the first component is the conifold modulus :math:`z_{\mathrm{cf}}` and the remaining components are bulk moduli :math:`z^a`.
        conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

    Returns:
        complex: Value of the coni-LCS prepotential :math:`F_{\mathrm{coni-LCS}}`.

    """


    return self.periods.F_coniLCS_series_per(self.moduli_to_periods(moduli,conj=conj),conj=conj)
