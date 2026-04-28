# ==============================================================================
# This code is written by Andreas Schachner.
#
# If any questions arise, please feel free to reach out to me (Andreas) either at
# andreas.schachner@gmx.net or at as3475@cornell.edu .
# ==============================================================================
#
# ------------------------------------------------------------------------------
# This file provides utilities for working with conifold limits in complex
# structure moduli space. It contains two layers of functionality:
#
#   1. Integer / lattice algebra (CYTools-independent):
#        extended_euclidean  — Bézout coefficients and unimodular basis change
#        orthogonal_lattice  — orthogonal-complement lattice generators
#        get_basis_change    — unimodular transformation to conifold basis
#        getAMatrix          — a-matrix from triple intersection numbers
#
#   2. Conifold geometry (requires CYTools):
#        Conifold            — class representing a conifold singularity
#        find_conifolds      — discover conifold loci from a toric polytope
# ------------------------------------------------------------------------------

#Standard libraries
# Important standard libraries
import os, sys, warnings
import numpy as np
from typing import Any, List, Optional
from functools import partial

# Import jax modules
import jax
import jax.numpy as jnp
from jax import jit
from jax import Array
from jax.scipy.special import zeta
from jax.numpy import pi as Pi
from jax.tree_util import register_pytree_node


# To load pickle files
import pickle
import gzip
from flint import fmpz, fmpz_mat


#Polylog imports
from jaxpolylog import jax_polylog_vmap

from cytools import Polytope
from cytools.triangulation import Triangulation
from jax import Array

from types import MethodType

class _ConifoldGated:
    r"""
    **Description:**
    Class-level descriptor: surfaces a method only when periods.limit ∈
    {coniLCS, coniLCS_series, coniLCS_bulk}; otherwise raises AttributeError
    so hasattr() returns False.
    
    This is used to gate conifold-specific methods in FluxVacuaFinder, so they only appear when the appropriate limit is set. The descriptor checks the periods.limit attribute of the instance to determine whether to allow access to the method or not. If the limit is not in the coniLCS family, an AttributeError is raised, effectively hiding the method from users of the class when it's not applicable.
     
    """
    def __init__(self, fn): self.fn = fn
    def __set_name__(self, owner, name): self.__name__ = name
    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        periods = instance.__dict__.get("periods")
        limit   = getattr(periods, "limit", None) if periods is not None else None
        if limit is None or "coniLCS" not in str(limit):
            raise AttributeError(
                f"{self.__name__!r} requires periods.limit ∈ coniLCS family "
                f"(got {limit!r})"
            )
        return MethodType(self.fn, instance)


#################### FROM PERIODS ####################

@partial(jit, static_argnums = (2,))
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
    XPer = jnp.append(jnp.array([X0,Xcf]),Xbulk)

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
    
    #return jax.grad(self.F_coniLCS_poly_per,holomorphic=True)(XPer,conj=conj)[1]
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

    #return jax.grad(self.dF_coniLCS_poly_per,holomorphic=True)(XPer,conj=conj)[1]
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

    #return jax.grad(self.ddF_coniLCS_poly_per,holomorphic=True)(XPer,conj=conj)[1]
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
def F_inst_per_coni(self,X0: complex ,XPerBulk: Array, conj: bool = False, n: int = 0) -> complex:
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
            bulk_invs = self.delete_coni_index(self.lcs_tree.gv_invariants, self.coni_index)
            bulk_charges = self.delete_coni_index(self.lcs_tree.gv_charges, self.coni_index)

    XPer = jnp.append(X0,jnp.append(0.+1j*0.,XPerBulk))
    
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
    res = jnp.matmul(bulk_invs*(bulk_charges[:,0]**n),Li_val)
    #res = jnp.sum(bulk_invs*(bulk_charges[:,0]**n)*Li_val)

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
    if conj:
        return self.lcs_tree.conifold.ncf*(X[1])**2/(-4*jnp.pi*1j)*jnp.log(2*jnp.pi*1j*X[1]/X[0])
    else:
        return self.lcs_tree.conifold.ncf*(X[1])**2/(4*jnp.pi*1j)*jnp.log(-2*jnp.pi*1j*X[1]/X[0])
    

@partial(jit, static_argnums = (4,5,))
def F_coniLCS_exp_per(self, 
                    X0: complex ,
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
    val = self.F_coni_per(XPer[:2], conj=conj)
    
    # ----------------------------------------
    # Extract periods
    # ----------------------------------------
    # XPer = (X^0, X_cf, X^2, ..., X^{h^{1,2}})
    XPerBulk = XPer[2:]   # Bulk (non-conifold) periods
    XConi    = XPer[1]    # Conifold period X_cf
    X0       = XPer[0]    # Fundamental period X^0
    

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
    
####################### END OF PERIODS ####################

##################### Complex structure ######################

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

    return self.F_LCS(moduli,conj=conj)+self.lcs_tree.conifold.ncf*self.lcs_tree.conifold.conifold_curve0@moduli/24.


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
    # Extract periods
    # ----------------------------------------
    # XPer = (X^0, X_cf, X^2, ..., X^{h^{1,2}})
    moduli = jnp.append(jnp.zeros(1), bulk_moduli)
    XPer = self.moduli_to_periods(moduli,conj=conj)
    XPerBulk = XPer[2:]   # Bulk (non-conifold) periods
    XConi    = 1. + 0*1j    # Conifold period X_cf
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
    
    
################### END OF COMPLEX STRUCTURE ###################




######################### Conifold utils #########################

def extended_euclidean(w):
    r"""
    **Description:**
    Computes Bézout's identity and a unimodular integer basis transformation
    for an integer array :math:`w`.

    .. admonition:: Details
        :class: dropdown

        Given an integer array :math:`w = (w_1,\ldots,w_n)`, the function
        returns integers :math:`b_i` (Bézout coefficients) satisfying

        .. math::
            \sum_{i=1}^{n} b_i \, w_i = \gcd(w_1,\ldots,w_n) \,,

        together with a unimodular integer matrix
        :math:`\Lambda \in \mathrm{GL}(n,\mathbb{Z})` such that

        .. math::
            \Lambda \, w = \bigl(\gcd(w_1,\ldots,w_n),\; 0,\;\ldots,\; 0\bigr)^T \,.

        The algorithm iteratively reduces pairs of entries via the Euclidean
        algorithm, tracking the accumulated integer row operations in
        :math:`\Lambda`.  Edge cases (single non-zero entry, all-zero input)
        are handled explicitly.

    Args:
        w (Array): Integer input array of length :math:`n`.

    Returns:
        tuple: ``(Bezout, GCD, Lambda)`` where

        - ``Bezout`` (``np.ndarray``, shape ``(n,)``, dtype ``int``) —
          Bézout coefficients satisfying
          :math:`\sum_i \text{Bezout}_i \cdot w_i = \gcd(w)`.
        - ``GCD`` (``int``) — Greatest common divisor of all non-zero
          entries of :math:`w`.
        - ``Lambda`` (``np.ndarray``, shape ``(n, n)``, dtype ``int``) —
          Unimodular transformation satisfying
          :math:`\Lambda w = (\gcd(w), 0,\ldots,0)^T`.

    See also: :func:`get_basis_change`, :func:`orthogonal_lattice`
    """
    
    # Ensure input is a NumPy array
    w = np.asarray(w)

    # Identify non-zero and zero entries
    nonvan_flag = (w != 0)   # Boolean mask for non-zero entries
    van_flag = (w == 0)      # Boolean mask for zero entries

    # Get indices of non-zero and zero entries
    nonvan_pos = np.where(nonvan_flag)[0]
    van_pos = np.where(van_flag)[0]

    # Initialize Bézout coefficients (same size as input)
    Bezout = np.zeros(len(w), dtype=int)

    # -----------------------------
    # Special case: only one non-zero entry
    # -----------------------------
    if sum(nonvan_flag) == 1:
        # The gcd is just that entry
        GCD = w[nonvan_flag][0]

        # Bézout coefficient is 1 for that entry
        Bezout[nonvan_flag] = 1

        # Construct transformation matrix
        Lambda_final = np.identity(len(w), dtype=int)

        # Swap first row with the non-zero position
        Lambda_final[0][0] = 0
        Lambda_final[nonvan_pos[0]][nonvan_pos[0]] = 0
        Lambda_final[0][nonvan_pos[0]] = 1
        Lambda_final[nonvan_pos[0]][0] = 1
        
    else:
        # -----------------------------
        # General case: multiple non-zero entries
        # -----------------------------

        # Extract non-zero entries
        v = w[nonvan_flag]

        # Work with absolute values for Euclidean algorithm
        acoeff = np.abs(v)

        # Sort entries in descending order (largest first)
        reordering = np.flip(np.argsort(acoeff))
        acoeffsorted = acoeff[reordering]

        # Initialize Lambda as permutation matrix corresponding to sorting
        Lambda = np.array([
            np.eye(1, len(reordering), i, dtype=int)[0]
            for i in reordering
        ])

        # Track how many dimensions have been reduced (zeros introduced)
        dim_red = 0

        # -----------------------------
        # Iterative Euclidean reduction
        # -----------------------------
        while True:
            # Divide all but last element by smallest element
            divs = acoeffsorted[:-1] / acoeffsorted[-1]

            # Integer quotients
            qs = divs.astype(int)

            # Remainders (careful rounding for integer stability)
            rs = np.rint(((divs - qs) * acoeffsorted[-1])).astype(int) \
                 + np.arange(len(divs)) * 1e-10

            # Sort remainders in descending order
            rssorted = np.flip(np.sort(rs))

            # Build permutation matrix mapping old remainders → sorted ones
            perm = np.array([i == rs for i in rssorted], dtype=int)

            # Update coefficients: smallest becomes first, followed by remainders
            acoeffsorted = np.rint(
                np.concatenate(([acoeffsorted[-1]], rssorted))
            ).astype(int)

            # Build next transformation block
            LambdaNext0 = np.block([
                [qs, np.transpose([[1]])],
                [perm, np.transpose([np.zeros(len(perm))])]
            ]).astype(int)

            # Expand transformation to include previously eliminated dimensions
            LambdaNext = np.block([
                [LambdaNext0, np.zeros([len(LambdaNext0), dim_red])],
                [np.zeros([dim_red, len(LambdaNext0)]), np.identity(dim_red)]
            ])

            # Update accumulated transformation
            Lambda = LambdaNext @ Lambda

            # Identify non-zero and zero positions
            posnonvan = np.where(acoeffsorted > 0)[0]
            posvan = np.where(acoeffsorted == 0)[0]

            # Remove zeros (dimension reduction)
            acoeffsorted = acoeffsorted[posnonvan]
            dim_red = dim_red + len(posvan)

            # Stop when only one value remains (the gcd)
            if len(acoeffsorted) == 1:
                break

        # -----------------------------
        # Recover Bézout coefficients
        # -----------------------------

        # First row of inverse transformation gives Bézout coefficients
        Bezout0 = (
            np.rint(np.transpose(np.linalg.inv(Lambda))[0]) * np.sign(v)
        ).astype(int)

        # Full inverse transformation (with signs restored)
        Lambda0 = (
            np.rint(np.transpose(np.linalg.inv(Lambda))) * np.sign(v)
        ).astype(int)

        # Embed into full dimension (including zeros)
        Lambda_tilde = np.block([
            [np.zeros([len(Lambda0), len(w) - len(Lambda0)], dtype=int), Lambda0],
            [np.identity(len(w) - len(Lambda0), dtype=int),
             np.zeros([len(w) - len(Lambda0), len(Lambda0)], dtype=int)]
        ])

        # -----------------------------
        # Reassemble full transformation
        # -----------------------------
        Lambda_final = np.identity(len(w), dtype=int)

        # Fill rows corresponding to non-zero entries
        Lambda_final[nonvan_pos] = Lambda_tilde.T[
            len(w) - len(Lambda0):len(Lambda_tilde)
        ]

        # Fill rows corresponding to zero entries
        Lambda_final[van_pos] = Lambda_tilde.T[
            0:len(w) - len(Lambda0)
        ]

        # Transpose to get final form
        Lambda_final = Lambda_final.T

        # Compute gcd from Bézout identity
        GCD = np.rint(sum(Bezout0 * v)).astype(int)

        # Place Bézout coefficients back into original positions
        Bezout[nonvan_flag] = Bezout0
        
    return (Bezout, GCD, Lambda_final)

def orthogonal_lattice(gens_in):
    r"""
    **Description:**
    Returns generators of the integer lattice orthogonal to the lattice
    spanned by ``gens_in``.

    .. admonition:: Details
        :class: dropdown

        Given :math:`d` generators :math:`g_1,\ldots,g_d \in \mathbb{Z}^n`
        with :math:`d < n`, the function computes generators of the
        *orthogonal complement lattice*

        .. math::
            L^\perp = \bigl\{ v \in \mathbb{Z}^n \;:\;
                v \cdot g_i = 0 \;\;\forall\, i=1,\ldots,d \bigr\} \,,

        which has rank :math:`n - d`.

        The algorithm constructs the augmented matrix

        .. math::
            B = \begin{pmatrix} c\,G \\ I_n \end{pmatrix}
            \in \mathbb{Z}^{(d+n)\times n} \,,

        where :math:`G` is the :math:`d\times n` generator matrix and
        :math:`c` is an integer scale chosen so that LLL reduction on
        :math:`B^T` separates the null-space rows.  The last :math:`n-d`
        rows of the LLL-reduced matrix (extracted from the :math:`I_n`
        block) are the desired generators.  The LLL computation uses
        ``flint.fmpz_mat`` for exact integer arithmetic.

    Args:
        gens_in (list): List of :math:`d` integer generator vectors of
            length :math:`n`, with :math:`d < n`.

    Returns:
        list: List of :math:`n-d` integer generators of :math:`L^\perp`,
        each of length :math:`n`.

    See also: :func:`extended_euclidean`, :func:`get_basis_change`
    """

    # Convert input list of generators into a NumPy array
    gens = np.array(gens_in)

    # d = number of input generators, n = ambient dimension
    d = len(gens)
    n = len(gens[0])

    # -----------------------------
    # Compute scaling factor c
    # -----------------------------
    # The exponent comes from bounds ensuring LLL separates
    # the orthogonal complement correctly
    exponent = (n - 1) / 2 + (n - d) * (n - d - 1) / 4

    # c scales the input generators so that LLL reduction
    # prioritizes orthogonality constraints over identity rows
    c = int(np.ceil(
        (2 ** exponent) *
        np.prod([np.linalg.norm(g) for g in gens])
    ))

    # -----------------------------
    # Build augmented matrix B^T
    # -----------------------------
    # Stack scaled generators on top of identity matrix:
    #   B^T = [ c * G ]
    #         [  I_n  ]
    #
    # Shape: (d + n) x n
    b_T = np.concatenate((c * gens, np.identity(n, dtype=int)))

    # Convert to FLINT integer matrix for exact LLL reduction
    b_T_mat = fmpz_mat(b_T.T.tolist())

    # -----------------------------
    # Perform LLL reduction
    # -----------------------------
    # Apply LLL to B^T (transposed form expected by FLINT)
    # Convert result back to NumPy array
    #
    # The first (n - d) rows correspond to short vectors,
    # which encode the orthogonal complement
    b_T_lll = [
        [int(ii) for ii in row][-n:]   # Extract last n entries (original coordinates)
        for row in np.array(b_T_mat.lll().tolist(), dtype=int)[:n - d]
    ]

    # -----------------------------
    # Return orthogonal lattice generators
    # -----------------------------
    return b_T_lll

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
    #return np.array([[intnumstensor[i][i][j]/2 if i<j else intnumstensor[i][j][j]/2 for j in range(h11)] for i in range(h11)])
    return np.array([[intnumstensor[i][j][j]/2 if i<j else intnumstensor[i][i][j]/2 for j in range(h11)] for i in range(h11)])








class Conifold:
    r"""
    **Description:**
    Represents a conifold singularity in complex structure moduli space dual
    to a flop transition in Kähler moduli space.

    .. admonition:: Details
        :class: dropdown

        A *conifold singularity* appears at the boundary of complex structure
        moduli space where a rational curve in the mirror Calabi-Yau shrinks
        to zero size.  On the mirror side, this corresponds to a topological
        transition — a *flop* — in Kähler moduli space.

        The flop is encoded in the triangulation of the toric polytope: a
        specific edge inside a 2-face of the dual polytope can be flipped
        (Whitehouse flip), and the corresponding curve is an extremal generator
        of the Mori cone.  The class stores the data needed to characterise
        this transition:

        * **Flop edge** :math:`e` — the edge in the dual triangulation that
          undergoes the flip.
        * **One-face divisors** — divisor indices interior to the 1-face of
          the polytope dual to the conifold (these become light in the
          coniLCS limit).
        * **Gopakumar-Vafa invariant** :math:`n_{\rm cf}` — the genus-0 GV
          invariant of the flopping curve.
        * **Conifold curve** :math:`q \in \mathbb{Z}^{h^{1,1}}` (original-basis
          charge) and **conifold_curve0** = :math:`q\,\Lambda^{T} = (1,0,\ldots,0)`
          (canonical-basis charge after the basis transformation
          :math:`\Lambda`); both are eagerly materialised at construction in
          geometric mode.

    The class supports two construction modes via the
    :meth:`from_geometry` and :meth:`from_data` classmethods.  After a JAX
    pytree round-trip, the cytools refs (:meth:`polytope`,
    :meth:`dual_triangulation`) return ``None``; the numerical state survives
    unchanged.
    """

    def __init__(self, *,
                 ncf: int = 0,
                 conifold_curve:     Optional[Array]         = None,
                 conifold_curve0:    Optional[Array]         = None,
                 basis_change:       Optional[Array]         = None,
                 flop_edge:          Optional[Array]         = None,
                 one_face_divisors:  Optional[Array]         = None,
                 polytope:           Optional[Polytope]      = None,
                 dual_triangulation: Optional[Triangulation] = None):
        r"""
        **Description:**
        Low-level keyword-only constructor; prefer :meth:`from_geometry` or
        :meth:`from_data` in user code.
        """
        # Eager / hot-path attributes (plain names, in pytree).
        self.ncf                 = int(ncf)
        self.conifold_curve      = conifold_curve     # original-basis charge
        self.conifold_curve0     = conifold_curve0    # canonical-basis charge
        # Lazy-getter backing storage (underscore-prefixed).
        self._basis_change       = basis_change
        self._flop_edge          = flop_edge
        self._one_face_divisors  = one_face_divisors
        # Cytools refs — dropped at the JAX pytree boundary by the custom flatten.
        self._polytope           = polytope
        self._dual_triangulation = dual_triangulation
        self._projection = None

    def __repr__(self):
        r"""
        **Description:**
        Returns a human-readable description of the :class:`Conifold` instance,
        including the Gopakumar-Vafa invariant.

        Returns:
            str: String representation.
        """

        return f"A conifold limit in complex structure moduli space dual to a flop transition with GV = {self.ncf}"

    # ------------------------------------------------------------------ #
    # Lazy / on-demand getter methods
    # ------------------------------------------------------------------ #
    
    def projection(self):
        
        if self._projection is None and self.conifold_curve is not None:
            q = np.array([0]+list(self.conifold_curve))
            w_proj_np  = extended_euclidean(q)[2][1:len(q)].T
            self._projection = jnp.asarray(w_proj_np)
        
        return self._projection
        

    def basis_change(self):
        r"""
        **Description:**
        Returns the unimodular basis change matrix :math:`\Lambda` that maps
        the :attr:`conifold_curve` to :math:`(1, 0, \ldots, 0)`.  Computed
        lazily on first call when the cache is empty and a
        ``conifold_curve`` is available.

        Returns:
            Array | None: Integer matrix
            :math:`\Lambda \in \mathrm{GL}(h^{1,1},\mathbb{Z})` of shape
            :math:`(h^{1,1}, h^{1,1})`, or ``None`` if neither cache nor
            ``conifold_curve`` is available.

        See also: :func:`get_basis_change`
        """
        if self._basis_change is None and self.conifold_curve is not None:
            self._basis_change = jnp.asarray(get_basis_change(self.conifold_curve))
        return self._basis_change

    def flop_edge(self):
        r"""
        **Description:**
        Returns the pair of point indices forming the flop edge in the dual
        triangulation.

        Returns:
            Array | None: Pair of integer indices defining the flop edge,
            or ``None`` if not available.
        """
        return self._flop_edge

    def one_face_divisors(self, as_index: bool = True):
        r"""
        **Description:**
        Returns the divisors interior to the 1-face dual to the conifold
        transition.

        Args:
            as_index (bool, optional): If ``True`` (default), returns the
                integer indices of the interior points.  If ``False``,
                returns the corresponding rows of the GLSM charge matrix
                (only valid when the cytools polytope is loaded).

        Returns:
            Array | None: Interior-point indices or GLSM charges, depending
            on ``as_index``.
        """
        if as_index:
            return self._one_face_divisors
        if self._polytope is None:
            raise RuntimeError(
                "polytope not loaded; cannot compute GLSM charge rows. "
                "Reconstruct via Conifold.from_geometry to use as_index=False."
            )
        return self._polytope.glsm_charge_matrix().T[self._one_face_divisors]

    def polytope(self):
        r"""
        **Description:**
        Returns the cytools polytope, or ``None`` after a JAX pytree
        round-trip (geometric data is dropped at the pytree boundary).

        Returns:
            cytools.Polytope | None: The polytope, or ``None`` if not loaded.
        """
        return self._polytope

    def dual_triangulation(self):
        r"""
        **Description:**
        Returns the cytools fine, regular, star triangulation of the dual
        polytope, or ``None`` after a JAX pytree round-trip.

        Returns:
            cytools.Triangulation | None: The dual triangulation, or ``None``
            if not loaded.
        """
        return self._dual_triangulation

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_geometry(cls,
                      polytope: Polytope,
                      dual_triangulation: Triangulation,
                      one_face_divisors: Array,
                      flop_edge: Array,
                      ncf: int) -> "Conifold":
        r"""
        **Description:**
        Construct a :class:`Conifold` from cytools geometry.  Eagerly
        computes :attr:`conifold_curve`, :attr:`conifold_curve0` and the
        basis-change matrix from the supplied polytope, triangulation and
        flop edge.

        Args:
            polytope (cytools.Polytope): Toric polytope of the CY model.
            dual_triangulation (cytools.Triangulation): Fine, regular, star
                triangulation of the dual polytope.
            one_face_divisors (Array): Indices of divisors interior to the
                1-face dual to the conifold transition.
            flop_edge (Array): Pair of point indices forming the flop edge.
            ncf (int): Gopakumar-Vafa invariant :math:`n_{\rm cf}` of the
                flopping curve (used to normalise the conifold charge).

        Returns:
            Conifold: A geometric-mode Conifold with all numerical fields
            populated and the cytools refs attached.
        """
        # Compute the mirror CY intersection numbers and conifold charge.
        dual_cy    = dual_triangulation.get_cy()
        intnums    = dual_cy.intersection_numbers()
        dual_basis = polytope.dual().glsm_basis()
        edge_list  = list(np.asarray(flop_edge))
        charge_np  = np.rint(np.array([
            intnums.get(tuple(np.sort(edge_list + [b])), 0)
            for b in dual_basis
        ]) / ncf).astype(int)

        bc_np  = np.asarray(get_basis_change(charge_np))   # int matrix
        curve  = jnp.asarray(charge_np)
        bc     = jnp.asarray(bc_np)
        curve0 = curve @ bc.T                              # = (1,0,...,0)

        return cls(
            ncf=int(ncf),
            conifold_curve=curve,
            conifold_curve0=curve0,
            basis_change=bc,
            flop_edge=jnp.asarray(flop_edge),
            one_face_divisors=jnp.asarray(one_face_divisors),
            polytope=polytope,
            dual_triangulation=dual_triangulation,
        )

    @classmethod
    def from_data(cls, **kwargs) -> "Conifold":
        r"""
        **Description:**
        Construct a numerical-only :class:`Conifold` from kwargs.  If
        ``conifold_curve`` and ``basis_change`` are supplied but
        ``conifold_curve0`` is omitted, it is derived as
        ``conifold_curve @ basis_change.T``.

        Args:
            **kwargs: Any subset of the :meth:`__init__` keyword arguments
                excluding ``polytope`` and ``dual_triangulation``.

        Returns:
            Conifold: A numerical-mode Conifold (cytools refs ``None``).
        """
        cc  = kwargs.get("conifold_curve")
        bc  = kwargs.get("basis_change")
        cc0 = kwargs.get("conifold_curve0")
        if cc0 is None and cc is not None and bc is not None:
            kwargs["conifold_curve0"] = jnp.asarray(cc) @ jnp.asarray(bc).T
        return cls(**kwargs)

    def to_data(self) -> "Conifold":
        r"""
        **Description:**
        Return a numerical-only copy of this :class:`Conifold` suitable for
        carrying across a JAX pytree boundary.  The cytools refs are
        stripped; the lazy basis-change cache is warmed first if a
        ``conifold_curve`` is available.

        Returns:
            Conifold: Numerical-mode copy.
        """
        # Warm the basis_change cache while geometry is still loaded.
        _ = self.basis_change()
        return Conifold.from_data(
            ncf=self.ncf,
            conifold_curve=self.conifold_curve,
            conifold_curve0=self.conifold_curve0,
            basis_change=self._basis_change,
            flop_edge=self._flop_edge,
            one_face_divisors=self._one_face_divisors,
        )


# --------------------------------------------------------------------------
# JAX pytree registration for Conifold.
#
# Cytools objects (Polytope, Triangulation) are not hashable and cannot live
# in aux_data, so the custom flatten silently drops them.  After a round-trip,
# ``polytope()`` and ``dual_triangulation()`` return ``None``.
# --------------------------------------------------------------------------

_CONIFOLD_DYNAMIC_KEYS = (
    # Hot-path attributes (plain names)
    'conifold_curve', 'conifold_curve0',
    # Lazy-getter backing storage (underscore-prefixed)
    '_basis_change', '_flop_edge', '_one_face_divisors',
)
_CONIFOLD_STATIC_KEYS = ('ncf',)


def _flatten_conifold(obj):
    children = tuple(getattr(obj, k, None) for k in _CONIFOLD_DYNAMIC_KEYS)
    aux_data = tuple((k, getattr(obj, k, None)) for k in _CONIFOLD_STATIC_KEYS)
    return children, aux_data


def _unflatten_conifold(aux_data, children):
    obj = object.__new__(Conifold)
    for k, v in aux_data:
        object.__setattr__(obj, k, v)
    for k, v in zip(_CONIFOLD_DYNAMIC_KEYS, children):
        object.__setattr__(obj, k, v)
    # Cytools refs are not part of the pytree — set to None on the way back.
    object.__setattr__(obj, '_polytope', None)
    object.__setattr__(obj, '_dual_triangulation', None)
    return obj


register_pytree_node(Conifold, _flatten_conifold, _unflatten_conifold)


def find_conifolds(polytope: Polytope,
                   FRSTs: List[Triangulation] | None = None,
                   n_conifolds: int = 2,
                   verbosity: int = 0):
    r"""
    **Description:**
    Identifies conifold limits in complex structure moduli space by searching
    for flop transitions in the triangulations of a toric polytope.

    .. admonition:: Details
        :class: dropdown

        A conifold transition corresponds to a *flop*: an edge inside a 2-face
        of the dual polytope that (i) can be flipped in the triangulation
        (Whitehouse flip) and (ii) whose associated curve is an extremal
        generator of the Mori cone.

        The search proceeds in three stages:

        1. **Face-pair selection** — pairs :math:`(f_1, f_2)` of primal 1-faces
           and dual 2-faces are retained when:

           * :math:`f_1` has exactly :math:`n_{\rm cf}-1` interior points
             (the conifold divisors).
           * :math:`f_2` has no interior points (for :math:`n_{\rm cf}>1`).
           * :math:`f_2` is not a simplex (:math:`\geq 4` vertices), so a
             flip is geometrically possible.

        2. **Triangulation loop** — for each FRST of the dual polytope, the
           simplices restricted to each candidate 2-face are extracted.
           Interior edges (shared by exactly two simplices) are enumerated,
           and flippable ones are identified by checking whether the convex
           hull of the four involved points has four vertices (a quadrilateral).

        3. **Mori cone check** — a candidate flop edge produces a charge
           vector :math:`q` (see :meth:`Conifold.conifold_charge`); only
           edges for which :math:`q` is an extremal ray of the Mori cone cap
           are accepted as genuine conifolds.

    Args:
        polytope (cytools.Polytope): Toric polytope of the CY model.
        FRSTs (list of cytools.Triangulation | None, optional): Pre-computed
            fine, regular, star triangulations of the dual polytope.  If
            ``None``, all NTFE classes are computed automatically. Defaults
            to ``None``.
        n_conifolds (int, optional): Number of conifold divisors — i.e. the
            number of interior points of the 1-face :math:`f_1`.  Defaults
            to ``2``.
        verbosity (int, optional): ``0`` — silent; ``1`` — prints candidate
            count; ``2`` — prints each conifold found. Defaults to ``0``.

    Returns:
        list of Conifold: All :class:`Conifold` instances found across all
        provided triangulations.
    """
    
    # ----------------------------------------
    # Basic polytope data
    # ----------------------------------------
    p     = polytope
    pdual = p.dual()                         # Dual polytope
    dpts  = pdual.points()                   # Coordinates of dual polytope points
    
    # All 1-faces of primal polytope and corresponding dual 2-faces
    one_faces      = p.faces(1)
    dual_two_faces = pdual.faces(2)
    
    # ----------------------------------------
    # Step 1: Select candidate face pairs
    # ----------------------------------------
    # We select (f1, f2) pairs satisfying:
    # - f1 has exactly (n_conifolds - 1) interior points
    # - f2 has no interior points (if n_conifolds > 1)
    # - f2 has at least 4 vertices (so a flip is possible)
    candidate_face_indices = []
    
    for i, (dual, one) in enumerate(zip(dual_two_faces, one_faces)):

        cond1 = (len(dual.interior_points()) == 0) if n_conifolds > 1 else True
        cond2 = (len(one.interior_points()) == n_conifolds - 1)
        cond3 = (len(dual.points()) > 3)
    
        if cond1 and cond2 and cond3:
            candidate_face_indices.append(i)

    if verbosity >= 1:
        print(f"Found {len(candidate_face_indices)} candidate face pairs for conifold limits.")

    # ----------------------------------------
    # Step 2: Prepare triangulations (FRSTs)
    # ----------------------------------------
    if FRSTs is None:
        if verbosity >= 1:
            print("No FRSTs provided. Computing all NTFEs...")
            
        # Warn user if computation may be expensive
        if pdual.h11("N") > 15:
            print("Warning: Computing all NTFEs for a polytope with h11 > 15 can be very time-consuming.")
            
        # Compute all fine, regular, star triangulations
        FRSTs = pdual.ntfe_frsts()
    
    # List of detected conifold objects
    conifolds = []

    # ----------------------------------------
    # Step 3: Loop over triangulations
    # ----------------------------------------
    for t in FRSTs:
        # Get simplices restricted to each 2-face (grouped by face)
        two_face_simplices = t.simplices(on_faces_dim=2, split_by_face=True)

        # Mori cone extremal rays (as tuples for fast lookup)
        Mcap = {
            tuple(q)
            for q in t.get_cy().mori_cone_cap(in_basis=True).extremal_rays()
        }

        # CY data associated to this triangulation
        dual_cy = t.get_cy()
        intnums = dual_cy.intersection_numbers()   # Intersection numbers dictionary
        dual_basis = p.dual().glsm_basis()         # Basis for charges
        
        # ----------------------------------------
        # Loop over candidate faces
        # ----------------------------------------
        for faceID in candidate_face_indices:
            # Interior lattice points of the 1-face (conifold divisors)
            one_face_divisors = np.array(
                one_faces[faceID].interior_points(as_indices=True)
            )

            # Simplices forming triangulation of this 2-face
            face_simplices = two_face_simplices[faceID]

            # ----------------------------------------
            # Extract edges in the 2-face
            # ----------------------------------------
            # Each triangle contributes 3 edges (remove one vertex at a time)
            edges_in_face = [
                np.delete(s, i, 0)
                for s in face_simplices
                for i in range(3)
            ]

            # Remove duplicate edges
            edges_in_face = np.unique(np.array(edges_in_face), axis=0)

            # ----------------------------------------
            # Identify interior edges
            # ----------------------------------------
            # Interior edges are those shared by exactly two simplices
            interior_edges = edges_in_face[
                np.where([
                    len(np.where([set(e).issubset(s) for s in face_simplices])[0]) == 2
                    for e in edges_in_face
                ])[0]
            ]

            # ----------------------------------------
            # Determine opposite vertices for each edge
            # ----------------------------------------
            # For each interior edge, find the two vertices opposite to it
            connected_points = np.array([
                [
                    [j for j in set(face_simplices[i]) - set(e)][0]
                    for i in np.where([
                        set(e).issubset(s) for s in face_simplices
                    ])[0]
                ]
                for e in interior_edges
            ])

            # ----------------------------------------
            # Identify flippable edges (Whitehouse flips)
            # ----------------------------------------
            # A flip is possible if the 4 involved points form a quadrilateral
            flop_indices = np.where([
                len(
                    Polytope(
                        np.array(
                            list(dpts[connected_points[i]]) +
                            list(dpts[interior_edges[i]])
                        )
                    ).vertices()
                ) == 4
                for i in range(len(interior_edges))
            ])[0]

            # Extract flippable edges
            flop_edges = interior_edges[flop_indices]

            # ----------------------------------------
            # Step 4: Mori cone check
            # ----------------------------------------
            for e in flop_edges:
                # Compute charge vector associated to the flop curve
                flop_charge = np.rint(np.array([
                    intnums.get(tuple(np.sort(list(e) + [b])), 0)
                    for b in dual_basis
                ]) / n_conifolds).astype(int)

                # Keep only if this is an extremal ray of the Mori cone
                if tuple(flop_charge) in Mcap:
                    conifolds.append(
                        Conifold.from_geometry(
                            polytope=p,
                            dual_triangulation=t,
                            one_face_divisors=one_face_divisors,
                            flop_edge=e,
                            ncf=n_conifolds,
                        )
                    )
                    
                    if verbosity >= 2:
                        print(f"Found conifold: {conifolds[-1]}")

    # ----------------------------------------
    # Return all detected conifolds
    # ----------------------------------------
    return conifolds


def conifold_fluxes(self,flux):
    
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
        q = self.lcs_tree.conifold.coni_curve
        # Ensure projection is cached before the JIT-compiled solver tries to access it.
        w_proj = self.lcs_tree.conifold.projection()  

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

def W_bulk(self, z, tau, flux, conj=False,normalise=False):
    
    if not self.lcs_tree.conifold_basis:
        raise NotImplementedError("compute_zcf_compact is only implemented for conifold_basis=True.")
    
    coeff = 2 * jnp.pi * 1j
    if conj:
        coeff = -coeff
        
    ncf = self.lcs_tree.conifold.ncf

    zbulk = z
    
    M0, H0, M1, H1, Malpha, Halpha, P1, K1, Palpha, Kalpha, P0, K0 = self.conifold_fluxes(flux)
    
    F0 = self.F_coniLCS_exp(zbulk, conj=conj, n=0)
    dF0 = self.dF_coniLCS_exp(zbulk, conj=conj, n=0)
    F1 = self.F_coniLCS_exp(zbulk, conj=conj, n=1)
    
    tmp  = (M0 - tau * H0) * (2*F0 - zbulk@dF0- 2*ncf*jax.scipy.special.zeta(3, q=1)/coeff**3)
    tmp += (M1 - tau * H1) * F1
    tmp += (Malpha - tau * Halpha) @ dF0
    tmp += (-1) * (P0 - tau * K0)
    tmp += (-1) * (Palpha - tau * Kalpha) @ zbulk
    
    if normalise:
        tmp = jnp.sqrt(2./jnp.pi)*tmp
    
    return tmp

def dK_cf_bulk(self, z, cz):
    
    coeff = 2 * jnp.pi * 1j
    
    zbulk = z
    czbulk = cz
    ncf = self.lcs_tree.conifold.ncf
    
    F0 = self.F_coniLCS_exp(zbulk, conj=False, n=0)
    cF0 = self.F_coniLCS_exp(czbulk, conj=True, n=0)
    dF0 = self.dF_coniLCS_exp(zbulk, conj=False, n=0)
    dcF0 = self.dF_coniLCS_exp(czbulk, conj=True, n=0)
    
    F1 = self.F_coniLCS_exp(zbulk, conj=False, n=1)
    dF1 = self.dF_coniLCS_exp(zbulk, conj=False, n=1)
    cF1 = self.F_coniLCS_exp(czbulk, conj=True, n=1)
    
    numerator = F1-cF1 - zbulk@dF1 + czbulk@dF1
    
    FF0 = 2*F0 - zbulk@dF0- 2*ncf*jax.scipy.special.zeta(3, q=1)/coeff**3
    cFF0 = 2*cF0 - czbulk@dcF0- 2*ncf*jax.scipy.special.zeta(3, q=1)/(-coeff)**3
    
    denom = czbulk@dF0 - zbulk@dcF0 + FF0 - cFF0
    
    dK = -numerator / denom
    
    return dK

def compute_zcf_correction(self, z, cz, tau, ctau, flux, conj=False):
    
    if not self.lcs_tree.conifold_basis:
        raise NotImplementedError("compute_zcf_correction is only implemented for conifold_basis=True.")
    
    coeff = 2 * jnp.pi * 1j
    if conj:
        coeff = -coeff
    
    ncf = self.lcs_tree.conifold.ncf

    zbulk = z
    czbulk = cz
    
    _, _, M1, H1, _, _, _, _, _, _, _, _, = self.conifold_fluxes(flux)
    
    if conj:
        Wbulk = self.W_bulk(czbulk, ctau, flux, conj=conj)
    else:
        Wbulk = self.W_bulk(zbulk, tau, flux, conj=conj)
    
    dK = self.dK_cf_bulk(zbulk, czbulk)
    
    tmp = dK*Wbulk
    
    if conj:
        denom = (M1 - ctau * H1)
    else:
        denom = (M1 - tau * H1)

    correction = -coeff * tmp / ncf / denom
    
    zcf1 = jnp.exp(correction)
    
    return zcf1

@partial(jit, static_argnames=["conj"])
def compute_zcf_explicit(self, z, tau, flux, conj=False):
    """Notebook-local: conifold modulus solver for ``conifold_basis=False``.

    Args:
        self: a FluxEFT-like object exposing ``_split_fluxes``, ``prange``.
        z: bulk moduli (already projected onto the basis-false bulk slice).
        tau: axio-dilaton.
        flux: full flux vector.
        lcs_tree: jaxvacua lcs_tree (must have ``conifold_basis=False``).
        mode: ``None`` for the full expression; reserved for ``"pfv"``.
        conj: if True, take the complex-conjugate branch.
    """
    #assert lcs_tree.conifold_basis is False, \
    #    "compute_zcf_new is the basis=False solver — pass an lcs_tree with conifold_basis=False."
    #if mode not in [None, "pfv", "analytic"]:
    #    raise ValueError("Mode must be one of [None, 'pfv', 'analytic'].")

    coeff = 2 * jnp.pi * 1j
    if conj:
        coeff = -coeff
    
    
    ncf = self.lcs_tree.conifold.ncf

    zbulk = z
    
    M0, H0, M1, H1, Malpha, Halpha, P1, K1, Palpha, Kalpha, P0, K0 = self.conifold_fluxes(flux)

    kappa = self.lcs_tree.intnums
    if self.lcs_tree.conifold_basis:
        kappa_011 = kappa[0,1:,1:]
        kappa_001 = kappa[0,0,1:]
        a_coni = self.lcs_tree.a_matrix[:,0]
        a00 = a_coni[0]
        a01 = a_coni[1:]
        b_coni = self.lcs_tree.b_vector[0]
    else:
        q = self.lcs_tree.conifold.coni_curve
        # Ensure this is cached before JIT compilation.
        w_proj = self.lcs_tree.conifold.projection()  
        kappa_011 = jnp.einsum("ijk,i,jl,km", kappa, q, w_proj, w_proj)
        kappa_001 = jnp.einsum("ijk,i,j,km",  kappa, q, q, w_proj)
        a_coni = self.lcs_tree.a_matrix @ q
        a00 = a_coni @ q
        a01 = a_coni @ w_proj
        b_coni = self.lcs_tree.b_vector @ q

    F1b  = ((kappa_011 @ zbulk) @ zbulk) / 2 + b_coni - jnp.pi**2 / 6 * ncf / coeff**2
    F2b  = -((kappa_001 @ zbulk)) + a00
    dF1b = -((kappa_011 @ zbulk)) + a01

    if self.periods.include_mirror_wsi:
        if self.lcs_tree.limit in ["coniLCS_series","coniLCS_bulk"]:
            bulk_invs = self.lcs_tree.gv_invariants
            bulk_charges = self.lcs_tree.gv_charges
        else:
            bulk_invs = self.periods.delete_coni_index(self.lcs_tree.gv_invariants, self.periods.coni_index)
            bulk_charges = self.periods.delete_coni_index(self.lcs_tree.gv_charges, self.periods.coni_index)

        if self.lcs_tree.conifold_basis:
            beta1 = jnp.asarray(bulk_charges[:,0])
            bulk_charges_proj = jnp.asarray(bulk_charges[:,1:])
            etpz = jnp.exp(coeff * jnp.einsum("ki,i", bulk_charges_proj, zbulk))
        else:
            q = self.lcs_tree.conifold.coni_curve
            beta1 = jnp.asarray(bulk_charges @ q)
            raise NotImplementedError("compute_zcf_explicit is not yet implemented for conifold_basis=False — the projection of the bulk charges must be cached first to avoid JIT issues.")    
            w_proj = self.lcs_tree.conifold.projection()
            bulk_charges_proj = jnp.asarray(bulk_charges @ w_proj.T)
            etpz = jnp.exp(coeff * jnp.einsum("ki,i", bulk_charges_proj, w_proj @ zbulk))
        
        Li1  = jax_polylog_vmap(etpz, 1, self.periods.prange)
        F2b  += -1 / coeff * jnp.sum(bulk_invs * beta1**2 * Li1)
        dF1b += -1 / coeff * jnp.sum((bulk_invs * Li1 * beta1)[:, None] * bulk_charges_proj, axis=0)
        
        # Li corrections to F1b
        Li2  = jax_polylog_vmap(etpz, 2, self.periods.prange)
        F1b  += -1 / coeff**2 * jnp.sum(bulk_invs * beta1 * (Li2 - (coeff*bulk_charges_proj@zbulk)*Li1))
        
        
    tmp  = (M0 - tau * H0) * F1b
    tmp += (M1 - tau * H1) * F2b
    tmp += (Malpha - tau * Halpha) @ dF1b
    tmp += (-1) * (P1 - tau * K1)

    exponent = -coeff * tmp / ncf / (M1 - tau * H1)
    zcf = (-1) / coeff * jnp.exp(exponent)
    
    return zcf

def compute_zcf_compact(self, z, tau, flux, conj=False):
    
    if not self.lcs_tree.conifold_basis:
        raise NotImplementedError("compute_zcf_compact is only implemented for conifold_basis=True.")
    
    coeff = 2 * jnp.pi * 1j
    if conj:
        coeff = -coeff
    
    ncf = self.lcs_tree.conifold.ncf

    zbulk = z
    
    M0, H0, M1, H1, Malpha, Halpha, P1, K1, Palpha, Kalpha, P0, K0 = self.conifold_fluxes(flux)
    
    F2 = self.F_coniLCS_exp(zbulk, conj=conj, n=2)
    F1 = self.F_coniLCS_exp(zbulk, conj=conj, n=1)
    dF1 = self.dF_coniLCS_exp(zbulk, conj=conj, n=1)
    
    F1b = F1 - zbulk@dF1
    F2b = F2 + ncf * 3 / 2 / coeff
    
    tmp  = (M0 - tau * H0) * F1b
    tmp += (M1 - tau * H1) * F2b
    tmp += (Malpha - tau * Halpha) @ dF1
    tmp += (-1) * (P1 - tau * K1)

    exponent = -coeff * tmp / ncf / (M1 - tau * H1)
    zcf = (-1) / coeff * jnp.exp(exponent)
    
    return zcf





#@partial(jit, static_argnums = (7,))
def W1_tilde(self,zbulk,tau,f2,h2,P1,K1,conj=False):
    r"""
    **Description:**
    Computes :math:`\widetilde{W}_1` as defined .
    
    Args:
        zbulk (Array): Values of the complex structure moduli excluding the conifold modulus.
        tau (complex): Value of the axio-dilaton :math:`\tau`.
        f2 (Array): Flux vector :math:`f_2`.
        h2 (Array): Flux vector :math:`h_2`.
        P1 (int): Flux :math:`P^1`.
        K1 (int): Flux :math:`K^1`.
        conj (bool, optional): Whether or not to conjugate the expression. Defaults to ``False``.
    
    Returns:
        complex: Value of :math:`\widetilde{W}_1`.
    
    """

    M0 = f2[0]
    H0 = h2[0]
    M1 = f2[1]
    H1 = h2[1]
    Malpha = f2[2:]
    Halpha = h2[2:]
    
    kappa = self.lcs_tree.intnums

    ncf = self.lcs_tree.conifold.ncf
    coeff = 2*jnp.pi*1j
    
    if conj:
        coeff = -coeff
        
    F1b = ((kappa[0,1:,1:]@zbulk)@zbulk)/2+self.lcs_tree.b_vector[0]-jnp.pi**2/6*ncf/coeff**2
    
    F2b = -((kappa[0,0,1:]@zbulk))+self.lcs_tree.a_matrix[0,0]
    
    dF1b = -((kappa[0,1:,1:]@zbulk))+self.lcs_tree.a_matrix[0,1:]
    
    tmp = (M0-tau*H0)*F1b
    
    tmp += (M1-tau*H1)*F2b
    
    tmp += (Malpha-tau*Halpha)@dF1b
    
    tmp += (-1)*(P1-tau*K1)
    
    return tmp

#@partial(jax.jit,static_argnums = (3,4,))
def compute_zcf(self,x,flux,mode=None,conj=False):
    r"""
    **Description:**
    Computes the value of the conifold modulus :math:`z_{cf}`.
    Args:
        x (Array): Real variables.
        flux (Array): Flux vector.
        mode (string, optional): Mode to be used. If ``"pfv"``, uses the PFV approximation. If ``None``, uses the full expression. Defaults to ``None``.
        conj (bool, optional): Whether or not to conjugate the expression. Defaults to ``False``.
        
    Returns:
        complex: Value of the conifold modulus :math:`z_{cf}`.
    
    """
    
    z,_,tau,_ = self._convert_real_to_complex(x)

    zbulk = z[1:]
    
    zcf = self.compute_zcf_explicit(zbulk,tau,flux,conj=conj)
    """
    
    if mode not in [None,"pfv","analytic"]:
        raise ValueError('Mode must be one of [None,"pfv"]!')
    
    f1,f2,h1,h2 = self._split_fluxes(flux)

    coeff = 2*jnp.pi*1j
        
    if conj:
        coeff = -coeff

    ncf = self.lcs_tree.conifold.ncf

    if mode == "pfv":

        Mvec = f2[1:]
        Kvec = h1[1:]
        M = Mvec[0]
        P1 = f1[1]
        N = self.lcs_tree.intnums@Mvec
        pvec = jnp.linalg.inv(N[1:,1:])@Kvec[1:]
        Kprime = Kvec[0]-N[0,1:]@pvec
        c0 = x[-2]
        s = x[-1]
        gs = 1/s

        phase_comb = -1j*(self.lcs_tree.a_matrix[0]@Mvec-P1+c0*Kprime)
        if conj:
            phase_comb = -phase_comb

        radial = Kprime/gs

        exponent = 2*jnp.pi/ncf/M*(phase_comb+radial)

    else:

        # x is the full real representation (including z_cf); use it directly
        z,_,tau,_ = self._convert_real_to_complex(x)

        zbulk = z[1:]   # z[0] = z_cf, z[1:] = bulk moduli
        f1,f2,h1,h2 = self._split_fluxes(flux)

        M1 = f2[1]
        H1 = h2[1]

        P1 = f1[1]
        K1 = h1[1]

        W1 = self.W1_tilde(zbulk,tau,f2,h2,P1,K1,conj=conj)

        exponent = -coeff*(W1)/ncf/(M1-tau*H1)
        
    zcf = (-1)/coeff * jnp.exp(exponent)
    """
    
    return zcf




@partial(jax.jit,static_argnums = (3,))
def zcf_handling(self,x,flux,mode=None):
    r"""
    **Description:**
    Handles the conifold modulus depending on the mode chosen.
    
    Args:
        x (Array): Real variables.
        flux (Array): Flux vector.
        mode (string, optional): Mode to be used. If ``"pfv"``, uses the PFV approximation. If ``None``, uses the full expression. Defaults to ``None``.
        
    Returns:
        Array: Real variables including the conifold modulus.
    
    """
    
    # TODO: this only works for conifold_basis=True
    
    if mode is None:
        xcz=jnp.zeros(2)
    else:
        
        #zcf = self.compute_zcf(x,flux,mode=mode)
        # CAREFUL: CHANGE WAS MADE!!!
        zcf = self.compute_zcf(jnp.append(jnp.zeros(2), x), flux, mode=mode)
        xcz = jnp.array([zcf.real,zcf.imag])
        
    return jnp.append(xcz,x)

@partial(jax.jit,static_argnums = (3,))
def DWbulk_x(self,x,flux,mode=None):
    r"""
    **Description:**
    Computes the gradient of the flux superpotential with respect to the bulk moduli.
    
    Args:
        x (Array): Real variables.
        flux (Array): Flux vector.
        mode (string, optional): Mode to be used. If ``"pfv"``, uses the PFV approximation. If ``None``, uses the full expression. Defaults to ``None``.
        
    Returns:
        Array: Gradient of the flux superpotential with respect to the bulk moduli.
    """
    
    X = self.zcf_handling(x,flux,mode=mode)

    return self.DW_x(X,flux)[2:]

@partial(jax.jit,static_argnums = (3,))
def dDWbulk_x(self,x,flux,mode=None):
    r"""
    **Description:**
    Computes the Hessian of the flux superpotential with respect to the bulk moduli.
    
    Args:
        x (Array): Real variables.
        flux (Array): Flux vector.
        mode (string, optional): Mode to be used. If ``"pfv"``, uses the PFV approximation. If ``None``, uses the full expression. Defaults to ``None``.
        
    Returns:
        Array: Hessian of the flux superpotential with respect to the bulk moduli.
    
    """

    X = self.zcf_handling(x,flux,mode=mode)

    return self.dDW_x(X,flux)[2:,2:]

@partial(jax.jit,static_argnums = (6,))
def DWbulk(self,z,cz,tau,ctau,flux,mode=None):
    
    x = self._convert_complex_to_real(z,cz,tau,ctau)
    
    X = self.zcf_handling(x,flux,mode=mode)
    
    # TODO: this only works for conifold_basis=True
    z0 = jnp.append(X[0]+1j*X[1],z)
    cz0 = jnp.append(X[0]-1j*X[1],cz)
    
    return self.DW(z0,cz0,tau,ctau,flux)[1:]

@partial(jax.jit,static_argnums = (6,))
def dDWbulk(self,z,cz,tau,ctau,flux,mode=None):
    
    x = self._convert_complex_to_real(z,cz,tau,ctau)
    
    X = self.zcf_handling(x,flux,mode=mode)
    
    # TODO: this only works for conifold_basis=True
    z0 = jnp.append(X[0]+1j*X[1],z)
    cz0 = jnp.append(X[0]-1j*X[1],cz)
    
    return self.dDW(z0,cz0,tau,ctau,flux)[1:,1:]

