# ==============================================================================
# This code is written by Andreas Schachner. Without the author's permission, 
# this code must not be shared with anyone else or used for any other projects 
# than those involving the author directly.
#
# If any questions arise, please feel free to reach out to me (Andreas) either at
# andreas.schachner@gmx.net or at as3475@cornell.edu or at a.schachner@lmu.de.
# ==============================================================================
#
# ------------------------------------------------------------------------------
# This file holds functions to construct complex structure moduli sector of
# Type IIB compactifications.
# ------------------------------------------------------------------------------


# Important standard libraries
import os, sys, warnings
import numpy as np
from functools import partial

# Important JAX libraries
import jax
from jax import jit, vmap, config
import jax.numpy as jnp
from jax import Array
#from jax.typing import ArrayLike
from numpy.typing import ArrayLike
from jax.scipy.special import zeta
from jax.numpy import pi as Pi

# Set CPU environment
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Enable 64 bit precision
config.update("jax_enable_x64", True)

#Polylog imports
from jaxpolylog import *

# JAXVacua custom imports
from .util import *
from .periods_LCS import periods_LCS # Needed for docstring ref?
from .periods import periods

class css_coniLCS:
    r"""
    **Description:**
    This class constructs the complex structure moduli sector of Type IIB orientifold compactifications at conifold-Large Complex Structure (coni-LCS) points in moduli space.
    """
    
    def __init__(self, prange = 500, use_gvs = False, **kwargs):
        r"""
        **Description:**
        Initializes the coni-LCS class for Type IIB orientifold compactifications.
        
        Args:
            prange (int, optional): Number of terms for polylogarithm computation. Defaults to 500.
            use_gvs (bool, optional): If ``True``, uses Gopakumar-Vafa invariants; 
            otherwise uses Gromov-Witten invariants. Defaults to ``False``.
            **kwargs: Additional keyword arguments passed to parent classes.
        
        Raises:
            ValueError: If validation of the 4th derivative of the polynomial prepotential fails.
        """
        #self.periods = periods_LCS(**kwargs)
        
        self.use_gvs = use_gvs
        self.prange = prange
        
        for _ in range(5):
            ReZ = np.random.uniform(-1,1,(self.periods.h12-1,))
            ImZ = np.random.uniform(-1,1,(self.periods.h12-1,))*1e-2+self.periods.tip_of_stretched_kahler_cone[1:]
            zbulk = ReZ+1j*ImZ

            ReZcf = np.random.uniform(-1,1)*1e-8
            ImZcf = np.random.uniform(-1,1)*1e-8
            zcf = ReZcf+1j*ImZcf

            val = self.ddddF_coniLCS_poly(zcf,zbulk,conj=False)
            cval = self.ddddF_coniLCS_poly(jnp.conj(zcf),jnp.conj(zbulk),conj=True)

            if jnp.max(jnp.abs(val))>1e-10 or jnp.max(jnp.abs(cval))>1e-10:
                raise ValueError("Test on 4th derivative of polynomial prepotential for csm failed! Please check input!")
  
    @partial(jit, static_argnums = (0,2,))
    def F_coniLCS_poly(self, moduli: ArrayLike, conj: bool = False) -> complex:
        r"""
        
        **Description:**
        Computes the polynomial constribution :math:`F_{\mathrm{poly}}` to the LCS prepotential 
        :math:`F_{\mathrm{LCS}}` in terms of the complex structure moduli :math:`z^i`.

        .. admonition:: Details
            :class: dropdown

            At Large Complex Structure (LCS), the polynomial part :math:`F_{\mathrm{poly}}`
            of the prepotential :math:`F_{\mathrm{LCS}}` is given by

            .. math::
                F_{\text{poly}}(Z)=\dfrac{1}{6}\widetilde{\kappa}_{ijk}\, z^i z^j z^k
                    +\dfrac{1}{2} a_{ij}z^i z^j+b_i\, z^i
                    +\dfrac{\text{i}}{2}\tilde{\xi}\, .

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
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            complex: Value of the polynomial constribution :math:`F_{\mathrm{poly}}` to 
                the LCS prepotential :math:`F_{\mathrm{LCS}}`.
            
        See also: :func:`F_LCS`
        
        """

        val = -jnp.dot(self.periods.kappa_tilde_sparse[:,-1],moduli[self.periods.kappa_tilde_sparse[:,0]]*moduli[self.periods.kappa_tilde_sparse[:,1]]*moduli[self.periods.kappa_tilde_sparse[:,2]])/6. 
        val += jnp.einsum('ij,i,j',self.periods.a_matrix,moduli,moduli)/(2.) 
        # Alternative sparse version, which does not seem to be faster....
        #val += jnp.dot(self.periods.a_matrix_sparse_values,moduli[self.periods.a_matrix_sparse[:,0]]*moduli[self.periods.a_matrix_sparse[:,1]])/(2.) 
        val += jnp.einsum('i,i',self.periods.b_vector,moduli)
        
        # Notice that we use K_0 = i * \tilde{\xi} in the code
        if not conj:
            return  val + self.periods.K0/2.
        else:
            return  val - self.periods.K0/2.
        

    @partial(jit, static_argnums = (0,2,3,))
    def F_coniLCS_inst(self,zbulk: ArrayLike, conj: bool = False, n: int = 0) -> complex:
        r"""
        
        **Description:**
        Returns the instanton part :math:`F_{\mathrm{inst}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
        in terms of the complex structure moduli :math:`z^i`.
        
        .. admonition:: Details
            :class: dropdown
        
            The sum over world-sheet instantons in the Type IIA language is given by
        
            .. math::
                F_{\text{inst}}(z^1,\ldots , z^{h^{1,2}})=-\frac{1}{(2\pi\mathrm{i})^3}\, \sum_{q\in\mathcal{M}(\widetilde{X})}\, 
                n_q^{0}\, \text{Li}_3\left (\text{e}^{2\pi \text{i} q_i z^i}\right )\; , \quad 
                \text{Li}_3\left (\right )=\sum_{m=1}^{\infty}\, \dfrac{x^{m}}{m^{3}}
        
            where the sum is performed over all effective curve classes :math:`q\in\mathcal{M}(\widetilde{X})`
            in the Mori cone :math:`\mathcal{M}(\widetilde{X})` of the mirror dual manifold :math:`\widetilde{X}`.
            Here, the :math:`n_q^{0}` are the genus-0 Gopakumar-Vafa (GV) invariants which can be computed
            systematically using methods described in `hep-th/9308122 <https://arxiv.org/pdf/hep-th/9308122.pdf>`_.
        
            The infinite sum appearing in the poly-logarithm :math:`\text{Li}_3` can be rewritten to arrive at
        
            .. math::
                \sum_{q\in\mathcal{M}(\widetilde{X})}\, n_q^{0}\, \text{Li}_3\left (\text{e}^{2\pi \text{i} q_i z^i}\right ) = \sum_{q\in\mathcal{M}(\widetilde{X})}\, N_q\, \text{e}^{2\pi \text{i} q_i z^i}
        
            in terms of genus-0 Gromov-Witten (GW) invariants :math:`N_q`. We typically work with the latter to simplify the calculation.
        
        .. note::
            This function return the sum over mirror worldsheet instantons wrapped on curves of degree smaller (or equal to) the provided maximal degree. 
            See :func:`F_inst_deg` for the option to return the value of the instanton pre-potential :math:`F_{\text{inst}}` for different choices of curve degrees.
        
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            complex: Value of the instanton part :math:`F_{\mathrm{inst}}` of the LCS prepotential 
                :math:`F_{\mathrm{LCS}}`.
            
        See also: :func:`F_LCS`
    
        See also: :func:`F_coniLCS_poly`
        
        """

        moduli = jnp.append(0.+1j*0.,zbulk)

        coeff = 2.*Pi*1j

        if conj:
            coeff = -coeff

        # Compute polylog
        Li_val = jax_polylog_vmap(jnp.exp(coeff*jnp.einsum("ki,i",self.periods.GV_charges_lim,moduli)),3-n,self.prange)
        
        # Compute sum over curves
        res = jnp.sum(self.periods.GV_inv_lim*(self.periods.GV_charges_lim[:,0]**n)*Li_val)

        # Add numerical prefactor
        return -res/(coeff)**(3-n)

    @partial(jit, static_argnums = (0,3,))
    def F_coniLCS_poly_split(self, zcf: ArrayLike, zbulk: ArrayLike, conj: bool = False) -> complex:
        r"""

        **Description:**
        Dummy function.

        """
        moduli = jnp.append(jnp.array([zcf]),zbulk)

        return self.F_coniLCS_poly(moduli,conj=conj)


    @partial(jit, static_argnums = (0,3,))
    def dF_coniLCS_poly(self, zcf: ArrayLike, zbulk: ArrayLike, conj: bool = False) -> complex:
        r"""
        **Description:**
        Computes the derivative of the conifold-LCS prepotential :math:`F_{\mathrm{LCS}}` with respect to the conifold modulus :math:`z_{\mathrm{cf}}`.
        
        Args:
            zcf (ArrayLike): Conifold modulus value.
            zbulk (ArrayLike): Bulk complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            complex: Value of the derivative of the conifold-LCS prepotential :math:`F_{\mathrm{LCS}}` with respect to the conifold modulus :math:`z_{\mathrm{cf}}`.
            
        """
        return jax.grad(self.F_coniLCS_poly_split,holomorphic=True,argnums=0)(zcf,zbulk,conj=conj)

    @partial(jit, static_argnums = (0,3,))
    def ddF_coniLCS_poly(self, zcf: ArrayLike, zbulk: ArrayLike, conj: bool = False) -> complex:
        r"""
        **Description:**
        Computes the second derivative of the conifold-LCS prepotential :math:`F_{\mathrm{LCS}}` with respect to the conifold modulus :math:`z_{\mathrm{cf}}`.
        
        Args:
            zcf (ArrayLike): Conifold modulus value.
            zbulk (ArrayLike): Bulk complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            complex: Value of the second derivative of the conifold-LCS prepotential :math:`F_{\mathrm{LCS}}` with respect to the conifold modulus :math:`z_{\mathrm{cf}}`.
        """
        return jax.grad(self.dF_coniLCS_poly,holomorphic=True,argnums=0)(zcf,zbulk,conj=conj)

    @partial(jit, static_argnums = (0,3,))
    def dddF_coniLCS_poly(self, zcf: ArrayLike, zbulk: ArrayLike, conj: bool = False) -> complex:
        r"""
        **Description:**
        Computes the third derivative of the conifold-LCS prepotential :math:`F_{\mathrm{LCS}}` with respect to the conifold modulus :math:`z_{\mathrm{cf}}`.
        
        Args:
            zcf (ArrayLike): Conifold modulus value.
            zbulk (ArrayLike): Bulk complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            complex: Value of the third derivative of the conifold-LCS prepotential :math:`F_{\mathrm{LCS}}` with respect to the conifold modulus :math:`z_{\mathrm{cf}}`.
        """
        return jax.grad(self.ddF_coniLCS_poly,holomorphic=True,argnums=0)(zcf,zbulk,conj=conj)

    @partial(jit, static_argnums = (0,3,))
    def ddddF_coniLCS_poly(self, zcf: ArrayLike, zbulk: ArrayLike, conj: bool = False) -> complex:
        r"""
        **Description:**
        Computes the fourth derivative of the conifold-LCS prepotential :math:`F_{\mathrm{LCS}}` with respect to the conifold modulus :math:`z_{\mathrm{cf}}`.
        
        Args:
            zcf (ArrayLike): Conifold modulus value.
            zbulk (ArrayLike): Bulk complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            complex: Value of the fourth derivative of the conifold-LCS prepotential :math:`F_{\mathrm{LCS}}` with respect to the conifold modulus :math:`z_{\mathrm{cf}}`.
        """
        
        return jax.grad(self.dddF_coniLCS_poly,holomorphic=True,argnums=0)(zcf,zbulk,conj=conj)
            
    
    @partial(jit, static_argnums = (0,2,))
    def F_coni(self, zcf: ArrayLike, conj: bool = False) -> complex:
        r"""
        **Description:**
        Computes the conifold prepotential :math:`F_{\mathrm{coni}}` in terms of the conifold modulus :math:`z_{\mathrm{cf}}`.
        
        .. admonition:: Details
            :class: dropdown
            
            Near the conifold point, the prepotential can be approximated by
            .. math::
                F_{\mathrm{coni}}(z_{\mathrm{cf}})=\dfrac{1}{2}\, \widetilde{n}_{\mathrm{cf}}\, \left ( \dfrac{z_{\mathrm{cf}}^2}{2\pi \mathrm{i}}\, \log \left ( -\dfrac{2\pi \mathrm{i} z_{\mathrm{cf}}}{\widetilde{n}_{\mathrm{cf}}} \right ) \right )\, ,
            where :math:`\widetilde{n}_{\mathrm{cf}}` is the number of conifold points in the complex structure moduli space. 
            
        Args:
            zcf (ArrayLike): Conifold modulus value.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            complex: Value of the conifold prepotential :math:`F_{\mathrm{coni}}`.
        """
        coeff = 2.*Pi*1j

        if conj:
            coeff = -coeff

        return self.periods.ncf*(zcf)**2/(2*coeff)*jnp.log(-coeff*zcf)

        
    @partial(jit, static_argnums = (0,2,3,))
    def F_coniLCS_exp(self, zbulk: ArrayLike, conj: bool = False, n: int = 0) -> complex:
        r"""
        **Description:**
        Computes the exponential corrections to the coni-LCS prepotential :math:`F_{\mathrm{coniLCS}}` in terms of the complex structure moduli :math:`z^i`.
        
        .. admonition:: Details
            :class: dropdown
            
            The exponential corrections to the coni-LCS prepotential :math:`F_{\mathrm{coniLCS}}` can be expressed as a series expansion in terms of the polylogarithm functions :math:`\text{Li}_n(x)` as
            .. math::
                F_{\mathrm{coniLCS,exp}}^{(n)}(z_{\mathrm{cf}},z_{\mathrm{bulk}})=\dfrac{1}{(2\pi \mathrm{i})^n}\, \sum_{q\in\mathcal{M}(\widetilde{X})}\, N_q\, \text{Li}_n\left (\text{e}^{2\pi \mathrm{i} q_i z^i}\right )\, ,
            where :math:`N_q` are the genus-0 Gromov-Witten (GW) invariants and :math:`n` indicates the order of the derivative with respect to the conifold modulus :math:`z_{\mathrm{cf}}`.
            
        Args:
            zbulk (ArrayLike): Bulk complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            n (int, optional): Order of derivative with respect to conifold modulus :math:`z_{\mathrm{cf}}`. Defaults to ``0``.
            
        Returns:
            complex: Value of the exponential corrections to the coni-LCS prepotential :math:`F_{\mathrm{coniLCS}}`.
        """

        # Numerical prefactor
        zeta_denom = (2*jnp.pi*1j)

        # Adjust for conjugation
        if conj:
            zeta_denom = -zeta_denom

        # Adjust for derivative order
        if n!=3:
            zeta_denom = zeta_denom**(3-n)
        else:
            zeta_denom = 1.

        # Coefficient in front of zeta functions / Bernoulli numbers
        coeff_cf = self.periods.ncf/zeta_denom

        # Conifold modulus set to zero
        zcf = 0.+1j*0.
        
        # Compute value depending on derivative order
        if n==0:
            val = self.F_coniLCS_poly_split(zcf,zbulk,conj=conj)-coeff_cf*jax.scipy.special.zeta(3,q=1)
        elif n==1:
            val = self.dF_coniLCS_poly(zcf,zbulk,conj=conj)-coeff_cf*jax.scipy.special.zeta(2,q=1)
        elif n==2:
            val = self.ddF_coniLCS_poly(zcf,zbulk,conj=conj)-coeff_cf*3/2
        elif n==3:
            val = self.dddF_coniLCS_poly(zcf,zbulk,conj=conj)+coeff_cf/2
        elif n>=4:
            val = coeff_cf*jax.scipy.special.bernoulli(n-2)[-1]/(n-2)

        # Adjust for factorial in denominator
        if n>1:
            val = val/jax.scipy.special.gamma(n+1)

        # Return value
        return val


    
    @partial(jit, static_argnums = (0,2,))
    def F_coniLCS(self, moduli: ArrayLike, conj: bool = False) -> complex:
        r"""
        **Description:**
        Calculates the full conifold-LCS prepotential :math:`F_{\mathrm{coniLCS}}` in terms of the complex structure moduli :math:`z^i`.
        
        
        .. admonition:: Details
            :class: dropdown
        
            At the conifold-LCS (coni-LCS) limit, the prepotential can be decomposed as
        
            .. math::
                F_{\mathrm{coni-LCS}}(z_{\mathrm{cf}}, z^a) = F_{\mathrm{coni}}(z_{\mathrm{cf}}) + \sum_{n=0}^{n_{\mathrm{max}}} F_{\mathrm{LCS}}^{(n)}(z_{\mathrm{cf}}, z^a) z_{\mathrm{cf}}^n
        
            where :math:`z_{\mathrm{cf}}` is the conifold modulus and :math:`z^a` are the bulk complex structure moduli.
            
            The conifold part :math:`F_{\mathrm{coni}}` encodes the singular behavior near the conifold point:
        
            .. math::
                F_{\mathrm{coni}}(z_{\mathrm{cf}}) = \dfrac{1}{2}\, \widetilde{n}_{\mathrm{cf}}\, \left ( \dfrac{z_{\mathrm{cf}}^2}{2\pi \mathrm{i}}\, \log \left ( -\dfrac{2\pi \mathrm{i} z_{\mathrm{cf}}}{\widetilde{n}_{\mathrm{cf}}} \right ) \right )
        
            The corrections :math:`F_{\mathrm{LCS}}^{(n)}` involve polynomials and polylogarithms and are summed over powers of :math:`z_{\mathrm{cf}}`.

        
        Args:
            moduli (ArrayLike): Complex structure moduli values, where the first component is the conifold modulus :math:`z_{\mathrm{cf}}` and the remaining components are bulk moduli :math:`z^a`.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            complex: Value of the coni-LCS prepotential :math:`F_{\mathrm{coni-LCS}}`.
            
        See also: :func:`F_coni`
        
        See also: :func:`F_coniLCS_exp`
        
        See also: :func:`F_coniLCS_inst`

        """

        # Split moduli into conifold and bulk part
        zbulk = moduli[1:]
        zcf = moduli[0]
        
        # Start with conifold part
        val = self.F_coni(zcf,conj=conj)
        
        # Expansion in the conifold modulus
        for n_exp in range(self.periods.nmax+1):
            
            # Compute polynomial and instanton part
            tmp = self.F_coniLCS_exp(zbulk,conj=conj,n=n_exp)
        
            # Add instanton part only if maximum degree >0
            if self.periods.maximum_degree>0:
                tmp = tmp + self.F_coniLCS_inst(zbulk,conj=conj,n=n_exp)

            # Add to total value
            if n_exp>0:
                val = val + tmp*zcf**(n_exp)
            else:
                val = val + tmp

        # Return value
        return val

    # DONE
    # -----------------------------------------------------------------------------------



    

       

