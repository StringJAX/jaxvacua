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
    
    
    def __init__(self, prange = 500, use_gvs = False, **kwargs):
        r"""
        **Description:**
        Defines class for the coni-LCS limits in Type IIB orientifold compactifications.
        
        Args:
            use_gvs (bool, optional): If set :var:`True`, GVs are being used and the corresponding polylogarithms evaluated. Otherwise, switch to GWs.
            prange (int, optional): Number of terms to be used when computing polylogarithms.
        
        """
        
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

        

    ###################################################################################################################################
    ################################### PREPOTENTIAL, KÄHLER POTENTIAL ETC. FOR LCS LIMIT #############################################
    ###################################################################################################################################
  
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

        """
        if not conj:
            return -jnp.sum(self.periods.GV_inv_lim*(self.periods.GV_charges_lim[:,0]**n)*jax_polylog_vmap(jnp.exp(2.*Pi*1j*jnp.einsum("ki,i",self.periods.GV_charges_lim,moduli)),3-n,self.prange))/(2.*Pi*1j)**(3-n)
        else:
            return -jnp.sum(self.periods.GV_inv_lim*(self.periods.GV_charges_lim[:,0]**n)*jax_polylog_vmap(jnp.exp(-2.*Pi*1j*jnp.einsum("ki,i",self.periods.GV_charges_lim,moduli)),3-n,self.prange))/(-2.*Pi*1j)**(3-n)
        """

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
        
        return jax.grad(self.F_coniLCS_poly_split,holomorphic=True,argnums=0)(zcf,zbulk,conj=conj)

    @partial(jit, static_argnums = (0,3,))
    def ddF_coniLCS_poly(self, zcf: ArrayLike, zbulk: ArrayLike, conj: bool = False) -> complex:
        
        return jax.grad(self.dF_coniLCS_poly,holomorphic=True,argnums=0)(zcf,zbulk,conj=conj)

    @partial(jit, static_argnums = (0,3,))
    def dddF_coniLCS_poly(self, zcf: ArrayLike, zbulk: ArrayLike, conj: bool = False) -> complex:
        
        return jax.grad(self.ddF_coniLCS_poly,holomorphic=True,argnums=0)(zcf,zbulk,conj=conj)

    @partial(jit, static_argnums = (0,3,))
    def ddddF_coniLCS_poly(self, zcf: ArrayLike, zbulk: ArrayLike, conj: bool = False) -> complex:
        
        return jax.grad(self.dddF_coniLCS_poly,holomorphic=True,argnums=0)(zcf,zbulk,conj=conj)
            
    
    @partial(jit, static_argnums = (0,2,))
    def F_coni(self, zcf: ArrayLike, conj: bool = False) -> complex:
        
        """
        if conj:
            return self.periods.ncf*(zcf)**2/(-4*jnp.pi*1j)*jnp.log(2*jnp.pi*1j*zcf)
        else:
            return self.periods.ncf*(zcf)**2/(4*jnp.pi*1j)*jnp.log(-2*jnp.pi*1j*zcf)
        """
        coeff = 2.*Pi*1j

        if conj:
            coeff = -coeff

        return self.periods.ncf*(zcf)**2/(2*coeff)*jnp.log(-coeff*zcf)

        
    @partial(jit, static_argnums = (0,2,3,))
    def F_coniLCS_exp(self, zbulk: ArrayLike, conj: bool = False, n: int = 0) -> complex:
        

        zeta_denom = (2*jnp.pi*1j)

        if conj:
            zeta_denom = -zeta_denom

        if n!=3:
            zeta_denom = zeta_denom**(3-n)
        else:
            zeta_denom = 1.

        coeff_cf = self.periods.ncf/zeta_denom

        zcf = 0.+1j*0.
        
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

        if n>1:
            val = val/jax.scipy.special.gamma(n+1)

        return val


    
    @partial(jit, static_argnums = (0,2,))
    def F_coniLCS(self, moduli: ArrayLike, conj: bool = False) -> complex:
        r"""
        
        **Description:**
        Calculates the value of the coni-LCS prepotential in terms of the complex structure moduli :math:`z^{i}`.
        
        
        .. admonition:: Details
            :class: dropdown
        
            At LCS, we can write the prepotential as
        
            .. math::
                F_{\text{LCS}}(z^1,\ldots , z^{h^{1,2}})=F_{\text{poly}}(z^1,\ldots , z^{h^{1,2}}) + F_{\text{inst}}(z^1,\ldots , z^{h^{1,2}})
        
            see e.g. Eq. (2.13) in `1312.0014 <https://arxiv.org/pdf/1312.0014.pdf>`_ for an equivalent formula. Here, the polynomial piece is given by
        
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
            Alternatively, the instanton contributions can be expressed as
            
            .. math::
                F_{\text{inst}}(z^1,\ldots , z^{h^{1,2}})=\sum_{q\in\mathcal{M}(\widetilde{X})}\, N_q\, \text{e}^{2\pi \text{i} q_i z^i}
        
            in terms of genus-0 Gromov-Witten (GW) invariants :math:`N_q`. 
            Both pieces are computed in separate functions, see :func:`F_coniLCS_poly` and :func:`F_inst` for details.

            TODO: CHANGE REFERENCES TO FUNCTIONS FROM OTHER CLASSES!!!

            TODO: change docstring!!

        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            complex: Value of the LCS prepotential :math:`F_{\text{LCS}}`.
            
        See also: :func:`F_coniLCS_poly`
    
        See also: :func:`F_inst`
    
        See also: :func:`periods_LCS.F_coniLCS_per`

        """

        zbulk = moduli[1:]
        zcf = moduli[0]
        
        val = self.F_coni(zcf,conj=conj)
        

        for n_exp in range(self.periods.nmax+1):
            tmp = self.F_coniLCS_exp(zbulk,conj=conj,n=n_exp)
        
            if self.periods.maximum_degree>0:
                tmp = tmp + self.F_coniLCS_inst(zbulk,conj=conj,n=n_exp)

            if n_exp>0:
                val = val + tmp*zcf**(n_exp)
            else:
                val = val + tmp


        return val

    # DONE
    # -----------------------------------------------------------------------------------



    

       

