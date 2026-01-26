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


class css_LCS:
    
    
    def __init__(self, prange = 500, use_gvs = False, **kwargs):
        r"""
        **Description:**
        Defines class for the Large Complex Structure limits in Type IIB orientifold compactifications.
        
        Args:
            use_gvs (bool, optional): If set :var:`True`, GVs are being used and the corresponding polylogarithms evaluated. Otherwise, switch to GWs.
            prange (int, optional): Number of terms to be used when computing polylogarithms.
            
        """
        
        self.use_gvs = use_gvs
        self.prange = prange
        

    ###################################################################################################################################
    ################################### PREPOTENTIAL, KÄHLER POTENTIAL ETC. FOR LCS LIMIT #############################################
    ###################################################################################################################################
  
    @partial(jit, static_argnums = (0,2,))
    def F_LCS_poly(self, 
                moduli: ArrayLike, 
                conj: bool = False
                ) -> complex:
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
        

    @partial(jit, static_argnums = (0,2,))
    def F_inst(self,
            moduli: ArrayLike, 
            conj: bool = False
            ) -> complex:
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
        
        
        
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            complex: Value of the instanton part :math:`F_{\mathrm{inst}}` of the LCS prepotential 
                :math:`F_{\mathrm{LCS}}`.
            
        See also: :func:`F_LCS`
    
        See also: :func:`F_LCS_poly`
        
        """
        if not conj:
            if self.use_gvs:
                return -jnp.sum(self.periods.GV_inv_lim*jax_polylog_vmap(jnp.exp(2.*Pi*1j*jnp.einsum("ki,i",self.periods.GV_charges_lim,moduli)),3,self.prange))/(2.*Pi*1j)**(3)
            else:
                return -jnp.sum(self.periods.GW_inv_lim*jnp.exp(2.*Pi*1j*jnp.einsum("ki,i",self.periods.GW_charges_lim,moduli)))/(2.*Pi*1j)**(3)
        else:
            if self.use_gvs:
                return -jnp.sum(self.periods.GV_inv_lim*jax_polylog_vmap(jnp.exp(-2.*Pi*1j*jnp.einsum("ki,i",self.periods.GV_charges_lim,moduli)),3,self.prange))/(-2.*Pi*1j)**(3)
            else:
                return -jnp.sum(self.periods.GW_inv_lim*jnp.exp(-2.*Pi*1j*jnp.einsum("ki,i",self.periods.GW_charges_lim,moduli)))/(-2.*Pi*1j)**(3)
    
    
    @partial(jit, static_argnums = (0,2,))
    def F_LCS(self, 
            moduli: ArrayLike, 
            conj: bool = False
            ) -> complex:
        r"""
        
        **Description:**
        Calculates the value of the LCS prepotential in terms of the complex structure moduli :math:`z^{i}`.
        
        
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
            Both pieces are computed in separate functions, see :func:`F_LCS_poly` and :func:`F_inst` for details.

        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            complex: Value of the LCS prepotential :math:`F_{\text{LCS}}`.
            
        See also: :func:`F_LCS_poly`
    
        See also: :func:`F_inst`
    
        See also: :func:`periods_LCS.F_LCS_per`

        """
        
        return self.F_LCS_poly(moduli,conj=conj)+self.F_inst(moduli,conj=conj)

    # DONE
    # -----------------------------------------------------------------------------------



    

       

