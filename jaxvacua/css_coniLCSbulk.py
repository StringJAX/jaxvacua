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
from .css_LCS import css_LCS


class css_coniLCSbulk:
    
    
    def __init__(self, prange = 500, use_gvs = False, **kwargs):
        r"""
        **Description:**
        Defines class for the bulk sector of coni-LCS limits in Type IIB orientifold compactifications.
        
        Args:
            use_gvs (bool, optional): If set :var:`True`, GVs are being used and the corresponding polylogarithms evaluated. Otherwise, switch to GWs.
            prange (int, optional): Number of terms to be used when computing polylogarithms.
        
        """
        
        self.use_gvs = use_gvs
        self.prange = prange

    
    @partial(jit, static_argnums = (0,2,))
    def F_coniLCSbulk(self, moduli: ArrayLike, conj: bool = False) -> complex:
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
            Both pieces are computed in separate functions, see :func:`css_LCS.F_LCS_poly` 
            and :func:`css_LCS.F_inst` for details.

        
        .. warning::
            The effective description for the bulk theory holds only provided that :math:`z^1 = z_{\text{cf}}\ll 1`.

        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            complex: Value of the LCS prepotential :math:`F_{\text{LCS}}`.
            
        See also: :func:`css_LCS.F_LCS_poly`
    
        See also: :func:`css_LCS.F_inst`
    
        See also: :func:`css_LCS.periods_LCS.F_LCS_per`

        """
        
        return self.F_LCS(moduli,conj=conj)+self.periods.ncf*self.periods.coninop@moduli/24

    # DONE
    # -----------------------------------------------------------------------------------



    

       

