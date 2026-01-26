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
# This file holds functions to compute periods and derived objects in CY threefold 
# compactifications.
# ------------------------------------------------------------------------------


# Important standard libraries
import os, sys, warnings
import numpy as np
import itertools
from functools import partial
import pandas as pd

# Important JAX libraries
import jax
from jax import jit, vmap, config
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jax.scipy.special import zeta
from jax.numpy import pi as Pi

# Enable 64 bit precision
config.update("jax_enable_x64", True)

#Polylog imports
from jaxpolylog import *

# JAXVacua custom imports
from .util import *
from .coniLCS_init import getAMatrix,get_basis_transformation
from .cicy_prepot import cicy_input
from .cytools_interface import cytools_instanton_data_init, cytools_model_data_init
from .periods_LCS import periods_LCS

# Some global variables
home_dir=os.path.dirname(os.path.realpath(__file__))
files_dir=home_dir+"/models"




class periods_coniLCSbulk:
    
    def __init__(self, h12=None,model_ID = None,model_type = "KS",
                 maximum_degree = 0, mirror_cy = None,ncf=2,
                 model_data = None,instanton_data = None,
                 use_cytools = False,basis_transformation = None,conifold_curve=None,
                 grading_vector = None, prange = 500, use_gvs = False,save_file=False, **kwargs):
        r"""

        **Description:**
        Period class for large complex structure limits.
        It provides functions for periods of Calabi-Yau threefolds :math:`X` at Large Complex Structure (LCS).

        Args:
            h12 (int): The number of moduli for the compactified geometry.
            model_type (string): The type of manifold considered for the compactification. Currently, ``"KS"`` and ``"CICY"`` are available.
            model_ID (int): ID specifying a certain model.
            model_data (dictionary): Contains model data like triple intersection numbers etc.
            instanton_data (list): List of GV and GW invariants.
            maximum_degree (int): Maximum degree used for the instanton sum.
            use_cytools (boolean): Whether or not to use CYTools to compute topological data of Calabi-Yau.
            mirror_cy (cytools.CalabiYau): Mirror Calabi-Yau threefold.
            basis_transformation (ArrayLike): Basis transformation to be applied to topological data of Calabi-Yau.
            grading_vector (ArrayLike): Grading vector to be used for the GV computation.
            save_file (bool, optional): Save files for new models. Defaults to ``False``.
            **kwargs: Extra inputs.

        Attributes:
            ncf (int): Number of conifolds.

        """

        self.coninop = jnp.identity(h12)[0]
        
        if ncf is None:
            raise ValueError("Need to provide value for the number of conifolds `ncf`!")

        
        
        periods_LCS.__init__(self, h12=h12, model_ID = model_ID,model_type = model_type, maximum_degree = maximum_degree, mirror_cy = mirror_cy,
                                model_data = model_data,instanton_data = instanton_data,use_cytools = use_cytools,
                                basis_transformation = basis_transformation,grading_vector = grading_vector, prange = prange, use_gvs = use_gvs,save_file=save_file,**kwargs)

        
        basis_transformation = self.basis_transformation
        
        if basis_transformation is None:
            if conifold_curve is None:
                raise ValueError("Need to provide conifold curve or basis transformation for coni-LCS!")
            basis_transformation = get_basis_transformation(conifold_curve)
        else:
            if conifold_curve is not None:
                coninop0 = conifold_curve@basis_transformation.T
                test = self.coninop == coninop0
                if not jnp.all(test):
                    raise ValueError(f"Input of basis transformation and conifold curve seems incompatible! \
                                        After applying the basis transformation, expect (1,0,0,...,0), but got {coninop0}!")
            
        if conifold_curve is None:
            self.conifold_curve = self.coninop@np.linalg.inv(basis_transformation.T)
        else:
            self.conifold_curve = conifold_curve

        self.ncf = ncf
        self.b_vector_prime = self.b_vector + self.ncf*self.coninop/24
        
        # Recompute a_matrix to match with conventions in paper
        self.a_matrix = jnp.array(getAMatrix(self.mirror_intersection_numbers))

        if self.max_deg>0:

            if len(self.GW_charges)>0:
                flag = jnp.all(self.GW_charges==self.coninop,axis=1)==False
                
                if np.max(np.abs(self.GW_charges))>0:
                    ind = np.where(flag)[0]
                    if len(ind)==0:
                        raise ValueError("Conifold curve not found in GW charges! Check basis transformation!")

                self.GW_charges = self.GW_charges[flag]
                self.GW_inv = self.GW_inv[flag]
                self.GW_curve_degrees = self.GW_curve_degrees[flag]
                flag = jnp.all(self.GW_charges_lim==self.coninop,axis=1)==False
                self.GW_charges_lim = self.GW_charges_lim[flag]
                self.GW_inv_lim = self.GW_inv_lim[flag]

            if len(self.GV_charges)>0:
                flag = jnp.all(self.GV_charges==self.coninop,axis=1)==False

                if np.max(np.abs(self.GV_charges))>0:
                    ind = np.where(flag)[0]
                    if len(ind)==0:
                        raise ValueError("Conifold curve not found in GV charges! Check basis transformation!")

                self.GV_charges = self.GV_charges[flag]
                self.GV_curve_degrees = self.GV_curve_degrees[flag]
                self.GV_inv = self.GV_inv[flag]
                flag = jnp.all(self.GV_charges_lim==self.coninop,axis=1)==False
                self.GV_charges_lim = self.GV_charges_lim[flag]
                self.GV_inv_lim = self.GV_inv_lim[flag]

    @partial(jit, static_argnums = (0,2,))
    def F_coniLCSbulk_per(self, XPer: ArrayLike, conj: bool = False) -> complex:
        r"""

        **Description:**
        Calculates the value of the LCS prepotential :math:`F_{\text{LCS}}` in terms of the periods :math:`X^I`.

        .. admonition:: Details
            :class: dropdown

            At Large Complex Structure (LCS), the prepotential can be expressed as

            .. math::
                F_{\mathrm{LCS}}(X) = F_{\mathrm{poly}}(X) + F_{\mathrm{inst}}(X)

            where

            .. math::
                F_{\mathrm{poly}}(X) = -\frac{1}{6X^0}\widetilde{\kappa}_{ijk}X^iX^jX^k+\frac{1}{2}a_{ij}X^iX^j
                    +b_{i}X^i\, X^0 + \dfrac{\text{i}}{2}\tilde{\xi}\, (X^0)^2\, ,

            and

            .. math::
                F_{\mathrm{inst}}(X) = -\frac{(X^0)^2}{(2\pi\mathrm{i})^3}\, \sum_{q\in\mathcal{M}(\widetilde{X})}\, 
                n_q^{0}\, \text{Li}_3\left (\text{e}^{2\pi \text{i} q_i X^i / X^0}\right )\; , \quad 
                \text{Li}_3\left (x\right )=\sum_{m=1}^{\infty}\, \dfrac{x^{m}}{m^{3}}\, .

            The former is computed via :func:`F_LCS_poly_per`, while the latter in :func:`F_inst_per`.


        Args:
            XPer (ArrayLike): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            complex: Value of the LCS prepotential :math:`F_{\text{LCS}}`.

        See also: :func:`F_LCS_poly_per`

        See also: :func:`F_inst_per`

        """
        
        return self.F_LCS_per(XPer,conj=conj)+self.ncf*self.coninop@XPer[1:]*XPer[0]/24

    # DONE
    # -----------------------------------------------------------------------------------

    
        


