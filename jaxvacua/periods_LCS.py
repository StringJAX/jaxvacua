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
#from jax.typing import ArrayLike
from numpy.typing import ArrayLike
from jax.scipy.special import zeta
from jax.numpy import pi as Pi

# Enable 64 bit precision
config.update("jax_enable_x64", True)

#Polylog imports
from jaxpolylog import *

# JAXVacua custom imports
from .util import *
from .utils_jaxvacua import *
from .lcs_init import LCS_init,instanton_init
from .cicy_prepot import cicy_input
from .cytools_interface import cytools_instanton_data_init, cytools_model_data_init



# Some global variables
home_dir=os.path.dirname(os.path.realpath(__file__))
files_dir=home_dir+"/models"

class periods_LCS:
    
    def __init__(self, h12=None,model_ID = None,model_type = "KS",
                 maximum_degree = 0, mirror_cy = None,
                 model_data = None,instanton_data = None,
                 use_cytools = False,basis_transformation = None,
                 grading_vector = None, prange = 500, use_gvs = False,save_file=False, **kwargs):
        r"""

        **Description:**
        Period class for large complex structure limits.
        It provides functions for periods of Calabi-Yau threefolds :math:`X` at Large Complex Structure (LCS).
        The periods are computed using mirror symmetry, i.e., from the Kähler potential of the mirror dual Calabi-Yau threefold :math:`\widetilde{X}`.
        

        Args:
            h12 (int): The number of moduli for the compactified geometry.
            model_type (string): The type of manifold considered for the compactification. Currently, ``"KS"`` and ``"CICY"`` are available.
            model_ID (int): ID specifying a certain model.
            model_data (dictionary): Contains model data like triple intersection numbers etc.
            instanton_data (list): List of GV and GW invariants.
            maximum_degree (int): Maximum degree used for the instanton sum.
            use_cytools (boolean): Whether or not to use CYTools to compute topological data of Calabi-Yau. Requires CYTools to be installed.
            prange (int): Range for polylogarithm computations.
            use_gvs (bool): Whether to use Gopakumar-Vafa invariants (``True``) or Gromov-Witten invariants (``False``) for the instanton sum.
            mirror_cy (cytools.CalabiYau, cytools.Triangulation, cytools.Polytope): Mirror Calabi-Yau threefold or associated objects like the underlying polytope or triangulation. 
            basis_transformation (ArrayLike): Basis transformation to be applied to topological data of Calabi-Yau.
            grading_vector (ArrayLike): Grading vector to be used for the GV computation.
            save_file (bool, optional): Save files for new models. Defaults to ``False``.
            **kwargs: Extra inputs.

        Attributes:
            mirror_intersection_numbers (ArrayLike): Triple intersection numbers of the mirror CY.  
            chi (int): Euler characteristic of the mirror CY.
            K0 (complex): :math:`(\alpha')^3` 1-loop correction on the mirror dual side.
            b_vector (ArrayLike): Linear piece in the prepotential.
            a_matrix (ArrayLike): Quadratic piece in the prepotential.
            kappa_tilde_sparse (ArrayLike): Sparse version of the mirror triple intersection number (including symmetry factors).
            hyperplanes (ArrayLike): Hyperplanes for the toric Kähler cone of the mirror CY.
            tip_of_stretched_kahler_cone (ArrayLike): Tip of the :math:`c=1` stretched Kähler cone of the mirror CY.
            generators_kahler_cone (ArrayLike): Extremal rays of the Kähler cone of the mirror CY.
            rays_kahler_cone (ArrayLike): Rays of the Kähler cone of the mirror CY.
            generators_mori_cone (ArrayLike): Extremal rays of the Mori cone of the mirror CY.
            polytope_points (ArrayLike): Polytope points.
            max_deg (int): Maximum degree for GV and/or GW invariants used for computations.  
            max_available_deg (int): Maximum degree for GV and/or GW invariants available in data. 
            grading_vector (ArrayLike): Grading vector used for the GV/GW computations.
            GW_inv (ArrayLike): GW invariants.
            GW_charges (ArrayLike): Curve charges for GW invariants.
            GV_inv (ArrayLike): GV invariants.
            GV_charges (ArrayLike): Curve charges for GV invariants.
            GW_inv_lim (ArrayLike): GW invariants limited to maximum degree.
            GW_charges_lim (ArrayLike): Curve charges for GW invariants limited to maximum degree.
            GV_inv_lim (ArrayLike): GV invariants limited to maximum degree.
            GV_charges_lim (ArrayLike): Curve charges for GV invariants limited to maximum degree.

        """

        
        # -----------------------------------------------------------------------------------
        # Testing inputs...
        ## Check correct model types
        model_types=["KS","CICY"]
        if model_type not in model_types:
            raise ValueError(f"Model type must be one of {model_types}!")
            
        if model_data is None:
            if use_cytools:
                ### Test that we can import CYTools
                try:
                    import cytools
                except ImportError:
                    raise ImportError("Cannot import CYTools! Please check your docker image!")
            
            else:
                ### Check for input files...
                if not os.path.isdir(files_dir):
                    raise ValueError(f"Could not find files directory {files_dir}")

        # DONE
        # -----------------------------------------------------------------------------------
        

        # -----------------------------------------------------------------------------------
        # Initialisation of class
        ## Setting general attributes from input
        self.max_deg = maximum_degree
        self.maximum_degree = maximum_degree

        self.prange = prange
        self.use_gvs = use_gvs

        file_KS = None
        
        ## Defining class based on model_type
        if self.model_type=="KS":
            
            if h12 is not None:
                self.h12 = h12
            else:
                if use_cytools:
                    self.h12 = mirror_cy.h11()
                else:
                    raise ValueError("For `model_type=KS`, the number of moduli should be provided explicitly!")
                
            if use_cytools:
                file_KS = None
                file_KS_inst = None
            else:
                if model_data is None:
                    if self.model_ID is None:
                        raise ValueError(f"Need to provide model_ID if grabbing model_data and instanton_data from files!")
                    
                    file_KS=files_dir+"/KS/h12_"+str(self.h12)+"/Model_"+str(self.model_ID)+"/"
                    
                    if not os.path.isfile(file_KS+"model_data.p"):
                        raise ValueError(f"File for `model_data` could not be found under path {file_KS+'model_data.p'}! Please check input!")
                        
                    if self.maximum_degree>0:
                        if not os.path.isfile(file_KS+"instanton_data.p"):# or not os.path.isfile(file_KS_inst+"_GVs.p"):
                            raise ValueError(f"File for `instanton_data` could not be found under path {file_KS+'instanton_data.p'}! Please check input!")
        
            
        ## Grabbing model data
        if self.model_type=="CICY":

            model_data,instanton_data = cicy_input(self.model_ID)

            self.h12 = model_data["h12"]

        elif self.model_type=="KS":
            if model_data is None:
                if use_cytools:

                    model_data = cytools_model_data_init(mirror_cy,basis_transformation=basis_transformation,model_ID=self.model_ID,save_file=save_file)
                    
                else:
                    model_data = load_zipped_pickle(file_KS+"model_data.p")
        else:
            raise ValueError("Could not obtain model_data!")
        
        if model_data is None:
            raise ValueError("Model data could not be initialised! Check input!")

        LCS_init(self,model_data)

        if "basis transformation" in model_data.keys():
            self.basis_transformation = model_data["basis transformation"]

        if 'amatrix' in kwargs:
            self.a_matrix = kwargs["amatrix"]

        if self.max_deg>0:
            if instanton_data is None:

                if self.model_type=="KS":
                    if use_cytools:
                        
                        instanton_data = cytools_instanton_data_init(mirror_cy,self.max_deg, grading_vector = grading_vector,model_ID=self.model_ID,save_file=save_file,basis_transformation=basis_transformation)
                            
                    else:
                        file = file_KS+"instanton_data.p"

                        if not os.path.isfile(file):
                            raise ValueError(f"File containing instanton data not found at {file}.")

                        instanton_data = load_zipped_pickle(file)

                elif self.model_type=="CICY":
                    if instanton_data is None:
                        raise ValueError("No instanton data found for CICY! Please check input!")
                else:
                    raise ValueError("No instanton data provided! Please check input!")
            else:
                if save_file:
                    save_model_data(instanton_data,"instanton_data.p",model_ID,self.h12)

            if instanton_data is None:
                raise ValueError("Instanton data could not be initialised! Check input!")



        instanton_init(self,instanton_data,mirror_cy,grading_vector,use_cytools)

        

        # DONE
        # -----------------------------------------------------------------------------------        

        
    ###################################################################################################################################
    #################################### PREPOTENTIAL ETC. AS FUNCTIONS OF PERIODS FOR LCS ############################################
    ###################################################################################################################################

    @partial(jit, static_argnums = (0,2,))
    def F_LCS_poly_per(self, XPer: ArrayLike, conj: bool = False) -> complex:
        r"""
        **Description:**
        Computes the polynomial part :math:`F_{\mathrm{poly}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
        in terms of the periods. 

        .. admonition:: Details
            :class: dropdown

            The polynomial part :math:`F_{\mathrm{poly}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
            can be expressed in terms of the periods :math:`X^I=(X^0,X^i)` as
            
            .. math::
                F_{\mathrm{poly}}(X) = -\frac{1}{6X^0}\widetilde{\kappa}_{ijk}X^iX^jX^k+\frac{1}{2}a_{ij}X^iX^j
                    +b_{i}X^i\, X^0 + \dfrac{\text{i}}{2}\tilde{\xi}\, (X^0)^2\, .

            Here, :math:`\widetilde{\kappa}_{ijk}` are the triple intersection numbers of
            the mirror dual Calabi-Yau threefold :math:`\widetilde{X}`.
            Here, we defined

            .. math::
                a_{ij} = \dfrac{1}{2}\begin{cases}
                                        \widetilde{\kappa}_{iij} & i\geq j\\[0.3em]
                                        \widetilde{\kappa}_{ijj} & i<j
                                    \end{cases} \, , \quad 
                b_i = \dfrac{1}{24} \int_{\tilde{D}^i}\, c_2(\widetilde{X})\, , \quad  
                \tilde{\xi}=\frac{\zeta(3)\, \chi(\widetilde{X})}{(2\pi)^3}\, .
                
            The :math:`a_{ij}` are rational numbers, while the :math:`b_i` are integers.
            The :math:`\tilde{D}^i` are the divisors dual to the basis of :math:`H^{1,1}(\widetilde{X},\mathbb{Z})`.
            Finally, :math:`\chi(\widetilde{X})` is the Euler characteristic of :math:`\widetilde{X}`.
            The last term in :math:`F_{\mathrm{poly}}` is the so-called :math:`(\alpha')^3` 1-loop correction
            on the mirror dual side.

        Args:
            XPer (ArrayLike): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            complex: Value of the polynomial constribution :math:`F_{\mathrm{poly}}` to 
                the LCS prepotential :math:`F_{\mathrm{LCS}}`.

        See also: :func:`F_LCS_per`

        """

        cubic = -jnp.dot(self.kappa_tilde_sparse[:,-1],XPer[self.kappa_tilde_sparse[:,0]+1]*XPer[self.kappa_tilde_sparse[:,1]+1]*XPer[self.kappa_tilde_sparse[:,2]+1])/6./XPer[0]
        quadratic = jnp.einsum('ij,i,j',self.a_matrix,XPer[1:],XPer[1:])/(2.)
        # Alternative sparse version, which does not seem to be faster....
        #quadratic = jnp.dot(self.a_matrix_sparse_values,XPer[self.a_matrix_sparse[:,0]+1]*XPer[self.a_matrix_sparse[:,1]+1])/(2.)
        linear = jnp.einsum('i,i',self.b_vector,XPer[1:])*XPer[0]
        val =  cubic + quadratic + linear
        
        if not conj:
            return val + self.K0/2.*XPer[0]**(2)
        else:
            return val - self.K0/2.*XPer[0]**(2)

    @partial(jit, static_argnums = (0,2,))
    def F_inst_per(self,XPer: ArrayLike, conj: bool = False) -> complex:
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
            XPer (ArrayLike): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            complex: Value of the instanton part :math:`F_{\mathrm{inst}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`.

        See also: :func:`F_LCS_per`

        """
        if not conj:
            if self.use_gvs:
                return -jnp.sum(self.GV_inv_lim*jax_polylog_vmap(jnp.exp(2.*Pi*1j*jnp.einsum("ki,i",self.GV_charges_lim,XPer[1:])/XPer[0]),3,self.prange))/(2.*Pi*1j)**(3)*XPer[0]**(2)
            else:
                return -jnp.sum(self.GW_inv_lim*jnp.exp(2.*Pi*1j*jnp.einsum("ki,i",self.GW_charges_lim,XPer[1:])/XPer[0]))/(2.*Pi*1j)**(3)*XPer[0]**(2)
        else:
            if self.use_gvs:
                return -jnp.sum(self.GV_inv_lim*jax_polylog_vmap(jnp.exp(-2.*Pi*1j*jnp.einsum("ki,i",self.GV_charges_lim,XPer[1:])/XPer[0]),3,self.prange))/(-2.*Pi*1j)**(3)*XPer[0]**(2)
            else:
                return -jnp.sum(self.GW_inv_lim*jnp.exp(-2.*Pi*1j*jnp.einsum("ki,i",self.GW_charges_lim,XPer[1:])/XPer[0]))/(-2.*Pi*1j)**(3)*XPer[0]**(2)


    

    @partial(jit, static_argnums = (0,2,))
    def F_LCS_per(self, XPer: ArrayLike, conj: bool = False) -> complex:
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
            See the respective function for more details.
            
        Args:
            XPer (ArrayLike): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            complex: Value of the LCS prepotential :math:`F_{\text{LCS}}`.

        See also: :func:`F_LCS_poly_per`

        See also: :func:`F_inst_per`

        """
        if self.maximum_degree>0:
            return self.F_LCS_poly_per(XPer,conj=conj)+self.F_inst_per(XPer,conj=conj)
        else:
            return self.F_LCS_poly_per(XPer,conj=conj)

    # DONE
    # -----------------------------------------------------------------------------------

    
        


