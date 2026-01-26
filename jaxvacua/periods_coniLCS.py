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
from .coniLCS_init import getAMatrix,get_basis_transformation
from .cicy_prepot import cicy_input
from .cytools_interface import cytools_instanton_data_init, cytools_model_data_init

# Some global variables
home_dir=os.path.dirname(os.path.realpath(__file__))
files_dir=home_dir+"/models"



class periods_coniLCS:
    
    def __init__(self, h12=None,model_ID = None,model_type = "KS",
                 maximum_degree = 0, mirror_cy = None,ncf=2,
                 model_data = None,instanton_data = None,
                 use_cytools = False,basis_transformation = None,conifold_curve=None,
                 grading_vector = None, prange = 500, use_gvs = False,save_file=False, **kwargs):
        r"""

        **Description:**
        Period class for the conifold large complex structure (coni-LCS) regime.
        This class inherits from the :class:`LCS_init` class and extends it by
        the conifold logarithmic structure. In particular, the prepotential
        :math:`F_{\mathrm{coni-LCS}}` is given by 
        
        .. math::
            F_{\mathrm{coni-LCS}} = F_{\mathrm{poly}} + F_{\mathrm{inst}} + F_{\mathrm{coni}}\, ,
            
        where :math:`F_{\mathrm{poly}}` and :math:`F_{\mathrm{inst}}` are the polynomial and instanton parts 
        of the large complex structure (LCS) prepotential :math:`F_{\mathrm{LCS}}` and
        :math:`F_{\mathrm{coni}}` is the conifold part. The polynomial and instanton parts 
        can be expressed in terms of the periods :math:`X^I=(X^0,X^{\mathrm{cf}},X^i)` as
        
        .. math::
            F_{\mathrm{poly}}(X) = -\frac{1}{6X^0}\widetilde{\kappa}_{ijk}X^iX^jX^k+\frac{1}{2}a_{ij}X^iX^j
                +b_{i}X^i\, X^0 + \dfrac{\text{i}}{2}\tilde{\xi}\, (X^0)^2\, ,
                
            F_{\mathrm{inst}}(X) = -\dfrac{(X^0)^2}{(2\pi \text{i})^3}\sum_{\beta} n_{\beta} \, \text{Li}_3\left(\exp\left(2\pi \text{i}\, \beta_i \frac{X^i}{X^0}\right)\right)\, ,
        
        where :math:`\widetilde{\kappa}_{ijk}` are the triple intersection numbers of 
        the mirror dual Calabi-Yau threefold :math:`\widetilde{X}`. Here, we defined
        .. math::
            a_{ij} = \dfrac{1}{2}\begin{cases}
                                    \widetilde{\kappa}_{iij} & i\geq j\\[0.3em]
                                    \widetilde{\kappa}_{ijj} & i<j
                                \end{cases} \, , \quad 
            b_i = \dfrac{1}{24} \int_{\tilde{D}^i}\, c_2(\widetilde{X})\, , \quad  
            \tilde{\xi}=\frac{\zeta(3)\, \chi(\widetilde{X})}{(2\pi)^3}\, .
            
        The conifold part :math:`F_{\mathrm{coni}}` is given by
        
        TODO: COMPLETE!

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
            prange (int): Range for polylog computations. Defaults to ``500``. Larger values increase accuracy but also computation time. 
            use_gvs (bool, optional): Whether to use GV invariants instead of GW invariants for the instanton sum. Defaults to ``False``. 
            ncf (int): Number of conifolds.
            conifold_curve (ArrayLike): Curve class of the conifold curve. If ``None``, the first basis vector is assumed to be the conifold curve.
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
        
        if 'nmax' in kwargs:
            self.nmax = kwargs["nmax"]
        else:
            self.nmax = 4

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


        self.coninop = jnp.identity(h12)[0]
        
        if ncf is None:
            raise ValueError("Need to provide value for the number of conifolds `ncf`!")

        self.ncf = ncf

        

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
            basis_transformation = model_data["basis transformation"]

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


        # Recompute a_matrix to match with conventions in paper
        self.a_matrix = jnp.array(getAMatrix(self.mirror_intersection_numbers))

        if self.max_deg>0:

            coninop = jnp.identity(h12)[0]

            if len(self.GW_charges)>0:
                flag = jnp.all(self.GW_charges==coninop,axis=1)==False
                
                if np.max(np.abs(self.GW_charges))>0:
                    ind = np.where(flag)[0]
                    if len(ind)==0:
                        raise ValueError("Conifold curve not found in GW charges! Check basis transformation!")

                self.GW_charges = self.GW_charges[flag]
                self.GW_inv = self.GW_inv[flag]
                self.GW_curve_degrees = self.GW_curve_degrees[flag]
                flag = jnp.all(self.GW_charges_lim==coninop,axis=1)==False
                self.GW_charges_lim = self.GW_charges_lim[flag]
                self.GW_inv_lim = self.GW_inv_lim[flag]

            if len(self.GV_charges)>0:
                flag = jnp.all(self.GV_charges==coninop,axis=1)==False

                if np.max(np.abs(self.GV_charges))>0:
                    ind = np.where(flag)[0]
                    if len(ind)==0:
                        raise ValueError("Conifold curve not found in GV charges! Check basis transformation!")


                self.GV_charges = self.GV_charges[flag]
                self.GV_curve_degrees = self.GV_curve_degrees[flag]
                self.GV_inv = self.GV_inv[flag]
                flag = jnp.all(self.GV_charges_lim==coninop,axis=1)==False
                self.GV_charges_lim = self.GV_charges_lim[flag]
                self.GV_inv_lim = self.GV_inv_lim[flag]


        for _ in range(5):
            ReZ = np.random.uniform(-1,1,(self.h12-1,))
            ImZ = np.random.uniform(-1,1,(self.h12-1,))*1e-2+self.tip_of_stretched_kahler_cone[1:]
            zbulk = ReZ+1j*ImZ

            ReZcf = np.random.uniform(-1,1)*1e-8
            ImZcf = np.random.uniform(-1,1)*1e-8
            zcf = ReZcf+1j*ImZcf

            X0 = 0.
            while jnp.abs(X0)<1e-10:
                X0 = np.random.uniform(-10,10)+1j*np.random.uniform(-10,10)

            val = self.ddddF_coniLCS_poly_per(X0,zcf,zbulk,conj=False)
            cval = self.ddddF_coniLCS_poly_per(jnp.conj(X0),jnp.conj(zcf),jnp.conj(zbulk),conj=True)

            if jnp.max(jnp.abs(val))>1e-10 or jnp.max(jnp.abs(cval))>1e-10:
                raise ValueError("Test on 4th derivative of polynomial prepotential for periods failed! Please check input!")

        # DONE
        # -----------------------------------------------------------------------------------        

        
    ###################################################################################################################################
    #################################### PREPOTENTIAL ETC. AS FUNCTIONS OF PERIODS FOR LCS ############################################
    ###################################################################################################################################

    @partial(jit, static_argnums = (0,2,))
    def F_coniLCS_poly_per(self, XPer: ArrayLike, conj: bool = False) -> complex:
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

        Args:
            XPer (ArrayLike): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            complex: Value of the polynomial constribution :math:`F_{\mathrm{poly}}` to 
                the LCS prepotential :math:`F_{\mathrm{LCS}}`.

        See also: :func:`F_coniLCS_per`

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


    @partial(jit, static_argnums = (0,4,))
    def F_coniLCS_poly_split_per(self, X0: ArrayLike, Xcf: ArrayLike, Xbulk: ArrayLike, conj: bool = False) -> complex:
        r"""
        **Description:**
        Dummy function to compute the polynomial part :math:`F_{\mathrm{poly}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
        in terms of the periods, where the periods are split into conifold and bulk parts.
        
        Args:
            X0 (ArrayLike): Value of period :math:`X^0`.
            Xcf (ArrayLike): Value of conifold period :math:`X^{\mathrm{cf}}`.
            Xbulk (ArrayLike): Values of bulk periods :math:`X^i`.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            complex: Value of the polynomial constribution :math:`F_{\mathrm{poly}}` to the LCS prepotential :math:`F_{\mathrm{LCS}}`.

        """
        XPer = jnp.append(jnp.array([X0,Xcf]),Xbulk)

        return self.F_coniLCS_poly_per(XPer,conj=conj)
        
        
    @partial(jit, static_argnums = (0,4,))
    def dF_coniLCS_poly_per(self, X0: ArrayLike, Xcf: ArrayLike, Xbulk: ArrayLike, conj: bool = False) -> complex:
        r"""
        **Description:**
        Computes the first derivative of the polynomial part :math:`F_{\mathrm{poly}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
        in terms of the periods.
        
        Args:
            X0 (ArrayLike): Value of period :math:`X^0`.
            Xcf (ArrayLike): Value of conifold period :math:`X^{\mathrm{cf}}`.
            Xbulk (ArrayLike): Values of bulk periods :math:`X^i`.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``. 
            
        Returns:
            complex: Value of the first derivative of the polynomial constribution :math:`F_{\mathrm{poly}}` to the LCS prepotential :math:`F_{\mathrm{LCS}}`.
            
        See also: :func:`F_coniLCS_per`
        """
        
        #return jax.grad(self.F_coniLCS_poly_per,holomorphic=True)(XPer,conj=conj)[1]
        return jax.grad(self.F_coniLCS_poly_split_per,holomorphic=True,argnums=1)(X0,Xcf,Xbulk,conj=conj)

    @partial(jit, static_argnums = (0,4,))
    def ddF_coniLCS_poly_per(self, X0: ArrayLike, Xcf: ArrayLike, Xbulk: ArrayLike, conj: bool = False) -> complex:
        r"""
        **Description:**
        Computes the second derivative of the polynomial part :math:`F_{\mathrm{poly}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
        in terms of the periods.
        
        Args:
            X0 (ArrayLike): Value of period :math:`X^0`.
            Xcf (ArrayLike): Value of conifold period :math:`X^{\mathrm{cf}}`.
            Xbulk (ArrayLike): Values of bulk periods :math:`X^i`.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            complex: Value of the second derivative of the polynomial constribution :math:`F_{\mathrm{poly}}` to the LCS prepotential :math:`F_{\mathrm{LCS}}`.
            
        See also: :func:`F_coniLCS_per`
        """

        #return jax.grad(self.dF_coniLCS_poly_per,holomorphic=True)(XPer,conj=conj)[1]
        return jax.grad(self.dF_coniLCS_poly_per,holomorphic=True,argnums=1)(X0,Xcf,Xbulk,conj=conj)

    @partial(jit, static_argnums = (0,4,))
    def dddF_coniLCS_poly_per(self, X0: ArrayLike, Xcf: ArrayLike, Xbulk: ArrayLike, conj: bool = False) -> complex:
        r"""
        **Description:**
        Computes the third derivative of the polynomial part :math:`F_{\mathrm{poly}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
        in terms of the periods.
        
        Args:
            X0 (ArrayLike): Value of period :math:`X^0`.
            Xcf (ArrayLike): Value of conifold period :math:`X^{\mathrm{cf}}`.
            Xbulk (ArrayLike): Values of bulk periods :math:`X^i`.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            complex: Value of the third derivative of the polynomial constribution :math:`F_{\mathrm{poly}}` to the LCS prepotential :math:`F_{\mathrm{LCS}}`.  
            
        See also: :func:`F_coniLCS_per`
        """

        #return jax.grad(self.ddF_coniLCS_poly_per,holomorphic=True)(XPer,conj=conj)[1]
        return jax.grad(self.ddF_coniLCS_poly_per,holomorphic=True,argnums=1)(X0,Xcf,Xbulk,conj=conj)

    @partial(jit, static_argnums = (0,4,))
    def ddddF_coniLCS_poly_per(self, X0: ArrayLike, Xcf: ArrayLike, Xbulk: ArrayLike, conj: bool = False) -> complex:
        r"""
        **Description:**
        Computes the fourth derivative of the polynomial part :math:`F_{\mathrm{poly}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
        in terms of the periods.
        
        Args:
            X0 (ArrayLike): Value of period :math:`X^0`.
            Xcf (ArrayLike): Value of conifold period :math:`X^{\mathrm{cf}}`.
            Xbulk (ArrayLike): Values of bulk periods :math:`X^i`.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            complex: Value of the fourth derivative of the polynomial constribution :math:`F_{\mathrm{poly}}` to the LCS prepotential :math:`F_{\mathrm{LCS}}`.
            
        See also: :func:`F_coniLCS_per`
        """

        # This function should vanish!
        return jax.grad(self.dddF_coniLCS_poly_per,holomorphic=True,argnums=1)(X0,Xcf,Xbulk,conj=conj)

    @partial(jit, static_argnums = (0,3,4,))
    def F_inst_per_coni(self,X0: complex ,XPerBulk: ArrayLike, conj: bool = False, n: int = 0) -> complex:
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

        See also: :func:`F_coniLCS_per`

        """
        #if n>4:
        #    
        #    raise ValueError("Value of `n` greater than 2 not tested!")
            
        XPer = jnp.append(X0,jnp.append(0.+1j*0.,XPerBulk))
        
        coeff = 2.*Pi*1j

        """
        if not conj:
            #return -jnp.sum(self.GV_inv_lim*(self.GV_charges_lim[:,0]**n)*jax_polylog_vmap(jnp.exp(2.*Pi*1j*jnp.einsum("ki,i",self.GV_charges_lim,XPer[1:])/XPer[0]),3-n,self.prange))/(2.*Pi*1j)**(3-n)*XPer[0]**(2)

            return -jnp.sum(self.GV_inv_lim*(self.GV_charges_lim[:,0]**n)*jax_polylog_vmap(jnp.exp(2.*Pi*1j*jnp.einsum("ki,i",self.GV_charges_lim,XPer[1:])/XPer[0]),3-n,self.prange))/(2.*Pi*1j)**(3-n)*XPer[0]**(2-n)
            
        else:
            #return -jnp.sum(self.GV_inv_lim*(self.GV_charges_lim[:,0]**n)*jax_polylog_vmap(jnp.exp(-2.*Pi*1j*jnp.einsum("ki,i",self.GV_charges_lim,XPer[1:])/XPer[0]),3-n,self.prange))/(-2.*Pi*1j)**(3-n)*XPer[0]**(2)

            return -jnp.sum(self.GV_inv_lim*(self.GV_charges_lim[:,0]**n)*jax_polylog_vmap(jnp.exp(-2.*Pi*1j*jnp.einsum("ki,i",self.GV_charges_lim,XPer[1:])/XPer[0]),3-n,self.prange))/(-2.*Pi*1j)**(3-n)*XPer[0]**(2-n)
        """

        if conj:
            coeff = -coeff

        # Compute polylog
        Li_val = jax_polylog_vmap(jnp.exp(coeff*jnp.einsum("ki,i",self.GV_charges_lim,XPer[1:])/XPer[0]),3-n,self.prange)
        
        # Compute sum over curves
        res = jnp.sum(self.GV_inv_lim*(self.GV_charges_lim[:,0]**n)*Li_val)

        # Add numerical prefactor
        return -res/(coeff)**(3-n)*XPer[0]**(2-n)

    @partial(jit, static_argnums = (0,2,))
    def F_coni_per(self, X: ArrayLike, conj: bool = False) -> complex:
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
            The basis transformation is stored in the attribute ``basis_transformation``.   
            
        Args:
            X (ArrayLike): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            complex: Value of the conifold part :math:`F_{\mathrm{conifold}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`. 
            
        """
        if conj:
            #return self.ncf*(X[1]/X[0])**2/(-4*jnp.pi*1j)*jnp.log(2*jnp.pi*1j*X[1]/X[0])
            return self.ncf*(X[1])**2/(-4*jnp.pi*1j)*jnp.log(2*jnp.pi*1j*X[1]/X[0])
        else:
            #return self.ncf*(X[1]/X[0])**2/(4*jnp.pi*1j)*jnp.log(-2*jnp.pi*1j*X[1]/X[0])
            return self.ncf*(X[1])**2/(4*jnp.pi*1j)*jnp.log(-2*jnp.pi*1j*X[1]/X[0])
        
    @partial(jit, static_argnums = (0,3,4,))
    def F_coniLCS_exp_per(self, 
                          X0: complex ,
                          XPerBulk: ArrayLike, 
                          conj: bool = False, 
                          n: int = 0
                          ) -> complex:
        r"""
        **Description:**
        Computes the expansion of the prepotential :math:`F` around the conifold point
        in terms of the periods. 
        
        .. admonition:: Details
            :class: dropdown
            
            The prepotential :math:`F` can be expanded around the conifold point as 
            
            .. math::
                F(X) = \sum_{n=0}^{\infty}\, \dfrac{1}{n!}\, \dfrac{\partial^n F}{\partial (X^{\mathrm{cf}})^n}\Bigg |_{X^{\mathrm{cf}}=0}\, (X^{\mathrm{cf}})^n\, .
                
            Here, :math:`X^{\mathrm{cf}}` is the period associated to the conifold curve.
            
            
            TODO : FINISH DOCSTRING WITH CONILCS DETAILS!!!
            
        Args:
            X0 (complex): Value of period :math:`X^0`.
            XPerBulk (ArrayLike): Values of bulk periods :math:`X^i`.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            n (int, optional): Order of derivative. Defaults to ``0``. 
            
        Returns:
            complex: Value of the :math:`n`-th derivative of the prepotential :math:`F` around the conifold point.
        """
        
        
        #if n>4:
        #    raise ValueError("Value of `n` greater than 2 not tested!")
        
        if conj:
            zeta_denom = (-2*jnp.pi*1j)
        else:
            zeta_denom = (2*jnp.pi*1j)

        if n!=3:
            zeta_denom = zeta_denom**(3-n)
        else:
            zeta_denom = 1.

        #coeff_cf = self.ncf/zeta_denom*X0**2
        coeff_cf = self.ncf/zeta_denom*X0**(2-n)

        #XPer = jnp.append(X0,jnp.append(0.+1j*0.,XPerBulk))
        Xcf = 0.+1j*0.
        
        if n==0:
            #return self.F_coniLCS_poly_per(XPer,conj=conj)-self.ncf/zeta_denom**3*jax.scipy.special.zeta(3,q=1)
            val = self.F_coniLCS_poly_split_per(X0,Xcf,XPerBulk,conj=conj)-coeff_cf*jax.scipy.special.zeta(3,q=1)
        elif n==1:
            #return self.dF_coniLCS_poly_per(XPer,conj=conj)-self.ncf/zeta_denom**2*jax.scipy.special.zeta(2,q=1)
            val = self.dF_coniLCS_poly_per(X0,Xcf,XPerBulk,conj=conj)-coeff_cf*jax.scipy.special.zeta(2,q=1)
        elif n==2:
            #return (self.ddF_coniLCS_poly_per(XPer,conj=conj)-self.ncf/zeta_denom*3/2)/2
            val = self.ddF_coniLCS_poly_per(X0,Xcf,XPerBulk,conj=conj)-coeff_cf*3/2
        elif n==3:
            #return (self.dddF_coniLCS_poly_per(XPer,conj=conj)-self.ncf*??????/zeta_denom**(??))/2/3
            # Using B_1 = 1/2 here and zeta(0)=-1/2
            val = self.dddF_coniLCS_poly_per(X0,Xcf,XPerBulk,conj=conj)+coeff_cf/2
        elif n>=4:

            val = coeff_cf*jax.scipy.special.bernoulli(n-2)[-1]/(n-2)

        if n>1:
            val = val/jax.scipy.special.gamma(n+1)

        return val
        

    @partial(jit, static_argnums = (0,2,))
    def F_coniLCS_per(self, XPer: ArrayLike, conj: bool = False) -> complex:
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

            The former is computed via :func:`F_coniLCS_poly_per`, while the latter in :func:`F_inst_per`.

            TODO: FINISH DOCSTRING WITH CONILCS DETAILS!!!

        Args:
            XPer (ArrayLike): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            complex: Value of the LCS prepotential :math:`F_{\text{LCS}}`.

        See also: :func:`F_coniLCS_poly_per`

        See also: :func:`F_coniLCS_exp_per`

        See also: :func:`F_inst_per_coni`

        """
        val = self.F_coni_per(XPer[:2],conj=conj)
        
        XPerBulk = XPer[2:]
        XConi = XPer[1]
        X0 = XPer[0]
        

        for n_exp in range(self.nmax+1):
            tmp = self.F_coniLCS_exp_per(X0,XPerBulk,conj=conj,n=n_exp)
        
            if self.maximum_degree>0:
                tmp = tmp + self.F_inst_per_coni(X0,XPerBulk,conj=conj,n=n_exp)


            if n_exp>0:
                val = val + tmp*(XConi)**(n_exp)
            else:
                val = val + tmp


        
        
        return val

    # DONE
    # -----------------------------------------------------------------------------------

    
        


