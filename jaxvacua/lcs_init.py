# ==============================================================================
# This code is written by Andreas Schachner.
#
# If any questions arise, please feel free to reach out to me (Andreas) either at
# andreas.schachner@gmx.net or at as3475@cornell.edu .
# ==============================================================================
#
# ------------------------------------------------------------------------------
# This file holds functions for ....
# ------------------------------------------------------------------------------

#Standard libraries
# Important standard libraries
import os, sys, warnings
import numpy as np
import itertools
import getopt
from functools import partial

# Import jax modules
import jax
import jax.numpy as jnp
from jax import jit
from jax import Array
from numpy.typing import ArrayLike
from jax.scipy.special import zeta
from jax.numpy import pi as Pi


# To load pickle files
import pickle
import gzip

jax.config.update("jax_enable_x64", True)



# -----------------------------------------------------------------------------------
# Define init functions

def LCS_init(self,model_data):
    r"""
    **Description:**
    Initializes the large complex structure limit (LCS) data of the Calabi-Yau model. 
        
    Args:
        model_data (dict): A dictionary containing the necessary data for the Calabi-Yau model.
            Required keys in the dictionary:
                - "intersection numbers": Triple intersection numbers of the Calabi-Yau manifold.
                - "a_matrix": The matrix 'a' used in the prepotential.
                - "second chern": Second Chern class of the Calabi-Yau manifold.
                - "chi": Euler characteristic of the Calabi-Yau manifold.
            Optional keys in the dictionary:
                - "intersection numbers coo": Coordinate representation of the intersection numbers.
                - "hyperplanes": Hyperplanes defining the Kähler cone.
                - "Q": Charge matrix.
                - "tip stretched KC": Tip of the stretched Kähler cone.
                - "generators KC": Generators of the Kähler cone.
                - "generators simplicial KC": Generators of the simplicial Kähler cone.
                - "generators MC": Generators of the Mori cone.
                - "rays KC": Rays of the Kähler cone.
                - "polytope points": Points of the polytope (for toric Calabi-Yau).
                - "mirror heights": Heights of the mirror polytope.
                - "extra_data": Any additional data.
                - "name": Name of the Calabi-Yau model. 
                
    Raises:
        ValueError: If the input dictionary is missing any of the required keys.
        
    Returns:
        None: The function initializes the attributes of the class instance.
    """
    
    model_data_keys=list(model_data.keys())
    required_keys=["intersection numbers","a_matrix","second chern","chi"]
    # Test if necessary input is available
    if not all(x in model_data_keys for x in required_keys):
        raise ValueError(f"The input dictionary misses necessary keys which must include all of {required_keys}!")
    
    # Grab input data:
    self.mirror_intersection_numbers = jnp.array(model_data["intersection numbers"])
    a_matrix = model_data["a_matrix"]
    self.chi = model_data["chi"]
    self.mirror_second_chern_class = model_data["second chern"]
    self.b_vector = jnp.array(self.mirror_second_chern_class)/24.
    
    # Prepare input
    self.K0 = zeta(3,q=1)*self.chi/(2.*jnp.pi*1j)**(3)

    # Get intersection numbers
    if "intersection numbers coo" in model_data_keys:
        dlist=[]
        for col in model_data["intersection numbers coo"]:
            perms=np.unique(list(itertools.permutations(col[:3])),axis=0)
            dlist.append([col[0],col[1],col[2],col[3]*len(perms)])
            
        self.kappa_tilde_sparse=jnp.array(dlist)
    
    else:
        # Compute coordinate representation for triple intersection numbers
        # Each row contains entries: [i,j,k,\kappa_{ijk}*perm({i,j,k})] with i,j,k labeling the different moduli 
        # and \kappa_{ijk} the corresponding triple intersection number.
        # The multiplicative factor perm({i,j,k}) takes care of symmetrisation over indices in the calculations below.
        dlist=[]
        for i in range(self.h12):
            for j in range(i,self.h12):
                for k in range(j,self.h12):
                    if self.mirror_intersection_numbers[i][j][k]!=0:
                        perms=np.unique(list(itertools.permutations([i,j,k])),axis=0)
                        dlist.append([i,j,k,int(self.mirror_intersection_numbers[i][j][k])*len(perms)])
                    
        self.kappa_tilde_sparse=jnp.array(dlist)
    
    
    # Update a_matrixs if necessary
    if len(a_matrix)>0:
        self.a_matrix = jnp.array(a_matrix)
    else:
        a_matrix = np.zeros((self.h12, self.h12))
        
        for I1 in range(self.h12):
            for I2 in range(self.h12):
                #Notice that we do not have the factor of (-1) as in eq. (2.15) in https://inspirehep.net/files/17fc09bdeef5d049b05b13d796badb84
                if self.h12!=51:
                    a_matrix[I1][I2] = self.mirror_intersection_numbers[I1][I2][I2]/2.
                else:
                    if I1>=I2:
                        a_matrix[I1][I2] = self.mirror_intersection_numbers[I1][I1][I2]/2.
                    elif I1<I2:
                        a_matrix[I1][I2] = self.mirror_intersection_numbers[I1][I2][I2]/2.
                
        
        # Take same convention as in https://inspirehep.net/files/d2f57319d398cfe81212b19c7f9d109f for CP11169
        if self.h12==2 and self.model_ID==1:
            a_matrix[0][1]=a_matrix[1][0]
        
        self.a_matrix = jnp.array(a_matrix)
        
    a_matrix = None
    del a_matrix

    dlist=[]
    blist = []
    for I1 in range(self.h12):
        for I2 in range(self.h12):
            if self.a_matrix[I1][I2]!=0:
                # ASSUMING IT IS SYMMETRIC?!
                #col = [I1,I2,a_matrix[I1][I2]]
                #perms=np.unique(list(itertools.permutations(col[:2])),axis=0)
                #dlist.append([col[0],col[1],col[2]*len(perms)])
                dlist.append([I1,I2])
                blist.append(self.a_matrix[I1][I2])
        
    #self.a_matrix_sparse = jnp.array(dlist).astype(int)
    #self.a_matrix_sparse_values = jnp.array(blist)
    
    if "hyperplanes" in model_data_keys:
        self.hyperplanes=model_data["hyperplanes"]
    else:
        if self.model_type=="CICY":
            self.hyperplanes=jnp.identity(self.h12)
        elif self.model_type=="KS":
            self.hyperplanes=jnp.array([[]])
    
    if "Q" in model_data_keys:
        self.Q=model_data["Q"]
    else:
        self.Q=None

    if "tip stretched KC" in model_data_keys:
        self.tip_of_stretched_kahler_cone=model_data["tip stretched KC"]
    else:
        if self.model_type=="CICY":
            self.tip_of_stretched_kahler_cone=jnp.ones(self.h12)
        elif self.model_type=="KS":
            self.tip_of_stretched_kahler_cone=[]
        
    if "generators KC" in model_data_keys:
        rays = model_data["generators KC"]
        if len(rays)>0:
            self.generators_kahler_cone=rays/jnp.linalg.norm(rays,axis=1)[:,None]
        else:
            self.generators_kahler_cone=jnp.array(rays)
    else:
        #if self.model_type=="CICY":
        #    self.generators_kahler_cone=jnp.identity(self.h12)
        #elif self.model_type=="KS":
            
        self.generators_kahler_cone=[]

    if "generators simplicial KC" in model_data_keys:
        rays = model_data["generators simplicial KC"]
        if len(rays)>0:
            self.generators_simplicial_kahler_cone=rays/jnp.linalg.norm(rays,axis=1)[:,None]
        else:
            self.generators_simplicial_kahler_cone=jnp.array(rays)
    else:
            
        self.generators_simplicial_kahler_cone=[]

    if "generators MC" in model_data_keys:
        rays = model_data["generators MC"]
        self.generators_mori_cone=jnp.array(rays)
    else:   
        self.generators_mori_cone=[]
        
    if "rays KC" in model_data_keys:
        rays = model_data["generators KC"]
        if len(rays)>0:
            self.rays_kahler_cone=rays/jnp.linalg.norm(rays,axis=1)[:,None]
        else:
            self.rays_kahler_cone=jnp.array(rays)
    else:
        self.rays_kahler_cone=[]
        
    if "polytope points" in model_data_keys:
        self.polytope_points=model_data["polytope points"]
    else:
        self.polytope_points=[]

    if "mirror heights" in model_data_keys:
        self.heights=model_data["mirror heights"]
    else:
        self.heights=[]

    if "extra_data" in model_data_keys:
        self.extra_data=model_data["extra_data"]
    else:
        self.extra_data=None

    if "name" in model_data_keys:
        self.name=model_data["name"]
    else:
        self.name=""

def instanton_init(self,instanton_data,mirror_cy,grading_vector,use_cytools):
    r"""
    **Description:**
    Initializes the instanton data of the Calabi-Yau model.
    
    Args:
        instanton_data (dict): A dictionary containing the necessary instanton data.
            Required keys in the dictionary:
                - "gws": Gromov-Witten invariants.
                - "gvs": Gopakumar-Vafa invariants.
            Optional keys in the dictionary:
                - "grading_vector": A vector used for grading the curve classes.
        mirror_cy (object): An object representing the mirror Calabi-Yau manifold.
        grading_vector (ArrayLike, optional): A vector used for grading the curve classes. If
            provided, it overrides the "grading_vector" in the instanton_data dictionary.
        use_cytools (bool): A flag indicating whether to use CYTools for certain computations.
        
    Raises:
        ValueError: If the input dictionary is missing any of the required keys or if no grading vector is provided when needed. 
        ImportError: If CYTools is not installed but required for the computations.
        
    Returns:
        None: The function initializes the attributes of the class instance.
    """
        
        
    if self.max_deg>0:

        dict_GW = instanton_data["gws"]

        if dict_GW is None:
            raise ValueError("No GW invariants provided!")

        dict_GV = instanton_data["gvs"]

        if grading_vector is None:
            self.grading_vector = instanton_data["grading_vector"]
        else:
            self.grading_vector = grading_vector
        
        if self.grading_vector is None:
            raise ValueError("No grading vector provided!")
            #warnings.warn("No grading vector provided!")
            #self.grading_vector = jnp.ones(self.h12)

        self.GW_curve_degrees = jnp.array(np.array(list(dict_GW.keys()))@self.grading_vector)
        self.GV_curve_degrees = jnp.array(np.array(list(dict_GV.keys()))@self.grading_vector)

        keys=list(dict_GW.keys())
        if self.h12==1:
            self.GW_charges=jnp.array([jnp.array([key for key in keys])])
            if len(self.GW_charges.shape)>2:
                self.GW_charges = self.GW_charges[0]
        else:
            self.GW_charges=jnp.array(keys)

        gw_values = list(map(float,list(dict_GW.values())))
        self.GW_inv=jnp.array(list(gw_values))


        keys=list(dict_GV.keys())
        if self.h12==1:
            self.GV_charges=jnp.array([jnp.array([key for key in keys])])
            if len(self.GV_charges.shape)>2:
                self.GV_charges = self.GV_charges[0]
        else:
            self.GV_charges=jnp.array(keys)
            
        # Below makes sure that we don't run into issues with too long integers!
        try:
            self.GV_inv=jnp.array(np.array(list(dict_GV.values())).astype(float))
        except:
            self.GV_inv=jnp.array(np.array(list(dict_GV.values())))

        self.max_available_deg=int(np.max(self.GV_curve_degrees))

        if self.max_deg>self.max_available_deg:
            self.max_deg = self.max_available_deg
            warnings.warn(f"Given maximal degree {self.max_deg} is larger than available degree {self.max_available_deg}! Using {self.max_available_deg} for the computation.")
            #raise ValueError(f"Given maximal degree {self.max_deg} is larger than available degree of {self.max_available_deg}!\
            #                    Compute higher order GV invariants or lower the maximal degree!")


        ind = jnp.where(self.GW_curve_degrees <= self.max_deg)[0]
        self.GW_charges_lim = self.GW_charges[ind]
        self.GW_inv_lim = self.GW_inv[ind]
        
        ind = jnp.where(self.GV_curve_degrees <= self.max_deg)[0]
        self.GV_charges_lim = self.GV_charges[ind]
        self.GV_inv_lim = self.GV_inv[ind]
        
    else:
        self.max_available_deg=0.
        self.charges = jnp.array([jnp.zeros(self.h12)])
        
        self.GW_inv = jnp.array([0.])
        self.GW_charges = self.charges
        self.GW_inv_lim = jnp.array([0.])
        self.GW_charges_lim = self.charges
        
        self.GV_inv = jnp.array([0.])
        self.GV_charges = self.charges
        self.GV_inv_lim = jnp.array([0.])
        self.GV_charges_lim = self.charges
        self.grading_vector = None

