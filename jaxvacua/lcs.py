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

"""Calabi-Yau model-data container for large-complex-structure workflows.

Purpose
-------
Define ``lcs_tree``, the JAX-pytree data container that gathers the
topological, cone, enumerative and conifold data needed to build periods and
flux EFTs.

Main public API
---------------
- ``coo_to_intnums``: convert sparse symmetric intersection-number data to a
  dense tensor.
- ``lcs_tree``: stores Hodge data, intersection numbers, Chern data,
  Gopakumar-Vafa/Gromov-Witten invariants, cone information, basis changes,
  conifold descriptors and metadata.
- Constructors and import helpers for dictionaries, saved model files and
  CYTools-derived geometries.

Design notes
------------
``lcs_tree`` is the shared geometry object passed into ``periods``, ``css``,
``FluxEFT`` and search classes.  It should remain lightweight to flatten and
stable across JAX transformations.
"""


# Important standard libraries
import os, sys, warnings
import numpy as np
import itertools
from typing import Any, Callable, Dict, Optional, Tuple, Union


import jax
from jax.tree_util import register_pytree_node
import jax.numpy as jnp
from jax.scipy.special import zeta
from jax import Array
from .conifold import compute_a_matrix, get_basis_change, Conifold
from .cytools_interface import compute_intersection_numbers_coo, cytools_model_data_init
from .util import load_zipped_pickle,flatten_func,unflatten_func_class


from itertools import permutations
import numpy as np

def coo_to_intnums(i, j, k, v, h11: int) -> np.ndarray:
    r"""
    **Description:**
    Converts intersection numbers from COO format to a dense 3D array format.
    
    Args:
        i (array-like): Array of first indices of the intersection numbers.
        j (array-like): Array of second indices of the intersection numbers.
        k (array-like): Array of third indices of the intersection numbers.
        v (array-like): Array of values of the intersection numbers corresponding to the indices (i, j, k).
        h11 (int): Hodge number h^{1,1} of the mirror CY, which determines the size of the output array.
        
    Returns:
        np.ndarray: A 3D array of shape (h11, h11, h11) containing the intersection numbers, where the entry at (i, j, k) is given by the corresponding value in `v` for the indices (i, j, k) in COO format.
    """
    intnums = np.zeros((h11, h11, h11), dtype=np.int32)
    i, j, k, v = np.asarray(i), np.asarray(j), np.asarray(k), np.asarray(v)
    for pi, pj, pk in set(permutations(range(3))):
        idx = (np.array([i, j, k])[[pi, pj, pk]])
        intnums[idx[0], idx[1], idx[2]] = v
    return intnums


class lcs_tree(object):
    """
    **Description:** Pytree structure to hold on to information about a Calabi-Yau model, including intersection numbers, Gopakumar-Vafa/Gromov-Witten invariants, cone data, and conifold curve information.
    
    """
    
    def __init__(self,
                 
                 # Basic data
                 h11 : int = 1,
                 h12 : int = 2,
                 Q: int | None = None,
                 
                 # Period data
                 intnums: Array | None = None,
                 intnums_coo: Array | None = None,
                 c2: Array | None = None,
                 a_matrix: Array | None = None,
                 b_vector: Array | None = None,
                 chi: int | None = None,
                 basis_change: Array | None = None,
                 
                 # Enumerative invariants
                 gvs: Dict | Array | None = None,
                 gws: Dict | Array | None = None,
                 grading_vector: Array | None = None,
                 maximum_degree: int | None = None,
                 
                 # Model and limit data
                 model_type: str = "KS",
                 model_ID: int = 1,
                 limit: str = "LCS",
                 name: str | None = None,
                 
                 # Conifold curve data for coniLCS limits
                 conifold: Conifold | None = None,
                 conifolds: Tuple[Conifold, ...] | None = None,
                 active_conifold_idx: int | None = None,
                 conifold_curve: Array | None = None,
                 ncf: int | None = 2,
                 conifold_basis: bool | None = True,
                 conifold_limits: Array | None = None,
                 n_conifolds: int | None = None,
                 
                 # Cone data
                 hyperplanes: Array | None = None,
                 tip_skc: Array | None = None,
                 kahler_generators: Array | None = None,
                 kahler_rays: Array | None = None,
                 mori_rays: Array | None = None,
                 
                 # KS data
                 polytope_points: Array | None = None,
                 heights: Array | None = None,
                 
                 # Non-pert. superpotential
                 Wnp: complex | float = 0.+1j*0.,
                 
                 # Extra data
                 extra_data: Dict | None = None,
                 
                 verbosity: int = 0,
                 **kwargs):
        r"""
        **Description:**
        Constructor for the lcs_tree class. Initializes the Calabi-Yau model data including intersection numbers, GV/GW invariants, Kähler cone data, and conifold curve information.

        Args:
            h11 (int): Hodge number h^{1,1} of the mirror CY. Default is 1.
            h12 (int): Hodge number h^{1,2} of the mirror CY. Default is 2.
            cy (cytools.CalabiYau | None, optional): CYTools Calabi-Yau object. Default is None.
            intnums (Array | None, optional): Triple intersection numbers.
            intnums_coo (Array | None, optional): Intersection numbers in COO format.
            c2 (Array | None, optional): Second Chern class.
            a_matrix (Array | None, optional): a-matrix for the prepotential.
            b_vector (Array | None, optional): b-vector for the prepotential. Defaults to c2/24.
            gvs (Dict | Array | None, optional): Gopakumar-Vafa invariants.
            gws (Dict | Array | None, optional): Gromov-Witten invariants.
            grading_vector (Array | None, optional): Grading vector for curve classes.
            chi (int | None, optional): Euler characteristic of the mirror CY.
            model_type (str, optional): Type of model, either "KS" or "CICY". Default is "KS".
            model_ID (int, optional): Model identifier. Default is 1.
            limit (str, optional): Moduli space limit, one of "LCS", "coniLCS", "coniLCS_series", "coniLCS_bulk". Default is "LCS".
            basis_change (Array | None, optional): Basis transformation matrix.
            conifold_curve (Array | None, optional): Conifold curve charges.
            ncf (int | None, optional): Number of conifolds.
            hyperplanes (Array | None, optional): Hyperplanes defining the Kähler cone.
            Q (int | None, optional): Charge of the orientifold.
            tip_skc (Array | None, optional): Tip of the stretched Kähler cone.
            kahler_generators (Array | None, optional): Generators of the Kähler cone.
            kahler_rays (Array | None, optional): Rays of the Kähler cone.
            mori_rays (Array | None, optional): Rays of the Mori cone.
            polytope_points (Array | None, optional): Points of the polytope.
            heights (Array | None, optional): Heights of the triangulation.
            extra_data (Dict | None, optional): Any additional data.
            name (str | None, optional): Name of the model.
            conifold_basis (bool | None, optional): Whether to use the conifold basis. Default is True.
            verbosity (int, optional): Verbosity level. Default is 0.

        Raises:
            ValueError: If neither `intnums` nor `c2` is provided, or if required data for the specified limit is missing.
        """

        if (intnums is None and intnums_coo is None) or c2 is None:
            raise ValueError("Please provide at least the intersection numbers `intnums` or the second Chern class `c2` to initialize the LCS-tree object!")
        
        if limit not in ["LCS","coniLCS","coniLCS_series","coniLCS_bulk",None]:
            raise ValueError(f"Invalid limit! Please choose one of 'LCS', 'coniLCS', 'coniLCS_series', 'coniLCS_bulk' or 'None'!")

        
        self.h11 = h11
        self.h12 = h12
        self.model_ID = model_ID
        self.model_type = model_type
        self.extra_data = extra_data
        self.name = name
        self.Wnp = Wnp
        
        if model_type=="KS":
            self.polytope_points=polytope_points
            self.heights=heights
        elif model_type=="CICY":
            weight_matrix = None
            self.weight_matrix=weight_matrix
            
        
        
        self.limit = limit
        self.basis_change = basis_change
        
        
        # ---------- Intersection data ----------
        if intnums is None:
            if intnums_coo is None:
                raise ValueError("Please provide either `intnums` or `intnums_coo`!")
            
            intnums = coo_to_intnums(intnums_coo[:,0], intnums_coo[:,1], intnums_coo[:,2], intnums_coo[:,3], h12)
            
            intnums_coo = jnp.array(intnums_coo)
            
        
        self.intnums = jnp.array(intnums)
        self.intnums_coo = intnums_coo
        self.c2 = c2
        self.a_matrix = a_matrix
        
        # Set the mirror chi here
        if chi is None:
            # mirror chi
            self.chi = -2*(self.h11-self.h12)
        else:
            # The relation chi = -2*(h11-h12) holds for KS/CICY mirror pairs
            # but NOT for hypergeometric one-modulus models (which are not
            # KS/CICY threefolds), so the consistency check is skipped there.
            if self.model_type != "hypergeometric" and chi != -2*(self.h11-self.h12):
                raise ValueError(f"Input value for chi seems inconsistent with h11 and h12! For the (mirror) Calabi-Yau threefold, expect chi=-2*(h11-h12)={-2*(self.h11-self.h12)}, but got {chi} instead!")

            self.chi = chi

        self.chi = jnp.int32(self.chi)

        self.K0 = jnp.complex_(zeta(3,q=1)*self.chi/(2.*jnp.pi*1j)**(3))
        
        
        # ---------- GV/GW data ----------
            
        self.grading_vector = grading_vector
        
        self.gw_charges, self.gw_invariants, self.gw_degrees = self._charges_from_gv_gw(gws)
        self.gv_charges, self.gv_invariants, self.gv_degrees = self._charges_from_gv_gw(gvs)
        
        if maximum_degree is not None and maximum_degree > 0:
            if self.grading_vector is None:
                raise ValueError("Cannot apply maximum degree cutoff without grading vector! Please provide a grading vector to apply the maximum degree cutoff!")
            
            self.maximum_degree = maximum_degree
            if self.gv_charges is not None and self.gv_invariants is not None:
                gv_degrees = self.gv_charges@self.grading_vector
                gv_degrees_mask = gv_degrees <= self.maximum_degree
                self.gv_charges = self.gv_charges[gv_degrees_mask]
                self.gv_invariants = self.gv_invariants[gv_degrees_mask]
            
            if self.gw_charges is not None and self.gw_invariants is not None:
                gw_degrees = self.gw_charges@self.grading_vector
                gw_degrees_mask = gw_degrees <= self.maximum_degree
                self.gw_charges = self.gw_charges[gw_degrees_mask]
                self.gw_invariants = self.gw_invariants[gw_degrees_mask]
        
        # ---------- Cone data ----------
        
        self.tip_skc=tip_skc
        self.hyperplanes = hyperplanes
        
        if kahler_generators is None:
            self.generators_kahler_cone = None
        else:
            if len(kahler_generators)>0:
                self.generators_kahler_cone=kahler_generators/jnp.linalg.norm(kahler_generators,axis=1)[:,None]
            else:
                self.generators_kahler_cone=jnp.array(kahler_generators)
        
        if kahler_rays is None:
            self.rays_kahler_cone = None
        else:
            if len(kahler_rays)>0:
                self.rays_kahler_cone=kahler_rays/jnp.linalg.norm(kahler_rays,axis=1)[:,None]
            else:
                self.rays_kahler_cone=jnp.array(kahler_rays)
            
        
        self.rays_mori_cone = mori_rays
        
        # ---------- Conifold container ----------
        # Resolve self.conifolds, self.conifold, self._active_conifold_idx by
        # priority order: conifolds=, conifold=, legacy conifold_limits=,
        # else None (LCS) or raise (coniLCS).
        self.conifolds = None
        self._active_conifold_idx = None
        if conifold_curve is not None:
            self.conifold = Conifold.from_data(
                ncf=ncf,
                conifold_curve=jnp.asarray(conifold_curve),
            )
        elif conifold is not None:
            self.conifold = conifold
        elif conifolds is not None and active_conifold_idx is not None:
            if active_conifold_idx < 0 or active_conifold_idx >= len(conifolds):
                raise ValueError(f"active_conifold_idx={active_conifold_idx} is out of bounds for conifolds with length {len(conifolds)}! Please provide a valid index for the active conifold!")
            self.conifold = conifolds[active_conifold_idx]
            self._active_conifold_idx = active_conifold_idx
        else:
            self.conifold = None
            
        if self.conifold is not None:
            if "coniLCS" not in self.limit:
                raise ValueError("Input for conifold curve provided but limit is not coniLCS! Please make sure that the input is consistent with the limit!")
            
        if conifolds is not None:
            self.conifolds = tuple(conifolds)
            idx = 0
            if self.conifold is not None and active_conifold_idx is None:
                for c in self.conifolds:
                    if c.conifold_curve is not None:
                        if self.conifold is not None and not jnp.allclose(c.conifold_curve, self.conifold.conifold_curve):
                            break
                        idx += 1
                        
                self._active_conifold_idx = int(idx)
            else:
                self._active_conifold_idx = None
        else:
            self.conifolds = None
            self._active_conifold_idx = None
        
        self.conifold_basis = conifold_basis

        # coniLCS-specific bookkeeping on the active conifold.
        if "coniLCS" in (self.limit or ""):
            if self.conifold is None:
                raise ValueError("coniLCS limit requires an active conifold.")

            # Propagate global lcs_tree.basis_change down to the active conifold
            # if its own _basis_change is unset.
            if self.conifold._basis_change is None and self.basis_change is not None:
                self.conifold._basis_change = jnp.asarray(self.basis_change)

            # Stamp legacy conifold_curve= kwarg if the active conifold lacks one.
            if self.conifold.conifold_curve is None and conifold_curve is not None:
                self.conifold.conifold_curve = jnp.asarray(conifold_curve)
                if self.conifold._basis_change is None and conifold_basis:
                    self.conifold._basis_change = jnp.asarray(get_basis_change(conifold_curve))

            # If lcs_tree has no global basis_change yet but the active conifold
            # does (or has just been computed), adopt it.
            bc_curr = self.conifold.basis_change()
            if self.basis_change is None and bc_curr is not None and conifold_basis:
                self.basis_change = bc_curr

            # Materialise canonical-basis charge if missing.
            if (self.conifold.conifold_curve0 is None
                    and self.conifold.conifold_curve is not None
                    and bc_curr is not None):
                self.conifold.conifold_curve0 = self.conifold.conifold_curve @ bc_curr.T
            
            """
            if self.conifold_basis:
                self.conifold.conifold_curve0 = jnp.identity(self.h12)[0]
            else:
                self.conifold.conifold_curve0 = self.conifold.conifold_curve
            """

            # Validate: conifold_curve @ basis_change.T must equal (1,0,...,0).
            if (self.conifold.conifold_curve is not None and bc_curr is not None):
                coninop0 = self.conifold.conifold_curve @ bc_curr.T
                expected = jnp.identity(self.h12)[0]
                if not jnp.allclose(coninop0, expected):
                    bc_curr = -bc_curr
                    self.basis_change = bc_curr
                    coninop0 = self.conifold.conifold_curve @ bc_curr.T
                    if not jnp.allclose(coninop0, expected):
                    
                        raise ValueError(
                            f"Input of basis transformation and conifold curve seems "
                            f"incompatible! After applying the basis transformation, "
                            f"expect (1,0,0,...,0), but got {coninop0}!"
                        )
        
        # ---------- Basis change ----------
        
        if self.basis_change is not None:
            self._perform_basis_change()
        
        
        # ---------- Prepare extra data ----------
        
        if b_vector is not None:
            if b_vector.shape != (self.h12,):
                raise ValueError(f"Input for b_vector has wrong shape! Expected shape {(self.h12,)}, but got {b_vector.shape} instead!")
            if np.all(b_vector - jnp.array(self.c2)/24. == 0):
                raise ValueError("Input for b_vector seems to be the same as c2/24! Please make sure that the input is consistent with the basis transformation and the limit!")
            
            self.b_vector = b_vector
        else:
            self.b_vector = jnp.array(self.c2)/24.
        
        self._prepare_prepot()
        
        # Remove conifold curve when required for limit from
        # set of charges
        # Compute the conifold index
        if "coniLCS" in self.limit and maximum_degree>0:
            
            self._update_conifold_curve_and_index()
            
        # Store additional keyword arguments
        self.__dict__.update(kwargs)
        
    def _update_conifold_curve_and_index(self):
        r"""
        **Description:**
        Updates the conifold curve and its index in the GV charge data, and removes the conifold curve from the GV charges and invariants if required by the limit.
        
        Args:
            None
            
        Returns:
            None
        
        """
        
        gv_charges = self.gv_charges
        gv_invariants = self.gv_invariants
        
        if self.conifold_basis:
            coninop0 = self.conifold.conifold_curve0
        else:
            coninop0 = self.conifold.conifold_curve
            if coninop0 is None:
                raise ValueError("Conifold curve data not found in lcs_tree! Please check conifold curve input and basis choice!")
            
        # Find index of conifold curve in GV charge data 
        flag = jnp.any(gv_charges!=coninop0,axis=1)
        # Test if conifold curve is found in GV charge data
        self.coni_index = jnp.where(flag==False)[0]
        if len(self.coni_index)==0:
            raise ValueError("Could not find conifold curve in GV charge data! Please check conifold curve input and basis choice!")
        
        if len(self.coni_index)>1:
            raise ValueError("Found multiple matches for conifold curve in GV charge data! Please check conifold curve input and basis choice!")
        
        self.coni_index = int(self.coni_index[0])
        
        # Remove conifold curve from gvs!
        if self.limit in ["coniLCS_series","coniLCS_bulk"]:
            gv_charges = gv_charges[flag]
            gv_invariants = gv_invariants[flag]
            self.gv_charges = gv_charges
            self.gv_invariants = gv_invariants
            self.gv_degrees = self.gv_degrees[flag]
        
    def _perform_basis_change(self):
        r"""
        **Description:**
        Performs a basis change on the CY data, including intersection numbers, second Chern class, cone generators and rays, hyperplanes, Mori cone rays and GV/GW charges.
        
        Args:
            None
            
        Returns:
            None
        
        """
        
        L = self.basis_change
        
        if L is None:
            raise ValueError("Bais change matrix is None! Please provide a valid basis change matrix to perform the basis transformation!")
        
        if L.shape != (self.h12, self.h12):
            raise ValueError(f"Basis change matrix has wrong shape! Expected shape {(self.h12, self.h12)}, but got {L.shape} instead!")
        
        Linv = jnp.linalg.inv(L)

        # Transform intersection numbers by basis change
        self.intnums = jnp.einsum('ai,ibc->abc', L, jnp.einsum('bj,ijc->ibc', L, jnp.einsum('ck,ijk->ijc', L, self.intnums)))
        # Transform second Chern class by basis change
        self.c2 = jnp.matmul(L, self.c2)
        
        self.intnums_coo = compute_intersection_numbers_coo(self.intnums).astype(int)
        
        # Transform cone generators and rays to new basis
        if self.generators_kahler_cone is not None:
            self.generators_kahler_cone = jnp.matmul(self.generators_kahler_cone, Linv)

        if self.rays_kahler_cone is not None:
            self.rays_kahler_cone = jnp.matmul(self.rays_kahler_cone, Linv)

        # Transform stretched cone point
        if self.tip_skc is not None:
            self.tip_skc = jnp.matmul(self.tip_skc, Linv)
            
        # Transform hyperplanes
        if self.hyperplanes is not None:
            self.hyperplanes = jnp.matmul(self.hyperplanes, L.T)
        
        # Transform mori cone rays
        if self.rays_mori_cone is not None:
            self.rays_mori_cone = jnp.matmul(self.rays_mori_cone, L.T)
        
        # Transform Gopakumar-Vafa charges
        if self.gv_charges is not None:
            self.gv_charges = jnp.matmul(self.gv_charges,L.T)

        # Transform Gromov-Witten charges
        if self.gw_charges is not None:
            self.gw_charges = jnp.matmul(self.gw_charges,L.T)
            
        # Transform grading vector
        if self.grading_vector is not None:
            self.grading_vector=jnp.matmul(self.grading_vector,Linv)
        
        
    def _prepare_prepot(self):
        r"""
        **Description:**
        Prepares the prepotential data, including intersection numbers, a_matrix, and b_vector.
        
        Args:
            None
            
        Returns:
            None
            
        """

        # Get intersection numbers
        if self.intnums_coo is not None:
            dlist=[]
            for col in self.intnums_coo:
                perms=np.unique(list(itertools.permutations(col[:3])),axis=0)
                dlist.append([col[0],col[1],col[2],col[3]*len(perms)])
                
            self.intnums_coo_sym=jnp.array(dlist)
        
        else:
            # Compute coordinate representation for triple intersection numbers
            # Each row contains entries: [i,j,k,\kappa_{ijk}*perm({i,j,k})] with i,j,k labelling the different moduli
            # and \kappa_{ijk} the corresponding triple intersection number.
            # The multiplicative factor perm({i,j,k}) takes care of symmetrisation over indices in the calculations below.
            dlist=[]
            for i in range(self.h12):
                for j in range(i,self.h12):
                    for k in range(j,self.h12):
                        if self.intnums[i][j][k]!=0:
                            perms=np.unique(list(itertools.permutations([i,j,k])),axis=0)
                            dlist.append([i,j,k,int(self.intnums[i][j][k])*len(perms)])
                        
            self.intnums_coo_sym=jnp.array(dlist)
            
        # Update a_matrixs if necessary
        if self.a_matrix is not None:
            
            if self.basis_change is not None:
                warnings.warn("Input of `a_matrix` overrides the basis transformation! Please make sure that the input is consistent with the basis transformation!")
                
            self.a_matrix = jnp.array(self.a_matrix)
        else:
            
            a_matrix = np.zeros((self.h12, self.h12))
            
            for I1 in range(self.h12):
                for I2 in range(self.h12):
                    """
                    if type(self.model_ID)==int:
                        a_matrix[I1][I2] = self.intnums[I1][I2][I2]/2.
                    else:
                    """
                    if I1>=I2:
                        a_matrix[I1][I2] = self.intnums[I1][I1][I2]/2.
                    elif I1<I2:
                        a_matrix[I1][I2] = self.intnums[I1][I2][I2]/2.
                            
            self.a_matrix = jnp.array(a_matrix)
                
            # Take same convention as in https://inspirehep.net/files/d2f57319d398cfe81212b19c7f9d109f for CP11169
            if self.h12==2 and self.model_ID==1:
                self.a_matrix = jnp.array([[4.5,1.5],[1.5,0.]])
            
        
        
    def _charges_from_gv_gw(self,x):
        r"""
        **Description:**
        Extracts curve charges, invariant values, and curve degrees from a GV or GW invariant dictionary or array.

        Args:
            x (Dict | Array | None): GV or GW invariants. Can be a dictionary mapping curve charges to invariant values, an array with charges in all-but-last columns and invariants in the last column, or None.

        Returns:
            tuple: A tuple (charges, invariants, degrees) where charges is an Array of curve charges, invariants is an Array of invariant values, and degrees is an Array of curve degrees (or None if no grading vector is set).

        Raises:
            ValueError: If `x` is not a dict, Array, np.ndarray, or None.
        """
        
        if x is None:
            charges = None
            invariants = None
        elif type(x)==dict:
            
            keys=list(x.keys())
            
            if "charges" in keys and "invariants" in keys:
                X = x["charges"]
                Y = x["invariants"]
                if X is None or Y is None:
                    charges = None
                    invariants = None
                else:
                    if X.dtype==object:
                        X = np.array(X.tolist(),dtype=float)
                    else:
                        X = np.array(X,dtype=float)

                    if Y.dtype==object:
                        Y = np.array(Y.tolist(),dtype=float)
                    else:
                        Y = np.array(Y,dtype=float)

                    X = X.astype(int)

                    charges = jnp.array(X)
                    invariants = jnp.array(Y)
                    
                    
                
            else:
                if self.h12==1:
                    charges=jnp.array([jnp.array([key for key in keys])])
                    if len(charges.shape)>2:
                        charges = charges[0]
                else:
                    charges=jnp.array(keys)
                    
                #values = list(map(float,list(x.values())))
                #invariants=jnp.array(list(values))
                # Below makes sure that we don't run into issues with too long 
                # integers!
                try:
                    invariants=jnp.array(np.array(list(x.values())).astype(np.float64))
                except:
                    try:
                        invariants=jnp.array(np.array(list(x.values())))
                    except:
                        warnings.warn("Could not convert GV/GW invariant values to float64! Please make sure that the input is consistent with the basis transformation and the limit!")
                        invariants=jnp.array(np.array(list(x.values())[:10]).astype(np.float64))
                        if charges.shape[0]>10:
                            charges = charges[:10]
                        else:
                            charges = charges[0][:10]
                            charges = charges.reshape(-1,1)
                
        elif type(x)==Array or type(x)==np.ndarray:
            charges = x[:,:-1]
            invariants = x[:,-1]
        else:
            raise ValueError(f"Input for GVs/GWs must be either a dictionary or an array! Got type {type(x)} instead!")
        
        if x is not None:
            if self.h12==1:
                if len(charges.shape)>1:
                    if charges.shape[1] == invariants.shape[0]:
                            charges = charges[0]
                            charges = charges.reshape(-1,1)
            
            if len(charges.shape)!=2:
                raise ValueError(f"Input charges have wrong shape! Expected 2-dimensional array, but got array with shape {charges.shape} instead!")
            
            if charges.shape[1] != self.h12:
                raise ValueError(f"Input charges have wrong shape! Expected second dimension of size {self.h12}, but got {charges.shape[1]} instead!")
            
            if invariants.shape[0] != charges.shape[0]:
                raise ValueError(f"Number of invariant values does not match number of charge entries! Expected {charges.shape[0]} invariant values, but got {invariants.shape[0]} instead!")
            
            if self.grading_vector is not None:
                
                if self.grading_vector.shape[0] != self.h12:
                    raise ValueError(f"Grading vector has wrong shape! Expected first dimension of size {self.h12}, but got {self.grading_vector.shape[0]} instead!")
        
        
        
        if self.grading_vector is None or charges is None:
            degrees = None
        else:
            if len(charges)>0:
                degrees = charges@self.grading_vector
            else:
                degrees = None
            
        return charges, invariants, degrees
        
        
    
    def __repr__(self):
        r"""
        **Description:**
        String representation of the LCS-tree object.

        Returns:
            str: A string summarizing h11 and h12 of the model.
        """

        return (f"LCS-tree object with \n"
                f"    h11  = {self.h11}\n"
                f"    h12  = {self.h12}\n")

    
    @classmethod
    def filter_dict(cls, d):
        """
        **Description:**
        
        """
        keys = list(d.keys())
        for key in keys:
            x = d[key]
            if type(x)==np.ndarray or type(x)==list:
                
                if type(x)==list:
                    x = np.array(x,dtype=float)
                elif x.dtype==object:
                    x = np.array(x.tolist(),dtype=float)
                else:
                    x = np.array(x,dtype=float)
                    
                if "intnums" in key or "c2" in key or "conifold" in key:
                    x = x.astype(np.int32)

                d[key] = jnp.array(x)

        return d
    
    @classmethod
    def from_dict(cls, d):
        r"""
        **Description:**
        Creates a LCS-tree instance from a dictionary.
        
        Args:
            d (dict): A dictionary containing the attributes of the LCS-tree instance.
        
        Returns:
            LCS-tree: A LCS-tree instance created from the dictionary.
        """
        
        d = cls.filter_dict(d)
        
        return cls(**d)
    
    @classmethod
    def from_file(cls, file, 
                  maximum_degree: int | None = None, 
                  conifold_basis: bool | None = True,
                  limit: str = "LCS"):
        r"""
        **Description:**
        Creates a lcs_tree instance from a zipped pickle file.

        Args:
            file (str): Path to the zipped pickle file containing model data.
            maximum_degree (int | None): If not ``None``, applies a maximum degree cutoff to the GV/GW invariants after loading the data. Default is ``None``, which means that no cutoff is applied.
            conifold_basis (bool | None): If not ``None``, whether to use the conifold basis for the conifold curve after loading the data. Default is ``True``.
            limit (str): The limit for which to compute periods. Default is ``"LCS"``.

        Returns:
            lcs_tree: An lcs_tree instance loaded from the file.
        """

        d = load_zipped_pickle(file)
        d["maximum_degree"] = maximum_degree
        d["conifold_basis"] = conifold_basis
        d["limit"] = limit
        
        return cls.from_dict(d)
    
    @classmethod
    def from_cytools(cls, 
                     cy : "cytools.CalabiYau", 
                     basis_change: Array | None = None,
                     maximum_degree: int = 0,
                     limit: str = "LCS",
                     grading_vector: Array | None = None,
                     ncf: int | None = None,
                     conifold_curve: Array | None = None,
                     conifold_basis: bool | None = True,
                     model_ID: int | None = None,
                     save_file: bool = False,
                     time_out: int = 10,
                     compute_gws: bool = False,
                     remove_axis: int | None = None):
        r"""
        **Description:**
        Creates a LCS-tree instance from a CYTools Calabi-Yau object.
        
        Args:
            cy (cytools.CalabiYau): A CYTools Calabi-Yau object containing the geometric data of the model.
            basis_change (Array | None): A basis transformation matrix to be applied to the CY data. If None, no basis transformation is applied.
            maximum_degree (int): The maximum degree up to which Gopakumar-Vafa and Gromov-Witten invariants are computed. Default is 0, which means that no invariants are computed.
            limit (str): The limit for which to compute periods. Default is "LCS".
            grading_vector (Array | None): A grading vector to be used for computing degrees of GV/GW invariants. If ``None``, degrees are not computed.
            ncf (int | None): Number of conifolds for the coniLCS limit. Defaults to ``None``.
            conifold_curve (Array | None): Conifold curve charges for the coniLCS limit. Defaults to ``None``.
            model_ID (int | None): An integer ID for the model, used for saving data. If None, data is not saved.
            save_file (bool): Whether to save the model data to a file. Default is False.
            time_out (int): Time limit in seconds for computing Gopakumar-Vafa and Gromov-Witten invariants. Default is 10 seconds.
            compute_gws (bool): Whether to compute Gromov-Witten invariants in addition to Gopakumar-Vafa invariants. Default is False.
            remove_axis (int | None): If not ``None``, the specified axis is removed from the GV/GW charge data. Default is ``None``, which means that no axis is removed.
            
        Returns:
            LCS-tree: A LCS-tree instance created from the CYTools Calabi-Yau object.     
        
        """
        
        # We use basis_change=None for cytools_init and apply the basis change inside the lcs_tree constructor, to make sure that the basis change is applied consistently to all data, including the conifold curve and the conifold index in the GV charge data.
        md = cytools_model_data_init(cy,
                                     basis_change=None,
                                     grading_vector=grading_vector,
                                     model_ID=model_ID,
                                     save_file=save_file,
                                     time_out=time_out,
                                     compute_gws=compute_gws,
                                     remove_axis=remove_axis,
                                     maximum_degree=maximum_degree)
        
        if ncf is not None:
            md["ncf"]=ncf
            
        if conifold_curve is not None:
            md["conifold_curve"]=conifold_curve
            
        md["conifold_basis"] = conifold_basis
        md["basis_change"] = basis_change
        
        md["maximum_degree"]=maximum_degree
        md["limit"] = limit
            
        return cls.from_dict(md)

    def update(self,**kwargs):
        r"""
        **Description:**
        Updates attributes of the LCS-tree object.
        This can be used to update any optional attribute of the object, including those not set during initialization.
        
        Args:
            **kwargs: Keyword arguments to be updated in the object.
            
        Returns:
            None
        """
        
        limit0 = self.limit

        self.__dict__.update(kwargs)
        
        if "gv_charges" in kwargs or "limit" in kwargs:
            # Remove conifold curve when required for limit from
            # set of charges
            # Compute the conifold index
            if "coniLCS" in self.limit and len(self.gv_charges)>0:
                
                if self.limit != limit0:
                    if "bulk" in limit0 or "series" in limit0:
                        warnings.warn(f"Changing limit from {limit0} to {self.limit} may lead to inconsistencies in the GV charge data! Please make sure that the input is consistent with the basis transformation and the limit!")
                
                self._update_conifold_curve_and_index()
                
    
    def __copy__(self):
        r"""
        **Description:**
        Creates a copy of the LCS-tree object.
        
        Args:
            None
            
        Returns:
            lcs_tree: A copy of the LCS-tree object.
            
        """
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj


# Register the lcs_tree class as a pytree node for JAX transformations
unflatten_func = lambda aux_data, children: unflatten_func_class(aux_data, children, lcs_tree)

# Register lcs_tree
register_pytree_node(lcs_tree, flatten_func, unflatten_func)
