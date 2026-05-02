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
# This file holds functions to interface with CYTools. 
# ------------------------------------------------------------------------------

## Important standard libraries
import os, sys, warnings
import numpy as np

# Important JAX libraries
import jax
from jax import jit, vmap, config
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from jax.scipy.special import zeta
from jax.numpy import pi as Pi

# Enable 64 bit precision
import itertools
from functools import partial

try:
    import cytools
    from cytools import Cone
except ImportError:
    warnings.warn("Cannot import CYTools!")
    
    
# JAXVacua custom imports
from .util import *
from .util import save_model_data
from .conifold import find_conifolds

# Some global variables
home_dir=os.path.dirname(os.path.realpath(__file__))
files_dir=home_dir+"/models"

def compute_intersection_numbers_coo(int_nums: np.ndarray) -> np.ndarray:
    r"""
    **Description:**
    Converts intersection numbers from dense to coo format.
    
    Args:
        int_nums (np.ndarray): Intersection numbers in dense format.
        
    Returns:
        np.ndarray: Intersection numbers in coo format.
    
    """
    
    int_nums_coo = []
    h11 = int_nums.shape[0]
    for i1 in range(h11):
        for i2 in range(i1,h11):
            for i3 in range(i2,h11):

                int_num = int_nums[i1][i2][i3]

                if int_num!=0:
                    int_nums_coo.append([i1,i2,i3,int_num])

    return np.array(int_nums_coo)

def remove_zeros(a,axis=1):
    """
    **Description:**
    Removes zero rows from a numpy array.
    
    Args:
        a (np.ndarray): Input array.
        axis (int, optional): Axis along which to check for zero rows. Default is 1.
        
    Returns:
        tuple: A tuple containing the array with zero rows removed and a boolean flag array indicating which rows were kept.
    
    """
    
    flag = np.all(a == 0.,axis=axis)==False
    return a[flag], flag

def cytools_model_data_init(
                            input_data: "cytools.polytope.Polytope | cytools.triangulation.Triangulation | cytools.calabiyau.CalabiYau",
                            basis_change: np.ndarray | None = None,
                            maximum_degree: int = 0,
                            grading_vector: Array | None = None,
                            model_ID: int | None = None,
                            save_file: bool = False,
                            time_out: int = 10,
                            compute_gws: bool = False,
                            remove_axis: int | None = None
                        ) -> dict:
    r"""
    **Description:**
    Initializes model data from a CYTools input (Polytope, Triangulation, or CalabiYau).
    Computes geometric data including intersection numbers, Chern classes, and Kähler cone information.
    Optionally applies basis transformations and removes specified axes. Saves data to file if requested.
    
    Args:
        input_data (cytools.polytope.Polytope | cytools.triangulation.Triangulation | cytools.calabiyau.CalabiYau): 
            Input Calabi-Yau geometry (polytope, triangulation, or CY object).
        basis_change (np.ndarray | None, optional): Basis transformation matrix for coordinates. Default is None.
        maximum_degree (int, optional): Maximum degree for computing Gopakumar-Vafa and Gromov-Witten invariants. Default is 0 (no invariants computed).
        grading_vector (Array | None, optional): Grading vector for computing degrees of GV/GW invariants. Default is None.
        model_ID (int | None, optional): Model identifier for saving the data. Default is None.
        save_file (bool, optional): Whether to save the model data to a file. Default is False.
        time_out (int, optional): Time limit for cone computations in seconds. Default is 10.
        compute_gws (bool, optional): Whether to compute Gromov-Witten invariants. Default is False.
        remove_axis (int | None, optional): Starting axis index for removing trailing dimensions. Default is None.
        
    Returns:
        dict: Dictionary containing geometric and topological data of the mirror CY manifold.
    
    """
    
    # Convert input to CalabiYau object if necessary
    if type(input_data) == cytools.polytope.Polytope:
        mirror_cy = input_data.triangulate().get_cy()
    elif type(input_data) == cytools.triangulation.Triangulation:
        mirror_cy = input_data.get_cy()
    elif type(input_data) == cytools.calabiyau.CalabiYau:
        mirror_cy = input_data
    else:
        raise ValueError("Cannot interpret input type for CYTools interface.")

    # Initialize model data dictionary
    model_data: dict = {}
    
    # Extract basic geometric data from mirror CY
    h12: int = mirror_cy.h11()
    h11: int = mirror_cy.h12()
    points = mirror_cy.polytope().points()
    int_nums = mirror_cy.intersection_numbers(in_basis=True, format="dense")
    sec_chern = mirror_cy.second_chern_class(in_basis=True)

    # Compute Kähler cone and related data (skip for high h11)
    if h12 < 100:
        # Define timed computation of Kähler cone from Mori cone
        @exit_after(time_out)
        def compute_K_cup(cy):
            """Compute the Kahler cone as the dual of the Mori cone cap."""
            m_cap = cy.mori_cone_cap(in_basis=True)
            m_cap_ext_rays = m_cap.extremal_rays()
            return Cone(rays=m_cap_ext_rays).dual()
        
        @exit_after(time_out)
        def compute_M_cap_generators(cy):
            """Compute the extremal ray generators of the Mori cone cap."""
            m_cap = cy.mori_cone_cap(in_basis=True)
            m_cap_ext_rays = m_cap.extremal_rays()
            return m_cap_ext_rays

        # Compute Kähler cone with fallback to toric cone
        try:
            KC = compute_K_cup(mirror_cy)
        except:
            warnings.warn("Computation for K_cup took too long. Working with toric Kähler cone instead. Increase value of `time_out` if necessary.")
            KC = mirror_cy.toric_kahler_cone()
            
        try:
            M_cap_generators = compute_M_cap_generators(mirror_cy)
        except:
            warnings.warn("Computation for M_cap_generators took too long.")
            M_cap_generators = None

        # Define timed cone property computations
        @exit_after(time_out)
        def compute_extremal_rays(cone):
            """Compute the extremal rays of a given cone."""
            return cone.extremal_rays()

        @exit_after(time_out)
        def compute_extremal_hyperplanes(cone):
            """Compute the extremal hyperplanes (facets) of a given cone."""
            return cone.extremal_hyperplanes()

        @exit_after(time_out)
        def compute_rays(cone):
            """Compute all rays of a given cone."""
            return cone.rays()

        # Compute cone rays with warning on timeout
        try:
            rays_KC = compute_rays(KC)
        except:
            warnings.warn("Computation for rays took too long. Increase value of `time_out` if necessary.")
            rays_KC = []

        # Compute extremal generators with warning on timeout
        try:
            generators_KC = compute_extremal_rays(KC)
        except:
            warnings.warn("Computation for extremal rays took too long. Increase value of `time_out` if necessary.")
            generators_KC = []

        # Compute extremal hyperplanes with fallback to all hyperplanes
        try:
            hyperplanes = compute_extremal_hyperplanes(KC)
        except:
            warnings.warn("Computation for extremal hyperplanes took too long. Working with hyperplanes instead. Increase value of `time_out` if necessary.")
            hyperplanes = KC.hyperplanes()

    else:
        # For large h11, use only toric cone without expensive computations
        KC = mirror_cy.toric_kahler_cone()
        rays_KC = []
        generators_KC = []
        hyperplanes = KC.hyperplanes()

    # Compute reference point on stretched Kähler cone
    points_for_model = KC.tip_of_stretched_cone(c=1)

    # Apply basis transformation if provided
    #if basis_change is not None:
    if False:
        L = basis_change
        Linv = np.linalg.inv(L)

        # Transform intersection numbers by basis change
        int_nums = np.einsum('ai,ibc->abc', L, np.einsum('bj,ijc->ibc', L, np.einsum('ck,ijk->ijc', L, int_nums)))
        # Transform second Chern class by basis change
        sec_chern = np.matmul(L, sec_chern)

        # Transform cone generators and rays to new basis
        if len(generators_KC) > 0:
            generators_KC = np.matmul(generators_KC, Linv)

        if len(rays_KC) > 0:
            rays_KC = np.matmul(rays_KC, Linv)

        # Transform Mori cone generators to new basis if they were computed
        M_cap_generators = np.matmul(M_cap_generators, L.T)
        
        # Transform stretched cone point and hyperplanes
        points_for_model = np.matmul(points_for_model, Linv)
        hyperplanes = np.matmul(hyperplanes, L.T)

    # Remove trailing axes if specified
    if remove_axis is not None:
        # Remove axes from intersection numbers and Chern class
        int_nums = int_nums[remove_axis:, remove_axis:, remove_axis:]
        sec_chern = sec_chern[remove_axis:]
        points_for_model = points_for_model[remove_axis:]

        # Remove axes from hyperplanes and eliminate zero rows
        hyperplanes = hyperplanes[:, remove_axis:]
        hyperplanes, _ = remove_zeros(hyperplanes)

        # Remove axes from cone generators with zero-row elimination
        if len(generators_KC) > 0:
            generators_KC = generators_KC[:, remove_axis:]
            generators_KC, _ = remove_zeros(generators_KC)

        # Remove axes from cone rays with zero-row elimination
        if len(rays_KC) > 0:
            rays_KC = rays_KC[:, remove_axis:]
            rays_KC, _ = remove_zeros(rays_KC)

    # Compute or extract intersection numbers in COO format
    if False:
        if (basis_change is not None) or (remove_axis is not None):
            int_nums_coo = compute_intersection_numbers_coo(int_nums)
        else:
            int_nums_coo = mirror_cy.intersection_numbers(in_basis=True, format="coo")
        
    int_nums_coo = mirror_cy.intersection_numbers(in_basis=True, format="coo")

    # Populate model data dictionary with all computed data
    model_data["h12"] = h12
    model_data["h11"] = h11
    model_data["polytope_points"] = points
    model_data["heights"] = mirror_cy.triangulation().heights()
    model_data["intnums"] = int_nums
    model_data["intnums_coo"] = int_nums_coo
    model_data["c2"] = sec_chern
    model_data["chi"] = mirror_cy.chi()
    #model_data["a_matrix"] = []
    model_data["basis_change"] = basis_change
    model_data["hyperplanes"] = hyperplanes
    model_data["tip_skc"] = np.array(points_for_model)
    model_data["kahler_generators"] = np.array(generators_KC)
    model_data["kahler_rays"] = np.array(rays_KC)
    model_data["mori_rays"] = np.array(M_cap_generators) 


    if maximum_degree>0:
        inst_data = cytools_instanton_data_init(input_data,maximum_degree, grading_vector = grading_vector,basis_change = basis_change, remove_axis =remove_axis, model_ID=model_ID, save_file=save_file, time_out=time_out, compute_gws=compute_gws)
        id_keys = list(inst_data.keys())
        for key in id_keys:
            model_data[key] = inst_data[key]
    else:
        model_data["gvs"] = None
        model_data["gws"] = None
        model_data["grading_vector"] = None
    
    
    # Compute conifold limits with timeout and save to model data
    @exit_after(time_out)
    def compute_conifolds(cy):
        """Compute conifold limits from the dual polytope and triangulation."""
        conifolds = find_conifolds(cy.polytope().dual(), [cy.triangulation()] , verbosity=0)
        return conifolds
    
    try:
        conifolds = compute_conifolds(mirror_cy)
    except Exception as e:
        print(f"Computation for conifolds failed or timed out: {e!r}")
        conifolds = []

    if len(conifolds) > 0:
        # find_conifolds returns geometric Conifolds whose hot-path arrays
        # (conifold_curve, conifold_curve0, basis_change) are eagerly populated
        # by from_geometry. to_data() drops cytools refs so the result can be
        # carried through JIT.
        conifolds_data = tuple(c.to_data() for c in conifolds)
        active = conifolds_data[0]
    else:
        conifolds_data = None
        active = None

    model_data["conifold"]  = active             # singular, the active conifold
    model_data["conifolds"] = conifolds_data     # plural, full enumeration (optional)
    
    # Save to file if requested
    if save_file:
        save_model_data(model_data, f"model_{model_ID}.p", model_ID, mirror_cy.h11())

    return model_data


    
def cytools_instanton_data_init(input_data,max_deg, grading_vector = None,basis_change = None, remove_axis = None, model_ID=None,save_file=False,time_out=10,compute_gws=False) -> dict:
    r"""
    **Description:**
    Initializes instanton data from a CYTools input (Polytope, Triangulation, or CalabiYau). Computes Gopakumar-Vafa and Gromov-Witten invariants up to a specified maximum degree. Optionally applies a basis transformation and removes specified axes from the data. Saves the instanton data to a file if specified. 
    
    Args:
        input_data (cytools.polytope.Polytope, cytools.triangulation.Triangulation, or cytools.calabiyau.CalabiYau): Input data to initialize the instanton data.
        max_deg (int): Maximum degree for computing invariants.
        grading_vector (np.ndarray, optional): Grading vector for computing invariants. Default is None.
        basis_change (np.ndarray, optional): Basis transformation matrix. Default is None.
        remove_axis (list, optional): List of axes to remove from the data. Default is None.
        model_ID (int, optional): Model ID for saving the data. Default is None.
        save_file (bool, optional): Whether to save the instanton data to a file. Default is False.
        time_out (int, optional): Time limit for certain computations in seconds. Default is 100.
        compute_gws (bool, optional): Whether to compute Gromov-Witten invariants in addition to Gopakumar-Vafa invariants. Default is False.
        
    Returns:
        dict: A dictionary containing the instanton data.
    
    """

    if type(input_data) == cytools.polytope.Polytope:

        mirror_cy = input_data.triangulate().get_cy()

    elif type(input_data) == cytools.triangulation.Triangulation:

        mirror_cy = input_data.get_cy()

    elif type(input_data) == cytools.calabiyau.CalabiYau:

        mirror_cy = input_data

    else:
        raise ValueError("Cannot interpret input type for CYTools interface.")

    
    if grading_vector is None:
        m_cap = mirror_cy.mori_cone_cap(in_basis=True)
        grading_vector = m_cap.find_grading_vector()

    @exit_after(time_out)
    def compute_gvs(cy):
        """Compute Gopakumar-Vafa invariants up to max_deg in DOK format."""
        return cy.compute_gvs(max_deg=max_deg,grading_vec=grading_vector).dok

    gvs=compute_gvs(mirror_cy)

    if compute_gws:
        @exit_after(time_out)
        def compute_gws(cy):
            """Compute Gromov-Witten invariants up to max_deg in DOK format."""
            return cy.compute_gws(max_deg=max_deg,grading_vec=grading_vector).dok
        gws=compute_gws(mirror_cy)
    else:
        gws = None

    #if basis_change is not None:
    if False:
        def apply_basis_change(inv_dict,basis_matrix):
            """Apply a basis transformation to the curve classes in an invariant dictionary."""
            curvesOG,invariants=zip(*inv_dict.items())
            curvesOG,invariants = np.array(curvesOG).astype(np.float64),np.array(invariants).astype(np.float64)
            curves = np.matmul(np.array(curvesOG),basis_matrix.T)
            return dict(zip(map(tuple,curves),invariants))


        gvs = apply_basis_change(gvs,basis_change)

        gws = apply_basis_change(gws,basis_change)

        grading_vector=np.matmul(grading_vector,np.linalg.inv(basis_change))

    if remove_axis is not None:

        def remove_axis(inv_dict):
            """Remove leading axes from curve classes and drop all-zero entries."""
            curvesOG,invariants=zip(*inv_dict.items())
            curves = curvesOG[:,remove_axis:]
            curves,flag = remove_zeros(curves)
            return dict(zip(map(tuple,curves),invariants[flag]))

        gvs = remove_axis(gvs)

        gws = remove_axis(gws)

        grading_vector = grading_vector[remove_axis:]

    instanton_data = {}

    instanton_data["gws"] = gws
    instanton_data["gvs"] = gvs
    instanton_data["grading_vector"] = grading_vector

    #if save_file:
    #    save_model_data(instanton_data,"instanton_data.p",model_ID,#mirror_cy.h11())
            
    

    return instanton_data



############# CALABI-YAU AND KÄHLER CONE HANDLING #############
    
def eft_to_poly(fluxeft):
    r"""
    **Description:**
    Extracts the polytope from the given model.

    Returns:
        cytools.Polytope: The polytope of the Calabi-Yau threefold.
    """
    
    return Polytope(fluxeft.lcs_tree.polytope_points)

def eft_to_cy(fluxeft):
    r"""
    **Description:**
    Returns the Calabi-Yau manifold from the given model.
        
    Returns:
        cytools.CalabiYau: The Calabi-Yau manifold.
    """
    
    # Create a Polytope object using the polytope points from the periods
    poly = Polytope(fluxeft.lcs_tree.polytope_points)
    
    # Retrieve the heights associated with the periods
    heights = fluxeft.lcs_tree.heights
    
    # Triangulate the polytope using the specified heights and return the corresponding Calabi-Yau manifold
    return poly.triangulate(heights=heights).get_cy()

def eft_to_coninop(fluxeft):
    r"""
    **Description:**
    Returns the basis transformation and conifold curve from the given model.

    Returns:
        tuple: A tuple containing the basis transformation and conifold curve.
    """
    cf = fluxeft.lcs_tree.conifold
    conifold_curve = cf.conifold_curve if cf is not None else None
    return fluxeft.lcs_tree.basis_change, conifold_curve


def eft_to_cone(fluxeft,timeout: int = 60, generators: Array | None = None):
    r"""
    **Description:**
    Extracts the Kähler cone from the given model.
    
    Args:
        timeout (int, optional): Timeout in seconds for CYTools computation if generators are not provided. Defaults to ``60``.
        generators (Array, optional): Generators of the Kähler cone. If provided, these will be used directly. Defaults to ``None``.
    
    Returns:
        Cone: The Kähler cone.
    """
    
    if generators is None:
        # Retrieve the generators of the Kähler cone from the periods
        generators = fluxeft.lcs_tree.generators_kahler_cone
    
    if generators is None:
        
        @exit_after(timeout)
        def get_cone():
            """Construct the Kahler cone by dualising the Mori cone cap from CYTools."""
            cy = eft_to_cy(fluxeft)
            mcap = cy.mori_cone_cap(in_basis=True)
            xray = mcap.extremal_rays()
            return Cone(xray).dual()
        
        return get_cone()
    
    # Return a Cone object constructed from the generators
    return Cone(generators)

def eft_to_facet(fluxeft):
    r"""
    **Description:**
    Extracts the facet of the Kähler cone from the given model.
    
    Returns:
        Cone: The facet of the Kähler cone.
    """
    # Retrieve the generators of the Kähler cone from the periods
    generators = fluxeft.lcs_tree.generators_kahler_cone
    
    # Exclude the first column of generators, which is typically not part of the facet
    generators_facet = generators[:,1:]
    
    # Create a boolean flag array indicating which rows have non-zero entries
    flag = np.all(generators_facet==0, axis=1) == False
    
    # Filter the generators_facet to keep only those rows that are not all zeros
    generators_facet = generators_facet[flag]
    
    # Return a Cone object constructed from the filtered generators_facet
    return Cone(generators_facet)


