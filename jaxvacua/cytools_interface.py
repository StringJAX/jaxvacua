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

"""CYTools interoperability helpers.

Purpose
-------
Convert CYTools polytopes, triangulations and Calabi-Yau objects into the
model-data dictionaries consumed by ``lcs_tree`` and provide adapters back
from JAXVacua EFT objects to selected CYTools geometry objects.

Main public API
---------------
- ``cytools_model_data_init`` and ``cytools_instanton_data_init`` for
  extracting intersection data, Chern data, cone data and optional
  Gopakumar-Vafa/Gromov-Witten invariants.
- Sparse/dense intersection helpers such as
  ``compute_intersection_numbers_coo`` and ``remove_zeros``.
- Geometry adapters ``eft_to_poly``, ``eft_to_cy``, ``eft_to_coninop``,
  ``eft_to_cone`` and ``eft_to_facet``.

Design notes
------------
CYTools is an optional but heavy dependency.  This module should remain the
boundary layer for CYTools-specific objects so the core package can operate on
plain dictionaries and ``lcs_tree`` instances once model data has been loaded.
"""

## Important standard libraries
import os, sys, warnings
import numpy as np

# Important JAX libraries
import jax
from jax import jit, config
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from jax.scipy.special import zeta
from jax.numpy import pi as Pi

# Enable 64 bit precision
import itertools
from functools import partial
from typing import Callable

try:
    import cytools
    from cytools import Cone, Polytope
except ImportError:
    warnings.warn("Cannot import CYTools!")
    
    
from stringjax_tools import run_in_subprocess

# JAXVacua custom imports
from .util import *
from .util import save_model_data
from .conifold import find_conifolds

# Some global variables
home_dir=os.path.dirname(os.path.realpath(__file__))
files_dir=home_dir+"/models"


############# TIME-LIMITED CYTOOLS COMPUTATIONS #############
#
# A ``time_out`` of ``None`` runs a computation in this process without a limit. A number runs
# it in a child process (``stringjax_tools.run_in_subprocess``) that is killed when the limit
# expires. A timer that interrupts the main thread (``util.exit_after``) cannot be relied on:
# it does nothing under CYTools' own SIGINT handler (a script, or a notebook cell that imports
# CYTools) and acts only under Python's default handler (a later notebook cell). The tasks below
# are module-level functions so that the child can import them; the child's start-up (importing
# CYTools and JAXVacua, about 2 s on a laptop) counts towards the limit. The child runs on the
# CPU (``JAX_PLATFORMS=cpu``): it does CYTools work only and must not claim accelerator memory.

def _run_timed(task: Callable, args: tuple, time_out: float | None):
    r"""
    **Description:**
    Run ``task(*args)`` in-process if ``time_out`` is ``None``, otherwise in a child process
    killed after ``time_out`` seconds.

    Args:
        task (Callable): A module-level function of this module.
        args (tuple): Its (picklable) arguments.
        time_out (float | None): Time limit in seconds, or ``None`` for no limit.

    Returns:
        Any: The value of ``task(*args)``.

    Raises:
        TimeoutError: If the child did not finish within ``time_out`` seconds.
        stringjax_tools.ChildProcessFailed: If the child raised (a ``RuntimeError``).
    """
    if time_out is None:
        return task(*args)
    env = {"CYTOOLS_NO_UPDATE_CHECK": os.environ.get("CYTOOLS_NO_UPDATE_CHECK", "1"),
           "JAX_PLATFORMS": "cpu"}
    return run_in_subprocess(task, args, timeout=time_out, env=env)


def _check_time_out(time_out) -> None:
    """Raise ``ValueError`` unless ``time_out`` is ``None`` or a positive number."""
    if time_out is not None and not (isinstance(time_out, (int, float)) and time_out > 0):
        raise ValueError(f"time_out must be a positive number of seconds or None, got {time_out!r}")


def _capture(compute: Callable) -> tuple:
    """``("ok", compute())``, or ``("error", "<type>: <message>")`` if it raises an ``Exception``."""
    try:
        return ("ok", compute())
    except Exception as e:
        return ("error", f"{type(e).__name__}: {e}")


def _kahler_cone_from_mori_cap(cy) -> tuple:
    r"""
    **Description:**
    The Kähler cone as the dual of the Mori cone cap, and the cap's extremal rays, from one
    computation of the cap. Each item is ``("ok", value)`` or ``("error", message)``.

    Args:
        cy (cytools.calabiyau.CalabiYau): The Calabi-Yau threefold.

    Returns:
        tuple: ``(kahler_cone, mori_cap_rays)``.
    """
    rays = _capture(lambda: cy.mori_cone_cap(in_basis=True).extremal_rays())
    if rays[0] != "ok":
        return rays, rays
    return _capture(lambda: Cone(rays=rays[1]).dual()), rays


def _kahler_cone_properties(cone) -> dict:
    r"""
    **Description:**
    The rays, extremal rays and extremal hyperplanes of ``cone``, each computed on its own and
    returned as ``("ok", value)`` or ``("error", message)``.

    Args:
        cone (cytools.Cone): The Kähler cone.

    Returns:
        dict: Keys ``"rays"``, ``"extremal_rays"``, ``"extremal_hyperplanes"``.
    """
    return {
        "rays": _capture(cone.rays),
        "extremal_rays": _capture(cone.extremal_rays),
        "extremal_hyperplanes": _capture(cone.extremal_hyperplanes),
    }


def _conifold_data(cy) -> tuple:
    r"""
    **Description:**
    The conifold limits from the dual polytope and the triangulation of ``cy``, as numerical
    :class:`Conifold` copies (``to_data``, without CYTools references).

    Args:
        cy (cytools.calabiyau.CalabiYau): The Calabi-Yau threefold.

    Returns:
        tuple: The conifolds' data, possibly empty.
    """
    conifolds = find_conifolds(cy.polytope().dual(), [cy.triangulation()], verbosity=0)
    return tuple(c.to_data() for c in conifolds)


def _invariants_dok(cy, kind: str, max_deg: int, grading_vector) -> tuple:
    r"""
    **Description:**
    Gopakumar-Vafa (``kind="gvs"``) or Gromov-Witten (``kind="gws"``) invariants of ``cy`` up
    to degree ``max_deg``, in DOK format. Without a grading vector, the Mori cone cap's is used.

    Args:
        cy (cytools.calabiyau.CalabiYau): The Calabi-Yau threefold.
        kind (str): ``"gvs"`` or ``"gws"``.
        max_deg (int): Maximum degree.
        grading_vector: Grading vector, or ``None``.

    Returns:
        tuple: ``({curve class: invariant}, grading vector)``.
    """
    if grading_vector is None:
        grading_vector = cy.mori_cone_cap(in_basis=True).find_grading_vector()
    compute = cy.compute_gvs if kind == "gvs" else cy.compute_gws
    return dict(compute(max_deg=max_deg, grading_vec=grading_vector).dok), grading_vector


def _kahler_cone_of_triangulation(points, heights):
    """The Kähler cone of the CY of the triangulation with ``heights`` of the polytope ``points``,
    as the dual of its Mori cone cap (raises on failure)."""
    cy = Polytope(points).triangulate(heights=heights).get_cy()
    return Cone(cy.mori_cone_cap(in_basis=True).extremal_rays()).dual()


def _time_out_message(what: str, time_out) -> str:
    return (f"{what} took longer than time_out={time_out} s (including the start of the child process, "
            "about 2 s on a laptop); increase `time_out`, or pass None for no limit")

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
                            time_out: float | None = None,
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
        time_out (float | None, optional): Time limit in seconds. ``None`` (default) runs the CYTools steps in this process without a limit. A number runs them in child processes killed at the limit, each with its own limit (the child's start-up, about 2 s on a laptop, counts): (1) the Mori cone cap and the Kähler cone, falling back to the toric Kähler cone; (2) the Kähler cone's rays, extremal rays and extremal hyperplanes together, falling back to none, none and all hyperplanes (a warning names the reason); (3) the conifold search, falling back to none (printed); (4) each GV/GW table, raising ``TimeoutError``. ``ValueError`` unless ``None`` or positive.
        compute_gws (bool, optional): Whether to compute Gromov-Witten invariants. Default is False.
        remove_axis (int | None, optional): Starting axis index for removing trailing dimensions. Default is None.
        
    Returns:
        dict: Dictionary containing geometric and topological data of the mirror CY manifold.
    
    """
    
    _check_time_out(time_out)

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
        # The Kähler cone (the dual of the Mori cone cap) and the cap's generators, from one
        # computation of the cap; on a timeout or an error, the toric Kähler cone.
        try:
            kc, m_cap = _run_timed(_kahler_cone_from_mori_cap, (mirror_cy,), time_out)
        except TimeoutError:
            kc = m_cap = ("error", _time_out_message("The Mori cone cap", time_out))
        except Exception as e:  # the child failed (ChildProcessFailed) or could not be started
            kc = m_cap = ("error", f"{type(e).__name__}: {e}")
        if kc[0] == "ok":
            KC = kc[1]
        else:
            warnings.warn(f"Kähler cone from the Mori cone cap not computed ({kc[1]}). Working with the toric Kähler cone instead.")
            KC = mirror_cy.toric_kahler_cone()
        if m_cap[0] == "ok":
            M_cap_generators = m_cap[1]
        else:
            warnings.warn(f"Mori cone cap generators not computed ({m_cap[1]}).")
            M_cap_generators = None

        # Rays, extremal rays and extremal hyperplanes of the Kähler cone, each with its fallback.
        try:
            props = _run_timed(_kahler_cone_properties, (KC,), time_out)
        except TimeoutError:
            reason = ("error", _time_out_message("The Kähler cone's rays and hyperplanes", time_out))
            props = dict.fromkeys(("rays", "extremal_rays", "extremal_hyperplanes"), reason)
        except Exception as e:
            reason = ("error", f"{type(e).__name__}: {e}")
            props = dict.fromkeys(("rays", "extremal_rays", "extremal_hyperplanes"), reason)
        if props["rays"][0] == "ok":
            rays_KC = props["rays"][1]
        else:
            warnings.warn(f"Rays of the Kähler cone not computed ({props['rays'][1]}).")
            rays_KC = []
        if props["extremal_rays"][0] == "ok":
            generators_KC = props["extremal_rays"][1]
        else:
            warnings.warn(f"Extremal rays of the Kähler cone not computed ({props['extremal_rays'][1]}).")
            generators_KC = []
        if props["extremal_hyperplanes"][0] == "ok":
            hyperplanes = props["extremal_hyperplanes"][1]
        else:
            warnings.warn(f"Extremal hyperplanes of the Kähler cone not computed ({props['extremal_hyperplanes'][1]}). Working with its hyperplanes instead.")
            hyperplanes = KC.hyperplanes()

    else:
        # For large h11, use only toric cone without expensive computations
        KC = mirror_cy.toric_kahler_cone()
        rays_KC = []
        generators_KC = []
        hyperplanes = KC.hyperplanes()
        M_cap_generators = None

    # Compute reference point on stretched Kähler cone
    points_for_model = KC.tip_of_stretched_cone(c=1)

    # Apply basis transformation if provided
    if basis_change is not None:
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
        if M_cap_generators is not None:
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
    if (basis_change is not None) or (remove_axis is not None):
        int_nums_coo = compute_intersection_numbers_coo(int_nums)
    else:
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
    
    
    # Compute conifold limits (time-limited if time_out is a number) and save to model data.
    # find_conifolds returns geometric Conifolds whose hot-path arrays (conifold_curve,
    # conifold_curve0, basis_change) are eagerly populated by from_geometry; to_data() drops the
    # cytools refs, so the result can be carried through JIT (and back from a child process).
    try:
        conifolds_data = _run_timed(_conifold_data, (mirror_cy,), time_out)
    except TimeoutError:
        print(f"Computation for conifolds timed out: {_time_out_message('The conifold search', time_out)}")
        conifolds_data = ()
    except Exception as e:
        print(f"Computation for conifolds failed: {e!r}")
        conifolds_data = ()

    if len(conifolds_data) == 0:
        conifolds_data = None

    model_data["conifold"]  = None               # singular, the active conifold
    model_data["conifolds"] = conifolds_data     # plural, full enumeration (optional)
    
    # Save to file if requested
    if save_file:
        save_model_data(model_data, f"model_{model_ID}.p", model_ID, mirror_cy.h11())

    return model_data


    
def cytools_instanton_data_init(input_data,max_deg, grading_vector = None,basis_change = None, remove_axis = None, model_ID=None,save_file=False,time_out=None,compute_gws=False) -> dict:
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
        time_out (float | None, optional): Time limit in seconds for each invariant computation, enforced in a child process (its start-up, about 2 s, counts); ``None`` (default) for no limit.
        compute_gws (bool, optional): Whether to compute Gromov-Witten invariants in addition to Gopakumar-Vafa invariants. Default is False.
        
    Returns:
        dict: A dictionary containing the instanton data.

    Raises:
        TimeoutError: If a computation exceeds ``time_out``.
        stringjax_tools.ChildProcessFailed: If ``time_out`` is a number and the computation failed
            in the child process (a ``RuntimeError``; without a limit, CYTools' own exception).
        ValueError: If ``time_out`` is neither ``None`` nor a positive number.
    
    """

    _check_time_out(time_out)

    if type(input_data) == cytools.polytope.Polytope:

        mirror_cy = input_data.triangulate().get_cy()

    elif type(input_data) == cytools.triangulation.Triangulation:

        mirror_cy = input_data.get_cy()

    elif type(input_data) == cytools.calabiyau.CalabiYau:

        mirror_cy = input_data

    else:
        raise ValueError("Cannot interpret input type for CYTools interface.")

    def _invariants(kind, grading_vector):
        """GVs or GWs up to max_deg in DOK format and the grading vector used, time-limited if
        time_out is a number (the grading vector, if not given, is computed within the limit)."""
        try:
            return _run_timed(_invariants_dok, (mirror_cy, kind, max_deg, grading_vector), time_out)
        except TimeoutError as e:
            raise TimeoutError(_time_out_message(f"The {kind[:-1].upper()} computation to degree {max_deg}", time_out)) from e

    gvs, grading_vector = _invariants("gvs", grading_vector)

    if compute_gws:
        gws, _ = _invariants("gws", grading_vector)
    else:
        gws = None

    if basis_change is not None:
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


def eft_to_cone(fluxeft,timeout: float | None = None, generators: Array | None = None):
    r"""
    **Description:**
    Extracts the Kähler cone from the given model.
    
    Args:
        timeout (float | None, optional): Time limit in seconds for the CYTools computation if generators are not provided, enforced in a child process (its start-up, about 2 s, counts). Defaults to ``None`` (no limit).
        generators (Array, optional): Generators of the Kähler cone. If provided, these will be used directly. Defaults to ``None``.
    
    Returns:
        Cone: The Kähler cone.

    Raises:
        TimeoutError: If the computation exceeds ``timeout``.
        stringjax_tools.ChildProcessFailed: If ``timeout`` is a number and the computation failed
            in the child process (a ``RuntimeError``; without a limit, CYTools' own exception).
        ValueError: If ``timeout`` is neither ``None`` nor a positive number.
    """
    
    if generators is None:
        # Retrieve the generators of the Kähler cone from the periods
        generators = fluxeft.lcs_tree.generators_kahler_cone
    
    if generators is None:
        
        _check_time_out(timeout)
        tree = fluxeft.lcs_tree
        try:
            return _run_timed(_kahler_cone_of_triangulation, (tree.polytope_points, tree.heights), timeout)
        except TimeoutError as e:
            raise TimeoutError(_time_out_message("The Kähler cone", timeout)) from e
    
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
