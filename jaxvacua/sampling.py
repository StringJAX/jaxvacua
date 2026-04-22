# ==============================================================================
# This code is written by Andreas Schachner. Without the author's permission, this 
# code must not be shared with anyone else or used for any other projects than 
# those involving the author directly.
#
# If any questions arise, please feel free to reach out to me (Andreas) either at
# andreas.schachner@gmx.net or at as3475@cornell.edu or at a.schachner@lmu.de.
# ==============================================================================
#
# ------------------------------------------------------------------------------
# This file holds the class to sample fluxes and moduli values.
# ------------------------------------------------------------------------------


#Standard libraries
import os, sys, warnings, time, getopt, itertools
import numpy as np
from functools import partial
import gurobipy as gp

# Import jax modules
import jax
import jax.numpy as jnp
from jax import jit, vmap, Array
from numpy.typing import ArrayLike
from typing import Tuple, Any, Callable
from jax.numpy import pi
# Self-made modules
from .util import random_integer, random_uniform, vmapping_func, vmapping_func_cached, PRNGSequence


home_dir=os.path.dirname(os.path.realpath(__file__))+"/.."


class data_sampler():

    def __init__(self,
        model: Any,
        flux_bounds: Tuple[float, float] = (-10, 10),
        axion_bounds: Tuple[float, float] = (-0.5, 0.5),
        dilaton_bounds: Tuple[float, float] = (2., 10.),
        moduli_bounds: Tuple[float, float] = (1., 5.),
        use_jax: bool = False,
        seed: int = 42
        ):
        r"""
        **Description:**
        A class to sample initial data for the construction of flux vacua.

        Args:
            self.model (jaxvacua.flux_sector.flux_sector): Model class for flux compactifications.
            flux_bounds (Tuple[float, float]): Bounds for fluxes.
            axion_bounds (Tuple[float, float]): Bounds for axions.
            dilaton_bounds (Tuple[float, float]): Bounds for dilaton.
            moduli_bounds (Tuple[float, float]): Bounds for moduli.
            use_jax (bool, optional): Use JAX environment for random number generation.
            seed (int, optional): Seed for the internal PRNG used when ``rns_key`` is not
                passed explicitly to sampling methods. Defaults to 42.
            **kwargs: Additional keyword arguments.


        Attributes:
            axion_lower (float): Lower bound universal axion.
            axion_upper (float): Upper bound universal axion.
            axions_lower (float): Lower bound complex structure axions.
            axions_upper (float): Upper bound complex structure axions.
            flux_lower (int): Lower bound fluxes.
            flux_upper (int): Upper bound fluxes.
            moduli_lower (float): Lower bound complex structure moduli.
            moduli_upper (float): Upper bound complex structure moduli.
            s_lower (float): Lower bound dilaton.
            s_upper (float): Lower bound dilaton.
            _n_fluxes (int): Number of fluxes.
            _generators_kahler_cone (Array): Generators Kähler cone.
            _h12 (float): Number of complex structure moduli.
            _rays_kahler_cone (Array): Rays Kähler cone.
            _tip_ray (Array): Tip of c=1 stretched Kähler cone.
            _F_inst (callable): Instanton prepotential.
            _prepot (callable): Prepotential.
            _tadpole (callable): D3-tadpole contribution of fluxes.
            _rng (PRNGSequence): Internal PRNG sequence used as default key source.

        """

        self.use_jax = use_jax
        self._rng = PRNGSequence(seed)
        self.model = model
        self.flux_upper = flux_bounds[1]
        self.flux_lower = flux_bounds[0]
        self._n_fluxes = self.model.n_fluxes
        self._h12 = self.model.h12
        self._tadpole = jit(vmap(self.model.tadpole))
        self._D3_tadpole = self.model.D3_tadpole

        self._kahler_metric = jit(vmap(model.kahler_metric))
        
        _gen = self.model.lcs_tree.generators_kahler_cone
        _hp = self.model.lcs_tree.hyperplanes
        _rays = self.model.lcs_tree.rays_kahler_cone
        _tip = self.model.lcs_tree.tip_skc

        self._extremal_rays = np.array(_gen) if _gen is not None else None
        self._hyperplanes = np.array(_hp) if _hp is not None else None
        self._rays = np.array(_rays) if _rays is not None else None
        self._tip = np.array(_tip) if _tip is not None else None

        if self._rays is not None and len(self._rays) == 0:
            self._rays = None
        if self._extremal_rays is not None and len(self._extremal_rays) == 0:
            self._extremal_rays = None
        
        self.axions_lower = axion_bounds[0]
        self.axions_upper = axion_bounds[1]
        
        self.axion_lower = axion_bounds[0]
        self.axion_upper = axion_bounds[1]

        self.moduli_lower = moduli_bounds[0]
        self.moduli_upper = moduli_bounds[1]

        self.s_lower = dilaton_bounds[0]
        self.s_upper = dilaton_bounds[1]
        
        self._F_inst = jit(vmap(lambda x: self.model.F_inst(x)))
        self._prepot = jit(vmap(lambda x: self.model.F(x)))

        try:
            self._cone_points = self.find_interior_points(N=1000, stretching=0.1)
        except Exception:
            self._cone_points = []
            
    def __repr__(self):
        r"""Return a string representation of the data sampler."""
        return f"Data sampler for {self.model}"


    # -----------------------------------------------------------------------
    # Private random-dispatch helpers
    # One dispatch decision replaces the repeated if self.use_jax / else
    # branches scattered throughout the public sampling methods.
    #
    # When use_jax=True and rns_key is None, all helpers fall back to
    # self._rng (a PRNGSequence seeded at construction time), so callers
    # never need to manage keys explicitly.
    # -----------------------------------------------------------------------

    def _get_jax_key(self, rns_key):
        r"""
        **Description:**
        Return a raw JAX PRNGKey from *rns_key*, falling back to ``self._rng``.

        Handles three cases:

        * ``None``            → draw the next key from ``self._rng``.
        * ``PRNGSequence``    → draw the next key from the caller's sequence.
        * raw JAX array       → use it directly (no splitting).
        """
        if rns_key is None:
            return next(self._rng)
        if isinstance(rns_key, PRNGSequence):
            return next(rns_key)
        return rns_key  # raw JAX PRNGKey array

    def _rand_uniform(self, lo, hi, shape, rns_key):
        r"""Return a uniform float array of the given *shape* on ``[lo, hi)``."""
        if self.use_jax:
            return random_uniform(lo, hi, rns_key=rns_key if rns_key is not None else self._rng, shape=shape)
        return np.random.uniform(lo, hi, shape)

    def _rand_integer(self, lo, hi, shape, rns_key):
        r"""Return a uniform integer array of the given *shape* on ``[lo, hi)``."""
        if self.use_jax:
            return random_integer(lo, hi, rns_key=rns_key if rns_key is not None else self._rng, shape=shape)
        return np.random.randint(lo, hi, shape)

    def _rand_permutation(self, arr, rns_key):
        r"""Return a randomly permuted copy of *arr* (non-destructive)."""
        if self.use_jax:
            return jax.random.permutation(self._get_jax_key(rns_key), arr)
        return np.random.permutation(arr)

    def _rand_choice(self, n, k, replace, rns_key):
        r"""Return *k* random indices drawn from ``[0, n)``."""
        if self.use_jax:
            return jax.random.choice(self._get_jax_key(rns_key), n, shape=(k,), replace=replace)
        return np.random.choice(n, size=k, replace=replace)

    def _collect_batches(self, batches: list, n: int):
        r"""
        **Description:**
        Concatenate *batches* along axis 0, trim to *n* rows, and return.

        This helper encapsulates the ``list-append → single concatenate``
        accumulation pattern used throughout the sampling methods.  Collecting
        in a Python list and concatenating once produces O(N) total allocation,
        as opposed to the O(N²) cost of repeated ``np.append`` / ``jnp.append``
        inside a loop.

        Args:
            batches (list[Array]): Non-empty list of arrays with matching trailing
                dimensions.
            n (int): Maximum number of rows to return.

        Returns:
            Array | np.ndarray: Concatenated and trimmed result; a JAX array when
            ``self.use_jax`` is ``True``, otherwise a NumPy array.
        """
        if self.use_jax:
            return jnp.concatenate(batches, axis=0)[:n]
        return np.concatenate(batches, axis=0)[:n]


    def update_interior_points(
                               self, 
                               num_pts: int, 
                               rns_key: Any | None = None,
                               maxval: float | None = None, 
                               stretching : float = 0.,
                               perturbation: float = 1e-1,
                               time_out: float = 60.,
                               verbosity: int = 0
                               ) -> None:
        r"""
        **Description:**
        Updates the interior points used for sampling in the Kähler cone.
        
        Args:
            num_pts (int): Number of interior points to sample.
            rns_key (Any | None, optional): PRNG random key. Defaults to ``None``.
            maxval (float, optional): Maximum value for rescaling points. Default is None.
            stretching (float, optional): Stretching parameter for the Kähler cone. Default is 0.
            perturbation (float, optional): Perturbation applied to interior points. Default is 1e-1.
            time_out (float, optional): Time-out duration in seconds. Default is 60.
            verbosity (int, optional): Verbosity level for logging. Default is 0.
            
        Raises:
            RuntimeError: If sampling points in the cone fails within the time-out duration.
            
        Returns:
            None
        """

        # Set maxval to flux_upper if not provided
        if maxval is None:
            maxval = self.flux_upper

        # Find initial interior points in the Kähler cone
        pts = self.find_interior_points(N=2*num_pts, stretching=stretching)
        H = self._hyperplanes
        chunks: list = []
        n_collected: int = 0
        tic = time.time()

        while n_collected < num_pts:
            if verbosity >= 1 and n_collected > 1:
                print(f"#samples: {n_collected}          ", flush=True, end="\r")

            pts = self._rand_permutation(pts, rns_key)
            pts0 = np.asarray(pts[:num_pts])
            coeffs = self._rand_uniform(-1, 1, pts0.shape, rns_key)
            pts0 = pts0 + np.asarray(coeffs) * perturbation
            pts0 = self.rescale_points(pts0, norm="l2", maxval=maxval, rns_key=rns_key)

            flag = np.all(np.asarray(pts0) @ H.T >= stretching, axis=1)
            valid = np.asarray(pts0)[flag]
            if valid.shape[0] > 0:
                chunks.append(valid)
                n_collected += valid.shape[0]

            if time.time() - tic > time_out:
                raise RuntimeError("Failed to sample points in cone!")

        cone_point = self._collect_batches(chunks, num_pts)

        # Append existing cone points to the new cone points if available
        if len(self._cone_points) > 0:
            cone_point = np.concatenate([np.asarray(cone_point), self._cone_points], axis=0)

        # Update the internal state with the new cone points
        self._cone_points = cone_point


    def get_fluxes(
        self, 
        num_pts: int, 
        mode: str = "full", 
        rns_key: Any | None = None, 
        minval: int | None = None, 
        maxval: int | None = None, 
        sampling_mode: str = "box", 
        radius: float = 10.0
    ) -> Array:
        r"""
        **Description:**
        Returns a random sample of flux choices used for ISD biased sampling.

        .. note::
            Note that, even though this function is not jit-compiled, it is faster than the equivalent 
            function using `jax.random.randint` (at least for more than 100 fluxes)!

        Args:
            num_pts (int): Number of points to sample.
            mode (str, optional): Mode of sampling ("full", "half", or None). Default is "full".
            rns_key (PRNGKey, optional): PRNG random key. Defaults to ``None``.
            minval (int | None, optional): Minimum value for flux sampling. Default is None.
            maxval (int | None, optional): Maximum value for flux sampling. Default is None.
            sampling_mode (str, optional): Mode of sampling ("box", "sphere", "tadpole_bound", "tadpole_cancel"). Default is "box".
            radius (float, optional): Radius constraint for certain sampling modes. Default is 10.0.

        Returns:
            ArrayLike: Array of sampled fluxes.

        Raises:
            ValueError: If `sampling_mode` or `mode` is not recognized.
        """
        
        sampling_modes = ["box","sphere","tadpole_bound","tadpole_cancel"]
        if sampling_mode not in sampling_modes:
            raise ValueError(f"`sampling_mode` should be one of {sampling_modes}, but is {sampling_mode}!")

        modes = [None,"full","half"]
        if mode not in modes:
            raise ValueError(f"`mode` should be one of {modes}, but is {mode}!")

        if mode is None or mode == "full":
            dim = self._n_fluxes*2
        elif mode=="half":
            dim = self._n_fluxes

        if maxval is None:
            maxval = self.flux_upper
        if minval is None:
            minval = self.flux_lower

        # "box" sampling produces exactly num_pts samples in one shot
        if sampling_mode == "box":
            return self._rand_integer(minval, maxval + 1, (num_pts, dim), rns_key)

        # For constrained modes: generate batches, keep those satisfying the constraint
        chunks: list = []
        n_collected: int = 0
        while n_collected < num_pts:
            flux = self._rand_integer(minval, maxval + 1, (num_pts, dim), rns_key)

            if sampling_mode == "tadpole_bound":
                flag = self._tadpole(flux) <= radius
            elif sampling_mode == "tadpole_cancel":
                flag = self._tadpole(flux) == radius
            else:  # "sphere"
                flux_np = np.asarray(flux)
                flag = np.sqrt(np.sum(flux_np * flux_np, axis=1)) <= radius

            valid = np.asarray(flux)[np.asarray(flag)]
            if valid.shape[0] > 0:
                chunks.append(valid)
                n_collected += valid.shape[0]

        return self._collect_batches(chunks, num_pts)


    def sample_ray(self, rns_key: Any | None = None):
        r"""
        **Description:**
        Samples a random ray from the Kähler cone.
        
        Returns:
            np.ndarray or Array: A random ray from the Kähler cone.
        """
        
        return self._random_direction_in_cone(rns_key=rns_key)

    def _random_direction_in_cone(self,rns_key: Any | None = None):
        r"""
        **Description:**
        Samples a random direction in the Kähler cone.

        Returns:
            np.ndarray or Array: A random interior point of the Kähler cone.
        """
        return self.sample_interior_point(rns_key=rns_key)

    def sample_rays(self, k: int, rns_key: Any | None = None):
        r"""
        **Description:**
        Samples `k` random rays from the Kähler cone.
        
        Args:
            k (int): Number of rays to sample.
            
        Returns:
            np.ndarray or Array: Array of sampled rays.
            
        Raises:
            RuntimeError: If ray data is not available.
            ValueError: If `k` is larger than the number of available rays.
        """
        if self._rays is None:
            raise RuntimeError("Cannot sample rays: ray data not available.")

        if len(self._rays) < k:
            raise ValueError(
                f"Requested number of random rays {k} larger than "
                f"number of available rays {len(self._rays)}."
            )

        inds = self._rand_choice(self._rays.shape[0], k, replace=False, rns_key=rns_key)
        return self._rays[inds]

    def sample_interior_point(self, rns_key: Any | None = None):
        r"""
        **Description:**
        Samples a random interior point of the Kähler cone.
        
        Returns:
            np.ndarray or Array: A random interior point of the Kähler cone.
        """

        pts = self.find_interior_points(N=100, normalise=True, verbosity=0)
        return pts[self._rand_choice(len(pts), 1, replace=True, rns_key=rns_key)[0]]

    def find_interior_points(
        self, 
        N: int = 1,
        stretching: float = 0.1, 
        normalise: bool = False,
        verbosity: int = 0
    ) -> np.ndarray | Array:
        r"""
        **Description:**
        Finds interior points of the Kähler cone using integer programming.
        
        Args:
            N (int, optional): Number of interior points to find. Default is 1.
            stretching (float, optional): Stretching parameter for the Kähler cone. Default is 0.1.
            normalise (bool, optional): Whether to normalise the points to L2-norm = 1. Default is False.
            verbosity (int, optional): Verbosity level for logging. Default is 0.
            
        Returns:
            np.ndarray or Array: Array of interior points.
            
        Raises:
            RuntimeError: If the optimization fails.
        """
        
        # Retrieve hyperplanes defining the Kähler cone
        H = self._hyperplanes

        # Create a new optimization model
        m = gp.Model("interior point finder")
        m.setParam('OutputFlag', verbosity > 0)  # Set verbosity for output
        
        # Configure the model to find multiple solutions
        m.setParam('PoolSearchMode', 2)
        m.setParam('PoolSolutions', N)
        
        # Define decision variables for the interior points
        p = m.addMVar(shape=H.shape[1], lb=-1000, ub=1000, vtype=gp.GRB.INTEGER)
        p_norm = m.addVar(ub=10000)  # Variable for the norm of the points
        
        # Add constraints to ensure points lie within the Kähler cone
        m.addConstr(H @ p >= stretching)
        m.addGenConstrNorm(p_norm, p, 2.0)  # Add constraint for L2 norm
        m.setObjective(p_norm, gp.GRB.MINIMIZE)  # Minimize the norm
        m.optimize()  # Solve the optimization problem
        
        # Retrieve and print the number of solutions found
        nSolutions = m.SolCount
        if verbosity >= 1:
            print(f"Found {nSolutions} solutions")

        m.Params.outputFlag = 0  # Suppress output clutter
        
        sols = []  # List to store solutions
        for i in range(nSolutions):
            m.setParam('SolutionNumber', i)  # Set the solution number to retrieve

            # Round the solution and convert to integer
            sols.append(np.rint(p.xn).astype(int))

        # Stack all solutions into a single array
        pts = np.vstack(sols)

        if normalise:
            # Normalize points to have L2-norm = 1
            pts = pts / np.linalg.norm(pts, axis=1)[:, None]

        return pts

    def rescale_points(self,pts,norm="l2",maxval=None,rns_key: Any | None = None):
        r"""
        **Description:**
        Rescales points to ensure they lie within a specified norm bound.
        
        Args:
            pts (np.ndarray): Points to be rescaled.
            norm (str, optional): Norm type for rescaling ("l2", "l1", or "inf"). Default is "l2".
            maxval (float, optional): Maximum value for the norm. Default is None.
            rns_key (Any | None, optional): PRNG random key for reproducibility. Default is None.
            
        Returns:
            np.ndarray or Array: Rescaled points.
            
        Raises:
            ValueError: If `norm` is not one of the supported types.
        """

        norms = ["l2","l1","inf"]
        if norm not in norms:
            raise ValueError(f"Norm for rescaling should be one of {norms}, but is {norm}.")

        if maxval is None:
            maxval = self.moduli_upper

        # Use jnp for all deterministic math — works correctly on both NumPy and JAX arrays
        pts_arr = jnp.asarray(pts)
        if norm == "l2":
            norms_vec = jnp.sqrt(jnp.sum(pts_arr * pts_arr, axis=1))
        elif norm == "l1":
            norms_vec = jnp.sum(jnp.abs(pts_arr), axis=1)
        else:  # "inf"
            norms_vec = jnp.max(jnp.abs(pts_arr), axis=1)

        rescalings = maxval / norms_vec
        # Points already within the target region keep their original scale
        rand = self._rand_uniform(0., 1., (len(rescalings),), rns_key)
        rescalings = jnp.where(rescalings >= 1.0, 1.0, rescalings * jnp.asarray(rand))
        return rescalings[:, None] * pts_arr

    def filter_points(self,
                      x: np.ndarray,
                      filter: Callable | None = None,
                      stretching: float = 0.0
                      ) -> np.ndarray | Array:
        r"""
        **Description:**
        Filters points to ensure they lie within the Kähler cone and applies an optional custom filter.
        
        Args:
            x (np.ndarray): Points to be filtered.
            filter (callable, optional): Custom filter function. Defaults to None.
            stretching (float, optional): Stretching parameter for the Kähler cone. Default is 0.
            
        Returns:
            np.ndarray or Array: Filtered points.
            
        """
        
        H = self._hyperplanes
        flag = jnp.all(jnp.asarray(x) @ H.T >= stretching, axis=1)
        x = x[flag]
        if filter is not None:
            x = filter(x)
        return x

    def get_moduli(
                self, 
                num_pts: int, 
                rns_key: Any | None = None, 
                minval: float | None = None, 
                maxval: float | None = None, 
                sampling_mode: str = "cone",
                stretching: float = 0.0,
                filter: Callable | None = None,
                n_rays: int = 2,
                perturbation: float = 1e-1,
                use_rays: bool = False,
                time_out: float = 60.0,
                verbosity: int = 0
            ) -> np.ndarray | Array:
        r"""
        **Description:**
        Samples moduli values within specified bounds using the selected sampling mode.

        Args:
            num_pts (int): 
                Number of points to sample.
            rns_key (Any | None): 
                PRNG random key for reproducibility. Defaults to None.
            minval (float | None): 
                Minimum value for sampling. Default is None.
            maxval (float | None): 
                Maximum value for sampling. Default is None.
            sampling_mode (str): 
                Mode of sampling ("box", "sphere", "cone", "stretched_cone", "tip_ray"). Default is "box".
            stretching (float): 
                Stretching parameter for the Kähler cone. Default is 0.
            n_rays (int): 
                Number of rays to use if `use_rays` is True. Default is 2.
            perturbation (float): 
                Perturbation applied to interior points. Default is 1e-1.
            use_rays (bool): 
                Whether to use rays for sampling. Default is False.
            time_out (float): 
                Time-out duration in seconds. Default is 60.
            verbosity (int): 
                Verbosity level for logging. Default is 0.

        Returns:
            np.ndarray or Array: 
                Array of sampled moduli.

        Raises:
            ValueError: 
                If `sampling_mode` is not recognized or required data is missing.
        """
        
        # Define supported sampling modes
        sampling_modes = ["box", "sphere", "cone", "stretched_cone", "tip_ray", "random_ray", "random_rays"]
        if sampling_mode not in sampling_modes:
            raise ValueError(f"`sampling_mode` should be one of {sampling_modes}, but is {sampling_mode}!")

        # Determine the ray basis depending on the sampling mode
        if sampling_mode in ["cone", "stretched_cone", "random_ray"]:
            if self._extremal_rays is None:
                if self._rays is None:
                    if self._hyperplanes is None:
                        raise ValueError("Need to provide information about the Kähler cone.")
                    if sampling_mode == "random_rays":
                        rays = self.sample_rays(n_rays, rns_key=rns_key)
                        use_rays = True
                    else:
                        rays = self._rays_kahler_cone
                else:
                    rays = self.find_interior_points(N=100, verbosity=verbosity)
            else:
                rays = self._extremal_rays

            if verbosity >= 1:
                print(f"Rays: {rays}")
            n_rays = len(rays)

        if sampling_mode in ["tip_ray", "stretched_cone"]:
            if self._tip is None:
                raise ValueError("Please provide tip of stretched cone!")
            ray = self._tip

        if sampling_mode == "random_ray":
            ray = self.sample_ray(rns_key=rns_key)

        if maxval is None:
            maxval = self.moduli_upper
        if minval is None:
            minval = self.moduli_lower

        # ------------------------------------------------------------------
        # "box": one-shot sampling — no loop needed
        # ------------------------------------------------------------------
        if sampling_mode == "box":
            return self._rand_uniform(minval, maxval, (num_pts, self._h12), rns_key)

        # ------------------------------------------------------------------
        # "cone" / "stretched_cone" / "random_rays" with ray coefficients
        # ------------------------------------------------------------------
        elif sampling_mode in ["cone", "stretched_cone", "random_rays"]:
            if use_rays:
                u = self._rand_uniform(0., maxval, (num_pts, n_rays), rns_key)
                cone_point = jnp.matmul(u, jnp.asarray(self._rays)) if self.use_jax \
                    else np.matmul(np.asarray(u), self._rays)
                if sampling_mode == "stretched_cone":
                    cone_point = cone_point + stretching * self._tip
                return cone_point[:num_pts]

            # Interior-point perturbation loop
            pts = self._rand_permutation(self._cone_points, rns_key)
            H = self._hyperplanes
            chunks: list = []
            n_collected: int = 0
            tic = time.time()
            if verbosity >= 1:
                print(f"#samples: 0        time: 0.00s       ", flush=True, end="\r")

            while n_collected < num_pts:
                if verbosity >= 1 and n_collected > 1:
                    toc = time.time()
                    print(
                        f"#samples: {n_collected}        time: {toc - tic:.2f}s       ",
                        flush=True, end="\r",
                    )

                coeffs = self._rand_uniform(-1, 1, pts.shape, rns_key)
                if self.use_jax:
                    pts0 = jnp.asarray(pts) + jnp.asarray(coeffs) * perturbation
                else:
                    pts0 = np.asarray(pts) + np.asarray(coeffs) * perturbation
                pts0 = self.rescale_points(pts0, norm="l2", maxval=maxval, rns_key=rns_key)
                pts0 = self.filter_points(pts0, filter=filter, stretching=stretching)

                if pts0.shape[0] > 0:
                    chunks.append(jnp.asarray(pts0) if self.use_jax else np.asarray(pts0))
                    n_collected += pts0.shape[0]

                if time.time() - tic > time_out:
                    raise RuntimeError("Failed to sample points in cone!")

            return self._collect_batches(chunks, num_pts)

        # ------------------------------------------------------------------
        # "tip_ray" / "random_ray": sample along a single ray direction
        # ------------------------------------------------------------------
        elif sampling_mode in ["tip_ray", "random_ray"]:
            tip_offset = ray if self._tip is None else self._tip
            chunks: list = []
            n_collected: int = 0
            tic = time.time()
            if verbosity >= 1:
                print(f"#samples: 0        time: 0.00s       ", flush=True, end="\r")

            while n_collected < num_pts:
                if verbosity >= 1 and n_collected > 1:
                    toc = time.time()
                    print(
                        f"#samples: {n_collected}        time: {toc - tic:.2f}s       ",
                        flush=True, end="\r",
                    )

                r = self._rand_uniform(0, maxval, (num_pts,), rns_key)
                if self.use_jax:
                    pts0 = jnp.asarray(ray) * jnp.asarray(r)[:, None] + stretching * jnp.asarray(tip_offset)
                else:
                    pts0 = np.asarray(ray) * np.asarray(r)[:, None] + stretching * np.asarray(tip_offset)
                pts0 = self.filter_points(pts0, filter=filter, stretching=stretching)

                if pts0.shape[0] > 0:
                    chunks.append(jnp.asarray(pts0) if self.use_jax else np.asarray(pts0))
                    n_collected += pts0.shape[0]

                if time.time() - tic > time_out:
                    raise RuntimeError("Failed to sample points along ray!")

            return self._collect_batches(chunks, num_pts)
    
    
    def get_axions(
            self,
            num_pts: int,
            rns_key: Any | None = None,
            minval: float = None,
            maxval: float = None,
            sampling_mode: str = "box"
        ) -> np.ndarray | Array:
        r"""
        **Description:**
        Generates random samples of axion values within specified bounds using the selected sampling mode.

        Args:
            num_pts (int): 
                Number of points to sample.
            rns_key (Any | None): 
                Random number seed key used for reproducibility. Default is None.
            minval (float): 
                Minimum value for axion sampling. If None, defaults to `self.axions_lower`.
            maxval (float): 
                Maximum value for axion sampling. If None, defaults to `self.axions_upper`.
            sampling_mode (str): 
                Mode of sampling, either "box" or "sphere". Default is "box".

        Returns:
            np.ndarray or Array: 
                Array of sampled axion values with shape `(num_pts, self._h12)`.

        Raises:
            ValueError: 
                If `sampling_mode` is not one of the supported modes ("box" or "sphere").
        """
        
        sampling_modes = ["box","sphere"]
        if sampling_mode not in sampling_modes:
            raise ValueError(f"`sampling_mode` should be one of {sampling_modes}, but is {sampling_mode}!")
            
        if maxval is None:
            maxval = self.axions_upper
        if minval is None:
            minval = self.axions_lower
        
        if sampling_mode == "box":
            return self._rand_uniform(minval, maxval, (num_pts, self._h12), rns_key).astype(complex)

    def get_complex_moduli(
            self,
            N: int,
            rns_key: Any | None = None,
            minval_axions: float = None,
            maxval_axions: float = None,
            minval_moduli: float = None,
            maxval_moduli: float = None,
            axion_sampling_mode: str = "box",
            moduli_sampling_mode: str = "cone"
        ) -> np.ndarray | Array:
        r"""
        **Description:**
        Generates complex samples for moduli, combining axion and moduli values.

        Args:
            N (int): 
                Number of complex moduli values to generate.
            rns_key (Any | None): 
                Random number seed key used for reproducibility. Default is None.
            minval_axions (float): 
                Minimum value for axion part of the moduli. Default is None.
            maxval_axions (float): 
                Maximum value for axion part of the moduli. Default is None.
            minval_moduli (float): 
                Minimum value for the real part of the moduli. Default is None.
            maxval_moduli (float): 
                Maximum value for the real part of the moduli. Default is None.
            axion_sampling_mode (str): 
                Sampling mode for axion values. Default is "box".
            moduli_sampling_mode (str): 
                Sampling mode for moduli values. Default is "box".

        Returns:
            np.ndarray or Array: 
                Array of complex moduli values.
        """
        
        moduli_val = self.get_axions(N,rns_key=rns_key,minval=minval_axions,maxval=maxval_axions,sampling_mode=axion_sampling_mode)
        
        moduli_val += 1j*self.get_moduli(N,rns_key=rns_key,minval=minval_moduli,maxval=maxval_moduli,sampling_mode=moduli_sampling_mode)
        
        return moduli_val

    
    
    def get_dilaton(
            self,
            num_pts: int,
            rns_key: Any | None = None,
            minval: float = None,
            maxval: float = None,
            sampling_mode: str = "box"
        ) -> np.ndarray | Array:
        r"""
        **Description:**
        Generates random samples of dilaton values within specified bounds using the selected sampling mode.

        Args:
            num_pts (int): 
                Number of points to sample.
            rns_key (Any | None): 
                Random number seed key used for reproducibility. Default is None.
            minval (float): 
                Minimum value for dilaton sampling. If None, defaults to `self.s_lower`.
            maxval (float): 
                Maximum value for dilaton sampling. If None, defaults to `self.s_upper`.
            sampling_mode (str): 
                Mode of sampling, either "box" or "sphere". Default is "box".

        Returns:
            np.ndarray or Array: 
                Array of sampled dilaton values.

        Raises:
            ValueError: 
                If `sampling_mode` is not one of the supported modes ("box" or "sphere").
        """
        
        sampling_modes = ["box","sphere"]
        if sampling_mode not in sampling_modes:
            raise ValueError(f"`sampling_mode` should be one of {sampling_modes}, but is {sampling_mode}!")
        
        if maxval is None:
            maxval = self.s_upper
        if minval is None:
            minval = self.s_lower
        
        if sampling_mode == "box":
            return self._rand_uniform(minval, maxval, (num_pts,), rns_key).astype(complex)

    def get_axion(
            self,
            num_pts: int,
            rns_key: Any | None = None,
            minval: float = None,
            maxval: float = None,
            sampling_mode: str = "box"
        ) -> np.ndarray | Array:
        r"""
        **Description:**
        Generates random samples of axion values within specified bounds using a selected sampling mode.

        Args:
            num_pts (int):
                Number of points to sample.
            rns_key (Any | None):
                Random number seed key used for reproducibility. Default is None.
            minval (float):
                Minimum value for axion sampling. If None, defaults to `self.axion_lower`.
            maxval (float):
                Maximum value for axion sampling. If None, defaults to `self.axion_upper`.
            sampling_mode (str): 
                Mode of sampling, either "box" or "sphere". Default is "box".

        Returns:
            np.ndarray or Array: 
                Array of sampled axion values.

        Raises:
            ValueError: 
                If `sampling_mode` is not one of the supported modes ("box" or "sphere").
        """
        
        sampling_modes = ["box","sphere"]
        if sampling_mode not in sampling_modes:
            raise ValueError(f"`sampling_mode` should be one of {sampling_modes}, but is {sampling_mode}!")
        
        if maxval is None:
            maxval = self.axion_upper
        if minval is None:
            minval = self.axion_lower
        
        if sampling_mode == "box":
            return self._rand_uniform(minval, maxval, (num_pts,), rns_key).astype(complex)


    def get_complex_tau(
            self,
            N: int,
            rns_key: Any | None = None,
            minval_axion: float = None,
            maxval_axion: float = None,
            minval_dilaton: float = None,
            maxval_dilaton: float = None,
            axion_sampling_mode: str = "box",
            dilaton_sampling_mode: str = "box"
        ) -> np.ndarray | Array:
        """
        **Description:**
        Generates complex samples for tau, which is a combination of axion and dilaton values.

        Args:
            N (int): 
                Number of complex tau values to generate.
            rns_key (Any | None): 
                Random number seed key used for reproducibility. Default is None.
            minval_axion (float): 
                Minimum value for axion part of tau. Default is None.
            maxval_axion (float): 
                Maximum value for axion part of tau. Default is None.
            minval_dilaton (float): 
                Minimum value for dilaton part of tau. Default is None.
            maxval_dilaton (float): 
                Maximum value for dilaton part of tau. Default is None.
            axion_sampling_mode (str): 
                Sampling mode for axion. Default is "box".
            dilaton_sampling_mode (str): 
                Sampling mode for dilaton. Default is "box".

        Returns:
            np.ndarray or Array: 
                Array of complex tau values.
        """
        
        tau_val = self.get_axion(N,rns_key=rns_key,minval=minval_axion,maxval=maxval_axion,sampling_mode=axion_sampling_mode)
        
        tau_val += 1j*self.get_dilaton(N,rns_key=rns_key,minval=minval_dilaton,maxval=maxval_dilaton,sampling_mode=dilaton_sampling_mode)
        
        return tau_val

    def sample_sphere(
            self,
            num_pts: int,
            dim: int = 1,
            rns_key: Any | None = None,
            angles: Tuple[float, float] = (0., 2. * np.pi),
            radius: float = 1.
        ) -> np.ndarray | Array:
        """
        **Description:**
        Samples points uniformly from the surface of a sphere or within a spherical volume.

        Args:
            num_pts (int): 
                Number of points to sample.
            dim (int): 
                Dimension of the sphere (1 for circle, 2 for sphere, etc.). Default is 1.
            rns_key (Any | None): 
                Random number seed key used for reproducibility. Default is None.
            angles (Tuple[float]): 
                Range of angles for sampling. Default is (0., 2.*pi).
            radius (float): 
                Radius of the sphere for sampling points. Default is 1.

        Returns:
            np.ndarray or Array: 
                Array of sampled complex values representing points on the sphere.
        """
        
        alpha = self._rand_uniform(angles[0], angles[1], (num_pts, dim), rns_key)
        r     = self._rand_uniform(0., radius,           (num_pts, dim), rns_key)
        val = np.asarray(r) * np.cos(np.asarray(alpha)) + 1j * np.asarray(r) * np.sin(np.asarray(alpha))
        
        # Flatten the result if the dimension is 1, otherwise return as is
        if dim == 1:
            return val.flatten()
        else:
            return val

    def filter_by_instantons(self, 
                             moduli: Array, 
                             inst_cutoff: float = 1e-1
                             ) -> Array:
        r"""
        **Description:**
        Filters starting points based on their instanton contributions to ensure they are sufficiently small.

        Args:
            moduli (Array): Array containing starting points for moduli.
            inst_cutoff (float, optional): Cutoff on the ratio of instanton to perturbative contributions to the pre-potential. Default is 1e-1.

        Returns:
            Array: Array containing starting points with sufficiently small instanton contributions.
        """

        # Compute the instanton contribution to the pre-potential for the given moduli
        Finstb = self._F_inst(moduli)
        
        # Compute the perturbative contribution to the pre-potential
        Fpertb = self._prepot(moduli) - Finstb

        # Return moduli where the ratio of instanton to perturbative contributions is below the cutoff
        return jnp.abs(Finstb) / jnp.abs(Fpertb) < inst_cutoff
    
    def filter_by_km(self, 
                    moduli: Array,
                    tau: Array,
                    ) -> Array:
        r"""
        **Description:**
        Filters starting points based on their Kähler metric to ensure they are positive definite.

        Args:
            moduli (Array): Array containing starting points for moduli.
            tau (Array): Array containing starting points for the axio-dilaton.

        Returns:
            Array: Array containing starting points with positive definite Kähler metric.
        """
        
        km_vals = self._kahler_metric(moduli, jnp.conj(moduli), tau, jnp.conj(tau))
        km_evals = jnp.linalg.eigvals(km_vals)
        rkm_evals = km_evals.real
        ikm_evals = km_evals.imag
        min_eval = jnp.min(rkm_evals, axis=1)

        return (min_eval > 0) & (jnp.max(jnp.abs(ikm_evals), axis=1) < 1e-5)
    
    def filter_moduli(self, 
                    moduli: Array,
                    tau: Array,
                    inst_cutoff: float = 1e-1
                    ) -> Array:
        r"""
        **Description:**
        Filters starting points for moduli based on both instanton contributions and Kähler metric positivity.

        Args:
            moduli (Array): Array containing starting points for moduli.
            tau (Array): Array containing starting points for the axio-dilaton.
            inst_cutoff (float, optional): Cutoff on the ratio of instanton to perturbative contributions to the pre-potential. Default is 1e-1.
            
        Returns:
            Array: Array containing starting points for moduli that satisfy both instanton and Kähler metric criteria.
        """
        
        flag_inst = self.filter_by_instantons(moduli, inst_cutoff=inst_cutoff)
        flag_km = self.filter_by_km(moduli, tau)
        flag = flag_inst & flag_km
        
        return moduli[flag]
    
    def initial_guesses(
        self,
        N: int,
        rns_key: Any | None = None,
        minval_dilaton: float | None = None,
        maxval_dilaton: float | None = None,
        minval_axions: float | None = None,
        maxval_axions: float | None = None,
        minval_moduli: float | None = None,
        maxval_moduli: float | None = None,
        moduli_sampling_mode: str = "cone",
        minval_fluxes: float | None = None,
        maxval_fluxes: float | None = None,
        fluxes_sampling_mode: str = "box",
        flux_mode: str = "full",
        flux_radius: float = 10.0,
        filter_moduli: bool = False,
        include_fluxes: bool = True
    ) -> Tuple[Array, Array] | Tuple[Array, Array, Array]:
        r"""
        **Description:**
        Generates initial guesses for moduli, dilaton, and optionally fluxes for sampling in moduli space.

        .. note::
            This function generates initial guesses for the complex structure moduli, dilaton, 
            and fluxes for a given number of samples, `N`. The function utilizes random sampling 
            within specified bounds and modes for dilaton, axions, moduli, and fluxes. If `filter_moduli` 
            is set to True, the generated moduli are filtered based on instanton effects. If `include_fluxes`
            is set to True, fluxes are also generated and returned.

        .. warning::
            For :math:`h^{1,2}>15`, we may not have the generators of the Kähler cone available. 
            In this case, we use the rays to sample points in the Kähler cone. Generically, this 
            could lead to a strong sampling bias in the starting points because we sample more points 
            in the regions of highest density of rays.

        Args:
            N (int): 
                The number of initial points to generate.
            rns_key (Any | None): 
                Random number seed key used for reproducibility. Default is None.
            minval_dilaton (float | None): 
                Minimum value for dilaton sampling. Default is None.
            maxval_dilaton (float | None): 
                Maximum value for dilaton sampling. Default is None.
            minval_axions (float | None): 
                Minimum value for axion sampling. Default is None.
            maxval_axions (float | None): 
                Maximum value for axion sampling. Default is None.
            minval_moduli (float | None): 
                Minimum value for moduli sampling. Default is None.
            maxval_moduli (float | None): 
                Maximum value for moduli sampling. Default is None.
            moduli_sampling_mode (str): 
                Sampling mode for moduli ("box", "cone", "stretched_cone", etc.). Default is "box".
            minval_fluxes (float | None): 
                Minimum value for flux sampling. Default is None.
            maxval_fluxes (float | None): 
                Maximum value for flux sampling. Default is None.
            fluxes_sampling_mode (str): 
                Sampling mode for fluxes ("box", "sphere", "tadpole_bound", "tadpole_cancel"). Default is "box".
            flux_mode (str): 
                Mode for flux generation ("full" or "half"). Default is "full".
            flux_radius (float): 
                Radius constraint for certain flux sampling modes. Default is 10.0.
            filter_moduli (bool):
                Whether to filter moduli based on instanton contributions. Default is False.
            include_fluxes (bool):
                Whether to include fluxes in the output. Default is True.

        Returns:
            Tuple[Array, Array] | Tuple[Array, Array, Array]:
                If ``include_fluxes`` is True, returns ``(moduli, tau, fluxes)`` where
                moduli has shape ``(N, h12)``, tau has shape ``(N,)``, and
                fluxes has shape ``(N, 2*n_fluxes)``.
                If ``include_fluxes`` is False, returns ``(moduli, tau)``.

        """
        
        # Sample axio-dilaton (tau) values uniformly within specified bounds.
        # Generated in one shot — no filtering needed.
        tau = self.get_complex_tau(
            N,
            rns_key=rns_key,
            minval_axion=minval_axions,
            maxval_axion=maxval_axions,
            minval_dilaton=minval_dilaton,
            maxval_dilaton=maxval_dilaton,
            axion_sampling_mode="box",
            dilaton_sampling_mode="box"
        )

        # Iteratively sample moduli until we have N valid points.
        # Without filtering the loop completes in one iteration; with filtering
        # it may require multiple passes.
        chunks: list = []
        n_collected: int = 0
        while n_collected < N:
            tmp = self.get_complex_moduli(
                N,
                rns_key=rns_key,
                minval_axions=minval_axions,
                maxval_axions=maxval_axions,
                minval_moduli=minval_moduli,
                maxval_moduli=maxval_moduli,
                axion_sampling_mode="box",
                moduli_sampling_mode=moduli_sampling_mode
            )
            if filter_moduli:
                tmp = self.filter_moduli(tmp, tau, inst_cutoff=1e-1)
            chunks.append(tmp)
            n_collected += int(tmp.shape[0])

        moduli = self._collect_batches(chunks, N)

        if include_fluxes:
            fluxes = self.get_fluxes(
                N,
                rns_key=rns_key,
                mode=flux_mode,
                minval=minval_fluxes,
                maxval=maxval_fluxes,
                sampling_mode=fluxes_sampling_mode,
                radius=flux_radius
            )
            return moduli, tau, fluxes
        else:
            return moduli, tau



    @partial(jit,static_argnums=(0,6,7,))
    def _ISD_sampling_FH(
        self,
        moduli: Array,
        moduli_c: Array,
        tau: Array,
        tau_c: Array,
        fluxes: Array,
        mode: str = "F",
        output: str = "full"
    ) -> Array:
        r"""
        **Description:**
        Computes RR-fluxes :math:`f=(f_1,f_2)` or NSNS-fluxes :math:`h=(h_1,h_2)` for given moduli 
        and axio-dilaton values via ISD sampling using the prepotential-based matrix method.

        .. admonition:: Details
            :class: dropdown

            For ISD fluxes, the RR-fluxes :math:`f=(f_1,f_2)` and NSNS-fluxes :math:`h=(h_1,h_2)` 
            are related through (see :func:`ISD_condition`):

            .. math::
                f=(s \, M(z^i,\overline{z}^i)\Sigma +  c_0)\, h\; ,\quad \tau=c_0 + \text{i} s \, .

            where the ISD-matrix :math:`M(z^i,\overline{z}^i)` is computed via :func:`ISD_matrix` 
            and :math:`\Sigma` is the symplectic form.
            
            **Mode "H"**: Given NSNS-fluxes :math:`h=(h_1,h_2)`, computes RR-fluxes :math:`f=(f_1,f_2)` via:

            .. math::
                f=(s \, M(z^i,\overline{z}^i)\Sigma +  c_0)\, h

            **Mode "F"**: Given RR-fluxes :math:`f=(f_1,f_2)`, computes NSNS-fluxes :math:`h=(h_1,h_2)` via:

            .. math::
                h=\dfrac{1}{|\tau|^2} (-s  M(z^i,\overline{z}^i)\Sigma + c_0)\, f

            The second relation exploits the symplectic property of the ISD-matrix.
            
        Args:
            moduli (Array): 
                Complex structure moduli values with shape (h^{1,2},).
            moduli_c (Array): 
                Complex conjugate of moduli with shape (h^{1,2},).
            tau (Array): 
                Axio-dilaton value (scalar complex).
            tau_c (Array): 
                Complex conjugate of axio-dilaton (scalar complex).
            fluxes (Array): 
                Input flux vector with shape (n_fluxes,) of either RR- or NSNS-fluxes.
            mode (str, optional): 
                ISD sampling mode. Either "F" (solve for RR-fluxes) or "H" (solve for NSNS-fluxes). 
                Defaults to "F".
            output (str, optional): 
                Output format. Either "full" (return complete flux vector) or "half" (return only 
                the fluxes determined by ISD condition). Defaults to "full".

        Returns:
            Array: 
                JAX array of sampled fluxes. Shape is (4*n_fluxes,) if output="full" or 
                (2*n_fluxes,) if output="half".

        Raises:
            ValueError: 
                If `mode` is not "F" or "H", or if `output` is not "full" or "half".
        
        See also: :func:`ISD_condition`, :func:`ISD_matrix`
        """
        
        # Validate mode parameter
        modes = ["F", "H"]
        if mode not in modes:
            raise ValueError(f"Unknown value for `mode`. Should be one of {modes}, but is {mode}!")
            
        # Validate output format parameter
        outputs = ["full", "half"]
        if output not in outputs:
            raise ValueError(f"Unknown value for `output`. Should be one of {outputs}, but is {output}!")
        
        # Extract real and imaginary parts of tau: tau = c0 + i*s
        c0 = (tau + tau_c) / 2
        s = (tau - tau_c) / 2 / 1j

        # Compute the symplectic action on the input flux vector: Sigma * fluxes
        SigmaFlux = jnp.matmul(self.model.periods.sigma, fluxes)

        # Compute the ISD matrix M(z^i, conj(z^i))
        M0 = self.model.ISD_matrix(moduli, moduli_c)

        # Compute the product M * Sigma * fluxes for use in both modes
        M0_SigmaFlux = jnp.matmul(M0, SigmaFlux)
        
        if mode == "H":
            # Mode "H": Given NSNS-fluxes, compute RR-fluxes
            # f = (s*M*Sigma + c0) * h
            f = M0_SigmaFlux * s + fluxes * c0

            # Return based on output format
            if output == "full":
                # Return full vector: [f_1, f_2, h_1, h_2]
                return jnp.append(f, fluxes)
            else:
                # Return only computed fluxes: [f_1, f_2]
                return f
        else:
            # Mode "F": Given RR-fluxes, compute NSNS-fluxes
            # First compute |tau|^2 = c0^2 + s^2
            tau_abs = c0**2 + s**2
            
            # h = (c0*f - s*M*Sigma*f) / |tau|^2
            h = (fluxes * c0 - M0_SigmaFlux * s) / tau_abs

            # Return based on output format
            if output == "full":
                # Return full vector: [f_1, f_2, h_1, h_2]
                return jnp.append(fluxes, h)
            else:
                # Return only computed fluxes: [h_1, h_2]
                return h


    @partial(jit,static_argnums=(0,6,7,))
    def _ISD_sampling_PM(
        self,
        moduli: Array,
        moduli_c: Array,
        tau: Array,
        tau_c: Array,
        fluxes: Array,
        mode: str = "ISD+",
        output: str = "full"
    ) -> Array:
        r"""
        **Description:**
        Solves the ISD condition to determine fluxes :math:`(f_1,h_1)` or :math:`(f_2,h_2)` using the 
        Picard-Fuchs approach for given moduli and axio-dilaton values.

        .. admonition:: Details
            :class: dropdown

            If the fluxes satisfy the Imaginary Self-Dual (ISD) condition :math:`\star G_3 = \text{i}G_3`, 
            then the RR-fluxes :math:`(f_1,f_2)` and NSNS-fluxes :math:`(h_1,h_2)` are related through:

            .. math::
                f_1-\tau h_1=\overline{\mathcal{N}}(z^i,\overline{z}^i)\, (f_2-\tau h_2)\, .

            **Mode ISD+**: Solves for fluxes :math:`(f_1,h_1)` given :math:`(f_2,h_2)` via:

            .. math::
                h_1=-\dfrac{\text{Im}(\Lambda)}{s}\; , \quad f_1=\text{Re}(\Lambda)-\dfrac{c_{0}}{s}\, \text{Im}(\Lambda)

            where :math:`\Lambda=\overline{\mathcal{N}}\, (f_2-\tau h_2)` and :math:`\tau=c_0+\text{i}\, s`.

            **Mode ISD-**: Solves for fluxes :math:`(f_2,h_2)` given :math:`(f_1,h_1)` via:

            .. math::
                h_2=-\dfrac{\text{Im}(\Lambda)}{s}\; , \quad f_2=\text{Re}(\Lambda)-\dfrac{c_{0}}{s}\, \text{Im}(\Lambda)

            where :math:`\Lambda=\overline{\mathcal{N}}^{-1}\, (f_1-\tau h_1)`.

            .. warning::
                For mode "ISD-", the gauge kinetic matrix :math:`\mathcal{N}` must be invertible. 
                This is only guaranteed inside the Kähler cone.

        Args:
            moduli (Array): 
                Complex structure moduli values with shape (h^{1,2},).
            moduli_c (Array): 
                Complex conjugate of moduli with shape (h^{1,2},).
            tau (Array): 
                Axio-dilaton value (scalar complex).
            tau_c (Array): 
                Complex conjugate of axio-dilaton (scalar complex).
            fluxes (Array): 
                Input flux vector with shape (2*n_fluxes,). First half contains RR-fluxes, 
                second half contains NSNS-fluxes.
            mode (str, optional): 
                ISD sampling mode. Either "ISD+" (solve for f_1,h_1) or "ISD-" (solve for f_2,h_2). 
                Defaults to "ISD+".
            output (str, optional): 
                Output format. Either "full" (return complete flux vector) or "half" (return only 
                the fluxes determined by ISD condition). Defaults to "full".

        Returns:
            Array: 
                JAX array of sampled fluxes. Shape is (4*n_fluxes,) if output="full" or 
                (2*n_fluxes,) if output="half".

        Raises:
            ValueError: 
                If `mode` is not "ISD+" or "ISD-", or if `output` is not "full" or "half".
        
        See also: :func:`ISD_condition`, :func:`gauge_kinetic_matrix`
        """
        
        # Validate mode parameter
        modes = ["ISD+", "ISD-"]
        if mode not in modes:
            raise ValueError(f"Unknown value for `mode`. Should be one of {modes}, but is {mode}!")
        
        # Validate output format parameter
        outputs = ["full", "half"]
        if output not in outputs:
            raise ValueError(f"Unknown value for `output`. Should be one of {outputs}, but is {output}!")

        # Compute the gauge kinetic matrix N and its complex conjugate
        NMatVal = self.model.gauge_kinetic_matrix(moduli, moduli_c)
        cNMatVal = self.model.gauge_kinetic_matrix(moduli, moduli_c, conj=True)
        
        # Split input fluxes into RR-fluxes (mu) and NSNS-fluxes (nu)
        mu = fluxes[:self.model.dimension_H3]
        nu = fluxes[self.model.dimension_H3:]
        
        # Compute RHS = f - tau*h for both tau and its conjugate
        RHS = mu - tau * nu
        cRHS = mu - tau_c * nu

        # Solve for Lambda depending on mode
        if mode == "ISD+":
            # ISD+ mode: Lambda = conj(N) * (f_2 - tau*h_2)
            Lambda = jnp.matmul(cNMatVal, RHS)
            cLambda = jnp.matmul(NMatVal, cRHS)
        else:
            # ISD- mode: Lambda = inv(conj(N)) * (f_1 - tau*h_1)
            Lambda = jnp.matmul(jnp.linalg.inv(cNMatVal), RHS)
            cLambda = jnp.matmul(jnp.linalg.inv(NMatVal), cRHS)

        # Extract real and imaginary parts of Lambda
        Re_Lambda = (Lambda + cLambda) / 2
        Im_Lambda = (Lambda - cLambda) / 2 / 1j
        
        # Extract real and imaginary parts of tau
        c0 = (tau + tau_c) / 2
        s = (tau - tau_c) / 2 / 1j

        # Compute the new flux values from Lambda decomposition
        ml = Re_Lambda - c0 / s * Im_Lambda
        nl = -1.0 / s * Im_Lambda

        # Return fluxes based on output format
        if output == "full":
            if mode == "ISD+":
                # Return: [f_1, f_2, h_1, h_2]
                return jnp.append(jnp.append(ml, mu), jnp.append(nl, nu))
            else:
                # Return: [f_1, f_2, h_1, h_2]
                return jnp.append(jnp.append(mu, ml), jnp.append(nu, nl))
        else:
            # Return only the computed fluxes: [f_i, h_i]
            return jnp.append(ml, nl)

    @partial(jit, static_argnums=(0, 6, 7, 8, 9, 10,))
    def ISD_sampling(
        self,
        moduli: Array,
        moduli_c: Array,
        tau: Array,
        tau_c: Array,
        flux0: Array,
        mode: str = "ISD+",
        output: str = "full",
        return_integer_flux: bool = False,
        in_axes: Tuple[int, int, int] = (0, 0, 0),
        vmap: bool = False
    ) -> Array:
        r"""
        **Description:**
        Determines half of the flux numbers such that the ISD condition
        :math:`\star G_3 = \text{i}G_3` is satisfied for given input values for the
        moduli :math:`z^i` and the axio-dilaton :math:`\tau`.

        .. admonition:: Details
            :class: dropdown

            This function implements *ISD sampling* as described in <>.
            The basic idea is to fix points in moduli space together with a subset of flux quanta,
            and fix the remaining fluxes through the ISD condition.

            There are four modi to solve the ISD condition for a given choices of half of the fluxes.
            Either we use the following form of the ISD condition (see :func:`ISD_condition`)

            .. math::
                f_1-\tau h_1=\overline{\mathcal{N}}(z^i,\overline{z}^i)\, (f_2-\tau h_2)\, .

            This expression can be solved for fluxes :math:`(f_1,h_1)` (corresponding to ``mode="ISD+"``)
            or :math:`(f_2,h_2)` (associated to ``mode="ISD-"``) corresponding to the components of the
            RR-flux vector :math:`f=(f_1,f_2)` and the NSNS-flux vector :math:`h=(h_1,h_2)`
            for given input values for the moduli :math:`z^i` and the axio-dilaton :math:`\tau`.
            The other two modi are obtained by rewriting the above expression in the following form

            .. math::
                f=(s \, M(z^i,\overline{z}^i)\Sigma +  c_0)\, h\; ,\quad \tau=c_0 + \text{i} s \, .

            This can then be solved for RR-flux vector :math:`f=(f_1,f_2)` (``mode="F"``)
            or the NSNS-flux vector :math:`h=(h_1,h_2)` (``mode="H"``)

            In general, the fluxes constructed in this way are non-integer.
            Upon rounding the fluxes to integers, the ISD condition is approximately satisfied:
            :math:`\star G_3 \approx \text{i}G_3` for given inputs.

        .. admonition:: Cached vmapped kernel
            :class: dropdown

            When ``vmap=True``, this method delegates to :func:`vmapping_func_cached`
            rather than constructing a new ``jax.jit(jax.vmap(...))`` closure on each
            call.  :func:`vmapping_func_cached` stores the compiled XLA kernel in an
            LRU cache keyed on ``(func, in_axes, frozen_kwargs)``.  Subsequent calls
            with the same arguments retrieve the pre-compiled kernel directly, avoiding
            recompilation overhead — particularly important when this method is called
            repeatedly inside the ``while`` loop of :func:`initial_guesses_ISD`.

        Args:
            moduli (Array): Complex structure moduli values with shape ``(h12,)``
                or ``(N, h12)`` if vmapped.
            moduli_c (Array): Complex conjugate of moduli, same shape as `moduli`.
            tau (Array): Axio-dilaton value; scalar or shape ``(N,)`` if vmapped.
            tau_c (Array): Complex conjugate of `tau`, same shape as `tau`.
            flux0 (Array): Input flux array with shape ``(n_fluxes,)`` or
                ``(N, n_fluxes)`` if vmapped.
            mode (str, optional): ISD sampling mode.  One of ``"ISD+"``, ``"ISD-"``,
                ``"F"``, ``"H"``.  Defaults to ``"ISD+"``.
            output (str, optional): Output format — ``"full"`` for the complete flux
                vector, ``"half"`` for only the ISD-fixed half.  Defaults to
                ``"full"``.
            return_integer_flux (bool, optional): Round fluxes to nearest integers.
                Defaults to ``False``.
            in_axes (Tuple[int, int, int], optional): Batch axes for vmap in the order
                ``(moduli_axis, tau_axis, flux_axis)``.  Defaults to ``(0, 0, 0)``.
            vmap (bool, optional): Vectorise over the batch dimension.  Defaults to
                ``False``.

        Returns:
            Array: Sampled flux array.  Shape ``(4*(h12+1),)`` or
                ``(2*(h12+1),)`` depending on ``output``; ``(N, flux_dim)`` when
                vmapped.

        Raises:
            ValueError: If ``mode`` or ``output`` are not among the recognised values.

        See also: :func:`ISD_condition`, :func:`vmapping_func_cached`,
            :func:`initial_guesses_ISD`, :func:`_ISD_sampling_PM`,
            :func:`_ISD_sampling_FH`
        """

        # --- vmap=True path: use cached jit(vmap(...)) to avoid recompilation ----
        if vmap:
            return vmapping_func_cached(
                self.ISD_sampling,
                in_axes=(in_axes[0], in_axes[0], in_axes[1], in_axes[1], in_axes[2]),
                mode=mode,
                output=output,
                return_integer_flux=return_integer_flux,
                vmap=False,
            )(moduli, moduli_c, tau, tau_c, flux0)

        # --- vmap=False path: single-point evaluation (identical to ISD_sampling) -
        modes = ["ISD+", "ISD-", "F", "H"]
        if mode not in modes:
            raise ValueError(
                f"Unknown value for `mode`. Should be one of {modes}, but is {mode}!"
            )

        outputs = ["full", "half"]
        if output not in outputs:
            raise ValueError(
                f"Unknown value for `output`. Should be one of {outputs}, but is {output}!"
            )

        if mode == "ISD+" or mode == "ISD-":
            flux = self._ISD_sampling_PM(
                moduli, moduli_c, tau, tau_c, flux0, mode=mode, output=output
            )
        elif mode == "F" or mode == "H":
            flux = self._ISD_sampling_FH(
                moduli, moduli_c, tau, tau_c, flux0, mode=mode, output=output
            )

        if return_integer_flux:
            flux = jnp.around(jnp.real(flux), 0).astype(jnp.int32)

        return flux


    def initial_guesses_ISD(
        self,
        N: int,
        rns_key: Any | None = None,
        minval_dilaton: float | None = None,
        maxval_dilaton: float | None = None,
        minval_axions: float | None = None,
        maxval_axions: float | None = None,
        minval_moduli: float | None = None,
        maxval_moduli: float | None = None,
        moduli_sampling_mode: str = "cone",
        minval_fluxes: float | None = None,
        maxval_fluxes: float | None = None,
        fluxes_sampling_mode: str = "box",
        flux_mode: str = "full",
        Nmax: float | None = None,
        filter_moduli: bool = False,
        mode: str = "ISD+",
        vmap_dim: int | None = None,
        ISD_oversample_factor: int = 10,
        print_progress: bool = False,
    ) -> Tuple[Array, Array, Array]:
        r"""
        **Description:**
        Generates initial guesses for moduli, dilaton, and fluxes using ISD
        (Imaginary Self-Dual) sampling.

        .. note::
            This function generates initial guesses for the complex structure moduli, dilaton,
            and fluxes for a given number of samples, `N` using ISD sampling. The function utilizes
            random sampling within specified bounds and modes for dilaton, axions, moduli, and fluxes.
            ISD sampling enforces the condition :math:`\star G_3 = \text{i}G_3` by fixing half of the
            flux quanta and deriving the remaining fluxes. If `filter_moduli` is set to True, the
            generated moduli are filtered based on instanton effects.

        .. warning::
            For :math:`h^{1,2}>15`, we may not have the generators of the Kähler cone available.
            In this case, we use the rays to sample points in the Kähler cone. Generically, this
            could lead to a strong sampling bias in the starting points because we sample more points
            in the regions of highest density of rays. To account for this, we assign a weight to each
            ray given by the inverse of the number of non-zero entries.

        .. warning::
            The fluxes generated via ISD sampling are generally non-integer. They are rounded to
            integers, so the ISD condition is only approximately satisfied: :math:`\star G_3 \approx \text{i}G_3`.

        .. admonition:: Implementation notes
            :class: dropdown

            **(a) Cached vmapped kernel via** :func:`vmapping_func_cached` **.**
            The vmapped+jitted ISD kernel is built once before the loop using
            :func:`vmapping_func_cached`, which stores the compiled XLA kernel in an
            LRU cache keyed on ``(func, in_axes, frozen_kwargs)``.  Subsequent
            iterations retrieve the cached callable directly, avoiding recompilation
            overhead on every loop iteration.

            **(b) O(N) list accumulation with** ``jnp.concatenate`` **.**
            Filtered batch arrays are accumulated in Python lists and concatenated
            exactly once after the loop via ``jnp.concatenate``, reducing total memory
            allocation to O(N) rather than the O(N²) behaviour of repeated
            ``jnp.append`` / ``np.append`` calls.

            **(c) Plain Python counter for async dispatch.**
            The collected-sample count is tracked with a plain Python ``int`` rather
            than evaluating ``len`` on a growing JAX array, which would force a device
            synchronisation on every iteration and stall JAX's asynchronous dispatch
            pipeline.

        Args:
            N (int): Number of initial points to generate.
            rns_key (Any | None): Random number seed key.  Default is ``None``.
            minval_dilaton (float | None): Lower bound for dilaton.  Default ``None``.
            maxval_dilaton (float | None): Upper bound for dilaton.  Default ``None``.
            minval_axions (float | None): Lower bound for axions.  Default ``None``.
            maxval_axions (float | None): Upper bound for axions.  Default ``None``.
            minval_moduli (float | None): Lower bound for moduli.  Default ``None``.
            maxval_moduli (float | None): Upper bound for moduli.  Default ``None``.
            moduli_sampling_mode (str): Sampling mode for moduli.  Default ``"box"``.
            minval_fluxes (float | None): Lower bound for flux sampling.  Default ``None``.
            maxval_fluxes (float | None): Upper bound for flux sampling.  Default ``None``.
            fluxes_sampling_mode (str): Sampling mode for fluxes.  Default ``"box"``.
            flux_mode (str): Flux generation mode (``"full"`` or ``"half"``).  Default ``"full"``.
            Nmax (float | None): Maximum D3-tadpole.  If ``None``, uses the model's
                ``D3_tadpole`` value.  Default ``None``.
            filter_moduli (bool): Filter moduli by instanton effects.  Default ``False``.
            mode (str): ISD mode — ``"ISD+"``, ``"ISD-"``, ``"F"``, or ``"H"``.
                Default ``"ISD+"``.
            vmap_dim (int | None): Batch size for vmapped operations.  Defaults to ``N``.
            ISD_oversample_factor (int): When ``use_jax=True``, process
                ``vmap_dim * ISD_oversample_factor`` points per outer iteration.
                More points per iteration → fewer iterations → fewer device syncs.
                Ignored when ``use_jax=False``.  Default ``10``.
            print_progress (bool): Print a running sample count.  Default ``False``.

        Returns:
            Tuple[Array, Array, Array]: ``(moduli, tau, fluxes)`` where
                moduli has shape ``(N, h12)``, tau has shape ``(N,)``, and
                fluxes has shape ``(N, 2*n_fluxes)`` with integer fluxes satisfying
                the ISD condition (approximately, after rounding).

        Raises:
            KeyboardInterrupt: Returns partial results if interrupted.

        See also: :func:`ISD_sampling`, :func:`vmapping_func_cached`,
            :func:`initial_guesses`
        """

        if vmap_dim is None:
            vmap_dim = N

        # When using JAX, process ISD_oversample_factor × more points per outer iteration
        # to reduce the total number of iterations (and therefore device syncs).
        # On CPU JAX random sampling is slower, but on GPU this amortises sync cost.
        effective_vmap: int = (vmap_dim * ISD_oversample_factor) if self.use_jax else vmap_dim

        # Fix 1: build the vmapped+jitted ISD kernel ONCE before the loop.
        # vmapping_func_cached stores the compiled XLA kernel in an LRU cache keyed
        # on (func, in_axes, frozen_kwargs), so the second and subsequent iterations
        # of the while loop retrieve the cached callable without any recompilation.
        _ISD_vmapped = vmapping_func_cached(
            self.ISD_sampling,
            in_axes=(0, 0, 0, 0, 0),
            mode=mode,
            output="full",
            return_integer_flux=True,
            vmap=False,
        )

        # Fix 2 & 3: accumulate results in Python lists; track count with a plain int.
        moduli_chunks: list = []
        tau_chunks: list = []
        fluxes_chunks: list = []
        n_collected: int = 0  # plain Python int — no device synchronisation

        try:
            while n_collected < N:

                if print_progress:
                    print(f"#samples: {n_collected}            ", flush=True, end="\r")

                z0, tau0 = self.initial_guesses(
                    effective_vmap,
                    rns_key=rns_key,
                    minval_dilaton=minval_dilaton,
                    maxval_dilaton=maxval_dilaton,
                    minval_axions=minval_axions,
                    maxval_axions=maxval_axions,
                    minval_moduli=minval_moduli,
                    maxval_moduli=maxval_moduli,
                    moduli_sampling_mode=moduli_sampling_mode,
                    filter_moduli=filter_moduli,
                    include_fluxes=False,
                )

                fluxes0 = self.get_fluxes(
                    effective_vmap,
                    rns_key=rns_key,
                    mode=flux_mode,
                    minval=minval_fluxes,
                    maxval=maxval_fluxes,
                    sampling_mode=fluxes_sampling_mode,
                    radius=Nmax,
                )[:, : self._n_fluxes]

                # Fix 1: call the pre-built cached kernel instead of re-entering
                # ISD_sampling(vmap=True) which would create a new closure each time.
                fluxes_ISD = _ISD_vmapped(
                    z0, jnp.conj(z0), tau0, jnp.conj(tau0), fluxes0
                )

                fluxes_ISD_integer = jnp.around(fluxes_ISD.real, 0).astype(jnp.int32)

                Nflux = self._tadpole(fluxes_ISD_integer)

                if Nmax is None:
                    Nmax = self._D3_tadpole

                flag = (Nflux <= Nmax) & (Nflux >= 0)
                z0_valid = z0[flag]
                tau0_valid = tau0[flag]
                fluxes0_valid = fluxes_ISD_integer[flag]

                if z0_valid.shape[0] > 0:
                    moduli_chunks.append(z0_valid)
                    tau_chunks.append(tau0_valid)
                    fluxes_chunks.append(fluxes0_valid)
                    n_collected += int(z0_valid.shape[0])  # plain Python int

        except KeyboardInterrupt:
            print("Interrupted by user.")

        # Fix 2: single concatenation — O(N) total allocation.
        if moduli_chunks:
            moduli = jnp.concatenate(moduli_chunks, axis=0)[:N]
            tau = jnp.concatenate(tau_chunks, axis=0)[:N]
            fluxes = jnp.concatenate(fluxes_chunks, axis=0)[:N]
        else:
            moduli = jnp.empty((0, self._h12), dtype=jnp.complex_)
            tau = jnp.empty((0,), dtype=jnp.complex_)
            fluxes = jnp.empty((0, 2 * self._n_fluxes), dtype=jnp.int32)

        return moduli, tau, fluxes
        

    


