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

# Import jax modules
import jax
import jax.numpy as jnp
from jax import jit, vmap, Array
from numpy.typing import ArrayLike
from typing import Optional, Tuple, Any
from jax.numpy import pi

# To load pickle files
import pickle
import gzip
import gurobipy as gp



jax.config.update("jax_enable_x64", True)

# Self-made modules
from .util import random_integer, random_uniform, random_uniform_jit, random_integer_jit, vmapping_func


home_dir=os.path.dirname(os.path.realpath(__file__))+"/.."


class data_sampler():

    def __init__(self, 
        model: Any, 
        flux_bounds: Tuple[float,float] = [-10, 10], 
        axion_bounds: Tuple[float,float] = [-0.5, 0.5], 
        dilaton_bounds: Tuple[float,float] = [2., 10.], 
        moduli_bounds: Tuple[float,float] = [1., 5.],
        use_jax : bool = False
        ) -> None:
        r"""
		**Description:**
        A class to sample initial data for the construction of flux vacua.
        
        Args:
            self.model (jaxvacua.flux_sector.flux_sector): Model class for flux compactifications.
            flux_bounds (ArrayLike): Bounds for fluxes.
            axion_bounds (ArrayLike): Bounds for axions.
            dilaton_bounds (ArrayLike): Bounds for dilaton.
            moduli_bounds (ArrayLike): Boundsd for moduli.
            use_jax (bool, optional): Use JAX environment for random number generation.
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

        """

        self.use_jax = use_jax
        self.model = model
        self.flux_upper = flux_bounds[1]
        self.flux_lower = flux_bounds[0]
        self._n_fluxes = self.model.n_fluxes
        self._h12 = self.model.h12
        self._tadpole = vmap(self.model.tadpole)
        self._D3_tadpole = self.model.D3_tadpole
        
        self._extremal_rays = np.array(self.model.periods.generators_kahler_cone)
        self._hyperplanes = np.array(self.model.periods.hyperplanes)
        self._rays = np.array(self.model.periods.rays_kahler_cone)
        self._tip = np.array(self.model.periods.tip_of_stretched_kahler_cone)

        if len(self._rays)==0:
            self._rays = None
        if len(self._extremal_rays)==0:
            self._extremal_rays = None
        
        self.axions_lower = axion_bounds[0]
        self.axions_upper = axion_bounds[1]
        
        self.axion_lower = axion_bounds[0]
        self.axion_upper = axion_bounds[1]

        self.moduli_lower = moduli_bounds[0]
        self.moduli_upper = moduli_bounds[1]

        self.s_lower = dilaton_bounds[0]
        self.s_upper = dilaton_bounds[1]
        
        self._F_inst = vmap(lambda x: self.model.F_inst(x))
        self._prepot = vmap(lambda x: self.model.F(x))

        try:
            self._cone_points = self.find_interior_points(N=1000,stretching=0.)
        except:
            self._cone_points = []


    def update_interior_points(
                               self, 
                               num_pts: int, 
                               rns_key: Optional[Any] = None,
                               maxval: float = None, 
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
            rns_key (Optional[Any], optional): PRNG random  key. Defaults to :c:`None`.
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

        if maxval is None:
            maxval = self.flux_upper

        pts = self.find_interior_points(N=2*num_pts,stretching=stretching)
        H = self._hyperplanes
        cone_point = []
        tic = time.time()

        while len(cone_point)<num_pts:
            if verbosity>=1 and len(cone_point)>1:
                print(f"#samples: {len(cone_point)}          ",flush=True,end="\r")

            np.random.shuffle(pts)

            pts0 = pts[:num_pts].copy()

            if self.use_jax:
                coeffs = random_uniform(-1,1,rns_key=rns_key,shape=pts0.shape)
            else:
                coeffs = np.random.uniform(-1,1,pts0.shape)

            pts0 = pts0+coeffs*perturbation

            # Rescale points
            pts0 = self.rescale_points(pts0,norm="l2",maxval=maxval)
            
            flag = np.all(pts0@H.T >= stretching,axis=1)

            if len(cone_point)==0:
                cone_point = pts0[flag]
            else:
                cone_point = np.append(cone_point,pts0[flag],axis=0)

            toc = time.time()
            if toc-tic>time_out:
                raise RuntimeError("Failed to sample points in cone!")

        if len(self._cone_points)>0:
            cone_point = np.append(cone_point,self._cone_points,axis=0)
            
        self._cone_points = cone_point


    def get_fluxes(
        self, 
        num_pts: int, 
        mode: str = "full", 
        rns_key: Optional[Any] = None, 
        minval: Optional[int] = None, 
        maxval: Optional[int] = None, 
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
            rns_key (PRNGKey, optional): PRNG random key. Defaults to :c:`None`.
            minval (Optional[int], optional): Minimum value for flux sampling. Default is None.
            maxval (Optional[int], optional): Maximum value for flux sampling. Default is None.
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
            
        tot_flux = []
        while len(tot_flux)<num_pts:
            
            if self.use_jax:
                flux = random_integer(minval,maxval+1,rns_key=rns_key,shape=(num_pts,dim))
            else:
                flux = np.random.randint(minval,maxval+1,(num_pts,dim))

            if sampling_mode == "tadpole_bound":
                flag = self._tadpole(flux) <= radius
            if sampling_mode == "tadpole_cancel":
                flag = self._tadpole(flux) == radius
            elif sampling_mode == "sphere":
                flag = np.sqrt(np.sum(flux*flux,axis=1)) <= radius

            if len(tot_flux)==0:
                if sampling_mode == "box":
                    tot_flux = flux
                    break
                else:
                    tot_flux = flux[flag]
            else:
                tot_flux = np.append(tot_flux,flux[flag],axis=0)

        return tot_flux[:num_pts]


    def sample_ray(self):
        r"""
        **Description:**
        Samples a random ray from the Kähler cone.
        
        Returns:
            np.ndarray: A random ray from the Kähler cone.
        """
        
        return self._random_direction_in_cone()

    def _random_direction_in_cone(self):
        r"""
        **Description:**
        Samples a random direction in the Kähler cone.

        Returns:
            np.ndarray: A random interior point of the Kähler cone.
        """
        return self.sample_interior_point()

    def sample_rays(self, k: int):
        r"""
        **Description:**
        Samples `k` random rays from the Kähler cone.
        
        Args:
            k (int): Number of rays to sample.
            
        Returns:
            np.ndarray: Array of sampled rays.
            
        Raises:
            RuntimeError: If ray data is not available.
            ValueError: If `k` is larger than the number of available rays.
        """
        if self._rays is None:
            raise RuntimeError("Cannot sample rays: ray data not available.")

        if len(self._rays)<k:
            raise ValueError(f"Requested number of random rays {k} larger than\
                                number of available rays {len(self._rays)}.")

        inds = np.random.choice(np.arange(len(self._rays)),replace=False,size=k)

        return self._rays[inds]

    def sample_interior_point(self):
        r"""
        **Description:**
        Samples a random interior point of the Kähler cone.
        
        Returns:
            np.ndarray: A random interior point of the Kähler cone.
        """

        pts = self.find_interior_points(N=100,normalise=True,verbosity=0)
        
        return pts[np.random.choice(np.arange(len(pts)))]

    def find_interior_points(
        self, 
        N=1,
        stretching=0.1, 
        normalise = False,
        verbosity=0
        ) -> np.ndarray:
        r"""
        **Description:**
        Finds interior points of the Kähler cone using integer programming.
        
        Args:
            N (int, optional): Number of interior points to find. Default is 1.
            stretching (float, optional): Stretching parameter for the Kähler cone. Default is 0.1.
            normalise (bool, optional): Whether to normalise the points to L2-norm = 1. Default is False.
            verbosity (int, optional): Verbosity level for logging. Default is 0.
            
        Returns:
            np.ndarray: Array of interior points.
            
        Raises:
            RuntimeError: If the optimization fails.
        """
        
        
        H = self._hyperplanes

        m = gp.Model("interior point finder")
        m.setParam('OutputFlag', verbosity>0)
        
        m.setParam('PoolSearchMode', 2)
        m.setParam('PoolSolutions', N)
        
        p = m.addMVar(shape=H.shape[1], lb=-1000, ub=1000, vtype=gp.GRB.INTEGER)
        p_norm = m.addVar(ub=10000)
        
        m.addConstr(H@p >= stretching)
        m.addGenConstrNorm(p_norm, p, 2.0)
        m.setObjective(p_norm, gp.GRB.MINIMIZE)
        m.optimize()
        
        # retrieve and print the solutions
        nSolutions = m.SolCount
        if verbosity>=1:
            print(f"Found {nSolutions} solutions")

        m.Params.outputFlag = 0 # avoid clutter
        
        sols = []
        for i in range(nSolutions):
            m.setParam('SolutionNumber', i)

            sols.append(np.rint(p.xn).astype(int))

        pts = np.vstack(sols)

        if normalise:
            # Normalise pts to L2-norm = 1
            pts = pts/np.linalg.norm(pts,axis=1)[:,None]

        return pts

    def rescale_points(self,pts,norm="l2",maxval=None):
        r"""
        **Description:**
        Rescales points to ensure they lie within a specified norm bound.
        
        Args:
            pts (np.ndarray): Points to be rescaled.
            norm (str, optional): Norm type for rescaling ("l2", "l1", or "inf"). Default is "l2".
            maxval (float, optional): Maximum value for the norm. Default is None.
            
        Returns:
            np.ndarray: Rescaled points.
            
        Raises:
            ValueError: If `norm` is not one of the supported types.
        """

        norms = ["l2","l1","inf"]
        if norm not in norms:
            raise ValueError(f"Norm for rescaling should be one of {norms}, but is {norm}.")

        if maxval is None:
            maxval = self.moduli_upper
        
        # Rescaling so that moduli values are smaller than fixed value set by maxval
        rescalings = maxval
        if norm=="l2":
            rescalings /= np.sqrt(np.sum(pts*pts,axis=1))
        elif norm=="l1":
            rescalings /= np.sum(np.abs(pts),axis=1)
        elif norm=="inf":
            rescalings /= np.max(np.abs(pts),axis=1)

        # Those that are already in target region, do nothing later
        flag = rescalings>0.99
        # Multiply by random number
        rescalings = rescalings*np.random.uniform(0,1,len(rescalings))
        # Those which were already in, do nothing by setting rescaling to 1.
        rescalings[flag] = 1.

        return rescalings[:,None]*pts

    def filter_points(self,
                      x,
                      filter=None,
                      stretching=0.):
        r"""
        Filters points to ensure they lie within the Kähler cone and applies an optional custom filter.
        
        Args:
            x (np.ndarray): Points to be filtered.
            filter (callable, optional): Custom filter function. Defaults to None.
            stretching (float, optional): Stretching parameter for the Kähler cone. Default is 0.
            
        Returns:
            np.ndarray: Filtered points.
            
        """
        
        H = self._hyperplanes

        flag = np.all(x@H.T >= stretching,axis=1)

        x = x[flag]

        if not filter is None:
            x = filter(x)

        return x

    def get_moduli(
                self, 
                num_pts: int, 
                rns_key: Optional[Any] = None, 
                minval: float = None, 
                maxval: float = None, 
                sampling_mode: str = "cone",
                stretching : float = 0.,
                filter = None,
                n_rays: int = 2,
                perturbation: float = 1e-1,
                use_rays: bool = False,
                time_out: float = 60.,
                verbosity: int = 0
            ) -> np.ndarray:
        """
        Samples moduli values within specified bounds using the selected sampling mode.

        Args:
            num_pts (int): Number of points to sample.
            rns_key (Optional[Any], optional): PRNG random key. Defaults to None.
            minval (float, optional): Minimum value for sampling. Default is None.
            maxval (float, optional): Maximum value for sampling. Default is None.
            sampling_mode (str, optional): Mode of sampling ("box", "sphere", "cone", "stretched_cone", "tip_ray"). Default is "box".
            stretching (float, optional): Stretching parameter for the Kähler cone. Default is 0.
            n_rays (int, optional): Number of rays to use if `use_rays` is True. Default is 2.
            perturbation (float, optional): Perturbation applied to interior points. Default is 1e-1.
            use_rays (bool, optional): Whether to use rays for sampling. Default is False.
            time_out (float, optional): Time-out duration in seconds. Default is 60.
            verbosity (int, optional): Verbosity level for logging. Default is 0.


        Returns:
            np.ndarray: Array of sampled moduli.

        Raises:
            ValueError: If `sampling_mode` is not recognized or required data is missing.
        
        """
        
        sampling_modes = ["box","sphere","cone","stretched_cone","tip_ray","random_ray","random_rays"]
        if sampling_mode not in sampling_modes:
            raise ValueError(f"`sampling_mode` should be one of {sampling_modes}, but is {sampling_mode}!")
            
        

        if sampling_mode=="cone" or sampling_mode=="stretched_cone" or "random_ray" in sampling_mode:
            if self._extremal_rays is None:
                if self._rays is None:
                    if self._hyperplanes is None:
                        raise ValueError("Need to provide information about the Kähler cone.")
                
                    if sampling_mode=="random_rays":
                        rays=self.sample_rays(n_rays)
                        use_rays = True
                    else:
                        rays=self._rays_kahler_cone
                    
                else:
                    rays = self.find_interior_points(N=100,verbosity=verbosity)

            else:
                rays = self._extremal_rays

            if verbosity>=1:
                print(f"Rays: {rays}")

            n_rays = len(rays)

        if sampling_mode=="tip_ray" or "stretched" in sampling_mode:
            if self._tip is None:
                raise ValueError("Please provide tip of stretched cone!")

            ray = self._tip

        if sampling_mode=="random_ray":
            ray = self.sample_ray()

        if maxval is None:
            maxval = self.moduli_upper
        if minval is None:
            minval = self.moduli_lower
            
        
        if sampling_mode == "box":
            
            if self.use_jax:
                return random_uniform(minval,maxval,rns_key=rns_key,shape=(num_pts,self._h12))
            else:
                return np.random.uniform(minval,maxval,(num_pts,self._h12))

        elif sampling_mode=="cone" or sampling_mode=="stretched_cone" or sampling_mode=="random_rays":

            if use_rays:
                if self.use_jax:
                    u = random_uniform(0.,maxval,rns_key=rns_key,shape=(num_pts,n_rays)) 
                else:
                    u = np.random.uniform(0,maxval,(num_pts,n_rays))

                # Contract with basis elements
                cone_point = np.matmul(u,rays)

                if sampling_mode=="stretched_cone":
                    # We add the tip of the stretched Kähler cone defined by some `c`
                    cone_point = cone_point + stretching*self._tip

            else:
                pts = self._cone_points

                np.random.shuffle(pts)

                H = self._hyperplanes
                cone_point = []
                tic = time.time()
                toc = tic
                if verbosity>=1:
                    print(f"#samples: {len(cone_point)}        time: {float(np.around(toc-tic,2))}s       ",flush=True,end="\r")
                while len(cone_point)<num_pts:

                    if verbosity>=1 and len(cone_point)>1:
                        print(f"#samples: {len(cone_point)}        time: {float(np.around(toc-tic,2))}s       ",flush=True,end="\r")

                    if self.use_jax:
                        coeffs = random_uniform(-1,1,rns_key=rns_key,shape=pts.shape)
                    else:
                        coeffs = np.random.uniform(-1,1,pts.shape)

                    pts0 = pts+coeffs*perturbation

                    # Rescale points
                    pts0 = self.rescale_points(pts0,norm="l2",maxval=maxval)
                    
                    pts0 = self.filter_points(pts0,filter=filter,stretching=stretching)
                    #flag = np.all(pts0@H.T >= stretching,axis=1)

                    if len(cone_point)==0:
                        cone_point = pts0#[flag]
                    else:
                        cone_point = np.append(cone_point,pts0,axis=0)#pts0[flag]

                    toc = time.time()
                    if toc-tic>time_out:
                        raise RuntimeError("Failed to sample points in cone!")

            return cone_point[:num_pts]

        elif sampling_mode=="tip_ray" or sampling_mode=="random_ray":
            
            H = self._hyperplanes
            cone_point = []
            tic = time.time()
            toc = tic
            if verbosity>=1:
                print(f"#samples: {len(cone_point)}        time: {float(np.around(toc-tic,2))}s       ",flush=True,end="\r")
            while len(cone_point)<num_pts:

                if verbosity>=1 and len(cone_point)>1:
                    print(f"#samples: {len(cone_point)}        time: {float(np.around(toc-tic,2))}s       ",flush=True,end="\r")

                if self.use_jax:
                    r = random_uniform(0,maxval,rns_key=rns_key,shape=(num_pts,))
                else:
                    r = np.random.uniform(0,maxval,(num_pts,))
                
                if self._tip is None:
                    pts0 = ray*r[:,None]+stretching*ray
                else:
                    pts0 = ray*r[:,None]+stretching*self._tip

                pts0 = self.filter_points(pts0,filter=filter,stretching=stretching)
                #flag = np.all(pts0@H.T >= stretching,axis=1)

                if len(cone_point)==0:
                    cone_point = pts0#[flag]
                else:
                    cone_point = np.append(cone_point,pts0,axis=0)#pts0[flag]

                toc = time.time()
                if toc-tic>time_out:
                    raise RuntimeError("Failed to sample points along ray!")
            
            return cone_point[:num_pts]
    
    
    def get_axions(
            self,
            num_pts: int,
            rns_key: Optional[Any] = None,
            minval: float = None,
            maxval: float = None,
            sampling_mode: str = "box"
        ) -> np.ndarray:
        r"""
        Generates random samples of axion values within specified bounds using the selected sampling mode.

        Args:
            num_pts (int): 
                Number of points to sample.
            rns_key (Optional[Any]): 
                Random number seed key used for reproducibility. Default is None.
            minval (float): 
                Minimum value for axion sampling. If None, defaults to `self.axions_lower`.
            maxval (float): 
                Maximum value for axion sampling. If None, defaults to `self.axions_upper`.
            sampling_mode (str): 
                Mode of sampling, either "box" or "sphere". Default is "box".

        Returns:
            np.ndarray: 
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
            if self.use_jax:
                return random_uniform(minval,maxval,rns_key=rns_key,shape=(num_pts,self._h12)).astype(jnp.complex128)
            else:
                return np.random.uniform(minval,maxval,(num_pts,self._h12)).astype(np.complex128)

    def get_complex_moduli(
            self,
            N: int,
            rns_key: Optional[Any] = None,
            minval_axions: float = None,
            maxval_axions: float = None,
            minval_moduli: float = None,
            maxval_moduli: float = None,
            axion_sampling_mode: str = "box",
            moduli_sampling_mode: str = "box"
        ) -> np.ndarray:
        r"""
        Generates complex samples for moduli, combining axion and moduli values.

        Args:
            N (int): 
                Number of complex moduli values to generate.
            rns_key (Optional[Any]): 
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
            np.ndarray: 
                Array of complex moduli values.
        """
        
        moduli_val = self.get_axions(N,rns_key=rns_key,minval=minval_axions,maxval=maxval_axions,sampling_mode=axion_sampling_mode)
        
        moduli_val += 1j*self.get_moduli(N,rns_key=rns_key,minval=minval_moduli,maxval=maxval_moduli,sampling_mode=moduli_sampling_mode)
        
        return moduli_val

    
    
    def get_dilaton(
            self,
            num_pts: int,
            rns_key: Optional[Any] = None,
            minval: float = None,
            maxval: float = None,
            sampling_mode: str = "box"
        ) -> np.ndarray:
        r"""
        Generates random samples of dilaton values within specified bounds using the selected sampling mode.

        Args:
            num_pts (int): 
                Number of points to sample.
            rns_key (Optional[Any]): 
                Random number seed key used for reproducibility. Default is None.
            minval (float): 
                Minimum value for dilaton sampling. If None, defaults to `self.s_lower`.
            maxval (float): 
                Maximum value for dilaton sampling. If None, defaults to `self.s_upper`.
            sampling_mode (str): 
                Mode of sampling, either "box" or "sphere". Default is "box".

        Returns:
            np.ndarray: 
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
            
            if self.use_jax:
                return random_uniform(minval,maxval,rns_key=rns_key,shape=(num_pts,)).astype(jnp.complex128)
            else:
                return np.random.uniform(minval,maxval,(num_pts,)).astype(np.complex128)
    
    def get_axion(
            self,
            num_pts: int,
            rns_key: Optional[Any] = None,
            minval: float = None,
            maxval: float = None,
            sampling_mode: str = "box"
        ) -> np.ndarray:
        r"""
        Generates random samples of axion values within specified bounds using a selected sampling mode.

        Args:
            num_pts (int): 
                Number of points to sample.
            rns_key (Optional[Any]): 
                Random number seed key used for reproducibility. Default is None.
            minval (float): 
                Minimum value for axion sampling. If None, defaults to `self.axion_lower`.
            maxval (float): 
                Maximum value for axion sampling. If None, defaults to `self.axion_upper`.
            sampling_mode (str): 
                Mode of sampling, either "box" or "sphere". Default is "box".

        Returns:
            np.ndarray: 
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
            
            #return random_uniform(minval,maxval,rns_key=rns_key,shape=(num_pts,))
            return np.random.uniform(minval,maxval,(num_pts,)).astype(np.complex128)

    
    
    def get_complex_tau(
            self,
            N: int,
            rns_key: Optional[Any] = None,
            minval_axion: float = None,
            maxval_axion: float = None,
            minval_dilaton: float = None,
            maxval_dilaton: float = None,
            axion_sampling_mode: str = "box",
            dilaton_sampling_mode: str = "box"
        ) -> np.ndarray:
        """
        Generates complex samples for tau, which is a combination of axion and dilaton values.

        Args:
            N (int): 
                Number of complex tau values to generate.
            rns_key (Optional[Any]): 
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
            np.ndarray: 
                Array of complex tau values.
        """
        
        tau_val = self.get_axion(N,rns_key=rns_key,minval=minval_axion,maxval=maxval_axion,sampling_mode=axion_sampling_mode)
        
        tau_val += 1j*self.get_dilaton(N,rns_key=rns_key,minval=minval_dilaton,maxval=maxval_dilaton,sampling_mode=dilaton_sampling_mode)
        
        return tau_val

    def sample_sphere(
            self,
            num_pts: int,
            dim: int = 1,
            rns_key: Optional[Any] = None,
            angles: Tuple[float,float] = [0., 2. * pi],
            radius: float = 1.
        ) -> np.ndarray:
        """
        Samples points uniformly from the surface of a sphere or within a spherical volume.

        Args:
            num_pts (int): 
                Number of points to sample.
            dim (int): 
                Dimension of the sphere (1 for circle, 2 for sphere, etc.). Default is 1.
            rns_key (Optional[Any]): 
                Random number seed key used for reproducibility. Default is None.
            angles (Tuple[float]): 
                Range of angles for sampling. Default is [0., 2.*pi].
            radius (float): 
                Radius of the sphere for sampling points. Default is 1.

        Returns:
            np.ndarray: 
                Array of sampled complex values representing points on the sphere.
        """
        
        if self.use_jax:
            alpha = random_uniform(angles[0],angles[1],rns_key=rns_key,shape=(num_pts,dim))
            r = random_uniform(0.,radius,rns_key=rns_key,shape=(num_pts,dim))
        else:
            alpha = np.random.uniform(angles[0],angles[1],(num_pts,dim))
            r = np.random.uniform(0.,radius,(num_pts,dim))
        
        val = r*np.cos(alpha)+1j*r*np.sin(alpha)
        
        if dim==1:
            return val.flatten()
        else:
            return val

    def filter_by_instantons(self,moduli, inst_cutoff = 1e-1):
        r"""

        **Description:**
        Returns starting points with sufficiently small instanton contributions.


        Args:
            F_inst_V (vmap function): Vmapped function to compute the instanton contribution to the pre-potential.
            Prepot_V (vmap function): Vmapped function to compute the perturbative contribution to the pre-potential
            initial (ArrayLike): Array containing starting points.
            inst_cutoff (float,optional): Cutoff on the ratio of instanton and perturbative contribution to the pre-potential.


        Returns:
            initial (ArrayLike): Array containing starting points having sufficiently small instanton contributions.

        """

        Finstb = self._F_inst(moduli)
        Fpertb = self._prepot(moduli) - Finstb

        return moduli[jnp.abs(Finstb)/jnp.abs(Fpertb) < inst_cutoff]
    
    def initial_guesses(
        self,
        N: int,
        rns_key: Optional[Any] = None,
        minval_dilaton: float = None,
        maxval_dilaton: float = None,
        minval_axions: float = None,
        maxval_axions: float = None,
        minval_moduli: float = None,
        maxval_moduli: float = None,
        moduli_sampling_mode: str = "box",
        minval_fluxes: float = None,
        maxval_fluxes: float = None,
        fluxes_sampling_mode: str = "box",
        flux_mode: str = "full",
        flux_radius: float = 10.0,
        filter_moduli: bool = False,
        include_fluxes: bool = True
    ) -> Tuple[Any, Any, Optional[Any]]:
        r"""
        **Description:**
        Generates initial guesses for moduli, dilaton, and fluxes.

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
            in the regions of highest density of rays. To account for this, we assign a weight to each 
            ray given by the inverse of the number of non-zero entries.

        Args:
            N (int): 
                The number of initial points to generate.
            rns_key (Optional[Any]): 
                Random number seed key used for reproducibility. Default is None.
            minval_dilaton (float): 
                Minimum value for dilaton sampling. Default is None.
            maxval_dilaton (float): 
                Maximum value for dilaton sampling. Default is None.
            minval_axions (float): 
                Minimum value for axion sampling. Default is None.
            maxval_axions (float): 
                Maximum value for axion sampling. Default is None.
            minval_moduli (float): 
                Minimum value for moduli sampling. Default is None.
            maxval_moduli (float): 
                Maximum value for moduli sampling. Default is None.
            moduli_sampling_mode (str): 
                Sampling mode for moduli. Default is "box".
            minval_fluxes (float): 
                Minimum value for flux sampling. Default is None.
            maxval_fluxes (float): 
                Maximum value for flux sampling. Default is None.
            fluxes_sampling_mode (str): 
                Sampling mode for fluxes. Default is "box".
            flux_mode (str): 
                Mode for flux generation. Default is "full".
            flux_radius (float): 
                Radius within which flux values are sampled. Default is 10.
            filter_moduli (bool): 
                Whether to filter moduli based on instanton effects. Default is True.
            include_fluxes (bool): 
                Whether to include fluxes in the output. Default is True.

        Returns:
            Tuple[Any, Any, Optional[Any]]:
                A tuple containing:
                    - moduli (Any): Array of sampled moduli values.
                    - tau (Any): Array of sampled dilaton values.
                    - fluxes (Optional[Any]): Array of sampled fluxes, if `include_fluxes` is True.

        """
        
        tau = self.get_complex_tau(N,rns_key=rns_key,minval_axion=minval_axions,maxval_axion=maxval_axions,minval_dilaton=minval_dilaton,
                            maxval_dilaton=maxval_dilaton,axion_sampling_mode="box",dilaton_sampling_mode="box")
        
        moduli = []
        while len(moduli)<N:
        
            tmp = self.get_complex_moduli(N,rns_key=rns_key,minval_axions=minval_axions,maxval_axions=maxval_axions,minval_moduli=minval_moduli,
                            maxval_moduli=maxval_moduli,axion_sampling_mode="box",moduli_sampling_mode=moduli_sampling_mode)
            
            if filter_moduli:
                tmp = self.filter_by_instantons(tmp, inst_cutoff = 1e-1)
                
            if len(moduli)==0:
                moduli = tmp
            else:
                moduli = np.append(moduli,tmp,axis=0)
                
        moduli = moduli[:N]
        
        if include_fluxes:
            fluxes = self.get_fluxes(N,rns_key=rns_key,mode=flux_mode,minval=minval_fluxes,maxval=maxval_fluxes,sampling_mode=fluxes_sampling_mode,radius=flux_radius)
        
            return jnp.array(moduli),jnp.array(tau),jnp.array(fluxes)
        else:
            return jnp.array(moduli),jnp.array(tau)



    @partial(jit,static_argnums=(0,6,7,))
    def _ISD_sampling_FH(self,moduli,moduli_c,tau,tau_c,fluxes,mode="F",output="full"):
        r"""

        **Description:**
        Returns RR-fluxes :math:`f=(f_1,f_2)` or NSNS-fluxes :math:`h=(h_1,h_2)` for given input values 
        for the moduli :math:`z^i` and the axio-dilaton :math:`\tau` via ISD sampling.

        .. admonition:: Details
            :class: dropdown

            If the fluxes are ISD, then the RR-fluxes :math:`f=(f_1,f_2)` and NSNS-fluxes :math:`h=(h_1,h_2)` are related 
            to each other through (see :func:`ISD_condition`)

            .. math::
                f=(s \, M(z^i,\overline{z}^i)\Sigma +  c_0)\, h\; ,\quad \tau=c_0 + \text{i} s \, .

            where the ISD-matrix :math:`M(z^i,\overline{z}^i)` is computed via :func:`ISD_matrix`.
            This function can return the values for the RR-fluxes :math:`f=(f_1,f_2)` for given choices of
            the NSNS-fluxes :math:`h=(h_1,h_2)` and values for the moduli :math:`z^i` and the axio-dilaton :math:`\tau`.
            
            Alternatively, we can also obtain the NSNS-fluxes :math:`h=(h_1,h_2)` for given choices of RR-fluxes :math:`f=(f_1,f_2)`
            and values for the moduli :math:`z^i` and the axio-dilaton :math:`\tau` by inverting the above equation and using that
            the ISD-matrix :math:`M(z^i,\overline{z}^i)` is symplectic. This leads to the condition
            
            .. math::
                h=\dfrac{1}{|\tau|^2} (-s  M(z^i,\overline{z}^i)\Sigma + c_0)\, f\; .
            
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): String specifying the type of ISD sampling to be used. Defaults to :math:`\mathrm{ISD}_+`.
            output (str, optional): String specifying the output of fluxes. If set to ``"full"``, the full flux vector is returned. \
                                    If set to ``"half"``, only that half of the fluxes is returned that is fixed through the ISD condition.

        Returns:
            Array: JAX array of shape (:math:`4(h^{1,2}+1)`, ) or (:math:`2(h^{1,2}+1)`, ) containing the sampled fluxes.
            
        See also: :func:`ISD_condition`

        """
        
        modes = ["F","H"]
        if mode not in modes:
            raise ValueError(f"Unknown value for `mode`. Should be one of {modes}, but is {mode}!")
            
        outputs = ["full","half"]
        if output not in outputs:
            raise ValueError(f"Unknown value for `output`. Should be one of {outputs}, but is {output}!")
            
        c0=(tau+tau_c)/2
        s=(tau-tau_c)/2/1j

        SigmaFlux = jnp.matmul(self.model.periods.sigma(),fluxes)

        M0 = self.model.ISD_matrix(moduli,moduli_c)

        M0_SigmaFlux = jnp.matmul(M0,SigmaFlux)
        
        if mode=="H":

            f = M0_SigmaFlux*s+fluxes*c0

            if output=="full":
                return jnp.append(f,fluxes)
            else:
                return f
        else:
            
            tau_abs = c0**2+s**2
            
            h = (fluxes*c0-M0_SigmaFlux*s)/tau_abs

            if output=="full":
                return jnp.append(fluxes,h)
            else:
                return h


    @partial(jit,static_argnums=(0,6,7,))
    def _ISD_sampling_PM(self,moduli,moduli_c,tau,tau_c,fluxes,mode="ISD+",output="full"):
        r"""

        **Description:**
        Returns fluxes :math:`(f_1,h_1)` or :math:`(f_2,h_2)` corresponding to the components of the 
        RR-flux vector :math:`f=(f_1,f_2)` and the NSNS-flux vector :math:`h=(h_1,h_2)`
        for given input values for the moduli :math:`z^i` and the axio-dilaton :math:`\tau` via ISD sampling.

        .. admonition:: Details
            :class: dropdown

            If the fluxes are ISD, then the RR-fluxes :math:`(f_1,f_2)` and NSNS-fluxes :math:`(h_1,h_2)` are related 
            to each other through (see :func:`ISD_condition`)

            .. math::
                f_1-\tau h_1=\overline{\mathcal{N}}(z^i,\overline{z}^i)\, (f_2-\tau h_2)\, .

            We can solve for the fluxes :math:`(f_1,h_1)` via

            .. math::
                h_1=-\dfrac{\text{Im}(\Lambda)}{s}\; , \quad f_1=\text{Re}(\Lambda)-\dfrac{c_{0}}{s}\, \text{Im}(\Lambda)

            where

            .. math::
                \Lambda=\overline{\mathcal{N}}\, (f_2-\tau h_2)\; , \quad \tau=c_0+\text{i}\, s \, .

            Generically, the returned values for :math:`(f_1,h_1)` are non-integer.


            We can solve for the fluxes :math:`(f_2,h_2)` via

            .. math::
                h_2=-\dfrac{\text{Im}(\Lambda)}{s}\; , \quad f_2=\text{Re}(\Lambda)-\dfrac{c_{0}}{s}\, \text{Im}(\Lambda)

            where

            .. math::
                \Lambda=\overline{\mathcal{N}}^{-1}\, (f_1-\tau h_1)\; , \quad \tau=c_0+\text{i}\, s \, .

            Generically, the returned values for :math:`(f_2,h_2) are non-integer.

            .. warning::
                In the latter case, we need to invert the matrix :math:`\mathcal{N}` computed in :func:`gauge_kinetic_matrix` 
                which may not necessarily be invertible outside the Kähler cone.

        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): String specifying the type of ISD sampling to be used. Defaults to :math:`\mathrm{ISD}_+`.
            output (str, optional): String specifying the output of fluxes. If set to ``"full"``, the full flux vector is returned. \
                                    If set to ``"half"``, only that half of the fluxes is returned that is fixed through the ISD condition.

        Returns:
            Array: JAX array of shape (:math:`4(h^{1,2}+1)`, ) or (:math:`2(h^{1,2}+1)`, ) containing the sampled fluxes.
            
        See also: :func:`ISD_condition`

        """
        
        modes = ["ISD+","ISD-"]
        if mode not in modes:
            raise ValueError(f"Unknown value for `mode`. Should be one of {modes}, but is {mode}!")
            
        outputs = ["full","half"]
        if output not in outputs:
            raise ValueError(f"Unknown value for `output`. Should be one of {outputs}, but is {output}!")

        NMatVal=self.model.gauge_kinetic_matrix(moduli,jnp.conj(moduli))
        cNMatVal=self.model.gauge_kinetic_matrix(moduli,jnp.conj(moduli),conj=True)
        
        mu = fluxes[:self.model.dimension_H3]
        nu = fluxes[self.model.dimension_H3:]
        
        RHS = mu-tau*nu
        cRHS = mu-tau_c*nu

        if mode == "ISD+":
            Lambda=jnp.matmul(cNMatVal,RHS)
            cLambda=jnp.matmul(NMatVal,cRHS)
        else:
            Lambda=jnp.matmul(jnp.linalg.inv(cNMatVal),RHS)
            cLambda=jnp.matmul(jnp.linalg.inv(NMatVal),cRHS)

        Re_Lambda=(Lambda+cLambda)/2
        Im_Lambda=(Lambda-cLambda)/2/1j
        c0=(tau+tau_c)/2
        s=(tau-tau_c)/2/1j

        ml,nl = Re_Lambda-c0/s*Im_Lambda, -1./s*Im_Lambda

        if output=="full":
            if mode == "ISD+":
                return jnp.append(jnp.append(ml,mu),jnp.append(nl,nu))
            else:
                return jnp.append(jnp.append(mu,ml),jnp.append(nu,nl))
        else:
            return ml,nl

    @partial(jit,static_argnums=(0,6,7,8,9,10,))
    def ISD_sampling(self,
                     moduli,
                     moduli_c,
                     tau,
                     tau_c,
                     flux0,
                     mode="ISD+",
                     output="full",
                     return_integer_flux=False,
                     in_axes=(0,0,0),
                     vmap=False
                     ) -> Array:
        r"""
        **Description:**
        Determines half of the flux numbers such that the ISD condition :math:`\star G_3 = \text{i}G_3` is satisfied 
        for given input values for the moduli :math:`z^i` and the axio-dilaton :math:`\tau`.

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
            Upon rounding the fluxes to integers, the ISD condition is approximately satisfied: :math:`\star G_3 \approx \text{i}G_3` for given inputs. 
            
            

        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): String specifying the type of ISD sampling to be used. Defaults to :math:`\mathrm{ISD}_+`.
            output (str, optional): String specifying the output of fluxes. If set to ``"full"``, the full flux vector is returned. \
                                    If set to ``"half"``, only that half of the fluxes is returned that is fixed through the ISD condition.
            return_integer_flux (boolean, optional): Whether to return integer or continuous fluxes.
            

        Returns:
            Array: JAX array of shape (:math:`4(h^{1,2}+1)`, ) or (:math:`2(h^{1,2}+1)`, ) containing the sampled fluxes.

        
        See also: :func:`ISD_condition`
        """

        if vmap:
            return vmapping_func(self.ISD_sampling,in_axes=(in_axes[0],in_axes[0],in_axes[1],in_axes[1],in_axes[2]),mode=mode,output=output,return_integer_flux=return_integer_flux,vmap=False)(moduli,moduli_c,tau,tau_c,flux0)
        
        modes = ["ISD+","ISD-","F","H"]
        if mode not in modes:
            raise ValueError(f"Unknown value for `mode`. Should be one of {modes}, but is {mode}!")
            
        outputs = ["full","half"]
        if output not in outputs:
            raise ValueError(f"Unknown value for `output`. Should be one of {outputs}, but is {output}!")

        if mode == "ISD+" or mode == "ISD-":

            flux = self._ISD_sampling_PM(moduli,moduli_c,tau,tau_c,flux0,mode=mode)
            
        elif mode == "F" or mode == "H":

            flux = self._ISD_sampling_FH(moduli,moduli_c,tau,tau_c,flux0,mode=mode)
            
        # Round the fluxes to integer values
        if return_integer_flux:
            flux = jnp.around(flux,0).astype(jnp.int64)

        return flux

    

    def initial_guesses_ISD(
                            self,
                            N: int,
                            rns_key: Optional[Any] = None,
                            minval_dilaton: float = None,
                            maxval_dilaton: float = None,
                            minval_axions: float = None,
                            maxval_axions: float = None,
                            minval_moduli: float = None,
                            maxval_moduli: float = None,
                            moduli_sampling_mode: str = "box",
                            minval_fluxes: float = None,
                            maxval_fluxes: float = None,
                            fluxes_sampling_mode: str = "box",
                            flux_mode: str = "full",
                            Nmax: float = None,
                            filter_moduli: bool = False,
                            mode="ISD+",
                            vmap_dim: int = None,
                            print_progress: bool = False
                            ) -> Tuple[Any, Any, Optional[Any]]:
        r"""
        **Description:**
        Generates initial guesses for moduli, dilaton, and fluxes.

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
            in the regions of highest density of rays. To account for this, we assign a weight to each 
            ray given by the inverse of the number of non-zero entries.

        Args:
            N (int): 
                The number of initial points to generate.
            rns_key (Optional[Any]): 
                Random number seed key used for reproducibility. Default is None.
            minval_dilaton (float): 
                Minimum value for dilaton sampling. Default is None.
            maxval_dilaton (float): 
                Maximum value for dilaton sampling. Default is None.
            minval_axions (float): 
                Minimum value for axion sampling. Default is None.
            maxval_axions (float): 
                Maximum value for axion sampling. Default is None.
            minval_moduli (float): 
                Minimum value for moduli sampling. Default is None.
            maxval_moduli (float): 
                Maximum value for moduli sampling. Default is None.
            moduli_sampling_mode (str): 
                Sampling mode for moduli. Default is "box".
            minval_fluxes (float): 
                Minimum value for flux sampling. Default is None.
            maxval_fluxes (float): 
                Maximum value for flux sampling. Default is None.
            fluxes_sampling_mode (str): 
                Sampling mode for fluxes. Default is "box".
            flux_mode (str): 
                Mode for flux generation. Default is "full".
            flux_radius (float): 
                Radius within which flux values are sampled. Default is 10.
            filter_moduli (bool): 
                Whether to filter moduli based on instanton effects. Default is True.
            include_fluxes (bool): 
                Whether to include fluxes in the output. Default is True.

        Returns:
            Tuple[Any, Any, Optional[Any]]:
                A tuple containing:
                    - moduli (Any): Array of sampled moduli values.
                    - tau (Any): Array of sampled dilaton values.
                    - fluxes (Optional[Any]): Array of sampled fluxes, if `include_fluxes` is True.

        """

        if vmap_dim is None:
            vmap_dim = N

        moduli = []
        tau = []
        fluxes = []
        while len(moduli)<N:

            if print_progress:
                print(f"#samples: {len(moduli)}            ",flush=True,end="\r")

            z0,tau0 = self.initial_guesses(vmap_dim,
                                          rns_key=rns_key,
                                          minval_dilaton=minval_dilaton,
                                          maxval_dilaton=maxval_dilaton,
                                          minval_axions=minval_axions,
                                          maxval_axions=maxval_axions,
                                          minval_moduli=minval_moduli,
                                          maxval_moduli=maxval_moduli,
                                          moduli_sampling_mode=moduli_sampling_mode,
                                          filter_moduli=filter_moduli,
                                          include_fluxes=False)
            
            fluxes0 = self.get_fluxes(vmap_dim,
                                     rns_key=rns_key,
                                     mode=flux_mode,
                                     minval=minval_fluxes,
                                     maxval=maxval_fluxes,
                                     sampling_mode=fluxes_sampling_mode,
                                     radius=Nmax)[:,:self._n_fluxes]

            fluxes_ISD = self.ISD_sampling(z0,jnp.conj(z0),tau0,jnp.conj(tau0),fluxes0,mode=mode,output="full",return_integer_flux=True,in_axes=(0,0,0),vmap=True)

            fluxes_ISD_integer = jnp.around(fluxes_ISD.real,0).astype(jnp.int64)

            Nflux = self._tadpole(fluxes_ISD_integer)
            
            if Nmax is None:
                Nmax = self._D3_tadpole

            flag = (Nflux<=Nmax)&(Nflux>=0)
            fluxes0 = fluxes_ISD_integer[flag]
            z0 = z0[flag]
            tau0 = tau0[flag]
            
            
            if len(moduli)==0:
                moduli = z0
                tau = tau0
                fluxes = fluxes0
            else:
                moduli = np.append(moduli,z0,axis=0)
                tau = np.append(tau,tau0,axis=0)
                fluxes = np.append(fluxes,fluxes0,axis=0)

        return moduli,tau,fluxes
        

    


