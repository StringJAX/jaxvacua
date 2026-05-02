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
# This file holds the class to construct flux vacua.
# ------------------------------------------------------------------------------


# Important standard libraries
import os, sys, warnings, time
import numpy as np
from typing import Tuple, Any, Callable
from functools import partial


# Important JAX libraries
import jax
from jax import jit, vmap, config
import jax.numpy as jnp
from jax import Array
from jax import pure_callback
from jax.tree_util import register_pytree_node

# Enable 64 bit precision
# JAXVacua custom imports
from .util import *
from .flux_eft import FluxEFT
from .sampling import data_sampler


class FluxVacuaFinder(FluxEFT):
    r"""
    **Description:**
    Class representing the effective field theory of the flux sector in 4D Type IIB string theory compactified on Calabi-Yau threefolds with 3-form flux backgrounds. 
    This class provides tools for computing and analyzing flux vacua, including superpotentials, Kähler potentials, F-term conditions, and various properties of the moduli space.
    
    """
    
    def __init__(
        self,
        h12: int | None = None,
        model_ID: int | None = None,
        model_type: str = "KS",
        limit: str = "LCS",
        maximum_degree: int = 0,
        mirror_cy: Any | None = None,
        model_data: dict | None = None,
        lcs_tree_input: object | None = None,
        model_file: str = "",
        use_cytools: bool = False,
        basis_change: Array | None = None,
        ncf: int | None = None,
        conifold_curve: Any | None = None,
        conifold_basis: bool | None = True,
        grading_vector: Array | None = None,
        period_input: Callable | None = None,
        Q: int | None = None,
        prepotential_input: Callable | None = None,
        gauge_choice: complex = 1.+0.*1j,
        prange: int = 5,
        use_gvs: bool = True,
        save_file: bool = False,
        flux_bounds: tuple = (-10, 10),
        axion_bounds: tuple = (-0.5, 0.5),
        dilaton_bounds: tuple = (2., 10.),
        moduli_bounds: tuple = (1., 5.),
        seed: int = 42,
        map_to_fd: bool = False,
        **kwargs
    ) -> None:
        r"""

        **Description:**
        Initializes a flux EFT class representing the effective field theory of the flux sector 
        in 4D Type IIB string theory compactified on Calabi-Yau threefolds with 3-form flux backgrounds.
        
        This class provides tools for computing and analyzing flux vacua, including superpotentials,
        Kähler potentials, F-term conditions, and various properties of the moduli space.
        
        Args:
            h12 (int, optional): Number of complex structure moduli :math:`h^{1,2}` for the compactified geometry.
            model_type (str, optional): Type of Calabi-Yau manifold. Currently supports ``"KS"`` (Kreuzer-Skarke) 
            and ``"CICY"`` (Complete Intersection Calabi-Yau). Defaults to ``"KS"``.
            model_ID (int, optional): Identifier for a specific model within the database.
            limit (str, optional): Type of moduli space limit for the periods. Currently supports ``"LCS"`` 
            (Large Complex Structure limit). Defaults to ``"LCS"``.
            model_data (dict, optional): Dictionary containing topological data of the Calabi-Yau, such as 
            triple intersection numbers, second Chern class, etc.
            instanton_data (list, optional): List of Gopakumar-Vafa (GV) and Gromov-Witten (GW) invariants 
            for instanton corrections.
            maximum_degree (int, optional): Maximum degree cutoff for instanton sum expansions. Defaults to ``0``.
            use_cytools (bool, optional): Whether to use CYTools library to compute topological data of the 
            Calabi-Yau automatically. Defaults to ``False``.
            mirror_cy (cytools.CalabiYau, optional): Mirror Calabi-Yau threefold object (from CYTools).
            basis_change (Array, optional): Basis transformation matrix to be applied to the topological 
            data of the Calabi-Yau.
            ncf (int, optional): Conifold number for conifold-enhanced models.
            conifold_curve (Array, optional): Specification of the conifold curve in the moduli space.
            grading_vector (Array, optional): Grading vector used for computing Gopakumar-Vafa invariants.
            period_input (Callable, optional): Custom function for computing periods.
            Q (int, optional): D3-brane tadpole bound constraint.
            prepotential_input (Callable, optional): Custom function for computing the prepotential.
            gauge_choice (complex, optional): Gauge choice parameter for the periods. Defaults to ``1.0+0.0j``.
            prange (int, optional): Period range parameter for convergence. Defaults to ``500``.
            use_gvs (bool, optional): Whether to use Gopakumar-Vafa invariants for instanton corrections. 
            Defaults to ``False``.
            save_file (bool, optional): Whether to save model data to disk for future use. Defaults to ``False``.
            flux_bounds (tuple, optional): ``(min, max)`` integer range for flux sampling. Defaults to ``(-10, 10)``.
            axion_bounds (tuple, optional): ``(min, max)`` range for axion sampling. Defaults to ``(-0.5, 0.5)``.
            dilaton_bounds (tuple, optional): ``(min, max)`` range for the axio-dilaton :math:`s = \mathrm{Im}(\tau)`. Defaults to ``(2., 10.)``.
            moduli_bounds (tuple, optional): ``(min, max)`` range for the imaginary parts of the complex structure moduli. Defaults to ``(1., 5.)``.
            seed (int, optional): Random seed for the sampler. Defaults to ``42``.
            map_to_fd (bool, optional): If True, all returned vacua are mapped to the
                fundamental domain via :func:`map_to_FD`. Defaults to ``False``.
            **kwargs: Additional keyword arguments passed to parent class :class:`FluxEFT`.

        """

        
        super().__init__(
            h12=h12,
            Q=Q,
            model_ID=model_ID,
            model_type=model_type,
            limit=limit,
            maximum_degree=maximum_degree,
            mirror_cy=mirror_cy,
            model_data=model_data,
            lcs_tree_input=lcs_tree_input,
            model_file=model_file,
            use_cytools=use_cytools,
            basis_change=basis_change,
            ncf=ncf,
            conifold_curve=conifold_curve,
            conifold_basis=conifold_basis,
            grading_vector=grading_vector,
            period_input=period_input,
            prepotential_input=prepotential_input,
            gauge_choice=gauge_choice,
            prange=prange,
            use_gvs=use_gvs,
            save_file=save_file,
            **kwargs
        )

        self._map_to_fd = map_to_fd
        self._sampler = None
        self._sampler_kwargs = dict(
            flux_bounds=flux_bounds,
            axion_bounds=axion_bounds,
            dilaton_bounds=dilaton_bounds,
            moduli_bounds=moduli_bounds,
            seed=seed,
        )

    @property
    def sampler(self) -> "data_sampler":
        r"""
        Description:
        Lazily-initialised :class:`data_sampler` for this model.

        The sampler is constructed on first access and cached for subsequent calls.
        Sampling bounds are configured via the constructor arguments ``flux_bounds``,
        ``axion_bounds``, ``dilaton_bounds``, ``moduli_bounds``, and ``seed``.
        To use a custom sampler instead, pass it explicitly to methods that accept
        a ``sampler`` keyword argument.
        """
        if self._sampler is None:
            self._sampler = data_sampler(self, **self._sampler_kwargs)
        return self._sampler

    def __repr__(self) -> str:
        r"""
        **Description:**
        Returns a string representation of the flux sector class.
        
        Returns:
            str: Description of the class.
        """
        
        return f"Flux EFT for h12={self.h12} complex structure moduli in the {self.periods.limit} limit."
        
    @partial(jit, static_argnums = (4,5,6,7,8,))
    def _newton_method_flux_vacua_complex(
        self, moduli: Array, tau: complex, fluxes: Array, mode: str = None, step_size_Newton: float = 1e-1,
        tol: float = 1e-10,max_iters: int = 100,print_progress: bool = False
        ) -> Tuple[int,Array,Array,float]:
        r"""

        **Description:**
        Solves the minimum conditions for flux vacua in terms of complex fields using Newton's method.

        Args:
            moduli (Array): Complex structure moduli values.
            tau (complex): Value of axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): Whether to solve F-term conditions :math:`D_IW=0` for SUSY minima or instead :math:`\partial_IV=0` for general minima.
                Defaults to ``None``.
            step_size_Newton (float, optional): Step size to be used in Newton's method. Defaults to ``0.1``.
            tol (float, optional): Tolerance for the residual. Defaults to ``1e-10``.
            max_iters (int, optional): Maximum number of iterations. Defaults to ``100``.
            print_progress (bool, optional): Whether to print the progress of the minimisation. Defaults to ``False``.
            
        Returns:
            int: Number of iterations.
            Array: Complex structure moduli values.
            Array: Value of axio-dilaton.
            float: Residual :math:`\sum_A |D_A W|` of the :math:`F`-terms.

        """

        modes=[None, "SUSY"]
        if mode not in modes:
            raise ValueError(f"Cannot determine mode for Newton's method!\
                    `mode` must be one of {modes}, but is {mode}!")

        def cond(arg):
            """Loop condition: continue while step < max_iters and residual > tol."""
            step, moduli, tau, res = arg
            return (step <max_iters) & (res > tol)

        def body(arg):
            """Single Newton step in real coordinates."""
            step, moduli, tau, res = arg

            if mode=="SUSY":
                m2 = self.dDW_real(moduli,tau,fluxes)
                DW = self.DW(moduli,jnp.conj(moduli),tau,jnp.conj(tau),fluxes)
                m1 = jnp.concatenate([jnp.real(DW),jnp.imag(DW)])
            else:
                m2 = self.ddV(moduli,jnp.conj(moduli),tau,jnp.conj(tau),fluxes,mode="real")
                m1 = self.dV(moduli,jnp.conj(moduli),tau,jnp.conj(tau),fluxes,mode="real")
                

            deltaPhi = -jnp.linalg.inv(m2)@m1

            moduli = moduli + step_size_Newton*(deltaPhi[self.h12+1:2*self.h12+1]+1j*deltaPhi[0:self.h12])
            tau = tau + step_size_Newton*(deltaPhi[2*self.h12+1]+1j*deltaPhi[self.h12])

            res = jnp.sum(jnp.abs(m1))

            if print_progress:
                out_type = jax.ShapeDtypeStruct(jnp.shape(step), jnp.result_type(step))
                step = jax.pure_callback(progress_bar_jax, out_type,(step, max_iters, res),step)
                
            return (step + 1,  moduli, tau, res)

        return jax.lax.while_loop(cond,body,(0,moduli,tau,10.))


    @partial(jit, static_argnums = (3,4,5,6,7,))
    def _newton_method_flux_vacua_real(
        self, x: Array, fluxes: Array, mode: str = None, step_size_Newton: float = 1e-1,
        tol: float = 1e-10, max_iters: int = 100, print_progress: bool = False
        ) -> Tuple[int,Array,float]:
        r"""
        **Description:**
        Solves the minimum conditions for flux vacua in terms of real fields using Newton's method.

        Args:
            x (Array): Array of shape (:math:`2(h^{1,2}+1)`, ) containing real and imaginary parts of
                the complex structure moduli :math:`z^i` and axio-dilaton :math:`\tau`.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): Whether to solve F-term conditions :math:`D_IW=0` for SUSY minima or instead :math:`\partial_I V=0` for general minima.
                Defaults to ``None``.
            step_size_Newton (float, optional): Step size to be used in Newton's method. Defaults to ``0.1``.
            tol (float, optional): Tolerance for the residual. Defaults to ``1e-10``.
            max_iters (int, optional): Maximum number of iterations. Defaults to ``100``.
            print_progress (bool, optional): Whether to print the progress of the minimisation. Defaults to ``False``.

        Returns:
            int: Number of iterations.
            Array: Values of real and imaginary parts of the complex structure moduli :math:`z^i`
                and axio-dilaton :math:`\tau`.
            float: Residual :math:`\sum_A |\partial_A V|`.

        """

        modes=[None, "SUSY"]
        if mode not in modes:
            raise ValueError(f"Cannot determine mode for Newton's method!\
                    `mode` must be one of {modes}, but is {mode}!")

        def cond(arg):
            """Loop condition: continue while step < max_iters and residual > tol."""
            step, x, res = arg
            return (step <max_iters) & (res > tol)

        def body(arg):
            """Single Newton step in complex coordinates."""
            step, x, res = arg

            if mode=="SUSY":
                m2 = self.dDW_x(x,fluxes)
                m1 = self.DW_x(x,fluxes)
            else:
                m2 = self.ddV_x(x,fluxes)
                m1 = self.dV_x(x,fluxes)

            deltaPhi = -jnp.linalg.inv(m2)@m1

            x = x + step_size_Newton*deltaPhi

            res = jnp.sum(jnp.abs(m1))

            if print_progress:
                out_type = jax.ShapeDtypeStruct(jnp.shape(step), jnp.result_type(step))
                step = jax.pure_callback(progress_bar_jax, out_type,(step, max_iters, res),step)

            return (step + 1,  x, res)

        return jax.lax.while_loop(cond,body,(0,x,10.))

    @partial(jit, static_argnums = (4,5,6,7,8,9,))
    def newton_method_flux_vacua(
        self, moduli: Array, tau: complex, fluxes: Array, mode: str = None, step_size_Newton: float = 1e-1,
        tol: float = 1e-10,max_iters: int = 100,print_progress: bool = False, solver_mode: str = "complex"
        ) -> Tuple[int,Array,Array,float]:
        r"""
        **Description:**
        Solves the minimum conditions for flux vacua using Newton's method.

        Args:
            moduli (Array): Complex structure moduli values.
            tau (complex): Value of axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): Whether to solve F-term conditions :math:`D_IW=0` for SUSY minima or instead :math:`\partial_IV=0` for general minima.
                Defaults to ``None``.
            step_size_Newton (float, optional): Step size to be used in Newton's method. Defaults to ``0.1``.
            tol (float, optional): Tolerance for the residual. Defaults to ``1e-10``.
            max_iters (int, optional): Maximum number of iterations. Defaults to ``100``.
            print_progress (bool, optional): Whether to print the progress of the minimisation. Defaults to ``False``.
            solver_mode (str, optional): Solver mode to use for the computation. Available options are ``"real"``
                or ``"complex"`` to solve equations in terms of real or complex fields. Defaults to ``"complex"``.

        Returns:
            Array: Complex structure moduli values.
            Array: Value of axio-dilaton.
            float: Residual :math:`\sum_A |D_A W|` of the :math:`F`-terms.
        """
        
        solver_modes=["real", "complex"]
        if solver_mode not in solver_modes:
            raise ValueError(f"Cannot determine solver mode for Newton's method!\
                    `solver_mode` must be one of {solver_modes}, but is {solver_mode}!")
            
        if solver_mode=="complex":
            
            _,moduli,tau,res = self._newton_method_flux_vacua_complex(moduli,tau,fluxes,mode=mode, step_size_Newton=step_size_Newton,tol=tol,max_iters=max_iters,print_progress=print_progress)
        
        elif solver_mode=="real":
            
            x = self._convert_complex_to_real(moduli,jnp.conj(moduli),tau,jnp.conj(tau))
            
            _, x, res = self._newton_method_flux_vacua_real(x,fluxes,mode=mode, step_size_Newton=step_size_Newton,tol=tol,max_iters=max_iters,print_progress=print_progress)
            
            moduli,_,tau,_ = self._convert_real_to_complex(x)
            
        return moduli,tau,res

    @partial(jit, static_argnums = (4,5,6,))
    def linearised_shifts_ISD(
        self,
        moduli: Array,
        tau: complex,
        fluxes: Array,
        mode: str = "ISD",
        return_shifts: bool = False,
        remove_NANs: bool = False
    ) -> Tuple[Array, complex, Array] | Tuple[Array, complex, Array, Array]:
        """
        **Description:**
        Computes the linearised shifts for the complex structure moduli and the axio-dilaton based on :math:`ISD_+`-sampling for fluxes.

        Args:
            moduli (Array): Values for the complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): Computation mode for obtaining the linearised shifts.
            return_shifts (bool, optional): Whether to return the shifts for the moduli or the shifted moduli. Defaults to ``False``.
            remove_NANs (bool, optional): Whether to remove NaN values from input. Defaults to ``False``.

        Returns:
            Array: Moduli shifts or shifted values of the complex structure moduli :math:`Z^i`.
            complex: Axio-dilaton shift or shifted value of the axio-dilaton :math:`\tau`.
            Array: Values of the fluxes.
        """

        modes = ["ISD","random"]

        if mode not in modes:
            raise ValueError(f"Mode must be one of {modes}, but `mode = {mode}` was given!")
            
        if remove_NANs:
            moduli = jnp.nan_to_num(moduli,nan=1.)
            tau = jnp.nan_to_num(tau,nan=1.)
            fluxes = jnp.nan_to_num(fluxes,nan=1.)

        NMatVal=self.gauge_kinetic_matrix(moduli,jnp.conj(moduli))
        IMat = jnp.imag(NMatVal)
        RMat = jnp.real(NMatVal)
        FF = fluxes[:self.n_fluxes]
        HF = fluxes[self.n_fluxes:]

        mu = FF[self.dimension_H3:]
        nu = HF[self.dimension_H3:]
        RHS = mu-tau*nu

        Lambda=jnp.matmul(jnp.conj(NMatVal),RHS)

        Re_Lambda=jnp.real(Lambda)
        Im_Lambda=jnp.imag(Lambda)
        c0=jnp.real(tau)
        s=jnp.imag(tau)

        ml_cont,nl_cont = Re_Lambda-c0/s*Im_Lambda, -1./s*Im_Lambda

        if mode=="ISD":
            ml = jnp.around(ml_cont,0)
            nl = jnp.around(nl_cont,0)
        else:
            ml = FF[:self.dimension_H3]
            nl = HF[:self.dimension_H3]

        delta_ml = ml-ml_cont
        delta_nl = nl-nl_cont

        delta_LHS = delta_ml-tau*delta_nl
        
        Lambdak = jnp.matmul(self.dN(moduli,jnp.conj(moduli),conj=True).transpose(2,0,1),RHS)
        tilde_Lambdak = jnp.matmul(self.dN_c(moduli,jnp.conj(moduli),conj=True).transpose(2,0,1),RHS)
        
        re_Lambdak = jnp.real(Lambdak)
        im_Lambdak = jnp.imag(Lambdak)
        re_tilde_Lambdak = jnp.real(tilde_Lambdak)
        im_tilde_Lambdak = jnp.imag(tilde_Lambdak)

        re_ak = re_Lambdak+re_tilde_Lambdak
        re_vk = -im_Lambdak+im_tilde_Lambdak
        im_ak = im_Lambdak+im_tilde_Lambdak
        im_vk = re_Lambdak-re_tilde_Lambdak
        re_c = nl-RMat@nu
        re_s = -IMat@nu
        im_c = IMat@nu
        im_s = nl-RMat@nu
        re_delta_LHS = jnp.real(delta_LHS)
        im_delta_LHS = jnp.imag(delta_LHS)
        b = jnp.append(re_delta_LHS,im_delta_LHS)


        A1 = jnp.append(jnp.append(jnp.append(re_ak,re_vk,axis=0),jnp.array([re_c]),axis=0),jnp.array([re_s]),axis=0)
        A2 = jnp.append(jnp.append(jnp.append(im_ak,im_vk,axis=0),jnp.array([im_c]),axis=0),jnp.array([im_s]),axis=0)
        A = jnp.append(A1.T,A2.T,axis=0)
        # warning: not guaranteed to be inveritble!!!
        # TODO: safer option to invert?

        # Solve for the shifts
        #shift = jnp.matmul(jnp.linalg.inv(A),b)
        shift = jnp.linalg.solve(A,b)
        
        flux_new = jnp.append(jnp.append(ml,mu),jnp.append(nl,nu))
        
        if return_shifts:
            return shift[:self.h12]+1j*shift[self.h12:2*self.h12],shift[-2]+1j*shift[-1],flux_new
        else:
            moduli_new = moduli+shift[:self.h12]+1j*shift[self.h12:2*self.h12]
            
            tau_new = tau+shift[-2]+1j*shift[-1]
            
            return moduli_new,tau_new,flux_new

    @partial(jit, static_argnums = (4,5,6,))
    def linearised_shifts_H(
        self,
        moduli: Array,
        tau: complex,
        fluxes: Array,
        mode: str = "Hflux",
        return_shifts: bool = False,
        remove_NANs: bool = False
    ) -> Tuple[Array, complex, Array] | Tuple[Array, complex, Array, Array]:
        """
        **Description:**
        Computes linearised shifts for the complex structure moduli and axio-dilaton based on given H-flux.

        Args:
            moduli (Array): Starting point for the complex structure moduli.
            tau (complex): Starting point for the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): Computation mode for obtaining the linearised shifts.
            return_shifts (bool, optional): Whether to return the shifts for the moduli or the shifted moduli. Defaults to ``False``.
            remove_NANs (bool, optional): Whether to remove NaN values from input. Defaults to ``False``.

        Returns:
            Array: Moduli shifts or shifted values of the complex structure moduli :math:`Z^i`.
            complex: Axio-dilaton shift or shifted value of the axio-dilaton :math:`\tau`.
            Array: Values of the fluxes.
        """

        modes = ["Hflux","Hflux_random"]

        if mode not in modes:
            raise ValueError(f"Mode must be one of {modes}, but `mode = {mode}` was given!")

        if remove_NANs:
            moduli = jnp.nan_to_num(moduli,nan=1.)
            tau = jnp.nan_to_num(tau,nan=1.)
            fluxes = jnp.nan_to_num(fluxes,nan=1.)

        FF = fluxes[:self.n_fluxes]
        HF = fluxes[self.n_fluxes:]

        c0=jnp.real(tau)
        s=jnp.imag(tau)

        SigmaH = jnp.matmul(self.periods.sigma,HF)
        RHS = s*SigmaH

        M0 = self.ISD_matrix(moduli,jnp.conj(moduli))

        M0_SigmaH = jnp.matmul(M0,SigmaH)

        f_cont = jnp.real(M0_SigmaH*s+HF*c0)

        # jnp.matmul((jnp.matmul(MMatVal,self.periods.sigma)*jnp.imag(tau)+jnp.identity(len(MMatVal))*jnp.real(tau)),HF)

        if mode=="Hflux":
            f = jnp.around(f_cont,0)
        else:
            f = FF     

        delta_f = f-f_cont

        delta_LHS = delta_f

        Theta_k = jnp.matmul(self.dM(moduli,jnp.conj(moduli)).transpose(2,0,1),RHS)
        Theta_bk = jnp.matmul(self.dM_c(moduli,jnp.conj(moduli)).transpose(2,0,1),RHS)


        re_Theta_k = jnp.real(Theta_k)
        im_Theta_k = jnp.imag(Theta_k)
        re_Theta_bk = jnp.real(Theta_bk)
        im_Theta_bk = jnp.imag(Theta_bk)

        re_ak = re_Theta_k+re_Theta_bk
        re_vk = -im_Theta_k+im_Theta_bk
        #im_ak = im_Theta_k+im_Theta_bk
        #im_vk = re_Theta_k-re_Theta_bk

        re_c = HF
        re_s = M0_SigmaH
        #im_c = jnp.zeros(re_ak.shape[1])
        #im_s = jnp.zeros(re_ak.shape[1])

        re_delta_LHS = jnp.real(delta_LHS)
        #im_delta_LHS = jnp.imag(delta_LHS) # THIS IS ZERO HERE!!!

        #b = jnp.append(re_delta_LHS,im_delta_LHS)

        b = re_delta_LHS

        A1 = jnp.append(jnp.append(jnp.append(re_ak,re_vk,axis=0),jnp.array([re_c]),axis=0),jnp.array([re_s]),axis=0)

        # SHOULD BE ZERO!!!
        #A2 = jnp.append(jnp.append(jnp.append(im_ak,im_vk,axis=0),jnp.array([im_c]),axis=0),jnp.array([im_s]),axis=0)
        
        A = A1.T

        # warning: not guaranteed to be inveritble!!!
        # TODO: Other version for inversion?
        #shift = jnp.matmul(jnp.linalg.inv(A),b)
        shift = jnp.linalg.solve(A,b)
        #print("TODO: CHANGED INVERSE!")
        #shift = jax.scipy.linalg.solve(A,b)
        #cfac = jax.scipy.linalg.lu_factor(A.astype(jnp.float64))
        #shift = jax.scipy.linalg.lu_solve(cfac, b)
        
        
        flux_new = jnp.append(f,HF)

        if return_shifts:
            return shift[:self.h12]+1j*shift[self.h12:2*self.h12],shift[-2]+1j*shift[-1],flux_new
        else:
            moduli_new = moduli+shift[:self.h12]+1j*shift[self.h12:2*self.h12]
            tau_new = tau+shift[-2]+1j*shift[-1]
            return moduli_new,tau_new,flux_new
        
    @partial(jit, static_argnums = (4,5,6,))
    @partial(jit, static_argnums = ( 4, 5, 6,))
    def linearised_shifts_F(
        self,
        moduli: Array,
        tau: complex,
        fluxes: Array,
        mode: str = "Fflux",
        return_shifts: bool = False,
        remove_NANs: bool = False
    ) -> Tuple[Array, complex, Array] | Tuple[Array, complex, Array, Array]:
        r"""
        **Description:**
        Computes linearised shifts for the complex structure moduli and axio-dilaton based on given F-flux.

        Args:
            moduli (Array): Starting point for the complex structure moduli.
            tau (complex): Starting point for the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): Computation mode for obtaining the linearised shifts. Defaults to ``"Fflux"``.
            return_shifts (bool, optional): Whether to return the shifts for the moduli or the shifted moduli. Defaults to ``False``.
            remove_NANs (bool, optional): Whether to remove NaN values from input. Defaults to ``False``.

        Returns:
            Array: Moduli shifts or shifted values of the complex structure moduli :math:`Z^i`.
            complex: Axio-dilaton shift or shifted value of the axio-dilaton :math:`\tau`.
            Array: Values of the fluxes.
        """
        
        modes: list = ["Fflux", "Fflux_random"]
        if mode not in modes:
            raise ValueError(f"Mode must be one of {modes}, but `mode = {mode}` was given!")

        # Handle NaN values if requested
        if remove_NANs:
            moduli = jnp.nan_to_num(moduli, nan=1.)
            tau = jnp.nan_to_num(tau, nan=1.)
            fluxes = jnp.nan_to_num(fluxes, nan=1.)
            
        FF = fluxes[:self.n_fluxes]
        HF = fluxes[self.n_fluxes:]

        c0=jnp.real(tau)
        s=jnp.imag(tau)
        
        tau_abs = c0**2+s**2

        SigmaF = jnp.matmul(self.periods.sigma,FF)
        RHS = s*SigmaF

        M0 = self.ISD_matrix(moduli,jnp.conj(moduli))

        M0_SigmaF = jnp.matmul(M0,SigmaF)
        
        h_cont = jnp.real(FF*c0-M0_SigmaF*s)/tau_abs

        if mode=="Fflux":
            h = jnp.around(h_cont,0)
        else:
            h = HF     

        delta_h = h-h_cont

        delta_LHS = delta_h

        Theta_k = jnp.matmul(self.dM(moduli,jnp.conj(moduli)).transpose(2,0,1),RHS)
        Theta_bk = jnp.matmul(self.dM_c(moduli,jnp.conj(moduli)).transpose(2,0,1),RHS)


        re_Theta_k = jnp.real(Theta_k)
        im_Theta_k = jnp.imag(Theta_k)
        re_Theta_bk = jnp.real(Theta_bk)
        im_Theta_bk = jnp.imag(Theta_bk)

        re_ak = re_Theta_k+re_Theta_bk
        re_vk = -im_Theta_k+im_Theta_bk
        #im_ak = im_Theta_k+im_Theta_bk
        #im_vk = re_Theta_k-re_Theta_bk

        re_c = FF-2*h_cont*c0
        re_s = -M0_SigmaF-2*h_cont*s
        #im_c = jnp.zeros(re_ak.shape[1])
        #im_s = jnp.zeros(re_ak.shape[1])

        re_delta_LHS = jnp.real(delta_LHS)
        #im_delta_LHS = jnp.imag(delta_LHS) # THIS IS ZERO HERE!!!

        #b = jnp.append(re_delta_LHS,im_delta_LHS)

        b = re_delta_LHS*tau_abs

        A1 = jnp.append(jnp.append(jnp.append(-re_ak,-re_vk,axis=0),jnp.array([re_c]),axis=0),jnp.array([re_s]),axis=0)

        # SHOULD BE ZERO!!!
        #A2 = jnp.append(jnp.append(jnp.append(im_ak,im_vk,axis=0),jnp.array([im_c]),axis=0),jnp.array([im_s]),axis=0)
        
        A = A1.T

        # warning: not guaranteed to be inveritble!!!
        # TODO: Other version for inversion?
        #shift = jnp.matmul(jnp.linalg.inv(A),b)
        shift = jnp.linalg.solve(A,b)
        #print("TODO: CHANGED INVERSE!")
        
        flux_new = jnp.append(FF,h)

        if return_shifts:
            return shift[:self.h12]+1j*shift[self.h12:2*self.h12],shift[-2]+1j*shift[-1],flux_new
        else:
            moduli_new = moduli+shift[:self.h12]+1j*shift[self.h12:2*self.h12]
            tau_new = tau+shift[-2]+1j*shift[-1]
            return moduli_new,tau_new,flux_new
        

    @partial(jit, static_argnums = (4,5,6,7,8,))
    def linearised_shifts(
        self,
        moduli: Array,
        tau: complex,
        fluxes: Array,
        Q: int | None = None,
        mode: str = "ISD",
        return_flag: bool = True,
        constraints: Callable | None = None,
        remove_NANs: bool = False
    ) -> Tuple[Array, complex, Array] | Tuple[Array, complex, Array, Array]:
        r"""
        **Description:**
        Computes the linearised shifts for the complex structure moduli and the axio-dilaton.

        Args:
            moduli (Array): Starting point for the complex structure moduli.
            tau (complex): Starting point for the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            Q (int, optional): Tadpole bound to be used in the computation for the constraints.
            mode (str, optional): Computation mode for obtaining the linearised shifts.
            constraints (None, optional): Additional constraints to be imposed on the solutions.
            remove_NANs (bool, optional): Whether to remove NaN values from input. Defaults to ``False``.

        Returns:
            Array: Shifted values for the complex structure moduli :math:`Z^i`.
            complex: Shifted value for the axio-dilaton :math:`\tau`.
            Array: Updated values for the fluxes based on the ISD-condition.
            bool, optional: Boolean specifying whether the constraints are satisfied.

        """

        modes = ["ISD","random","Hflux","Hflux_random","Fflux","Fflux_random"]

        if mode not in modes:
            raise ValueError(f"Mode must be one of {modes}, but `mode = {mode}` was given!")

        if mode == "ISD" or mode == "random":
            moduli_new,tau_new,flux_new = self.linearised_shifts_ISD(moduli,tau,fluxes,mode=mode,remove_NANs=remove_NANs)
        elif mode == "Hflux" or mode == "Hflux_random":
            moduli_new,tau_new,flux_new = self.linearised_shifts_H(moduli,tau,fluxes,mode=mode,remove_NANs=remove_NANs)
        elif mode == "Fflux" or mode == "Fflux_random":
            moduli_new,tau_new,flux_new = self.linearised_shifts_F(moduli,tau,fluxes,mode=mode,remove_NANs=remove_NANs)


        if return_flag:

            # Checks boundary conditions like Kähler cone conditions or tadpole
            hyperplane_dist=self.lcs_tree.hyperplanes@jnp.imag(moduli_new)
            hyperplane_check=jnp.all(hyperplane_dist>=0.,axis=0)

            filter_conditions = (hyperplane_check==True)

            NFlux = self.tadpole(flux_new)

            filter_conditions &= (NFlux>0)

            if Q is not None:

                filter_conditions &= (NFlux<=Q)

            if constraints is not None:
                filter_conditions &= (constraints(moduli_new,tau_new,flux_new))

            return moduli_new,tau_new,flux_new,filter_conditions
        else:

            return moduli_new,tau_new,flux_new

    @partial(jit, static_argnums = (2,))
    def compute_residual(self,x,axis=1):
        r"""
        **Description:**
        Computes residual.
        
        Args:
            x (Array): Array. 
        
        Returns: 
            Array: Sum of absolute values.
        
        """

        return jnp.sum(jnp.abs(x),axis=axis)

    @partial(jit, static_argnums = (4,5,6,7,8,9,))
    def fterm_solver(
                     self, 
                     moduli: Array, 
                     tau: complex, 
                     fluxes: Array, 
                     objective_fct: Callable | None = None, 
                     optimiser: Callable | None = None,
                     tol: float = 1e-10, 
                     max_iters: int = 100, 
                     print_progress: bool = False, 
                     mode: str = None
                     ) -> Tuple[int,Array,Array,Array,Array,Array]:
        r"""
        **Description:**
        Solves F-term conditions for a given optimiser.
        
        Args:
            moduli (Array): Complex structure moduli values.
            tau (complex): Value of axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            objective_fct (Callable, optional): Objective function to miminise.
            optimiser (Callable, optional): Optimiser to be used for the minimisation.
            tol (float, optional): Tolerance for the residual. Defaults to ``1e-10``.
            max_iters (int, optional): Maximum number of iterations. Defaults to ``100``.
            print_progress (bool, optional): Whether to print the progress of the minimisation. Defaults to ``False``.
            mode (str, optional): Mode for running the solver. Defaults to ``None``.    
        
        Returns: 
            int: Number of iterations.
            Array: Complex structure moduli values.
            Array: Value of axio-dilaton.
            Array: Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            Array: Boolean array specifying whether constraints are satisfied.
            Array: Residuals :math:`\sum_A |D_A W|` of the :math:`F`-terms.
        
        """

        modes=[None, "fluxes"]
        if mode not in modes:
            raise ValueError(f"CANNOT DETERMINE MODE FOR F-TERM SOLVER!\
                    Mode for filter computation must be one of {modes}!")
            
        if objective_fct is None:
            if mode is None:
                objective_fct = jax.vmap(self.DW)
            elif mode =="fluxes":
                objective_fct = jax.vmap(jax.vmap(self.DW))

        if mode is None:
            axis=1
                
        elif mode == "fluxes":
            axis=2

        def cond(arg):
            """Loop condition: continue while step < max_iters, residual > tol, and checks pass."""
            step, moduli, tau, fluxes, checks, res = arg
            return (step <max_iters) & (jnp.any(res > tol)) & (jnp.any(checks))


        def body(arg):
            """Single optimisation step applying the optimiser and recomputing residuals."""
            #step, moduli, tau, fluxes, checks, res = arg
            step, moduli, tau, fluxes, _ , res = arg

            moduli, tau, fluxes, checks = optimiser(moduli, tau, fluxes)

            obj = objective_fct(moduli,jnp.conj(moduli),tau,jnp.conj(tau),fluxes)

            res = self.compute_residual(obj,axis=axis)

            if print_progress:
                # DEPRECATED
                #step = host_callback.id_tap(progress_bar_jax, (step, max_iters, jnp.min(jnp.nan_to_num(res,nan=1.))), result=step)

                out_type = jax.ShapeDtypeStruct(jnp.shape(step), jnp.result_type(step))
                step = pure_callback(progress_bar_jax, out_type,(step, max_iters, jnp.min(jnp.nan_to_num(res,nan=1.))),step)

            return (step + 1,  moduli, tau, fluxes, checks, res)

        if mode is None:
            return jax.lax.while_loop(cond,body,(0,moduli,tau,fluxes,jnp.ones(moduli.shape[0], dtype=bool),jnp.ones(moduli.shape[0])*10.))
        elif mode == "fluxes":
            input_shape = (moduli.shape[0],moduli.shape[1])
            return jax.lax.while_loop(cond,body,(0,moduli,tau,fluxes,jnp.ones(input_shape, dtype=bool),jnp.ones(input_shape)*10.))
        

    def sample_SUSY_flux_vacua(
                               self,
                               N: int = 100,
                               sampler = None,
                               rns_key = None,
                               max_iters: int = 10,
                               moduli_sampling_mode: str = "cone",
                               max_tadpole: int | None = None,
                               objective_fct: Callable | None = None,
                               optimiser: Callable | None = None,
                               optimisers: list | None = None,
                               constraints: Callable | None = None,
                               mode: str | None = None,
                               tol: float = 1e-10,
                               vmap_dim: int = 10**2,
                               print_progress: bool = False,
                               deduplicate: bool = True,
                               ) -> Tuple[Array, Array, Array, Array]:
        r"""
        **Description:**
        Samples SUSY flux vacua.
        
        Args:
            N (int, optional): Defaults to ``100``.
            sampler (data_sampler, optional): Defaults to ``None``.
            rns_key (PRNGKey, optional): PRNG random key. Defaults to ``None``.
            max_iters (int, optional): Maximum number of iterations. Defaults to ``100``.
            moduli_sampling_mode (str, optional): Sampling mode for the moduli values. Defaults to ``"cone"``.
            max_tadpole (int, optional): Maximum tadpole to use for the sampling. Defaults to ``None``.
            objective_fct (Callable, optional): Objective function to miminise.
            optimiser (Callable, optional): Optimiser to be used for the minimisation.
            optimisers (list, optional): List of optimisers to be used for the minimisation.
            tol (float, optional): Tolerance for the residual. Defaults to ``1e-10``.
            print_progress (bool, optional): Whether to print the progress of the minimisation. Defaults to ``False``.
            constraints (Callable, optional): Defaults to ``None``.
            mode (str, optional): Solving mode specifying we want to solve F-term conditions with random fluxes (``mode="random"``)
                or using ISD bias (``mode="ISD"``) for the fluxes. Defaults to ``None``.
            vmap_dim (int, optional): Array dimension to use in vmapping. Defaults to ``100``.
        
        Returns: 
            Tuple[Array, Array, Array, Array]: Complex structure moduli, axio-dilaton values, fluxes, and residuals.
        
        """
        
        
        # map_to_fd=True implies deduplicate=True
        if self._map_to_fd:
            deduplicate = True

        modes = [None,"ISD","random"]

        if mode not in modes:
            raise ValueError(f"Mode must be one of {modes}, but `mode = {mode}` was given!")

        if max_tadpole is None:
            max_tadpole = self.D3_tadpole

        if sampler is None:
            sampler = self.sampler
        
        if optimisers is None:
            if optimiser is None:
                print("TODO: Should we make these separate functions?")

                kwargs = {"Q":max_tadpole,"return_flag":True,"constraints":constraints,"remove_NANs":True,"in_axes":(0,0,0)}
                
                if mode == "ISD":

                    linearised_shifts_ISD_v = vmapping_func(self.linearised_shifts,mode="ISD",**kwargs)
                    linearised_shifts_H_v = vmapping_func(self.linearised_shifts,mode="Hflux",**kwargs)
                    linearised_shifts_F_v = vmapping_func(self.linearised_shifts,mode="Fflux",**kwargs)

                elif mode == "random":

                    linearised_shifts_ISD_v = vmapping_func(self.linearised_shifts,mode="random",**kwargs)
                    linearised_shifts_H_v = vmapping_func(self.linearised_shifts,mode="Hflux_random",**kwargs)
                    linearised_shifts_F_v = vmapping_func(self.linearised_shifts,mode="Fflux_random",**kwargs)
                
                optimisers = [linearised_shifts_ISD_v,linearised_shifts_H_v,linearised_shifts_F_v]
                
            
            else:
                optimisers = [optimiser]
        
        if objective_fct is None:
            objective_fct = jax.vmap(self.DW)
        
        
        if rns_key is None:
            seed = 42
            rns_key = PRNGSequence(seed)
        
        moduli_sols = []
        tau_sols = []
        fluxes_sols = []
        res_sols = []
        count = 0
        num_vacua = 0
        tic = time.time()
        warnings.warn("TODO: Change the way we use checks. Seems like things are slowing done because we have to pull them back from CPU?")
        try:
            while num_vacua<N:
                
                
                for optimiser in optimisers:
                    
                    flag = True
                    tic0 = time.time()
                    c = 0
                    while flag:
                        moduli,tau,fluxes = sampler.initial_guesses(vmap_dim,rns_key=rns_key,moduli_sampling_mode=moduli_sampling_mode)
                        
                        _, moduli, tau, fluxes, checks,_ = self.fterm_solver(moduli,tau,fluxes.astype(float),objective_fct=objective_fct,
                                                            optimiser=optimiser,tol=tol,max_iters=1)

                        if jnp.any(checks):
                            flag = False

                        toc0 = time.time()
                        if np.abs(toc0-tic0)>60:
                            if c==0:
                                print(f"Sampling suitable initial guesses seems to take a long time.")


                            print(f"Iteration: {c}",flush=True,end="\r")
                            c += 1

                    _, moduli, tau, fluxes, checks,res = self.fterm_solver(moduli,tau,fluxes.astype(float),objective_fct=objective_fct,
                                                            optimiser=optimiser,tol=tol,max_iters=max_iters)
                    
                    checks &= (res<tol)
                    
                    moduli, tau, fluxes, res = moduli[checks], tau[checks], fluxes[checks], res[checks]
                    
                    
                    if len(moduli_sols)==0:
                        moduli_sols,tau_sols,fluxes_sols,res_sols = moduli, tau, fluxes, res
                    else:
                        moduli_sols = jnp.append(moduli_sols,moduli,axis=0)
                        tau_sols = jnp.append(tau_sols,tau)
                        fluxes_sols = jnp.append(fluxes_sols,fluxes,axis=0)
                        res_sols = jnp.append(res_sols,res)

                    # Deduplicate in-loop to avoid accumulating monodromy-equivalent solutions
                    if deduplicate and len(moduli_sols) > 0:
                        moduli_sols, tau_sols, fluxes_sols, keep = self.deduplicate_vacua(
                            moduli_sols, tau_sols, fluxes_sols)
                        res_sols = res_sols[keep]

                    num_vacua = len(moduli_sols)

                    toc = time.time()

                    if print_progress:
                    
                        print(f"Number vacua: {num_vacua}/{N}      finishing rate: {np.around(num_vacua/N*100,2)}%       counter: {count}         time: {np.around(toc-tic,2)}s           ",end="\r",flush=False)
                    
                    if num_vacua>N:
                        break
                    
                count += 1
        except KeyboardInterrupt:
            print("")
            print("Stopped sampling due to KeyboardInterrupt.")
            return moduli_sols,tau_sols,fluxes_sols,res_sols
        except Exception as error:
            print("")
            print('Caught this error: ' + repr(error))
            return moduli_sols,tau_sols,fluxes_sols,res_sols

        
        return moduli_sols,tau_sols,fluxes_sols,res_sols

    def sample_SUSY_vacua_from_fluxes(
                                      self,
                                      fluxes_init=None,
                                      initial_guesses = None,
                                      N=100,
                                      sampler = None,
                                      rns_key=None,
                                      max_iters = 10,
                                      moduli_sampling_mode = "cone",
                                      max_tadpole = None,
                                      objective_fct = None,
                                      optimiser_init: Callable | None = None,
                                      optimiser_steps: Callable | None = None,
                                      constraints=None,
                                      mode=None,
                                      tol = 1e-10,
                                      vmap_dim_flux = 10,
                                      vmap_dim_pts = 10,
                                      print_progress=False,
                                      deduplicate: bool = True,
                                      ) -> Tuple[Array,Array,Array,Array]:
        r"""
        **Description:**
        Samples SUSY flux vacua for given input fluxes.
        
        Args:
            fluxes_init (Array, optional): Input fluxes.
            initial_guesses (Tuple[Array,Array], optional): Initial guesses. Format: (moduli,tau).
            N (int, optional): Defaults to ``100``.
            sampler (data_sampler, optional): Defaults to ``None``.
            rns_key (PRNGKey, optional): PRNG random key. Defaults to ``None``.
            max_iters (int, optional): Maximum number of iterations. Defaults to ``100``.
            moduli_sampling_mode (str, optional): Sampling mode for the moduli values. Defaults to ``"cone"``.
            max_tadpole (int, optional): Maximum tadpole to use for the sampling. Defaults to ``None``.
            objective_fct (Callable, optional): Objective function to miminise.
            optimiser (Callable, optional): Optimiser to be used for the minimisation.
            optimisers (list, optional): List of optimisers to be used for the minimisation.
            tol (float, optional): Tolerance for the residual. Defaults to ``1e-10``.
            print_progress (bool, optional): Whether to print the progress of the minimisation. Defaults to ``False``.
            constraints (Callable, optional): Defaults to ``None``.
            mode (str, optional): Solving mode specifying we want to solve F-term conditions with random fluxes (``mode="random"``)
                or using ISD bias (``mode="ISD"``) for the fluxes. Defaults to ``None``.
            vmap_dim_flux (int, optional): Array dimension to use in vmapping over fluxes. Defaults to ``10``.
            vmap_dim_pts (int, optional): Array dimension to use in vmapping over initial guesses. Defaults to ``10``.
        
        Returns: 
            int: Number of iterations.
            Array: Complex structure moduli values.
            Array: Value of axio-dilaton.
            Array: Boolean array specifying whether constraints are satisfied.
            Array: Residuals :math:`\sum_A |D_A W|` of the :math:`F`-terms.
        
        """
        
        # map_to_fd=True implies deduplicate=True
        if self._map_to_fd:
            deduplicate = True

        modes = ["ISD","random","Hflux","Hflux_random","Fflux","Fflux_random"]

        if mode not in modes:
            raise ValueError(f"Mode must be one of {modes}, but `mode = {mode}` was given!")

        if max_tadpole is None:
            max_tadpole = self.D3_tadpole

        if sampler is None and initial_guesses is None:
            sampler = data_sampler(self)
        
        if objective_fct is None:
            objective_fct = jax.vmap(jax.vmap(self.DW))
        
        if rns_key is None:
            seed = 42
            rns_key = PRNGSequence(seed)
        
        if optimiser_init is None:
            kwargs = {"mode":mode,"Q":max_tadpole,"constraints":constraints,"remove_NANs":True}
            find_solution_init = vmapping_func(self.linearised_shifts,in_axes=(0,0,None),return_flag=False,**kwargs)
            optimiser_init = vmapping_func(find_solution_init,in_axes=(None,None,0))
        
        if optimiser_steps is None:
            kwargs = {"mode":mode,"Q":max_tadpole,"constraints":constraints,"remove_NANs":True}
            find_solution_steps = vmapping_func(self.linearised_shifts,in_axes=(0,0,0),return_flag=True,**kwargs)
            optimiser_steps = vmapping_func(find_solution_steps,in_axes=(0,0,0))

        #if not (fluxes_init is None):
        #    N = 0

        moduli_sols = []
        tau_sols = []
        fluxes_sols = []
        res_sols = []
        count = 0
        num_vacua = 0
        tic = time.time()
        try:
            while num_vacua<N:
                
                if fluxes_init is None:
                    fluxes = sampler.get_fluxes(vmap_dim_flux)
                else:
                    fluxes = fluxes_init
                    
                    
                if initial_guesses is None:
                    moduli,tau = sampler.initial_guesses(vmap_dim_pts,rns_key=rns_key,moduli_sampling_mode=moduli_sampling_mode,include_fluxes=False)
                else:
                    moduli,tau = initial_guesses
                
                # Initial step:
                ## Maps initial guesses for a single choice of half of the fluxes
                ## onto full choices of fluxes via ISD sampling.
                ## We use the linearised shift computation to also return new initial guesses
                ## closer to the actual F-term minimum.
                moduli0,tau0,fluxes0 = optimiser_init(moduli,tau,fluxes)
                
                # Optimisation step:
                ## Using linearised shift computation to solve the F-term conditions
                ## for the above choices of fluxes.
                _, moduli, tau, fluxes, checks, res = self.fterm_solver(moduli0,tau0,fluxes0.astype(float),objective_fct=objective_fct,
                                                        optimiser=optimiser_steps,tol=tol,max_iters=max_iters,mode="fluxes")#,print_progress=print_progress

                ## Add to the checks that DW<tol
                checks &= (res<tol)

                ## Use the checks as flags to collect final results.
                moduli, tau, fluxes, res = moduli[checks], tau[checks], fluxes[checks], res[checks]


                if len(moduli_sols)==0:
                    moduli_sols,tau_sols,fluxes_sols,res_sols = moduli, tau, fluxes, res
                else:
                    moduli_sols = jnp.append(moduli_sols,moduli,axis=0)
                    tau_sols = jnp.append(tau_sols,tau)
                    fluxes_sols = jnp.append(fluxes_sols,fluxes,axis=0)
                    res_sols = jnp.append(res_sols,res)

                # Deduplicate in-loop
                if deduplicate and len(moduli_sols) > 0:
                    moduli_sols, tau_sols, fluxes_sols, keep = self.deduplicate_vacua(
                        moduli_sols, tau_sols, fluxes_sols)
                    res_sols = res_sols[keep]

                num_vacua = len(moduli_sols)

                toc = time.time()
                if print_progress:
                    if N>0:
                        print(f"Number vacua: {num_vacua}/{N}      finishing rate: {np.around(num_vacua/N*100,2)}%       counter: {count}         time: {np.around(toc-tic,2)}s           ",end="\r",flush=False)
                    else:
                        print(f"Number vacua: {num_vacua}/{len(fluxes0)}      success rate: {np.around(num_vacua/len(fluxes0)*100,2)}%       time: {np.around(toc-tic,2)}s           ",end="\r",flush=False)

                count += 1
        except KeyboardInterrupt:
            print("")
            print("Stopped sampling due to KeyboardInterrupt.")
            return moduli_sols,tau_sols,fluxes_sols,res_sols
        except Exception as error:
            print("")
            print('Caught this error: ' + repr(error))
            return moduli_sols,tau_sols,fluxes_sols,res_sols

        return moduli_sols,tau_sols,fluxes_sols,res_sols


    
    
    
    
    
    def deduplicate_vacua(
        self,
        moduli: Array,
        tau: Array,
        fluxes: Array,
        n_digits: int = 8,
        boundary_tol: float = 1e-8,
        axion_FD: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Array, Array, Array, Array]:
        r"""
        **Description:**
        Removes duplicate vacua from a batch of solutions.

        If ``map_to_fd=True`` (set at construction), vacua are first mapped to
        the fundamental domain via :func:`map_to_FD` (monodromy +
        :math:`\text{SL}(2,\mathbb{Z})`), so that monodromy-equivalent solutions
        are identified.  If ``map_to_fd=False``, deduplication is performed on
        the raw ``(moduli, tau, fluxes)`` values directly.

        In both cases, duplicates are identified by rounding to ``n_digits``
        decimal places.  Note that a single flux vector can admit multiple
        distinct vacua, so the fingerprint always includes moduli and tau.

        Args:
            moduli (Array): Complex structure moduli, shape ``(N, h12)``.
            tau (Array): Axio-dilaton values, shape ``(N,)``.
            fluxes (Array): Flux vectors, shape ``(N, 2*n_fluxes)``.
            n_digits (int): Number of decimal digits for rounding when
                identifying duplicates. Default ``8``.
            boundary_tol (float): Points within this tolerance of the FD
                boundary are snapped to the boundary. Default ``1e-8``.
                Only used when ``map_to_fd=True``.
            axion_FD (tuple, optional): Axion fundamental domain ``(lo, hi)``.
                If ``None``, uses ``self.axion_FD``.  Only used when
                ``map_to_fd=True``.

        Returns:
            Tuple[Array, Array, Array, Array]:
                ``(moduli_unique, tau_unique, fluxes_unique, keep_indices)``

        See also: :func:`map_to_FD`
        """
        N = len(moduli)
        if N == 0:
            return moduli, tau, fluxes, jnp.array([], dtype=jnp.int32)

        if self._map_to_fd:
            lo, hi = axion_FD if axion_FD is not None else self.axion_FD

            # Step 1: Map all vacua to fundamental domain
            moduli_fd_list = []
            tau_fd_list = []
            fluxes_fd_list = []

            for i in range(N):
                m, t, f = self.map_to_FD(moduli[i], tau[i], fluxes[i], axion_FD=axion_FD)
                moduli_fd_list.append(m)
                tau_fd_list.append(t)
                fluxes_fd_list.append(f)

            moduli_fd = jnp.stack(moduli_fd_list)
            tau_fd = jnp.array(tau_fd_list)
            fluxes_fd = jnp.stack(fluxes_fd_list)

            # Step 2: Snap boundary points — Re(z) within boundary_tol of lo snaps to hi
            # Convention: (lo, hi] so lo is excluded and hi is included
            re_z = moduli_fd.real
            near_lo = jnp.abs(re_z - lo) < boundary_tol
            moduli_fd = jnp.where(near_lo, hi + 1j * moduli_fd.imag, moduli_fd)
        else:
            moduli_fd, tau_fd, fluxes_fd = moduli, tau, fluxes

        # Build fingerprint from (moduli, tau, fluxes) and find unique rows
        fingerprint = jnp.concatenate([
            jnp.around(moduli_fd.real, n_digits),
            jnp.around(moduli_fd.imag, n_digits),
            jnp.around(tau_fd.real[:, None], n_digits),
            jnp.around(tau_fd.imag[:, None], n_digits),
            jnp.around(fluxes_fd, 0),
        ], axis=1)

        _, keep = jnp.unique(fingerprint, axis=0, return_index=True, size=N)
        keep = jnp.sort(keep)
        # Remove padding from jnp.unique (pads with last index when size > n_unique)
        valid = jnp.concatenate([jnp.array([True]), keep[1:] > keep[:-1]])
        keep = keep[valid]

        removed = N - len(keep)
        #if removed > 0:
        #    print(f"deduplicate_vacua: removed {removed}/{N} duplicates, {len(keep)} unique remain.")

        return moduli_fd[keep], tau_fd[keep], fluxes_fd[keep], keep


    def CheckConstraints(model, roots, fluxes,flux_ids=None,uni_pts_ids=None,max_deg=1,mode=None,verbose=False,tolerance=1e-10):
        r"""
        
        **Description:**
        Tests filter conditions on the obtained roots of the F-term conditions for given fluxes. We currently check the following conditions
        
        
        .. admonition:: Details
            :class: dropdown
        
            1.) Positivity of the string coupling
        
            2.) Kahler cone conditions: real Kähler potential and positive definiteness of the Kähler metric
        
            3.) Positive definiteness of the Hessian of the Scalar potential
        
            To check the positive definiteness of the Hermitian matrices obtained from the Kähler metric and Hessian, we compute the Cholesky decomposition via `jnp.linalg.cholesky`.
        
            Flat directions or, stated otherwise, a non-trivial dimension of the vacuum moduli space are accounted for by computing `model.codim_vacuum_moduli_space`. Here, we use a tolerance set by the one used for checking roots of the F-term conditions `model.fterms`.
        
            Provide example here?
        
            TODO
        
            Let :math:`t^i` be the generators of the Kähler cone. For our vacuum solution :math:`Z` to be inside the Kähler cone, we demand that it can be written as
        
            .. math::
                Z=\sum_{i=1}^{h^{1,2}}\, c_i\, t^{i}\; ,\quad c_i\geq 0\, .
        
            The coefficients :math:`c_i` are a measure of the distance to one of the walls for the Kähler cone.
        
            If we search only for SUSY vacua, we look at 1st order obstructions to DW=0 for the fluctuations (see https://inspirehep.net/files/6bbb139c9328fbcd16e4b2ce50a8a567)
        
            TODO CONTINUE DESCRIPTIOn
        
        Args:
            model (class)
            roots (JAX array): Values of moduli at point in moduli space to compute Hessian.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            flux_ids (JAX array, optional): IDs of initial fluxes.
            uni_pts_ids (JAX array, optional): ID of unique start points for which roots have been obtained
            max_deg (int, optional): Description
            mode (None, optional): Description
            verbose (string, optional)
            tolerance (float, optional): Description
        
        Returns:
            filtered_roots (numpy array): Array of roots satisfying the filter conditions.
            filtered_fluxes (numpy array): Array of fluxes satisfying the filter conditions.
        
        """

        check_nan_V=vmap(check_nan)

        cholesky_V=vmap(jnp.linalg.cholesky)




        print("NOT WORKING YET!!!!")


        modes = [None, "SUSY"]
        if mode not in modes:
            warnings.warn(f"Mode for filter computation must be one of {modes}!")
            sys.exit()
            
            
        # Define vmap functions:
        F_inst_V = vmap(model.F_inst)
        
        Prepot_V = vmap(model.prepot)
        
        Kahler_Metric_V = vmap(model.kahler_metric)
        
        Pi_dagger_Sigma_Pi_V=vmap(model.A)
        
        # If we search only for SUSY vacua, we look at 1st order obstructions to DW=0 for the fluctuations (see https://inspirehep.net/files/6bbb139c9328fbcd16e4b2ce50a8a567)
        if mode=="SUSY":
            codim_vacuum_moduli_space_V=vmap(lambda mod, cmod, tau, ctau, f: model.codim_vacuum_moduli_space(mod,cmod,tau,ctau,f,mode=mode,tolerance=tolerance))
            rank_matrix_V=vmap(lambda x: rank_matrix(x,tolerance=tolerance*tolerance)) # Take tolerance^2 because we have F-term squared for SUSY minima which should be of order less than tolerance!
        else:
            rank_matrix_V=vmap(lambda x: rank_matrix(x,tolerance=tolerance))
            codim_vacuum_moduli_space_V=rank_matrix_V
        
        Scalar_Potential_Hess_V = vmap(lambda mod, cmod, tau, ctau, f: model.scalar_potential_hessian(mod,cmod,tau,ctau,f,mode=mode))
        
        
        # Define complex moduli from real roots
        mod = jnp.asarray(roots[:,0:-2:2]) + 1.j*jnp.asarray(roots[:,1:-2:2])
        
        # Define complex axio-dilaton from real roots
        tau = roots[:,-2] + 1.j*roots[:,-1]
        
        # Get only imaginary part
        imag_mod=jnp.imag(mod)

        Pi_D_Pi=Pi_dagger_Sigma_Pi_V(mod, jnp.conj(mod))
        
        Finstb = F_inst_V(mod)
        Fpertb = Prepot_V(mod) - Finstb
        
        
        filter_conditions=(jnp.imag(tau)>0) & (Pi_D_Pi>0)& (jnp.abs(Finstb/Fpertb)< 1e-1)
        
        if verbose:
            print("HAVE TO STILL IMPLEMENT THE RADIUS OF CONVERGENCE CHECK!!!")
            
        # TODO: Implement radius of convergence check!
        """
        if model.max_deg>0:
            
            lamCs=model.check_radius_conv(max_deg)
            radius_conv_check1 = jnp.all(imag_mod-jnp.array(lamCs)>0,axis=0)
            filter_conditions&=(radius_conv_check1==True)
            
            # Vmap the radius convergence check
            check_radius_conv_KC_V=vmap(model.check_radius_conv_KC, in_axes = (0,None))
            
            lamC=check_radius_conv_KC_V(imag_mod,max_deg)
            radius_conv_check2 = jnp.all(imag_mod-lamC.reshape(len(imag_mod),1)>0)
            
            filter_conditions&=(radius_conv_check2==True)
        """
            
        
        # FILTER FLAG TO SPECIFY HOW WE CHECK THE FILTERS!
        filter_flag=""
        
        if len(model.generators_kahler_cone)!=model.NumMod:
            # IF WE DON'T HAVE THE GENERTORS; WE TEST HYPERPLANES OR KAHLER METRIC
            if len(model.hyperplanes)==0:
                filter_flag="naive"
                KMM=Kahler_Metric_V(mod,jnp.conj(mod),tau,jnp.conj(tau))
                CKMM=cholesky_V(KMM)
                
                filter_conditions &= (check_nan_V(KMM)==False)& (check_nan_V(CKMM)==False)
            else:
                filter_flag="hyperplane"
                imaginary_parts=jnp.imag(mod)
                hyperplane_dist=jnp.einsum("ij,kj",model.hyperplanes,imaginary_parts)
                hyperplane_check=jnp.all(hyperplane_dist>=0.,axis=0)
                
                filter_conditions &= (hyperplane_check==True)
            
        else:
            filter_flag="span"
            inv_KC_generator=jnp.linalg.inv(model.generators_kahler_cone)
            
            imaginary_parts=jnp.imag(mod)
            span_dist=jnp.einsum("ji,kj",inv_KC_generator,imaginary_parts)
            
            # We demand that the coefficients are all positive.
            span_check=jnp.all(span_dist>=0.,axis=0)
            
            filter_conditions &= (span_check==True)
            
            
        print("Using flag: ",filter_flag)

        SP_Hess = Scalar_Potential_Hess_V(mod, jnp.conj(mod), tau, jnp.conj(tau), fluxes)
        
        CSP=cholesky_V(SP_Hess)
        
        if mode=="SUSY":
            codims_mod_space=codim_vacuum_moduli_space_V(mod, jnp.conj(mod), tau, jnp.conj(tau), fluxes)
            # If verbose is turned on, we make an additional test by looking
            if verbose:
                rank_hessian_test=rank_matrix_V(SP_Hess)
                if np.max(np.abs(codims_mod_space-rank_hessian_test))>0:
                    print("Rank test on Hessian does not agree with rank test on vacuum moduli space!")
                    print("Vacuum moduli space: ",codims_mod_space)
                    print("Hessian: ",rank_hessian_test)
        else:
            codims_mod_space=codim_vacuum_moduli_space_V(SP_Hess)
            
        
        filter_conditions &= (check_nan_V(SP_Hess)==False)& (check_nan_V(CSP)==False)& (codims_mod_space==2*(model.NumMod+1))
        
        
        ind0=jnp.where(filter_conditions)[0]
        
        roots_ret=roots[ind0]
        flux_ret=fluxes[ind0]
        
        
        if flux_ids is not None:
            flux_ids_ret=jnp.array(flux_ids)[ind0]
        else:
            flux_ids_ret=None
        
        if uni_pts_ids is not None:
            uni_pts_ids_ret=jnp.array(uni_pts_ids)[ind0]
        else:
            uni_pts_ids_ret=None
        
        if verbose:
        
            print("Number of solutions passing...")
            ind1=jnp.where(jnp.imag(tau)>0)
            print("... Im(tau)>0: ",len(ind1[0]))
            ind2=jnp.where(Pi_D_Pi>0)
            print("... Pi^{dagger}*Pi>0: ",len(ind2[0]))
            ind3=jnp.where(jnp.abs(Finstb/Fpertb)< 1e-1)
            print("... |F_inst/F_pert|<1e-1: ",len(ind3[0]))
            
            if filter_flag=="naive":
                ind4=jnp.where(check_nan_V(KMM)==False)
                print("... Nan Kähler Metric: ",len(ind4[0]))
                ind5=jnp.where(check_nan_V(CKMM)==False)
                print("... Nan Kähler Cholesky: ",len(ind5[0]))
                
            ind6=jnp.where(check_nan_V(SP_Hess)==False)
            print("... Nan Hessian: ",len(ind6[0]))
            ind7=jnp.where(check_nan_V(CSP)==False)
            print("... Nan Hessian Cholesky: ",len(ind7[0]))
            ind8=jnp.where(codims_mod_space==2*(model.NumMod+1))
            print("... flat direction: ",len(ind8[0]))
            
            if filter_flag=="hyperplane":
                
                ind9 = jnp.where(hyperplane_check==True)
                print("... hyperplane test: ",len(ind9[0]))
                
            if filter_flag=="span":
            
                ind9 = jnp.where(span_check==True)
                print("... span test: ",len(ind9[0]))
            
            """
            if model.max_deg>0:
                ind10 = jnp.where(radius_conv_check1==True)
                print("... radius convergence asymptotic: ",len(ind10[0]))
                ind11 = jnp.where(radius_conv_check1==True)
                print("... radius convergence point: ",len(ind11[0]))
            """
        
        return jnp.array(roots_ret),jnp.array(flux_ret),flux_ids_ret,uni_pts_ids_ret
    

unflatten_func = lambda aux_data, children: unflatten_func_class(aux_data, children, FluxVacuaFinder)

register_pytree_node(FluxVacuaFinder, flatten_func, unflatten_func)