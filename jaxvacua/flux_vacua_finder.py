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

"""Canonical flux-vacuum search interface.

Purpose
-------
Define ``FluxVacuaFinder``, the main entry point for constructing, sampling
and classifying flux vacua in Type IIB compactifications on Calabi-Yau
threefolds.

Main public API
---------------
- ``FluxVacuaFinder``: a subclass of ``FluxEFT``.  A finder is the EFT model,
  with vacuum-search methods added; inherited physics methods such as
  ``V_x``, ``DW``, ``ddV_x``, ``hessian`` and ``tadpole`` are callable
  directly on the finder.
- SUSY workflow: ``newton_method_flux_vacua(mode="SUSY")``,
  ``sample_SUSY_flux_vacua``, ``sample_SUSY_vacua_from_fluxes``,
  ``linearised_shifts_*`` and ``deduplicate_vacua``.
- Non-SUSY workflow: ``sample_critical_points`` with Newton, Adam, L-BFGS,
  Adam-on-potential, hybrid and SciPy backends.
- Gaussian-M prior calibration and persistence through
  ``calibrate_priors``, ``save_calibration`` and ``load_calibration``.

Design notes
------------
Stateless post-processing helpers delegate to ``jaxvacua.flux_utils``.  Use
``FluxVacuaFinder.from_model(model, sampler=None)`` to reuse an existing
``FluxEFT`` geometry without recomputing it.
"""


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
from . import flux_utils

# Optional dep: optax is required only for the first-order solvers
# (_solve_dV_optax_*, sample_critical_points(solver='adam'|'lbfgs'|'hybrid')).
# Lazy-import so users who only need Newton don't have to install optax.
try:
    import optax
    _HAS_OPTAX = True
except ImportError:
    _HAS_OPTAX = False


class FluxVacuaFinder(FluxEFT):
    r"""
    **Description:**
    Canonical entry point for searching, sampling, and classifying flux
    vacua in Type IIB string theory on a Calabi--Yau threefold.

    **Class relationship.** ``FluxVacuaFinder`` is a *subclass* of
    :class:`jaxvacua.flux_eft.FluxEFT` — it **is** the EFT model, with
    vacuum-finding methods bolted on.  There is **no separate "model"
    attribute**: every physics method (``scalar_potential``, ``V_x``,
    ``DW``, ``ddV_x``, ``hessian``, ``tadpole``, ``ISD_matrix``, …) is
    inherited and callable directly on the finder instance.  Existing
    code that uses ``FluxEFT`` works unchanged if you swap in a
    ``FluxVacuaFinder`` — Liskov-substitutable.

    Use :meth:`from_model` to construct a finder from an existing
    ``FluxEFT`` instance without re-doing the geometry computation
    (useful when running multiple finders against the same model).

    **Headline capabilities** (post-merge, 2026-05-17):

    *Newton solver* with B1+B2 guards (:meth:`newton_method_flux_vacua`).
    Handles SUSY (``mode="SUSY"``) and general extrema (``mode=None``).
    Optional runaway escape via ``moduli_max=`` kwarg + automatic
    singular-Hessian protection.

    *SUSY workflow* — :meth:`sample_SUSY_flux_vacua`,
    :meth:`sample_SUSY_vacua_from_fluxes`, :meth:`linearised_shifts_*`,
    :meth:`deduplicate_vacua`.  Solves :math:`D_I W = 0`.

    *Non-SUSY workflow* — :meth:`sample_critical_points`.  Six solver
    backends: ``"newton"`` (default), ``"adam"``, ``"lbfgs"``,
    ``"adam_v"``, ``"hybrid"``, ``"scipy"``.  Solves
    :math:`\partial_I V = 0`.

    *Gaussian-M prior calibration* — :meth:`_precompute_M_eigensystem` →
    :meth:`_estimate_sigmas` → :meth:`calibrate_priors` →
    :meth:`save_calibration` / :meth:`load_calibration`.  Tunes the
    integer-flux sampling prior per ISD mode.

    *Shared utilities* — :meth:`classify_solution`, :meth:`is_physical`,
    :meth:`dedup_key`, :meth:`to_fd`.  Stateless helpers for post-hoc
    analysis of any converged candidate.

    **Quickstart**:

    .. code-block:: python

        import jax.numpy as jnp
        import jaxvacua as jvc

        finder = jvc.FluxVacuaFinder(h12=2, model_ID=1)

        # 1. SUSY vacuum from a starting point:
        fluxes = jnp.array([4.,-3.,-2.,-2.,3.,2.,39.,-13.,-4.,0.,0.,0.])
        mod, tau, res = finder.newton_method_flux_vacua(
            jnp.array([2.+3j, 1.5+2j]), -0.3+5j, fluxes,
            mode="SUSY",
        )

        # 2. Non-SUSY scan (~100 critical points from a Gaussian-M prior):
        vacua = finder.sample_critical_points(
            Q=200, n_target=100, n_batch=2000, max_batches=5,
            solver="hybrid", isd_mode="ISD-", verbose=False,
        )

        # 3. Post-hoc classification of any converged (x, flux):
        info = finder.classify_solution(x_sol, fluxes)
        # → {'V', '|DW|', 'eigenvalues', 'is_susy', 'is_minimum', 'Nflux'}

    See ``documentation/source/notebooks/02_vacuum_finding/`` for full
    tutorials.
    
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
                fundamental domain via :func:`map_to_fd`. Defaults to ``False``.
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

    @classmethod
    def from_model(
        cls,
        model: FluxEFT,
        sampler: "data_sampler" = None,
        map_to_fd: bool = False,
        flux_bounds: tuple = (-10, 10),
        axion_bounds: tuple = (-0.5, 0.5),
        dilaton_bounds: tuple = (2., 10.),
        moduli_bounds: tuple = (1., 5.),
        seed: int = 42,
    ) -> "FluxVacuaFinder":
        r"""
        **Description:**
        Construct a :class:`FluxVacuaFinder` from an existing
        :class:`FluxEFT` instance, **reusing** all of its geometry data
        (period vector, prepotential, conifold tree, …) — no
        recomputation.  Composition-style alternative to the standard
        ``FluxVacuaFinder(h12=..., model_ID=...)`` constructor.

        Use this when:

        * You already have a ``FluxEFT`` and want to run a finder on it
          without rebuilding the geometry (which can be slow for large
          h12).
        * You want multiple finders on the same model (e.g. different
          ``sampler`` bounds, different ``map_to_fd``).
        * You have a custom :class:`FluxEFT` subclass and want the
          finder methods on top of it.

        Internally clones the ``model``'s ``__dict__`` into a freshly
        allocated ``FluxVacuaFinder`` instance.  The two instances
        thereafter share references to the same arrays / sub-objects
        (``lcs_tree``, ``periods``, ``css``, …).  Mutating the model
        will be visible on the finder; mutating finder-only state
        (``_map_to_fd``, ``_calibrated_sigmas``, …) does *not* leak
        back into the model.

        Args:
            model (FluxEFT): Instance whose state to clone.
            sampler (data_sampler, optional): If given, used instead of
                lazy-constructing one from the bounds kwargs.
            map_to_fd (bool, optional): Whether to map results to the
                fundamental domain. Defaults to ``False``.
            flux_bounds (tuple, optional): Forwarded to ``data_sampler``;
                only used if ``sampler is None``.
            axion_bounds (tuple, optional): Forwarded to ``data_sampler``;
                only used if ``sampler is None``.
            dilaton_bounds (tuple, optional): Forwarded to ``data_sampler``;
                only used if ``sampler is None``.
            moduli_bounds (tuple, optional): Forwarded to ``data_sampler``;
                only used if ``sampler is None``.
            seed (int, optional): Forwarded to ``data_sampler``; only used
                if ``sampler is None``.

        Returns:
            FluxVacuaFinder: New instance sharing geometry with ``model``.

        **Example:**

        .. code-block:: python

            model  = jvc.FluxEFT(h12=2, model_ID=1, maximum_degree=5)
            # ... use model for some FluxEFT-only work ...
            finder = jvc.FluxVacuaFinder.from_model(model)
            vacua  = finder.sample_critical_points(Q=200, n_target=10)
        """
        finder = cls.__new__(cls)
        finder.__dict__.update(model.__dict__)
        # Finder-only state (not inherited from the model)
        finder._map_to_fd = map_to_fd
        finder._sampler = sampler
        finder._sampler_kwargs = dict(
            flux_bounds=flux_bounds,
            axion_bounds=axion_bounds,
            dilaton_bounds=dilaton_bounds,
            moduli_bounds=moduli_bounds,
            seed=seed,
        )
        return finder

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
        
    @partial(jit, static_argnums = (4,5,6,7,8,9,))
    def _newton_method_flux_vacua_complex(
        self, moduli: Array, tau: complex, fluxes: Array, mode: str = None, step_size_Newton: float = 1e-1,
        tol: float = 1e-10,max_iters: int = 100,print_progress: bool = False, moduli_max: float = None
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
            """Loop condition: continue while step < max_iters and residual > tol
            AND residual is finite (so B1's runaway escape and B2's singular-Hessian
            escape — which set res=inf — actually terminate the loop instead of
            stalling to max_iters)."""
            step, moduli, tau, res = arg
            return (step < max_iters) & (res > tol) & jnp.isfinite(res)

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
                

            # Solve the Newton system m2 · deltaPhi = -m1 directly rather than
            # forming inv(m2) explicitly: a single solve is more numerically
            # stable and cheaper.  On a degenerate m2 the solve returns
            # non-finite entries, which the B2 guard below catches.
            deltaPhi = -jnp.linalg.solve(m2, m1)

            # Singular-Hessian guard (B2): if the solve blows up (NaN/inf
            # entries in deltaPhi from a degenerate m2), zero out the step so
            # the iterate stays in a valid pre-NaN snapshot, and flag the
            # iteration as failed via res=inf so cond() exits next iteration.
            delta_ok = jnp.all(jnp.isfinite(deltaPhi))
            deltaPhi = jnp.where(delta_ok, deltaPhi, jnp.zeros_like(deltaPhi))

            moduli = moduli + step_size_Newton*(deltaPhi[self.h12+1:2*self.h12+1]+1j*deltaPhi[0:self.h12])
            tau = tau + step_size_Newton*(deltaPhi[2*self.h12+1]+1j*deltaPhi[self.h12])

            # Residual reuses the already-computed m1 = (DW or dV) at the iterate
            # *before* this step (no extra gradient evaluation).  It is therefore
            # the pre-step residual: the returned (moduli, tau) have had one more
            # Newton step applied, so the point is at least as converged as `res`
            # reports.  The loop exits once this drops to tol, guaranteeing the
            # returned residual is never optimistic.
            res = jnp.sum(jnp.abs(m1))
            res = jnp.where(delta_ok, res, jnp.inf)

            # Runaway escape: if moduli_max is set and the iterate's magnitude
            # exceeds it, force res=inf so cond() exits on the next iteration.
            # Static `if` resolves at trace time; zero HLO cost when moduli_max=None.
            if moduli_max is not None:
                runaway = jnp.max(jnp.abs(jnp.append(moduli, tau))) > moduli_max
                res = jnp.where(runaway, jnp.inf, res)

            if print_progress:
                out_type = jax.ShapeDtypeStruct(jnp.shape(step), jnp.result_type(step))
                step = jax.pure_callback(progress_bar_jax, out_type,(step, max_iters, res),step)

            return (step + 1,  moduli, tau, res)

        return jax.lax.while_loop(cond,body,(0,moduli,tau,10.))


    @partial(jit, static_argnums = (3,4,5,6,7,8,))
    def _newton_method_flux_vacua_real(
        self, x: Array, fluxes: Array, mode: str = None, step_size_Newton: float = 1e-1,
        tol: float = 1e-10, max_iters: int = 100, print_progress: bool = False, moduli_max: float = None
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
            """Loop condition: continue while step < max_iters and residual > tol
            AND residual is finite (so B1/B2 escapes via res=inf terminate the loop
            promptly instead of stalling to max_iters)."""
            step, x, res = arg
            return (step < max_iters) & (res > tol) & jnp.isfinite(res)

        def body(arg):
            """Single Newton step in complex coordinates."""
            step, x, res = arg

            if mode=="SUSY":
                m2 = self.dDW_x(x,fluxes)
                m1 = self.DW_x(x,fluxes)
            else:
                m2 = self.ddV_x(x,fluxes)
                m1 = self.dV_x(x,fluxes)

            # Solve the Newton system m2 · deltaPhi = -m1 directly (see
            # _newton_method_flux_vacua_complex for the inv→solve rationale).
            deltaPhi = -jnp.linalg.solve(m2, m1)

            # Singular-Hessian guard (B2, see _newton_method_flux_vacua_complex
            # for rationale).  Zero out non-finite step entries and flag the
            # iteration as failed.
            delta_ok = jnp.all(jnp.isfinite(deltaPhi))
            deltaPhi = jnp.where(delta_ok, deltaPhi, jnp.zeros_like(deltaPhi))

            x = x + step_size_Newton*deltaPhi

            # Pre-step residual reusing m1 (see _newton_method_flux_vacua_complex
            # for the semantics: returned `x` has had one further Newton step, so
            # the reported residual is pessimistic, never optimistic).
            res = jnp.sum(jnp.abs(m1))
            res = jnp.where(delta_ok, res, jnp.inf)

            # Runaway escape (static `if`, see _newton_method_flux_vacua_complex
            # for rationale).  In real-coord form `x` already includes both
            # moduli and tau components, so a single max-abs check suffices.
            if moduli_max is not None:
                runaway = jnp.max(jnp.abs(x)) > moduli_max
                res = jnp.where(runaway, jnp.inf, res)

            if print_progress:
                out_type = jax.ShapeDtypeStruct(jnp.shape(step), jnp.result_type(step))
                step = jax.pure_callback(progress_bar_jax, out_type,(step, max_iters, res),step)

            return (step + 1,  x, res)

        return jax.lax.while_loop(cond,body,(0,x,10.))

    @partial(jit, static_argnums = (4,5,6,7,8,9,10,))
    def newton_method_flux_vacua(
        self, moduli: Array, tau: complex, fluxes: Array, mode: str = None, step_size_Newton: float = 1e-1,
        tol: float = 1e-10,max_iters: int = 100,print_progress: bool = False, solver_mode: str = "complex",
        moduli_max: float = None,
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

            _,moduli,tau,res = self._newton_method_flux_vacua_complex(moduli,tau,fluxes,mode=mode, step_size_Newton=step_size_Newton,tol=tol,max_iters=max_iters,print_progress=print_progress, moduli_max=moduli_max)

        elif solver_mode=="real":

            x = self._convert_complex_to_real(moduli,jnp.conj(moduli),tau,jnp.conj(tau))

            _, x, res = self._newton_method_flux_vacua_real(x,fluxes,mode=mode, step_size_Newton=step_size_Newton,tol=tol,max_iters=max_iters,print_progress=print_progress, moduli_max=moduli_max)

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
        remove_NANs: bool = False,
        step_size: float = 1.0
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
            step_size (float, optional): Damping factor applied to the linearised shift; the moduli/axio-dilaton
                are updated by ``step_size`` times the computed shift. ``1.0`` recovers the full (undamped) step.
                Defaults to ``1.0``.

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
        b = jnp.concatenate([re_delta_LHS, im_delta_LHS])

        # Assemble the linear system in a single concatenate rather than a chain
        # of jnp.append calls (fewer HLO nodes -> faster compile, identical result).
        A1 = jnp.concatenate([re_ak, re_vk, re_c[None], re_s[None]], axis=0)
        A2 = jnp.concatenate([im_ak, im_vk, im_c[None], im_s[None]], axis=0)
        A = jnp.concatenate([A1.T, A2.T], axis=0)

        # Solve for the shifts
        shift = jnp.linalg.solve(A,b)

        flux_new = jnp.concatenate([ml, mu, nl, nu])
        
        if return_shifts:
            return step_size*(shift[:self.h12]+1j*shift[self.h12:2*self.h12]),step_size*(shift[-2]+1j*shift[-1]),flux_new
        else:
            moduli_new = moduli+step_size*(shift[:self.h12]+1j*shift[self.h12:2*self.h12])
            
            tau_new = tau+step_size*(shift[-2]+1j*shift[-1])
            
            return moduli_new,tau_new,flux_new

    @partial(jit, static_argnums = (4,5,6,))
    def linearised_shifts_H(
        self,
        moduli: Array,
        tau: complex,
        fluxes: Array,
        mode: str = "Hflux",
        return_shifts: bool = False,
        remove_NANs: bool = False,
        step_size: float = 1.0
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
            step_size (float, optional): Damping factor applied to the linearised shift; the moduli/axio-dilaton
                are updated by ``step_size`` times the computed shift. ``1.0`` recovers the full (undamped) step.
                Defaults to ``1.0``.

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

        # --- old version: materialise the full rank-3 dM/dM_c tensor, then contract ---
        #Theta_k = jnp.matmul(self.dM(moduli,jnp.conj(moduli)).transpose(2,0,1),RHS)
        #Theta_bk = jnp.matmul(self.dM_c(moduli,jnp.conj(moduli)).transpose(2,0,1),RHS)
        # New: contract the ISD-matrix derivative with RHS via a single jacrev of
        # (M @ RHS) instead of forming the full tensor.  RHS is moduli-independent,
        # so (d_z M) @ RHS = d_z (M @ RHS); ~dim fewer reverse-mode passes,
        # verified to match the explicit tensor to ~1e-13.
        moduli_c = jnp.conj(moduli)
        Theta_k = jax.jacrev(lambda zz: jnp.matmul(self.ISD_matrix(zz, moduli_c), RHS), holomorphic=True)(moduli).T
        Theta_bk = jax.jacrev(lambda zzc: jnp.matmul(self.ISD_matrix(moduli, zzc), RHS), holomorphic=True)(moduli_c).T


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

        # Single concatenate rather than chained jnp.append (the imaginary block
        # A2 is identically zero in Hflux mode, so only the real block remains).
        A1 = jnp.concatenate([re_ak, re_vk, re_c[None], re_s[None]], axis=0)
        A = A1.T

        shift = jnp.linalg.solve(A,b)

        flux_new = jnp.concatenate([f, HF])

        if return_shifts:
            return step_size*(shift[:self.h12]+1j*shift[self.h12:2*self.h12]),step_size*(shift[-2]+1j*shift[-1]),flux_new
        else:
            moduli_new = moduli+step_size*(shift[:self.h12]+1j*shift[self.h12:2*self.h12])
            tau_new = tau+step_size*(shift[-2]+1j*shift[-1])
            return moduli_new,tau_new,flux_new
        
    @partial(jit, static_argnums = (4,5,6,))
    def linearised_shifts_F(
        self,
        moduli: Array,
        tau: complex,
        fluxes: Array,
        mode: str = "Fflux",
        return_shifts: bool = False,
        remove_NANs: bool = False,
        step_size: float = 1.0
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
            step_size (float, optional): Damping factor applied to the linearised shift; the moduli/axio-dilaton
                are updated by ``step_size`` times the computed shift. ``1.0`` recovers the full (undamped) step.
                Defaults to ``1.0``.

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

        # --- old version: materialise the full rank-3 dM/dM_c tensor, then contract ---
        #Theta_k = jnp.matmul(self.dM(moduli,jnp.conj(moduli)).transpose(2,0,1),RHS)
        #Theta_bk = jnp.matmul(self.dM_c(moduli,jnp.conj(moduli)).transpose(2,0,1),RHS)
        # New: contract the ISD-matrix derivative with RHS via a single jacrev of
        # (M @ RHS) instead of forming the full tensor.  RHS is moduli-independent,
        # so (d_z M) @ RHS = d_z (M @ RHS); ~dim fewer reverse-mode passes,
        # verified to match the explicit tensor to ~1e-13.
        moduli_c = jnp.conj(moduli)
        Theta_k = jax.jacrev(lambda zz: jnp.matmul(self.ISD_matrix(zz, moduli_c), RHS), holomorphic=True)(moduli).T
        Theta_bk = jax.jacrev(lambda zzc: jnp.matmul(self.ISD_matrix(moduli, zzc), RHS), holomorphic=True)(moduli_c).T


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

        # Single concatenate rather than chained jnp.append (imaginary block is
        # identically zero in Fflux mode, so only the real block remains).
        A1 = jnp.concatenate([-re_ak, -re_vk, re_c[None], re_s[None]], axis=0)
        A = A1.T

        shift = jnp.linalg.solve(A,b)

        flux_new = jnp.concatenate([FF, h])

        if return_shifts:
            return step_size*(shift[:self.h12]+1j*shift[self.h12:2*self.h12]),step_size*(shift[-2]+1j*shift[-1]),flux_new
        else:
            moduli_new = moduli+step_size*(shift[:self.h12]+1j*shift[self.h12:2*self.h12])
            tau_new = tau+step_size*(shift[-2]+1j*shift[-1])
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
        remove_NANs: bool = False,
        step_size: float = 1.0
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
            step_size (float, optional): Damping factor applied to the linearised shift; the moduli/axio-dilaton
                are updated by ``step_size`` times the computed shift. ``1.0`` recovers the full (undamped) step.
                Defaults to ``1.0``.

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
            moduli_new,tau_new,flux_new = self.linearised_shifts_ISD(moduli,tau,fluxes,mode=mode,remove_NANs=remove_NANs,step_size=step_size)
        elif mode == "Hflux" or mode == "Hflux_random":
            moduli_new,tau_new,flux_new = self.linearised_shifts_H(moduli,tau,fluxes,mode=mode,remove_NANs=remove_NANs,step_size=step_size)
        elif mode == "Fflux" or mode == "Fflux_random":
            moduli_new,tau_new,flux_new = self.linearised_shifts_F(moduli,tau,fluxes,mode=mode,remove_NANs=remove_NANs,step_size=step_size)


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
                               Q: int | None = None,
                               objective_fct: Callable | None = None,
                               optimiser: Callable | None = None,
                               optimisers: list | None = None,
                               constraints: Callable | None = None,
                               mode: str | None = None,
                               tol: float = 1e-10,
                               vmap_dim: int = 10**2,
                               print_progress: bool = False,
                               deduplicate: bool = True,
                               step_size: float = 1.0,
                               max_resample_attempts: int = 200,
                               max_batches: int = 1000,
                               errors: str = "raise",
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
            Q (int, optional): Maximum tadpole to use for the sampling. Defaults to ``None``.
            objective_fct (Callable, optional): Objective function to miminise.
            optimiser (Callable, optional): Optimiser to be used for the minimisation.
            optimisers (list, optional): List of optimisers to be used for the minimisation.
            tol (float, optional): Tolerance for the residual. Defaults to ``1e-10``.
            print_progress (bool, optional): Whether to print the progress of the minimisation. Defaults to ``False``.
            constraints (Callable, optional): Defaults to ``None``.
            mode (str, optional): Solving mode specifying we want to solve F-term conditions with random fluxes (``mode="random"``)
                or using ISD bias (``mode="ISD"``) for the fluxes. Defaults to ``None``.
            vmap_dim (int, optional): Array dimension to use in vmapping. Defaults to ``100``.
            step_size (float, optional): Damping factor forwarded to :func:`linearised_shifts`; the moduli/axio-dilaton
                are updated by ``step_size`` times the linearised shift each iteration. ``1.0`` is the full step.
                Defaults to ``1.0``.
            max_resample_attempts (int, optional): Maximum number of times initial guesses are re-sampled while
                searching for a batch in which at least one candidate passes the one-step filter. Bounds the inner
                loop so a geometry/mode that rarely yields passing candidates cannot spin forever. Defaults to ``200``.
            max_batches (int, optional): Maximum number of outer sampling batches before giving up on reaching ``N``.
                Defaults to ``1000``.
            errors (str, optional): Behaviour when ``max_resample_attempts`` or ``max_batches`` is exhausted —
                ``"raise"`` raises a ``RuntimeError`` (fail fast), ``"warn"`` emits a warning and returns the
                partial result. Defaults to ``"raise"``.

        Returns:
            Tuple[Array, Array, Array, Array]: Complex structure moduli, axio-dilaton values, fluxes, and residuals.
        
        """
        
        
        # map_to_fd=True implies deduplicate=True
        if self._map_to_fd:
            deduplicate = True

        modes = [None,"ISD","random"]

        if mode not in modes:
            raise ValueError(f"Mode must be one of {modes}, but `mode = {mode}` was given!")

        if Q is None:
            if self.Q() is None:
                raise ValueError("Q must be supplied or a global value for the D3-tadpole must be set in the FluxEFT constructor!")
            Q = self.Q()

        if sampler is None:
            sampler = self.sampler
        
        if optimisers is None:
            if optimiser is None:
                kwargs = {"Q":Q,"return_flag":True,"constraints":constraints,"remove_NANs":True,"step_size":step_size,"in_axes":(0,0,0)}
                
                if mode == "ISD":

                    linearised_shifts_ISD_v = vmapping_func_cached(self.linearised_shifts,mode="ISD",**kwargs)
                    linearised_shifts_H_v = vmapping_func_cached(self.linearised_shifts,mode="Hflux",**kwargs)
                    linearised_shifts_F_v = vmapping_func_cached(self.linearised_shifts,mode="Fflux",**kwargs)

                elif mode == "random":

                    linearised_shifts_ISD_v = vmapping_func_cached(self.linearised_shifts,mode="random",**kwargs)
                    linearised_shifts_H_v = vmapping_func_cached(self.linearised_shifts,mode="Hflux_random",**kwargs)
                    linearised_shifts_F_v = vmapping_func_cached(self.linearised_shifts,mode="Fflux_random",**kwargs)
                
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
        if errors not in ("raise", "warn"):
            raise ValueError(f"`errors` must be 'raise' or 'warn', got {errors!r}.")

        # Per-batch buffers, cleared in place each batch so the merge helper below
        # always references the same list objects (no rebinding).
        batch_moduli, batch_tau, batch_flux, batch_res = [], [], [], []

        def merge_batch():
            """Concatenate the current batch buffers into the running solution set,
            deduplicate, update ``num_vacua``, and clear the buffers.  Called at the
            end of each batch *and* from the KeyboardInterrupt handler, so an
            interrupt mid-run still returns (and deduplicates) everything found so
            far rather than discarding the in-progress batch."""
            nonlocal moduli_sols, tau_sols, fluxes_sols, res_sols, num_vacua
            if not batch_moduli:
                return
            new_moduli = jnp.concatenate(batch_moduli, axis=0)
            new_tau    = jnp.concatenate(batch_tau)
            new_flux   = jnp.concatenate(batch_flux, axis=0)
            new_res    = jnp.concatenate(batch_res)
            if len(moduli_sols) == 0:
                moduli_sols, tau_sols, fluxes_sols, res_sols = new_moduli, new_tau, new_flux, new_res
            else:
                moduli_sols = jnp.concatenate([moduli_sols, new_moduli], axis=0)
                tau_sols    = jnp.concatenate([tau_sols, new_tau])
                fluxes_sols = jnp.concatenate([fluxes_sols, new_flux], axis=0)
                res_sols    = jnp.concatenate([res_sols, new_res])
            # Deduplicate to drop monodromy-equivalent solutions.
            if deduplicate and len(moduli_sols) > 0:
                moduli_sols, tau_sols, fluxes_sols, keep = self.deduplicate_vacua(
                    moduli_sols, tau_sols, fluxes_sols)
                res_sols = res_sols[keep]
            num_vacua = len(moduli_sols)
            batch_moduli.clear(); batch_tau.clear(); batch_flux.clear(); batch_res.clear()

        try:
            while num_vacua < N and count < max_batches:

                for optimiser in optimisers:

                    # Re-sample initial guesses until at least one candidate passes
                    # the one-step filter, bounded by max_resample_attempts so a
                    # geometry/mode that rarely yields a passing candidate cannot
                    # spin forever (this was previously an unbounded `while` loop).
                    found = False
                    for _attempt in range(max_resample_attempts):
                        moduli,tau,fluxes = sampler.initial_guesses(vmap_dim,rns_key=rns_key,moduli_sampling_mode=moduli_sampling_mode)
                        _, moduli, tau, fluxes, checks, _ = self.fterm_solver(moduli,tau,fluxes.astype(float),objective_fct=objective_fct,
                                                            optimiser=optimiser,tol=tol,max_iters=1)
                        if jnp.any(checks):
                            found = True
                            break

                    if not found:
                        msg = (f"sample_SUSY_flux_vacua: no candidate passed the filter in "
                               f"{max_resample_attempts} resampling attempts (mode={mode}). "
                               f"Loosen Q, change mode, or increase vmap_dim.")
                        if errors == "raise":
                            raise RuntimeError(msg)
                        warnings.warn(msg)
                        continue

                    _, moduli, tau, fluxes, checks, res = self.fterm_solver(moduli,tau,fluxes.astype(float),objective_fct=objective_fct,
                                                            optimiser=optimiser,tol=tol,max_iters=max_iters)
                    checks &= (res < tol)
                    batch_moduli.append(moduli[checks])
                    batch_tau.append(tau[checks])
                    batch_flux.append(fluxes[checks])
                    batch_res.append(res[checks])

                    # Merge this optimiser's results and stop early once N vacua
                    # are reached — avoids compiling/running the remaining
                    # mode-optimisers when the first already supplies enough
                    # (preserves the original early-exit fast path).
                    merge_batch()
                    if num_vacua >= N:
                        break

                if print_progress:
                    toc = time.time()
                    print(f"Number vacua: {num_vacua}/{N}      finishing rate: {np.around(num_vacua/N*100,2)}%       counter: {count}         time: {np.around(toc-tic,2)}s           ",end="\r",flush=False)

                count += 1

            if num_vacua < N:
                msg = (f"sample_SUSY_flux_vacua: reached max_batches={max_batches} with only "
                       f"{num_vacua}/{N} vacua found. Increase max_batches/vmap_dim or loosen filters.")
                if errors == "raise":
                    raise RuntimeError(msg)
                warnings.warn(msg)

        except KeyboardInterrupt:
            print("")
            print("Stopped sampling due to KeyboardInterrupt; merging partial batch and returning found solutions.")
            merge_batch()
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
                                      Q = None,
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
                                      step_size: float = 1.0,
                                      max_batches: int = 1000,
                                      errors: str = "raise",
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
            Q (int, optional): Maximum tadpole to use for the sampling. Defaults to ``None``.
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
            step_size (float, optional): Damping factor forwarded to :func:`linearised_shifts`; the moduli/axio-dilaton
                are updated by ``step_size`` times the linearised shift each iteration. ``1.0`` is the full step.
                Defaults to ``1.0``.
            max_batches (int, optional): Maximum number of sampling batches before giving up on reaching ``N``.
                Bounds the loop so it cannot spin forever. Defaults to ``1000``.
            errors (str, optional): Behaviour when ``max_batches`` is exhausted — ``"raise"`` raises a
                ``RuntimeError`` (fail fast), ``"warn"`` emits a warning and returns the partial result.
                Defaults to ``"raise"``.

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

        if Q is None:
            if self.Q() is None:
                raise ValueError("Q must be supplied or a global value for the D3-tadpole must be set in the FluxEFT constructor!")
            Q = self.Q()

        if sampler is None and initial_guesses is None:
            sampler = data_sampler(self)
        
        if objective_fct is None:
            objective_fct = jax.vmap(jax.vmap(self.DW))
        
        if rns_key is None:
            seed = 42
            rns_key = PRNGSequence(seed)
        
        if optimiser_init is None:
            kwargs = {"mode":mode,"Q":Q,"constraints":constraints,"remove_NANs":True,"step_size":step_size}
            find_solution_init = vmapping_func_cached(self.linearised_shifts,in_axes=(0,0,None),return_flag=False,**kwargs)
            optimiser_init = vmapping_func_cached(find_solution_init,in_axes=(None,None,0))
        
        if optimiser_steps is None:
            kwargs = {"mode":mode,"Q":Q,"constraints":constraints,"remove_NANs":True,"step_size":step_size}
            find_solution_steps = vmapping_func_cached(self.linearised_shifts,in_axes=(0,0,0),return_flag=True,**kwargs)
            optimiser_steps = vmapping_func_cached(find_solution_steps,in_axes=(0,0,0))

        #if not (fluxes_init is None):
        #    N = 0

        moduli_sols = []
        tau_sols = []
        fluxes_sols = []
        res_sols = []
        count = 0
        num_vacua = 0
        tic = time.time()
        if errors not in ("raise", "warn"):
            raise ValueError(f"`errors` must be 'raise' or 'warn', got {errors!r}.")

        # Per-batch buffers, cleared in place each batch (see sample_SUSY_flux_vacua
        # for the rationale of the shared merge helper).
        batch_moduli, batch_tau, batch_flux, batch_res = [], [], [], []

        def merge_batch():
            """Concatenate the current batch buffers into the running solution set,
            deduplicate, update ``num_vacua``, and clear the buffers.  Called at the
            end of each batch and from the KeyboardInterrupt handler so an interrupt
            still returns (and deduplicates) everything found so far."""
            nonlocal moduli_sols, tau_sols, fluxes_sols, res_sols, num_vacua
            if not batch_moduli:
                return
            new_moduli = jnp.concatenate(batch_moduli, axis=0)
            new_tau    = jnp.concatenate(batch_tau)
            new_flux   = jnp.concatenate(batch_flux, axis=0)
            new_res    = jnp.concatenate(batch_res)
            if len(moduli_sols) == 0:
                moduli_sols, tau_sols, fluxes_sols, res_sols = new_moduli, new_tau, new_flux, new_res
            else:
                moduli_sols = jnp.concatenate([moduli_sols, new_moduli], axis=0)
                tau_sols    = jnp.concatenate([tau_sols, new_tau])
                fluxes_sols = jnp.concatenate([fluxes_sols, new_flux], axis=0)
                res_sols    = jnp.concatenate([res_sols, new_res])
            if deduplicate and len(moduli_sols) > 0:
                moduli_sols, tau_sols, fluxes_sols, keep = self.deduplicate_vacua(
                    moduli_sols, tau_sols, fluxes_sols)
                res_sols = res_sols[keep]
            num_vacua = len(moduli_sols)
            batch_moduli.clear(); batch_tau.clear(); batch_flux.clear(); batch_res.clear()

        try:
            # `count < max_batches` bounds the loop so it cannot spin forever if
            # the target N is never reached for the given fluxes/geometry.
            while num_vacua < N and count < max_batches:

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
                                                        optimiser=optimiser_steps,tol=tol,max_iters=max_iters,mode="fluxes")

                ## Add to the checks that DW<tol, then collect the accepted solutions.
                checks &= (res<tol)
                batch_moduli.append(moduli[checks])
                batch_tau.append(tau[checks])
                batch_flux.append(fluxes[checks])
                batch_res.append(res[checks])

                merge_batch()

                if print_progress:
                    toc = time.time()
                    if N>0:
                        print(f"Number vacua: {num_vacua}/{N}      finishing rate: {np.around(num_vacua/N*100,2)}%       counter: {count}         time: {np.around(toc-tic,2)}s           ",end="\r",flush=False)
                    else:
                        print(f"Number vacua: {num_vacua}/{len(fluxes0)}      success rate: {np.around(num_vacua/len(fluxes0)*100,2)}%       time: {np.around(toc-tic,2)}s           ",end="\r",flush=False)

                count += 1

            if N > 0 and num_vacua < N:
                msg = (f"sample_SUSY_vacua_from_fluxes: reached max_batches={max_batches} with only "
                       f"{num_vacua}/{N} vacua found. Increase max_batches/vmap_dim or loosen filters.")
                if errors == "raise":
                    raise RuntimeError(msg)
                warnings.warn(msg)

        except KeyboardInterrupt:
            print("")
            print("Stopped sampling due to KeyboardInterrupt; merging partial batch and returning found solutions.")
            merge_batch()
            return moduli_sols,tau_sols,fluxes_sols,res_sols

        return moduli_sols,tau_sols,fluxes_sols,res_sols


    
    
    
    
    
    def dedup_key(self, moduli, tau, fluxes, n_digits: int = 6):
        r"""
        **Description:**
        Hashable dedup key for a single ``(moduli, tau, fluxes)`` triple.
        Suitable for incremental ``set`` / ``dict`` membership tests in a
        streaming sampling loop.

        Distinct from :func:`deduplicate_vacua`, which is a *batched*
        ``jnp.unique`` pass over an entire solution set.

        Thin delegator over :func:`jaxvacua.flux_utils.dedup_key`.
        See its docstring for argument and return-value detail.

        **Note (2026-05-17 arg-order change):** flipped from legacy
        ``(flux, moduli, tau)`` to ``(moduli, tau, fluxes)`` for project-wide
        consistency.
        """
        return flux_utils.dedup_key(moduli, tau, fluxes, n_digits=n_digits)

    def classify_solution(self, x, flux, noscale: bool = True, min_tol: float = 1e-6) -> dict:
        r"""
        **Description:**
        Classify a converged critical-point candidate ``(x, flux)``: returns
        ``{V, |DW|, eigenvalues, is_susy, is_minimum, Nflux}``.

        Thin delegator over :func:`jaxvacua.flux_utils.classify_solution`.
        See its docstring for argument and return-value detail.
        """
        return flux_utils.classify_solution(self, x, flux, noscale=noscale, min_tol=min_tol)

    def is_physical(self, x, moduli_max: float = None, verbose: bool = False) -> bool:
        r"""
        **Description:**
        Decide whether the converged candidate ``x`` lies in the physical
        region of moduli space.  Runs a five-tier cascade: runaway bound →
        dilaton floor → Kähler-cone hyperplanes → Kähler-metric positivity →
        basic ``Im(z_i) > 0``.

        Thin delegator over :func:`jaxvacua.flux_utils.is_physical`, passing
        ``self`` as both the model and the source of the sampler (via
        ``self.sampler``).  See the helper's docstring for full cascade
        detail.

        Pass ``verbose=True`` to print which check rejected the point
        (useful for debugging "why was my candidate filtered?").
        """
        return flux_utils.is_physical(
            self, self.sampler, x, moduli_max=moduli_max, verbose=verbose,
        )

    def to_fd(self, moduli, tau, fluxes) -> tuple:
        r"""
        **Description:**
        Optionally map ``(moduli, tau, fluxes)`` to the fundamental domain
        via :func:`map_to_fd`, with numpy/JAX marshalling at the boundary.

        Whether the mapping is applied is controlled by the constructor
        flag ``self._map_to_fd`` (set via ``map_to_fd=`` kwarg at finder
        construction).  If ``False``, the inputs are returned unchanged.

        Thin delegator over :func:`jaxvacua.flux_utils.map_to_fd`.  See its
        docstring for full detail and the rationale for keeping this as a
        separate wrapper over :func:`map_to_fd` (the actual math method).
        """
        return flux_utils.map_to_fd(self, moduli, tau, fluxes, enabled=self._map_to_fd)

    def deduplicate_vacua(
        self,
        moduli: Array,
        tau: Array,
        fluxes: Array,
        n_digits: int = 8,
        boundary_tol: float = 1e-8,
        axion_fd: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Array, Array, Array, Array]:
        r"""
        **Description:**
        Removes duplicate vacua from a batch of solutions.

        If ``map_to_fd=True`` (set at construction), vacua are first mapped to
        the fundamental domain via :func:`map_to_fd` (monodromy +
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
            axion_fd (tuple, optional): Axion fundamental domain ``(lo, hi)``.
                If ``None``, uses ``self.axion_fd``.  Only used when
                ``map_to_fd=True``.

        Returns:
            Tuple[Array, Array, Array, Array]:
                ``(moduli_unique, tau_unique, fluxes_unique, keep_indices)``

        See also: :func:`map_to_fd`
        """
        N = len(moduli)
        if N == 0:
            return moduli, tau, fluxes, jnp.array([], dtype=jnp.int32)

        if self._map_to_fd:
            lo, hi = axion_fd if axion_fd is not None else self.axion_fd

            # Step 1: Map all vacua to fundamental domain
            moduli_fd_list = []
            tau_fd_list = []
            fluxes_fd_list = []

            for i in range(N):
                m, t, f = self.map_to_fd(moduli[i], tau[i], fluxes[i], axion_fd=axion_fd)
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


    # ==========================================================================
    #  ISD-flux Gaussian-prior calibration
    # ==========================================================================

    def _precompute_M_eigensystem(self, n_sample: int = 50, Q: int = None, sampler=None):
        r"""
        **Description:**
        Pre-compute the ISD-matrix eigensystem used by the Gaussian-M flux
        prior in :func:`_generate_flux_candidates`.

        Procedure:

        1. Sample ``n_sample`` complex-moduli points from ``sampler`` (defaults
           to ``self.sampler``).
        2. At each point, compute :math:`M = \mathrm{ISD\_matrix}(z, \bar z)`
           and its absolute eigenvalues.  Track the condition number
           :math:`\lambda_{\max}/\lambda_{\min}` and
           :math:`\mathrm{tr}(M^{-1})`.
        3. Pick the **best-conditioned** ``M`` as the representative for the
           sampling prior; eigen-decompose ``M = V Λ Vᵀ``.
        4. Build per-direction variance scales
           :math:`\sigma^2 = N_{\max}/(n_{\mathrm{fluxes}} \cdot s_{\min})`,
           then ``_M_scales = sqrt(σ² · |Λ|)``.

        Persists to ``self``:

        - ``_M_eigvecs`` ``(d, d)``: eigenvectors of the chosen ``M``
        - ``_M_scales`` ``(d,)``:    per-direction sampling sqrt-variances
        - ``_s_min``:                dilaton floor used for σ²
        - ``_tr_Minv_median``:       median over samples of ``tr(M⁻¹)``
        - ``_M_cond``:               condition number of the chosen ``M``

        The first two are JAX-pytree-static via ``_STATIC_KEYS``; the
        scalars are added there as well, so subsequent jit'd calls on
        ``self`` work after calibration.

        Args:
            n_sample (int, optional): number of moduli points to scan.
                Default: ``50``.
            Q (int): target tadpole bound :math:`N_{\max}`.  Used
                for the σ² scale.  Default: ``self.Q()`` (the
                geometry's natural maximum.
            sampler (data_sampler, optional): sampler instance.  If
                ``None`` (default), uses ``self.sampler``.

        Returns:
            ``None``.  Result lands on the finder instance as the five
            attributes listed above.
        """
        if Q is None:
            if self.Q() is None:
                raise ValueError("Q must be supplied or a global value for the D3-tadpole must be set in the FluxEFT constructor!")
            Q = int(self.Q())
        if sampler is None:
            sampler = self.sampler

        mod = np.array(sampler.get_complex_moduli(n_sample))
        mod_jax = jnp.array(mod, dtype=complex)

        # Vectorised: compute the ISD matrix at all sampled moduli in a single
        # XLA call.  ~800x faster than the per-row Python loop on h12=2 at
        # n_sample=50 (700 ms → 0.9 ms).  Compilation is one-shot per
        # (model, n_sample) shape combination.
        M_all   = np.asarray(
            jax.vmap(self.ISD_matrix, in_axes=(0, 0))(mod_jax, jnp.conj(mod_jax))
        )                                      # shape (n_sample, d, d)
        eigs_all   = np.abs(np.linalg.eigvalsh(M_all))   # shape (n_sample, d)
        conds      = eigs_all.max(axis=1) / np.maximum(eigs_all.min(axis=1), 1e-30)
        all_tr_Minv = list(np.sum(1.0 / np.maximum(eigs_all, 1e-30), axis=1))

        best_idx  = int(np.argmin(conds))
        best_cond = float(conds[best_idx])
        best_M    = M_all[best_idx]

        eigvals_M, eigvecs_M = np.linalg.eigh(best_M)
        eigvals_M = np.abs(eigvals_M)

        s_min = float(getattr(sampler, 's_lower', np.sqrt(3) / 2))
        sigma_sq = float(Q) / (self.n_fluxes * s_min)
        self._M_scales = np.sqrt(sigma_sq * eigvals_M)
        self._M_eigvecs = eigvecs_M

        # Summary stats for analytical σ estimation (used by _estimate_sigmas, C2).
        self._s_min = s_min
        self._tr_Minv_median = float(np.median(all_tr_Minv))
        self._M_cond = float(best_cond)


    def _estimate_sigmas(self, Q: int = None) -> None:
        r"""
        **Description:**
        Estimate the default Gaussian-prior σ per ISD mode analytically
        from the M-eigensystem precomputed by :func:`_precompute_M_eigensystem`.

        Closed forms:

        * **H mode**  (input = h directly):
          ``σ_H² = Q / (s_min · tr(M⁻¹))``.
          Derived from ``E[s·hᵀM⁻¹h] = s·σ²·tr(M⁻¹) ≈ Q`` for
          ``h ~ N(0, σ²I)``.
        * **F mode**  (input = f, completion ``h ~ M⁻¹f``): heuristic
          ``σ_F = σ_H`` — the M⁻¹ acts as a natural regulariser.
        * **ISD−**: ``σ = 2·σ_H`` — completion via N⁻¹ amplifies inputs.
        * **ISD+**: ``σ = 0.5·σ_H`` — completion via N amplifies outputs.

        The four sigmas are written to ``self._calibrated_sigmas`` as a
        ``dict[str, float]``.  These are the *defaults*; ``calibrate_priors``
        (C7) can refine them empirically via binary-search.

        Args:
            Q (int, optional): target tadpole bound :math:`N_{\max}`.
                Default: ``self.Q()`` (the geometry's natural maximum).

        Returns:
            ``None``.  Result lands on ``self._calibrated_sigmas``.

        **Raises:**

            ``ValueError`` if ``Q`` is not supplied.

        **Pre-requisite:**

            :func:`_precompute_M_eigensystem` must have run first
            (populates ``self._s_min`` and ``self._tr_Minv_median``).
            Calling this without prior calibration raises ``AttributeError``.
        """
        if Q is None:
            
            if self.Q() is None:
                raise ValueError("Q must be supplied or a global value for the D3-tadpole must be set in the FluxEFT constructor!")
            
            Q = int(self.Q())
        
        Q_f = float(Q)
        s_min = self._s_min
        tr_Minv = self._tr_Minv_median

        # H mode (h ~ N(0,σ²I), tadpole ≈ s·σ²·tr(M⁻¹))
        sigma_H = np.sqrt(Q_f / (s_min * tr_Minv))

        # F mode: heuristic
        sigma_F = sigma_H

        # ISD− mode: completion via N⁻¹ amplifies inputs
        sigma_ISDm = 2.0 * sigma_H

        # ISD+ mode: completion via N amplifies outputs
        sigma_ISDp = 0.5 * sigma_H

        self._calibrated_sigmas = {
            "F":    float(sigma_F),
            "H":    float(sigma_H),
            "ISD-": float(sigma_ISDm),
            "ISD+": float(sigma_ISDp),
        }


    # --------------------------------------------------------------------------
    #  Gaussian-prior flux candidate generator
    # --------------------------------------------------------------------------

    def _generate_flux_candidates(
        self,
        n_candidates: int,
        moduli_pts,
        tau_pts,
        isd_mode: str = "ISD-",
        rng: "np.random.Generator" = None,
        Q: int = None,
        flux_prior: str = "gaussian",
        flux_prior_sigma: float = None,
        sampler = None,
    ):
        r"""
        **Description:**
        Generate integer flux candidates via ISD sampling in the specified mode.

        Pipeline:

        1. Draw ``N`` input vectors from ``flux_prior``:

           * ``"gaussian"`` — isotropic ``N(0, σ²I)`` with σ from
             ``_calibrated_sigmas[isd_mode]`` (set by :func:`_estimate_sigmas`
             or :func:`calibrate_priors`), or a per-mode hard-coded fallback
             if uncalibrated.
           * ``"M_weighted"`` (``H`` mode only) — ``h ~ N(0, σ²·M)`` via the
             precomputed eigensystem; falls back to isotropic for other modes.
           * ``"uniform"`` — integers in ``[-3, 4)``.

        2. Round to integers.
        3. ISD-complete each input via
           ``sampler.ISD_sampling(z, conj(z), tau, conj(tau), input, mode=isd_mode)``.
        4. Filter: drop on ISD-completion failure, non-finite flux, all-zero
           input, ``|tadpole| > Q`` or ``|tadpole| ≤ 0``.
        5. Build the real-coord starting point
           ``x0 = _convert_complex_to_real(...)`` for each survivor.

        Args:
            n_candidates (int): maximum number of candidate flux vectors
                to attempt.
            moduli_pts (Array): starting complex-structure moduli, shape
                ``(N, h12)``.  Must satisfy ``len(moduli_pts) >= n_candidates``.
            tau_pts (Array): starting axio-dilatons, shape ``(N,)``.
            isd_mode (str, optional): ISD completion mode, one of
                ``"F"``, ``"H"``, ``"ISD+"``, ``"ISD-"``.  Default: ``"ISD-"``.
            rng (np.random.Generator, optional): RNG for the prior
                draws.  If ``None``, uses ``np.random.default_rng(0)`` for
                determinism.
            Q (int, optional): tadpole bound.  Default:
                ``self.Q()`` (the geometry's natural maximum).
            flux_prior (str, optional): ``"gaussian"`` (default),
                ``"M_weighted"``, or ``"uniform"``.
            flux_prior_sigma (float, optional): per-call σ override.
                When ``None``, uses ``_calibrated_sigmas`` or the per-mode
                hard-coded fallback.
            sampler (data_sampler, optional): sampler instance.  If
                ``None`` (default), uses ``self.sampler``.

        Returns:
            ``Tuple[ndarray, ndarray, ndarray]``:
            ``(x0_array, flux_array, indices)`` for the surviving candidates.
            Shapes ``(K, 2*(h12+1))``, ``(K, 2*n_fluxes)``, ``(K,)``.  When
            no candidate survives, returns shape-``(0, …)`` arrays of the
            corresponding dtype rather than ``None``.
        """
        if Q is None:
            if self.Q() is None:
                raise ValueError("Q must be supplied or a global value for the D3-tadpole must be set in the FluxEFT constructor!")
            
            Q = int(self.Q())
        if sampler is None:
            sampler = self.sampler
        if rng is None:
            rng = np.random.default_rng(0)

        n_fl = self.n_fluxes
        dim_H3 = self.dimension_H3
        N = min(n_candidates, len(moduli_pts))

        # Input length per mode
        if isd_mode in ("F", "H"):
            inp_len = n_fl
        elif isd_mode in ("ISD+", "ISD-"):
            inp_len = 2 * dim_H3
        else:
            raise ValueError(
                f"Unknown isd_mode: {isd_mode}. "
                f"Must be one of 'F', 'H', 'ISD+', 'ISD-'."
            )

        # Per-mode default σ table.
        _DEFAULT_SIGMA_BY_MODE = {"F": 1.0, "H": 1.0, "ISD+": 0.5, "ISD-": 2.0}

        # Draw input vectors.  σ resolution priority:
        #   1. explicit flux_prior_sigma override
        #   2. self._calibrated_sigmas[isd_mode]  (after C2/C7)
        #   3. _DEFAULT_SIGMA_BY_MODE[isd_mode]
        calibrated = getattr(self, "_calibrated_sigmas", None) or {}
        if flux_prior in ("gaussian", "M_weighted"):
            if flux_prior_sigma is not None:
                sigma = flux_prior_sigma
            elif isd_mode in calibrated:
                sigma = calibrated[isd_mode]
            else:
                sigma = _DEFAULT_SIGMA_BY_MODE.get(isd_mode, 1.0)

            M_eigvecs = getattr(self, "_M_eigvecs", None)
            M_scales  = getattr(self, "_M_scales",  None)
            if (flux_prior == "M_weighted" and isd_mode == "H"
                    and M_eigvecs is not None and M_scales is not None):
                # h ~ N(0, scale² · σ²_M · M) via eigensystem.  _M_scales
                # already encodes sqrt(Q/(d·s_min) · λ_M).
                scale_factor = flux_prior_sigma or 0.3
                z = rng.standard_normal((N, inp_len))
                inputs = np.round(
                    (z * M_scales[None, :] * scale_factor) @ M_eigvecs.T
                )
            else:
                inputs = np.round(rng.normal(0, sigma, (N, inp_len)))

        elif flux_prior == "uniform":
            inputs = rng.integers(-3, 4, (N, inp_len)).astype(float)
        else:
            raise ValueError(
                f"Unknown flux_prior: {flux_prior}. "
                f"Must be 'gaussian', 'M_weighted', or 'uniform'."
            )

        x0_list = []
        flux_list = []
        for i in range(N):
            inp = jnp.array(inputs[i], dtype=float)
            if jnp.all(inp == 0):
                continue
            try:
                fl = sampler.ISD_sampling(
                    moduli_pts[i], jnp.conj(moduli_pts[i]),
                    tau_pts[i], jnp.conj(tau_pts[i]),
                    inp, mode=isd_mode,
                )
            except Exception:
                continue

            if fl is None or not jnp.all(jnp.isfinite(fl)):
                continue

            fl_int = np.round(np.array(jnp.array(fl).real, copy=True)).astype(float)
            tad = abs(float(jnp.real(self.tadpole(jnp.array(fl_int)))))
            if tad <= 0 or tad > Q:
                continue

            x0 = np.asarray(self._convert_complex_to_real(
                moduli_pts[i], jnp.conj(moduli_pts[i]),
                tau_pts[i], jnp.conj(tau_pts[i]),
            ))
            x0_list.append(x0)
            flux_list.append(fl_int)

        if not x0_list:
            n_fields = 2 * (self.h12 + 1)
            return (np.zeros((0, n_fields)),
                    np.zeros((0, 2 * n_fl)),
                    np.array([], dtype=int))

        return np.array(x0_list), np.array(flux_list), np.arange(len(x0_list))


    # --------------------------------------------------------------------------
    #  Batched JIT-vmap'd classification
    # --------------------------------------------------------------------------

    def _build_classify_kernel(self, noscale: bool = True):
        r"""
        **Description:**
        Build a JIT-compiled, vmapped classification kernel for batched use.

        Returns a callable
        ``(x_batch, flux_batch) -> (V, dw_norm, eigs, tad)``
        where each output is an array over the batch dimension.  Equivalent
        per-row to :func:`jaxvacua.flux_utils.classify_solution`, but
        processed under a single XLA call per chunk for speed (~1000× over
        a Python loop at N≈10⁴).

        Used by :func:`_classify_batch`.

        Args:
            noscale (bool, optional): pass-through to
                :func:`scalar_potential` and :func:`ddV_x`.  Default: ``True``.
        """
        def classify_single(x, fl):
            z, zb, tau, taub = self._convert_real_to_complex(x)
            V       = self.scalar_potential(z, zb, tau, taub, fl, noscale=noscale).real
            DW      = self.DW(z, zb, tau, taub, fl)
            dw_norm = jnp.sum(jnp.abs(DW))
            ddV     = self.ddV_x(x, fl, noscale=noscale)
            eigs    = jnp.sort(jnp.linalg.eigvalsh(ddV).real)
            tad     = jnp.abs(jnp.real(self.tadpole(fl)))
            return V, dw_norm, eigs, tad

        return jax.jit(vmap(classify_single))

    def _classify_batch(
        self,
        x_batch,
        flux_batch,
        sub_batch_size: int = 64,
        noscale: bool = True,
        min_tol: float = 1e-6,
    ) -> list:
        r"""
        **Description:**
        Vectorised classification of a batch of critical-point candidates.

        Computes :math:`V`, :math:`|DW|`, Hessian eigenvalues, and tadpole
        for each of the ``N`` candidates.  Returns a list of dicts matching
        the schema of :func:`jaxvacua.flux_utils.classify_solution`.

        Processed in sub-batches of ``sub_batch_size`` rows for memory
        control (the per-row Hessian eigvals computation is memory-hungry
        under vmap for large ``N``).

        Args:
            x_batch (Array): solutions, shape ``(N, 2*(h12+1))``.
            flux_batch (Array): fluxes, shape ``(N, 2*n_fluxes)``.
            sub_batch_size (int, optional): chunk size for the vmap'd
                kernel.  Default: ``64``.  Chunking is semantically transparent
                (independent of chunk size).
            noscale (bool, optional): pass-through.  Default: ``True``.
            min_tol (float, optional): minimum-classification threshold
                (matches :func:`classify_solution`).  ``is_minimum`` is True
                iff all Hessian eigenvalues exceed ``min_tol``.  Default:
                ``1e-6``.

        Returns:
            list of dicts with keys: ``V``, ``|DW|``, ``eigenvalues``,
            ``is_susy``, ``is_minimum``, ``Nflux`` — same schema as
            :func:`jaxvacua.flux_utils.classify_solution`.
        """
        kernel = self._build_classify_kernel(noscale=noscale)
        N = len(x_batch)

        all_V, all_dw, all_eigs, all_tad = [], [], [], []
        for i in range(0, N, sub_batch_size):
            j = min(i + sub_batch_size, N)
            V, dw, eigs, tad = kernel(
                jnp.array(x_batch[i:j]), jnp.array(flux_batch[i:j]),
            )
            all_V.append(np.asarray(V))
            all_dw.append(np.asarray(dw))
            all_eigs.append(np.asarray(eigs))
            all_tad.append(np.asarray(tad))

        if N == 0:
            return []

        V_arr    = np.concatenate(all_V)
        dw_arr   = np.concatenate(all_dw)
        eigs_arr = np.concatenate(all_eigs, axis=0)
        tad_arr  = np.concatenate(all_tad)

        results = []
        for k in range(N):
            results.append({
                'V':           float(V_arr[k]),
                '|DW|':        float(dw_arr[k]),
                'eigenvalues': eigs_arr[k],
                'is_susy':     bool(dw_arr[k] < 1e-6),
                'is_minimum':  bool(np.all(eigs_arr[k] > min_tol)),
                'Nflux':       float(tad_arr[k]),
            })
        return results


    # --------------------------------------------------------------------------
    #  optax-based first-order solver (per-candidate, eager Python loop)
    # --------------------------------------------------------------------------

    def _solve_dV_optax_single(
        self,
        x0,
        flux,
        optimiser=None,
        n_steps: int = 1000,
        tol: float = 1e-10,
        noscale: bool = True,
    ) -> Tuple[Any, float, bool]:
        r"""
        **Description:**
        Per-candidate first-order minimiser of :math:`|\nabla V|^2` using an
        ``optax`` optimiser.  Eager Python loop; ``n_steps`` updates with an
        every-50-step convergence check.

        First-order alternative to :func:`newton_method_flux_vacua`: slower
        per-iteration convergence but **robust** to bad initial conditions
        where the Hessian is singular (e.g. far from a critical point in
        non-SUSY mode).  Common pattern: warm-start with this solver for a
        few hundred steps, then hand off to Newton for the final
        quadratic-convergence polish (this is what
        ``sample_critical_points(solver='hybrid', …)`` does, C8).

        Args:
            x0 (Array-like): real-coord starting point, shape
                ``(2*(h12+1),)``.
            flux (Array-like): flux vector, shape ``(2*n_fluxes,)``.
            optimiser (optax.GradientTransformation, optional): if
                ``None`` (default), uses
                ``optax.adam(learning_rate=exponential_decay(1e-3, 200, 0.5))``.
            n_steps (int, optional): max optimisation steps.
                Default: ``1000``.
            tol (float, optional): convergence tolerance on
                :math:`\Sigma|\partial_A V|`.  Default: ``1e-10``.
            noscale (bool, optional): pass-through to :func:`dV_x`.
                Default: ``True``.

        Returns:
            ``Tuple[np.ndarray, float, bool]``:
            ``(x_final, residual, converged)`` where
            ``residual = Σ_A |∂_A V(x_final)|`` and ``converged = (residual < tol)``.

        **Raises:**

            ``ImportError`` if ``optax`` is not installed.

        **Notes:**

            * Non-finite-gradient escape (same idea as B2): if the gradient
              becomes non-finite at any iterate, the loop breaks and
              ``x_cur`` is returned at the last valid snapshot (no NaN
              propagation).
            * The default optimiser uses an exponential-decay LR schedule
              (init=1e-3, halves every 200 steps).
        """
        if not _HAS_OPTAX:
            raise ImportError(
                "optax is required for the first-order optax solver "
                "(_solve_dV_optax_single).  Install with: pip install optax"
            )

        fl_jax = jnp.array(flux)

        @jax.jit
        def loss_fn(x):
            dV = self.dV_x(x, fl_jax, noscale=noscale)
            return jnp.sum(dV ** 2)

        grad_fn = jax.jit(jax.grad(loss_fn))

        if optimiser is None:
            schedule = optax.exponential_decay(
                init_value=1e-3, transition_steps=200, decay_rate=0.5,
            )
            optimiser = optax.adam(learning_rate=schedule)

        x_cur = jnp.array(x0)
        opt_state = optimiser.init(x_cur)

        for step in range(n_steps):
            g = grad_fn(x_cur)
            if not jnp.all(jnp.isfinite(g)):
                break
            updates, opt_state = optimiser.update(g, opt_state, x_cur)
            x_cur = optax.apply_updates(x_cur, updates)

            if step % 50 == 49:
                loss = float(loss_fn(x_cur))
                if loss < tol ** 2:
                    break

        dV_final = self.dV_x(x_cur, fl_jax, noscale=noscale)
        res = float(jnp.sum(jnp.abs(dV_final)))
        return np.asarray(x_cur), res, res < tol


    # --------------------------------------------------------------------------
    #  optax-based first-order solver (batched, JIT-vmap'd via lax.scan)
    # --------------------------------------------------------------------------

    def _build_optax_kernel(
        self,
        optimiser,
        n_steps: int = 2000,
        objective: str = "dV2",
        noscale: bool = True,
    ):
        r"""
        **Description:**
        Build a JIT-compiled, vmapped optax-based minimiser kernel.

        Returned signature:
        ``(x0_batch, flux_batch) -> (x_final_batch, residual_batch)`` — fully
        vectorised over the leading batch dimension.  Uses ``lax.scan`` for
        the per-element step loop (JIT-traceable) with NaN-guarded updates.

        Used by :func:`_solve_dV_optax_batch`.

        Args:
            optimiser (optax.GradientTransformation): the optax
                optimiser (e.g. ``optax.adam(...)``).
            n_steps (int, optional): fixed scan length.  Default: ``2000``.
            objective (str, optional): loss-function variant —
                ``"dV2"`` (default), ``"log_dV2"``, or ``"V"``.  ``"V"`` finds
                minima only (descends ``V`` directly); the others find any
                critical point.
            noscale (bool, optional): pass-through to :func:`dV_x` /
                :func:`V_x`.  Default: ``True``.
        """
        if not _HAS_OPTAX:
            raise ImportError(
                "optax is required for the batched optax kernel. "
                "Install with: pip install optax"
            )

        if objective == "dV2":
            @jax.jit
            def loss_fn(x, fl):
                dV = self.dV_x(x, fl, noscale=noscale)
                return jnp.sum(dV ** 2)
        elif objective == "log_dV2":
            @jax.jit
            def loss_fn(x, fl):
                dV = self.dV_x(x, fl, noscale=noscale)
                return jnp.log(jnp.sum(dV ** 2) + 1e-30)
        elif objective == "V":
            @jax.jit
            def loss_fn(x, fl):
                return self.V_x(x, fl, noscale=noscale)
        else:
            raise ValueError(
                f"Unknown objective: {objective!r}. "
                f"Must be one of 'dV2', 'log_dV2', 'V'."
            )

        grad_fn = jax.grad(loss_fn)

        def solve_single(x0, fl):
            """Optimise one (x0, fl) pair via lax.scan with NaN guards."""
            opt_state = optimiser.init(x0)

            def step_fn(carry, _):
                x, opt_st = carry
                g = grad_fn(x, fl)
                # NaN-gradient guard: zero out non-finite gradients so the
                # step becomes a no-op (we can't break out of a lax.scan).
                g = jnp.where(jnp.isfinite(g), g, 0.0)
                updates, new_opt_st = optimiser.update(g, opt_st, x)
                new_x = optax.apply_updates(x, updates)
                # NaN-x guard: freeze the iterate at the last finite value.
                new_x = jnp.where(jnp.isfinite(new_x), new_x, x)
                return (new_x, new_opt_st), None

            (x_final, _), _ = jax.lax.scan(
                step_fn, (x0, opt_state), None, length=n_steps,
            )
            dV_final = self.dV_x(x_final, fl, noscale=noscale)
            residual = jnp.sum(jnp.abs(dV_final))
            return x_final, residual

        return jax.jit(vmap(solve_single))

    def _solve_dV_optax_batch(
        self,
        x0_batch,
        flux_batch,
        optimiser=None,
        n_steps: int = 2000,
        objective: str = "dV2",
        tol: float = 1e-10,
        sub_batch_size: int = 64,
        noscale: bool = True,
    ) -> Tuple[Any, Any, Any]:
        r"""
        **Description:**
        Vectorised first-order solver for ``∂V/∂φ = 0`` across a batch of
        candidates.  Uses ``lax.scan`` for the per-candidate step loop and
        ``vmap`` for the batch dimension, compiled into a single XLA kernel
        per sub-batch.

        Powers the ``solver="adam_v"`` and warm-start half of
        ``solver="hybrid"`` paths in :func:`sample_critical_points` (C8).

        Args:
            x0_batch (Array): starting points, shape ``(N, 2*(h12+1))``.
            flux_batch (Array): integer fluxes, shape ``(N, 2*n_fluxes)``.
            optimiser (optax.GradientTransformation, optional): default
                is ``optax.chain(clip_by_global_norm(10.0), adam(cosine_decay(0.5, n_steps)))``.
                Note this differs from :func:`_solve_dV_optax_single`'s default
                (exp-decay only).
            n_steps (int, optional): fixed scan length per candidate.
                Default: ``2000``.
            objective (str, optional): ``"dV2"`` (default),
                ``"log_dV2"``, or ``"V"``.
            tol (float, optional): convergence threshold on
                ``Σ|∂V|``.  Default: ``1e-10``.
            sub_batch_size (int, optional): chunk size for memory
                control.  Default: ``64``.
            noscale (bool, optional): pass-through.  Default: ``True``.

        Returns:
            ``Tuple[ndarray, ndarray, ndarray]``:
            ``(x_finals, residuals, converged)`` with shapes ``(N, …)``,
            ``(N,)``, ``(N,)`` (bool).
        """
        if not _HAS_OPTAX:
            raise ImportError(
                "optax is required for the batched optax solver "
                "(_solve_dV_optax_batch).  Install with: pip install optax"
            )

        if optimiser is None:
            schedule = optax.cosine_decay_schedule(
                init_value=0.5, decay_steps=n_steps,
            )
            optimiser = optax.chain(
                optax.clip_by_global_norm(10.0),
                optax.adam(learning_rate=schedule),
            )

        kernel = self._build_optax_kernel(
            optimiser, n_steps=n_steps, objective=objective, noscale=noscale,
        )

        N = len(x0_batch)
        all_x = []
        all_res = []
        for i in range(0, N, sub_batch_size):
            j = min(i + sub_batch_size, N)
            x_sub = jnp.array(x0_batch[i:j])
            fl_sub = jnp.array(flux_batch[i:j])
            x_out, res_out = kernel(x_sub, fl_sub)
            all_x.append(np.asarray(x_out))
            all_res.append(np.asarray(res_out))

        if N == 0:
            n_fields = 2 * (self.h12 + 1)
            return (np.zeros((0, n_fields)),
                    np.zeros((0,)),
                    np.zeros((0,), dtype=bool))

        x_finals  = np.concatenate(all_x,   axis=0)
        residuals = np.concatenate(all_res, axis=0)
        converged = residuals < tol
        return x_finals, residuals, converged


    # --------------------------------------------------------------------------
    #  Empirical calibration of the Gaussian-M prior σ (per ISD mode)
    # --------------------------------------------------------------------------

    def run_calibration(
        self,
        Q: int = None,
        n_sample: int = 50,
        n_test: int = 200,
        target_acceptance: float = 0.8,
        modes: list = None,
        sampler=None,
        verbose: bool = True,
    ) -> dict:
        r"""
        **Description:**
        Convenience wrapper that runs the full Gaussian-M-prior calibration
        pipeline in one call:

        1. :func:`_precompute_M_eigensystem` — eigen-decomposes the
           best-conditioned ISD matrix across ``n_sample`` moduli points.
        2. :func:`_estimate_sigmas` — analytical σ-per-mode defaults from
           the eigensystem.
        3. :func:`calibrate_priors` — empirical binary-search refinement
           to hit ``target_acceptance``.

        Roughly 1-5 s per ISD mode on h12=2 (longer for larger geometries).
        Recommended before any :func:`sample_critical_points` run with
        ``flux_prior="gaussian"`` (the default) for non-trivial Q.

        Args:
            Q (int, optional): tadpole bound for both stages.
                Default: ``self.Q()``.
            n_sample (int, optional): moduli points scanned in step 1.
                Default: ``50``.
            n_test (int, optional): candidates per σ trial in step 3.
                Default: ``200``.
            target_acceptance (float, optional): target acceptance rate
                for the binary-search.  Default: ``0.8``.
            modes (list[str], optional): ISD modes to calibrate.
                Default: ``["F", "H", "ISD+", "ISD-"]``.
            sampler (data_sampler, optional): defaults to
                ``self.sampler``.
            verbose (bool, optional): print per-step progress.

        Returns:
            ``dict[str, float]``: the calibrated σ per ISD mode (also
            stored on ``self._calibrated_sigmas``).

        **Example:**

        .. code-block:: python

            finder = jvc.FluxVacuaFinder(h12=2, model_ID=1)
            finder.run_calibration(Q=200)
            vacua = finder.sample_critical_points(
                Q=200, n_target=100, solver="hybrid",
            )
        """
        if Q is None:
            if self.Q() is None:
                raise ValueError("Q must be supplied or a global value for the D3-tadpole must be set in the FluxEFT constructor!")
            Q = int(self.Q())
        if verbose:
            print(f"[run_calibration] Q={Q}, n_sample={n_sample}, "
                  f"n_test={n_test}, target_acceptance={target_acceptance}")
        self._precompute_M_eigensystem(n_sample=n_sample, Q=Q, sampler=sampler)
        self._estimate_sigmas(Q=Q)
        if verbose:
            print(f"  analytical σ defaults: {self._calibrated_sigmas}")
        result = self.calibrate_priors(
            Q=Q, modes=modes, n_test=n_test,
            target_acceptance=target_acceptance, sampler=sampler,
            verbose=verbose,
        )
        return result

    def calibrate_priors(
        self,
        Q: int = None,
        modes=None,
        n_test: int = 200,
        target_acceptance: float = 0.8,
        sampler=None,
        verbose: bool = True,
    ) -> dict:
        r"""
        **Description:**
        Empirically refine the Gaussian-prior σ per ISD mode by binary-search.

        For each mode in ``modes``, scans σ in ``[0.1, 20.0]`` over 12 binary-
        search iterations to find the σ where the measured ISD-completion
        acceptance rate ≈ ``target_acceptance``.  "Acceptance" = fraction of
        random ``N(0, σ²I)`` inputs that, when ISD-completed via the
        sampler and tadpole-filtered against ``Q``, give a valid flux.

        Result overrides the analytical defaults from :func:`_estimate_sigmas`
        (C2).  This is the empirical refinement step: ~1–5 s per mode,
        produces model-adaptive parameters that can be cached via
        :func:`save_calibration`.

        Note: ``calibrate_priors`` ends
        with an optional "runaway diagnostic" that calls
        ``sample_critical_points`` on 50 candidates.  That is omitted here —
        the σ-calibration math is byte-for-byte equivalent; only the
        diagnostic print is dropped.  Users wanting runaway stats can run
        :func:`sample_critical_points` directly after calibration.

        Args:
            Q (int): tadpole bound (required; method-level here, per the C-cluster convention).
            modes (list[str], optional): modes to calibrate.  Default:
                ``["F", "H", "ISD+", "ISD-"]``.
            n_test (int, optional): number of candidates per σ trial.
                Default: ``200``.
            target_acceptance (float, optional): target acceptance
                rate.  Default: ``0.8``.
            sampler (data_sampler, optional): sampler instance.  If
                ``None`` (default), uses ``self.sampler``.
            verbose (bool, optional): print per-mode results.

        Returns:
            ``dict[str, float]``: copy of the calibrated σ values per mode.

        **Side effects:**

            * Writes ``self._calibrated_sigmas[mode] = σ`` for each mode.
            * Writes ``self._calibration_isd_modes = tuple(modes)`` to
              record which modes are empirically calibrated.
        """
        if Q is None:
            if self.Q() is None:
                raise ValueError("Q must be supplied or a global value for the D3-tadpole must be set in the FluxEFT constructor!")
            Q = int(self.Q())
        if sampler is None:
            sampler = self.sampler
        if modes is None:
            modes = ["F", "H", "ISD+", "ISD-"]

        mod_pts = jnp.array(sampler.get_complex_moduli(n_test), dtype=complex)
        tau_pts = jnp.array(sampler.get_complex_tau(n_test),    dtype=complex)

        def _acceptance(sigma, mode):
            rng = np.random.default_rng(42)
            inp_len = (self.n_fluxes if mode in ("F", "H")
                       else 2 * self.dimension_H3)
            inputs = np.round(rng.normal(0, sigma, (n_test, inp_len)))
            valid = 0
            for i in range(n_test):
                inp = jnp.array(inputs[i], dtype=float)
                if jnp.all(inp == 0):
                    continue
                try:
                    fl = sampler.ISD_sampling(
                        mod_pts[i], jnp.conj(mod_pts[i]),
                        tau_pts[i], jnp.conj(tau_pts[i]),
                        inp, mode=mode,
                    )
                except Exception:
                    continue
                if fl is None or not jnp.all(jnp.isfinite(fl)):
                    continue
                fl_int = np.round(np.array(jnp.array(fl).real)).astype(float)
                tad = abs(float(jnp.real(self.tadpole(jnp.array(fl_int)))))
                if 0 < tad <= Q:
                    valid += 1
            return valid / n_test

        if verbose:
            print("[calibrate_priors] Calibrating σ for each ISD mode...")

        # Initialise _calibrated_sigmas if not present
        if not hasattr(self, "_calibrated_sigmas") or self._calibrated_sigmas is None:
            self._calibrated_sigmas = {}

        for mode in modes:
            lo, hi = 0.1, 20.0
            best_sigma = self._calibrated_sigmas.get(mode, 1.0)
            best_diff  = abs(_acceptance(best_sigma, mode) - target_acceptance)
            for _ in range(12):
                mid = (lo + hi) / 2
                acc = _acceptance(mid, mode)
                diff = abs(acc - target_acceptance)
                if diff < best_diff:
                    best_diff  = diff
                    best_sigma = mid
                if acc > target_acceptance:
                    lo = mid   # σ too small → increase
                else:
                    hi = mid   # σ too large → decrease

            self._calibrated_sigmas[mode] = best_sigma
            if verbose:
                final_acc = _acceptance(best_sigma, mode)
                print(f"  {mode:>5}: σ = {best_sigma:.3f} "
                      f"(acceptance = {100*final_acc:.0f}%)")

        self._calibration_isd_modes = tuple(modes)
        return dict(self._calibrated_sigmas)


    def _get_calibration_dir(self) -> str:
        r"""
        **Description:**
        Default directory for calibration JSON files: builds
        ``jaxvacua/models/{model_type}/h12_{h12}/``.

        The directory may not exist on disk — the caller (typically
        :func:`save_calibration`) checks ``os.path.isdir`` and falls back
        to the current working directory if not.

        Returns:
            str: the canonical path
            ``jaxvacua/models/{model_type}/h12_{h12}/``.  Always returns a
            path; existence is the caller's concern.
        """
        import os
        home_dir   = os.path.dirname(os.path.realpath(__file__))
        model_type = getattr(self, "model_type", "KS")
        return os.path.join(home_dir, "models", model_type, f"h12_{self.h12}")

    def save_calibration(
        self,
        Q: int,
        path: str = None,
        flux_prior: str = "gaussian",
    ) -> str:
        r"""
        **Description:**
        Save calibrated prior parameters to a JSON file for reuse.

        Args:
            Q (int): tadpole bound the calibration is for.
                Required; used in the default
                filename and recorded in the JSON.
            path (str, optional): explicit save path.  If ``None``
                (default), uses
                ``jaxvacua/models/{model_type}/h12_{h12}/critical_points_prior_Q{Q}.json``,
                falling back to the current directory.
            flux_prior (str, optional): ``"gaussian"``,
                ``"M_weighted"``, or ``"uniform"`` — recorded in the JSON.
                Default: ``"gaussian"``.

        Returns:
            str: path to the saved file.

        **Pre-requisite:**

            ``self._calibrated_sigmas`` must be populated (via
            :func:`_estimate_sigmas` or :func:`calibrate_priors`).
        """
        import os, json
        
        fname = f"critical_points_prior_Q{Q}_{flux_prior}.json"
        if path is None:
            cal_dir = self._get_calibration_dir()
            if os.path.isdir(cal_dir):
                path = os.path.join(cal_dir, fname)
            else:
                path = fname

        sigmas = getattr(self, "_calibrated_sigmas", {}) or {}
        data = {
            "Q":           int(Q),
            "n_fluxes":       int(self.n_fluxes),
            "flux_prior":     flux_prior,
            "sigmas":         {k: float(v) for k, v in sigmas.items()},
            "M_cond":         float(self._M_cond) if getattr(self, "_M_cond", None) else None,
            "tr_Minv_median": float(getattr(self, "_tr_Minv_median", 0.0)),
            "s_min":          float(getattr(self, "_s_min", 0.0)),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def load_calibration(self, path: str) -> dict:
        r"""
        **Description:**
        Load calibrated prior parameters from a JSON file produced by
        :func:`save_calibration`.

        Sets ``self._calibrated_sigmas`` from the loaded ``"sigmas"`` block;
        also restores ``self._calibration_isd_modes`` so subsequent code
        can tell which modes were calibrated.

        Args:
            path (str): JSON file path.

        Returns:
            ``dict[str, float]``: the loaded σ values per mode (also stored
            on ``self``).
        """
        import json
        with open(path) as f:
            data = json.load(f)
        sigmas = data.get("sigmas", {}) or {}
        self._calibrated_sigmas = {k: float(v) for k, v in sigmas.items()}
        self._calibration_isd_modes = tuple(sorted(self._calibrated_sigmas.keys()))
        return dict(self._calibrated_sigmas)


    # ==========================================================================
    #  Main non-SUSY entry point: critical-point sampling
    # ==========================================================================

    def _make_critical_point_entry(
        self,
        x,
        fluxes,
        residual,
        seen: set,
        moduli_max: float = None,
        classify: bool = True,
        deduplicate: bool = True,
        map_to_fd: bool = False,
        noscale: bool = True,
    ):
        r"""
        **Description:**
        Post-process one converged Newton/optax/scipy candidate into the
        ``sample_critical_points`` result-dict form.  Centralises the
        physicality filter → FD mapping → streaming dedup → optional
        classification pipeline.  Used internally by all three solver
        branches in :func:`sample_critical_points`.

        Returns ``None`` when the candidate is filtered out (unphysical or
        seen-before duplicate); otherwise returns the per-vacuum dict
        with keys ``'flux'``, ``'moduli'``, ``'tau'``, ``'residual'`` and,
        when ``classify=True``, the six classification fields from
        :func:`classify_solution`.

        Side effect: when ``deduplicate=True`` and the candidate is new,
        its key is added to the ``seen`` set.

        Args:
            x (Array): Real-coord solution vector.
            fluxes (Array): Flux vector at the candidate.
            residual (float): Newton/optax/scipy residual at ``x``.
            seen (set): Caller-owned set of already-seen dedup keys
                (mutated when this candidate is accepted as new).
            moduli_max (float, optional): Runaway bound forwarded to
                :func:`is_physical`. Defaults to ``None``.
            classify (bool, optional): Same meaning as on
                :func:`sample_critical_points`.
            deduplicate (bool, optional): Same meaning as on
                :func:`sample_critical_points`.
            map_to_fd (bool, optional): Same meaning as on
                :func:`sample_critical_points`.
            noscale (bool, optional): Same meaning as on
                :func:`sample_critical_points`.

        Returns:
            dict | None: The entry dict, or ``None`` if filtered out.
        """
        if not self.is_physical(x, moduli_max=moduli_max):
            return None
        z_sol, _, tau_sol, _ = self._convert_real_to_complex(jnp.array(x))
        mv, tv, fv = flux_utils.map_to_fd(
            self, np.asarray(z_sol), complex(tau_sol), fluxes,
            enabled=map_to_fd,
        )
        if deduplicate:
            key = flux_utils.dedup_key(mv, tv, fv)
            if key in seen:
                return None
            seen.add(key)
        entry = {"flux": fv, "moduli": mv, "tau": tv,
                 "residual": float(residual)}
        if classify:
            entry.update(flux_utils.classify_solution(
                self, x, fluxes, noscale=noscale,
            ))
        return entry

    def sample_critical_points(
        self,
        Q: int = None,
        n_target: int = 100,
        n_batch: int = 10_000,
        max_batches: int = 20,
        isd_mode: str = "ISD-",
        solver: str = "newton",
        step_size: float = 1.0,
        newton_tol: float = 1e-10,
        newton_max_iters: int = 300,
        optax_steps: int = 1000,
        optax_objective: str = "dV2",
        optimiser=None,
        classify: bool = True,
        deduplicate: bool = True,
        noscale: bool = True,
        flux_prior: str = "gaussian",
        flux_prior_sigma: float = None,
        moduli_max: float = None,
        map_to_fd: bool = None,
        sampler=None,
        verbose: bool = True,
    ) -> list:
        r"""
        **Description:**
        Sample critical points of the scalar potential :math:`V` by drawing
        Gaussian-M-prior flux candidates, ISD-completing them, refining via
        the chosen solver, then filtering by physicality and deduplicating.

        Ties together :func:`_generate_flux_candidates` (C3), the chosen refinement
        solver, :func:`is_physical` (A3), :func:`to_fd` (A4),
        :func:`dedup_key` (A1), and :func:`classify_solution` (A2).

        **Solver dispatch** (``solver`` kwarg):

          * ``"newton"`` — FVF Newton via :func:`jax.vmap` over the
            converged batch.  Uses the B1+B2 guards (runaway escape +
            singular-Hessian protection).  Best when starting points are
            already near a critical point (e.g. SUSY-mode initialisations).
          * ``"adam"`` / ``"lbfgs"`` — per-candidate
            :func:`_solve_dV_optax_single`.  Python loop; slow but
            robust to bad initial conditions.  Useful for one-off
            debugging or when ``optax``-only convergence is desired.
          * ``"adam_v"`` — batched :func:`_solve_dV_optax_batch` with
            cosine-decay schedule + gradient clipping.  Vectorised via
            ``lax.scan`` + ``vmap``.  First-order so wider basin of
            attraction; pair with ``optax_steps`` in the high hundreds.
          * ``"hybrid"`` — adam_v warm-start (fast vmapped optax) followed
            by per-candidate Newton polish on the near-converged subset.
            **Default-recommended for large non-SUSY scans** — combines
            the robustness of optax with Newton's quadratic convergence.
          * ``"scipy"`` — per-candidate ``scipy.optimize.root(method='hybr')``
            with analytical Jacobian.  Sequential Python loop.  Robust;
            useful when the JAX-side solvers struggle.

        **Quick decision table** (typical for ``h12=2``, Q=200, n_batch≈1000):

        +-----------+-----------+-------------+------------------------------+
        | solver    | speed     | robustness  | when to use                  |
        +===========+===========+=============+==============================+
        | newton    | fastest   | moderate    | warm starts; SUSY-like inits |
        +-----------+-----------+-------------+------------------------------+
        | adam_v    | fast      | high (1st-  | cold starts; large batch     |
        |           |           | order)      |                              |
        +-----------+-----------+-------------+------------------------------+
        | hybrid    | moderate  | very high   | **default for non-SUSY**     |
        |           |           |             | sampling at scale            |
        +-----------+-----------+-------------+------------------------------+
        | scipy     | slow      | very high   | small N, debugging, when     |
        |           | (no JAX)  |             | JAX paths fail               |
        +-----------+-----------+-------------+------------------------------+
        | adam,     | slowest   | high (1st-  | single-candidate diagnostics |
        | lbfgs     | (per-cand)| order)      |                              |
        +-----------+-----------+-------------+------------------------------+

        The C-cluster convention applies: formerly-instance kwargs are
        now method-level.

        Args:
            Q (int, optional): tadpole bound. Default:
                ``self.Q()`` (the geometry's natural maximum).
            n_target (int, optional): stop once this many critical
                points are found. Default: ``100``.
            n_batch (int, optional): candidates per batch. Default:
                ``10_000``.
            max_batches (int, optional): batch cap. Default: ``20``.
            isd_mode (str, optional): one of ``"F"``, ``"H"``,
                ``"ISD+"``, ``"ISD-"``. Default: ``"ISD-"``.
            solver (str, optional): see dispatch table above.
                Default: ``"newton"``.
            step_size (float, optional): Newton step size.
                Default: ``1.0``.
            newton_tol (float, optional): convergence tolerance.
                Default: ``1e-10``.
            newton_max_iters (int, optional): Newton iteration cap.
                Default: ``300``.
            optax_steps (int, optional): step count for optax solvers.
                Default: ``1000``.
            optax_objective (str, optional): ``"dV2"``, ``"log_dV2"``,
                or ``"V"``. Default: ``"dV2"``.
            optimiser (optax.GradientTransformation, optional):
                custom optimiser overrides ``solver``.
            classify (bool, optional): attach Hessian classification.
                Default: ``True``.
            deduplicate (bool, optional): streaming dedup via
                :func:`dedup_key`. Default: ``True``.
            noscale (bool, optional): pass-through to V/dV/ddV.
                Default: ``True``.
            flux_prior (str, optional): ``"gaussian"`` (default),
                ``"M_weighted"``, or ``"uniform"``.
            flux_prior_sigma (float, optional): per-call σ override.
            moduli_max (float, optional): runaway bound for Newton's
                B1 guard. Also used post-warm-start in ``solver="hybrid"``.
            map_to_fd (bool, optional): if ``None`` (default), uses
                ``self._map_to_fd``. Explicit ``True``/``False`` overrides.
            sampler (data_sampler, optional): defaults to
                ``self.sampler``.
            verbose (bool, optional): print per-batch + final summary.
                Default: ``True``.

        Returns:
            ``list[dict]``: one dict per surviving critical point with keys
            ``flux``, ``moduli``, ``tau``, ``residual``, and (when
            ``classify=True``) ``V``, ``|DW|``, ``eigenvalues``,
            ``is_susy``, ``is_minimum``, ``Nflux``.
        """
        if Q is None:
            if self.Q() is None:
                raise ValueError("Q must be supplied or a global value for the D3-tadpole must be set in the FluxEFT constructor!")
            Q = int(self.Q())
        if sampler is None:
            sampler = self.sampler
        if map_to_fd is None:
            map_to_fd = getattr(self, "_map_to_fd", False)

        t0 = time.perf_counter()
        def _elapsed():
            s = time.perf_counter() - t0
            if s < 120: return f"{s:.1f}s"
            if s < 3600: return f"{s/60:.1f}m"
            h, m = int(s // 3600), int((s % 3600) // 60)
            return f"{h}h {m}m"

        if verbose:
            ns_label = "no-scale" if noscale else "full SUGRA"
            print(f"[sample_critical_points] Searching for critical points of V ({ns_label})")
            print(f"  Q={Q}, mode={isd_mode}, solver={solver}"
                  + (f", moduli_max={moduli_max}" if moduli_max is not None else ""))

        results = []
        seen = set()
        n_tried = 0
        n_valid = 0
        n_runaway_total = 0
        rng = np.random.default_rng()

        # Build optax optimiser for adam/lbfgs paths (other paths build their own)
        optax_opt = optimiser
        if solver == "adam" and optax_opt is None and _HAS_OPTAX:
            schedule = optax.exponential_decay(
                init_value=1e-3, transition_steps=200, decay_rate=0.5,
            )
            optax_opt = optax.adam(learning_rate=schedule)
        elif solver == "lbfgs" and optax_opt is None and _HAS_OPTAX:
            optax_opt = optax.lbfgs()

        # Pre-build the vmapped Newton solver if using 'newton'. step_size/tol/max_iters/moduli_max are
        # closed over as compile-time constants.
        vmap_newton = None
        if solver == "newton":
            def _newton_one(m, t, f):
                return self.newton_method_flux_vacua(
                    m, t, f, mode=None,
                    step_size_Newton=step_size, tol=newton_tol,
                    max_iters=newton_max_iters,
                    solver_mode="real", moduli_max=moduli_max,
                )
            vmap_newton = jax.vmap(_newton_one, in_axes=(0, 0, 0))

        for batch_idx in range(max_batches):
            if len(results) >= n_target:
                break

            # Starting points
            mod_pts = jnp.array(sampler.get_complex_moduli(n_batch), dtype=complex)
            tau_pts = jnp.array(sampler.get_complex_tau(n_batch),    dtype=complex)

            # C3: Gaussian-M-prior flux candidates
            x0_arr, flux_arr, _ = self._generate_flux_candidates(
                n_batch, mod_pts, tau_pts, isd_mode=isd_mode, rng=rng,
                Q=Q, flux_prior=flux_prior,
                flux_prior_sigma=flux_prior_sigma, sampler=sampler,
            )
            n_tried += n_batch
            n_valid += len(x0_arr)
            if len(x0_arr) == 0:
                if verbose:
                    print(f"  Batch {batch_idx+1}: 0 valid flux candidates  [{_elapsed()}]")
                continue

            batch_results = []
            n_runaway_batch = 0

            if solver in ("adam_v", "hybrid"):
                # --- C6 batched optax path ---
                x_finals, residuals, conv_arr = self._solve_dV_optax_batch(
                    x0_arr, flux_arr,
                    optimiser=optax_opt, n_steps=optax_steps,
                    objective=optax_objective, tol=newton_tol, noscale=noscale,
                )
                x_finals = np.asarray(x_finals); residuals = np.asarray(residuals); conv_arr = np.asarray(conv_arr)

                if solver == "hybrid":
                    # Filter runaways before Newton refinement
                    if moduli_max is not None:
                        max_abs = np.max(np.abs(x_finals), axis=1)
                        runaway = max_abs > moduli_max
                        n_runaway_batch = int(runaway.sum())
                        conv_arr = conv_arr & (~runaway)
                        residuals = np.where(runaway, np.inf, residuals)

                    # Phase 2: per-candidate FVF Newton polish on near-converged
                    # (sequential is faster than vmapped fixed-iter Newton on CPU
                    # because median convergence is ~14 iters).
                    near_mask = (residuals < 1.0) & (~conv_arr)
                    for j in np.where(near_mask)[0]:
                        z0_j, _, t0_j, _ = self._convert_real_to_complex(jnp.array(x_finals[j]))
                        mod_w, tau_w, res_w = self.newton_method_flux_vacua(
                            z0_j, t0_j, jnp.array(flux_arr[j]), mode=None,
                            step_size_Newton=step_size, tol=newton_tol,
                            max_iters=newton_max_iters,
                            solver_mode="real", moduli_max=moduli_max,
                        )
                        res_w = float(jnp.abs(res_w))
                        if res_w < newton_tol:
                            x_w = np.asarray(self._convert_complex_to_real(
                                mod_w, jnp.conj(mod_w), tau_w, jnp.conj(tau_w),
                            ))
                            x_finals[j] = x_w
                            residuals[j] = res_w
                            conv_arr[j] = True

                # Filter converged → physical → dedup → classify
                for j in np.where(conv_arr)[0]:
                    entry = self._make_critical_point_entry(
                        x_finals[j], flux_arr[j], residuals[j], seen,
                        moduli_max=moduli_max, classify=classify,
                        deduplicate=deduplicate, map_to_fd=map_to_fd,
                        noscale=noscale,
                    )
                    if entry is not None:
                        batch_results.append(entry)

            elif solver == "newton":
                # --- Vectorised FVF Newton path (D1-D3's replacement) ---
                x0_j = jnp.asarray(x0_arr)
                mod_batch = jax.vmap(lambda x: self._convert_real_to_complex(x)[0])(x0_j)
                tau_batch = jax.vmap(lambda x: self._convert_real_to_complex(x)[2])(x0_j)
                fl_batch  = jnp.asarray(flux_arr)

                mod_finals, tau_finals, residuals_j = vmap_newton(
                    mod_batch, tau_batch, fl_batch,
                )
                residuals = np.asarray(jnp.abs(residuals_j))
                conv_arr  = residuals < newton_tol

                # Convert back to real coords for is_physical / classify
                x_finals = np.asarray(jax.vmap(
                    lambda m, t: self._convert_complex_to_real(m, jnp.conj(m), t, jnp.conj(t))
                )(mod_finals, tau_finals))

                for j in np.where(conv_arr)[0]:
                    entry = self._make_critical_point_entry(
                        x_finals[j], flux_arr[j], residuals[j], seen,
                        moduli_max=moduli_max, classify=classify,
                        deduplicate=deduplicate, map_to_fd=map_to_fd,
                        noscale=noscale,
                    )
                    if entry is not None:
                        batch_results.append(entry)

            else:
                # --- Per-candidate Python loop (adam / lbfgs / scipy / custom optimiser) ---
                for j in range(len(x0_arr)):
                    if solver in ("adam", "lbfgs") or optimiser is not None:
                        x_sol, res, conv = self._solve_dV_optax_single(
                            x0_arr[j], flux_arr[j],
                            optimiser=optax_opt, n_steps=optax_steps,
                            tol=newton_tol, noscale=noscale,
                        )
                    elif solver == "scipy":
                        from scipy.optimize import root as scipy_root
                        fl_np = flux_arr[j]
                        def f(x):
                            return np.asarray(self.dV_x(jnp.array(x), jnp.array(fl_np), noscale=noscale))
                        def jac(x):
                            return np.asarray(self.ddV_x(jnp.array(x), jnp.array(fl_np), noscale=noscale))
                        res_sp = scipy_root(f, x0=x0_arr[j], method="hybr", jac=jac)
                        x_sol = res_sp.x
                        res = max(abs(res_sp.fun)) if res_sp.success else float("inf")
                        conv = res_sp.success and res < newton_tol
                    else:
                        raise ValueError(f"Unknown solver: {solver}")

                    if not conv:
                        continue
                    entry = self._make_critical_point_entry(
                        x_sol, flux_arr[j], res, seen,
                        moduli_max=moduli_max, classify=classify,
                        deduplicate=deduplicate, map_to_fd=map_to_fd,
                        noscale=noscale,
                    )
                    if entry is not None:
                        batch_results.append(entry)

            results.extend(batch_results)

            if verbose:
                n_susy = sum(1 for r in batch_results if r.get("is_susy", False))
                n_ns   = len(batch_results) - n_susy
                n_min  = sum(1 for r in batch_results if r.get("is_minimum", False))
                run_str = f", {n_runaway_batch} runaways" if n_runaway_batch > 0 else ""
                print(
                    f"  Batch {batch_idx+1}: {len(x0_arr)} candidates → "
                    f"{len(batch_results)} critical pts "
                    f"({n_susy} SUSY, {n_ns} non-SUSY, {n_min} minima{run_str}) "
                    f"| total: {len(results)}  [{_elapsed()}]"
                )

            n_runaway_total += n_runaway_batch

        if verbose:
            n_susy_tot = sum(1 for r in results if r.get("is_susy", False))
            n_ns_tot   = len(results) - n_susy_tot
            n_min_tot  = sum(1 for r in results if r.get("is_minimum", False))
            n_sad_tot  = len(results) - n_min_tot
            run_str = (f", {n_runaway_total} runaways filtered"
                       if n_runaway_total > 0 else "")
            print(
                f"\n[sample_critical_points] Done: {len(results)} critical points "
                f"({n_susy_tot} SUSY, {n_ns_tot} non-SUSY, "
                f"{n_min_tot} minima, {n_sad_tot} saddle{run_str}) "
                f"from {n_valid} valid candidates ({n_tried} tried)  [{_elapsed()}]"
            )

        return results


unflatten_func = lambda aux_data, children: unflatten_func_class(aux_data, children, FluxVacuaFinder)

register_pytree_node(FluxVacuaFinder, flatten_func, unflatten_func)
