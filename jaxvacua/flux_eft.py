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

"""Flux effective field theory for Type IIB compactifications.

Purpose
-------
Define ``FluxEFT``, the flux-physics layer above ``css``.  It evaluates the
Gukov-Vafa-Witten superpotential, F-terms, scalar potential, tadpole and
stability data for integer three-form flux backgrounds.

Main public API
---------------
- ``FluxEFT``: extends ``css`` with flux splitting, superpotential,
  Kähler-covariant derivatives, real-coordinate derivatives, Hessians and mass
  matrices.
- Tadpole, ISD and physicality helpers used by the vacuum-search algorithms.
- Fundamental-domain and monodromy utilities used to compare equivalent
  solutions.

Design notes
------------
This module contains the reusable EFT model.  Search, sampling and
classification workflows live in ``jaxvacua.flux_vacua_finder`` and operate
directly on this class through inheritance.
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
from jax.tree_util import register_pytree_node


# Enable 64 bit precision
# JAXVacua custom imports
from .util import *
from .css import css

class FluxEFT(css):
    r"""
    **Description:**
    A class representing the flux sector in a 4D EFT obtained from Type IIB
    compactification on orientifolds of CY threefolds with 3-form flux backgrounds.
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
        prepotential_input: Callable | None = None,
        Q: int | None = None,
        gauge_choice: complex = 1.0 + 0.0j,
        prange: int = 5,
        use_gvs: bool = True,
        save_file: bool = False,
        **kwargs: Any,
    ) -> None:
        r"""

        **Description:**
        A class representing the flux sector in a 4D EFT obtained from Type IIB 
        compactification on CY threefolds with 3-form flux backgrounds.
        
        Args:
            h12 (int): The number of moduli for the compactified geometry.
            model_type (string): The type of manifold considered for the compactification. Currently, ``"KS"`` and ``"CICY"`` are available.
            model_ID (int): ID specifying a certain model.
            limit (string): String identifying the type of periods to be considered. Currently, only ``"LCS"`` is available.
            model_data (dictionary): Contains model data like triple intersection numbers etc.
            instanton_data (list): List of GV and GW invariants.
            maximum_degree (int): Maximum degree used for the instanton sum.
            use_cytools (boolean): Whether or not to use CYTools to compute topological data of Calabi-Yau.
            mirror_cy (cytools.CalabiYau): Mirror Calabi-Yau threefold.
            basis_change (Array): Basis transformation to be applied to topological data of Calabi-Yau.
            grading_vector (Array): Grading vector to be used for the GV computation.
            period_input (function): Input for periods.
            prepotential_input (function): Input for prepotential.
            gauge_choice (complex, optional): Gauge choice parameter.
            save_file (bool, optional): Save files for new models. Defaults to ``False``.
            **kwargs: Additional keyword arguments.
            
        Attributes:
            D3_tadpole (int): Maximum allowed D3-tadpole.
            n_fluxes (int): Number of fluxes.
            
        """

        # Initialise flux sector as subclass of complex structure sector!
        super(FluxEFT, self).__init__(
            h12=h12,
            model_ID=model_ID,
            model_type=model_type,
            limit=limit,
            maximum_degree=maximum_degree,
            mirror_cy=mirror_cy,
            model_data=model_data,
            lcs_tree_input = lcs_tree_input,
            model_file = model_file,
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
        
        if Q is None:
            warnings.warn("Tadpole was set to the maximum possible value, which may not be the true D3-tadpole for the orientifold!")
            self.D3_tadpole = self.lcs_tree.h11+self.lcs_tree.h12+2
        else:
            self.D3_tadpole = Q

        self.n_fluxes = self._dimension_H3_tot
        self.axion_fd = kwargs.get("axion_fd", (-0.5, 0.5))
        
        

    def __repr__(self) -> str:
        r"""
        **Description:**
        Returns a string representation of the flux sector class.
        
        Returns:
            str: Description of the class.
        """
        return f"Flux sector with h12={self.h12} complex structure moduli in the {self.periods.limit} limit."
            
    def map_to_fd_tau(
        self, 
        tau: complex, 
        fluxes: Array,
        return_SL2Z_matrix: bool = False, 
        cutoff: float = 1e15,
        verbose: bool = False
        ):
        r"""
        
        **Description:**
        Map of the axio-dilaton value and the flux vector to the fundamental domain (FD) of :math:`\text{SL}(2,\mathbb{Z})`.
        
        .. admonition:: Details
            :class: dropdown
        
            The axio-dilaton transforms as
        
            .. math::
                \tau \rightarrow \dfrac{a\tau+b}{c\tau+d}\; ,\quad ad-bc=1\; ,\quad \Sigma = \left (\begin{array}{cc}a & b \ c & d \end{array} \right )\in \text{SL}(2,\mathbb{Z})\, .
        
            The flux vector of RR-flux :math:`f` and NSNS-flux :math:`h` transform as
        
            .. math::
                \left (\begin{array}{c}f \ h \end{array} \right )\rightarrow\left (\begin{array}{c}a\, f+b\, h \\ c\, f+d\, h\end{array} \right )\, .
        
            Given a value :math:`\tau_0` for the axio-dilaton, we apply translations and S-duality transformations
        
            .. math::
                 \tau_0 \rightarrow \tau_0+b\; ,\quad \tau_0 \rightarrow -\dfrac{1}{\tau_0}\, ,
        
            until we find a value of :math:`\tau` in the fundamental domain :math:`\mathcal{F}_{\text{SL}(2,\mathbb{Z})}` of :math:`\text{SL}(2,\mathbb{Z})` given by
        
            .. math::
                \mathcal{F}_{\text{SL}(2,\mathbb{Z})} = \lbrace \tau\in\mathbb{C}:\, |\text{Re}(\tau)|\leq 0.5\, ,\, |\tau|\leq 1\rbrace \, .
        
        
        .. note::
            Per default, this function only returns the value of the axio-dilaton and the fluxes. It is however also possible to return the :math:`\text{SL}(2,\mathbb{Z})` transformation taking the input value :math:`\tau` of the axio-dilaton to the fundamental domain.
        
        Args:
            tau (complex): Axio-dilaton value.
            fluxes (Array): The flux vector used in the mapping.
            return_SL2Z_matrix (bool, optional): If ``True``, also returns the
                :math:`\text{SL}(2,\mathbb{Z})` matrix. Defaults to ``False``.
            cutoff (float, optional): Safety cutoff for the iteration count.
                Defaults to ``1e15``.
            verbose (bool, optional): If ``True``, enables verbose mode.
                Defaults to ``False``.

        Returns:
            Tuple: ``(tau_fd, fluxes_fd)`` or ``(tau_fd, fluxes_fd, SL2Z_matrix)``
                if ``return_SL2Z_matrix=True``. Returns ``(None, None)`` if the
                input is invalid.

        """
        
        

        # --- input validation ------------------------------------------------

        # helper: emit a warning and return the appropriate None tuple
        def _fail(msg):
            """Emit a warning and return a None tuple for early exit."""
            warnings.warn(msg)
            if return_SL2Z_matrix:
                return None, None, None
            return None, None

        # Check fluxes length
        expected_len = 2 * self.n_fluxes
        if len(fluxes) != expected_len:
            return _fail(
                f"map_to_fd_tau: fluxes has length {len(fluxes)}, "
                f"expected {expected_len}."
            )

        # Convert tau to a plain Python complex so that all subsequent scalar
        # arithmetic uses Python/NumPy floats instead of JAX-traced values.
        # This prevents the "Cannot convert float NaN to integer" crash and
        # avoids JAX overhead for purely scalar work.
        try:
            tau_py = complex(tau)
        except (TypeError, ValueError) as exc:
            return _fail(f"map_to_fd_tau: tau cannot be cast to complex: {exc}.")

        tau1 = tau_py.real
        tau2 = tau_py.imag

        # Guard against non-finite input (NaN, ±inf)
        if not (jnp.isfinite(tau1) and jnp.isfinite(tau2)):
            return _fail(
                f"map_to_fd_tau: tau is not finite "
                f"(Re={tau1}, Im={tau2}). Skipping."
            )

        # SL(2,Z) action requires Im(tau) > 0
        if tau2 <= 0.0:
            return _fail(
                f"map_to_fd_tau: Im(tau) = {tau2} is non-positive. "
                "SL(2,Z) fundamental domain requires Im(tau) > 0."
            )

        # --- initialise ------------------------------------------------------

        if return_SL2Z_matrix:
            sig = jnp.array([[1, 0], [0, 1]])

        # Early exit if already in the fundamental domain
        if abs(tau1) <= 0.5 and abs(tau_py) >= 1.0:
            if return_SL2Z_matrix:
                return tau_py, jnp.asarray(fluxes), sig
            return tau_py, jnp.asarray(fluxes)

        FFlux = jnp.array(fluxes[: self.n_fluxes], dtype=float)
        HFlux = jnp.array(fluxes[self.n_fluxes :], dtype=float)

        count = 0
        end_loop = False

        # --- main reduction loop ----------------------------------------------

        while not end_loop:

            # safety: detect divergence or non-convergence
            if count > 10_000:
                return _fail(
                    f"map_to_fd_tau: did not converge after {count} iterations. "
                    "Stopping."
                )

            if not (jnp.isfinite(tau1) and jnp.isfinite(tau2)):
                return _fail(
                    f"map_to_fd_tau: tau became non-finite "
                    f"(Re={tau1}, Im={tau2}) after {count} iterations."
                )

            if tau1 > cutoff or tau2 > cutoff:
                return _fail(
                    f"map_to_fd_tau: tau components exceeded cutoff {cutoff} "
                    f"after {count} iterations."
                )

            if tau2 < 0.0:
                return _fail(
                    f"map_to_fd_tau: Im(tau) became negative ({tau2}) "
                    f"after {count} iterations."
                )

            # T-transformation: shift real part into (-1, 0] or [0, 1)
            temp1 = int(jnp.floor(tau1))   # safe: tau1 is a finite Python float

            if tau1 < 0.0:
                p0 = float(jnp.fmod(tau1, 1.0)) + 1.0
            else:
                p0 = float(jnp.fmod(tau1, 1.0))
            p1 = tau2

            FFlux = FFlux - temp1 * HFlux

            if return_SL2Z_matrix:
                #b=-temp1
                #a=1
                #c=0
                #d=1
                
                sig = jnp.matmul(jnp.array([[1, -temp1], [0, 1]]), sig)
                if verbose:
                    print("Sigma 1:", sig)

            # additional half-unit shift if |Re| > 0.5
            if jnp.abs(p0) > 0.5:
                temp2 = int(jnp.sign(p0))
                p0 = p0 - temp2
                FFlux = FFlux - temp2 * HFlux

                if return_SL2Z_matrix:
                    #b=-temp2
                    #a=1
                    #c=0
                    #d=1
                    sig = jnp.matmul(jnp.array([[1, -temp2], [0, 1]]), sig)
                    if verbose:
                        print("Sigma 2:", sig)

            # S-transformation: invert if |tau| < 1
            norm_p_sq = p0 * p0 + p1 * p1
            if jnp.sqrt(norm_p_sq) < 1.0:
                if norm_p_sq == 0.0:
                    return _fail(
                        f"map_to_fd_tau: |tau| = 0 encountered after "
                        f"{count} iterations; inversion undefined."
                    )
                FFlux_old = FFlux.copy()
                FFlux = HFlux.copy()
                HFlux = -FFlux_old
                p0 = -p0 / norm_p_sq
                p1 = p1 / norm_p_sq

                if return_SL2Z_matrix:
                    #a=0
                    #b=1
                    #c=-1
                    #d=0
                    sig = jnp.matmul(jnp.array([[0, 1], [-1, 0]]), sig)
                    if verbose:
                        print("Sigma 3:", sig)

            tau1 = p0
            tau2 = p1
            count += 1

            if jnp.abs(tau1) <= 0.5 and jnp.sqrt(tau1 * tau1 + tau2 * tau2) >= 1.0:
                end_loop = True

        # --- assemble output --------------------------------------------------

        fluxes_out = jnp.hstack((FFlux, HFlux))
        tau_out = tau1 + 1j * tau2

        if return_SL2Z_matrix:
            return tau_out, fluxes_out, sig
        return tau_out, fluxes_out


    def map_to_fd(
        self,
        moduli: Array,
        tau: complex,
        fluxes: Array,
        axion_fd: Optional[Tuple[float, float]] = None,
        boundary_tol: float = 1e-8,
    ) -> Tuple[Array, complex, Array]:
        r"""
        **Description:**
        Maps a vacuum solution ``(moduli, tau, fluxes)`` to the fundamental domain by:

        1. Mapping the axio-dilaton :math:`\tau` to the :math:`\text{SL}(2,\mathbb{Z})`
           fundamental domain via :func:`map_to_fd_tau`.
        2. Shifting the axion :math:`\text{Re}(z^a)` into the range ``(lo, hi]`` via
           integer monodromy transformations :math:`z^a \to z^a + n^a`.
        3. Snapping values within ``boundary_tol`` of the excluded boundary ``lo``
           to the included boundary ``hi``.

        For conifold models (``coniLCS_series`` or ``coniLCS_bulk``), the conifold
        modulus (index 0 in the conifold basis) is left untouched because
        the logarithmic monodromy breaks integer periodicity.

        Args:
            moduli (Array): Complex structure moduli values.
            tau (complex): Axio-dilaton value.
            fluxes (Array): Flux vector.
            axion_fd (tuple, optional): Range ``(lo, hi)`` for :math:`\text{Re}(z^a)`.
                Convention: :math:`\text{Re}(z^a) \in (lo, hi]`.
                If ``None``, uses ``self.axion_fd`` (default ``(-0.5, 0.5)``).
            boundary_tol (float): Points within this tolerance of the excluded
                boundary ``lo`` are snapped to ``hi``. Default ``1e-8``.

        Returns:
            Tuple[Array, complex, Array]: ``(moduli_fd, tau_fd, fluxes_fd)``
                mapped to the fundamental domain.

        See also: :func:`map_to_fd_tau`, :func:`monodromy_matrix`
        """

        lo, hi = axion_fd if axion_fd is not None else self.axion_fd

        # Step 1: Map tau to SL(2,Z) fundamental domain
        result_tau = self.map_to_fd_tau(tau, fluxes)
        if result_tau[0] is None:
            return moduli, tau, fluxes
        tau_fd = result_tau[0]
        fluxes_fd = result_tau[1]

        # Step 2: Snap Re(tau) = -0.5 boundary to +0.5 via T-transformation
        if jnp.abs(tau_fd.real - (-0.5)) < boundary_tol:
            H = fluxes_fd[self.n_fluxes:]
            fluxes_fd = jnp.append(fluxes_fd[:self.n_fluxes] + H, H).astype(int)
            tau_fd = tau_fd + 1

        # Step 3: Map Re(z^a) into (lo, hi] via monodromy
        # For coniLCS_series/bulk: the conifold log term breaks integer periodicity.
        # With conifold_basis=True, the conifold modulus is at index 0 — skip it.
        # With conifold_basis=False, the conifold modulus is a linear combination
        # of coordinate moduli, so we cannot safely apply monodromy to any direction.
        _is_conifold_limit = self.periods.limit in ("coniLCS_series", "coniLCS_bulk")
        _conifold_basis = getattr(self.lcs_tree, 'conifold_basis', True)

        n = -jnp.ceil(moduli.real - hi).astype(int)
        if _is_conifold_limit:
            if not _conifold_basis:
                import warnings
                warnings.warn(
                    "map_to_fd: conifold_basis=False — cannot identify the conifold "
                    "direction in coordinate basis. Skipping monodromy shifts for all "
                    "moduli. Set conifold_basis=True for proper FD mapping."
                )
                n = jnp.zeros_like(n)
            else:
                n = n.at[0].set(0)
        moduli_fd, fluxes_fd = self.apply_monodromy(moduli, fluxes_fd, n)

        # Step 4: Snap Re(z) near lo to hi (boundary identification)
        near_lo = jnp.abs(moduli_fd.real - lo) < boundary_tol
        if _is_conifold_limit:
            if not _conifold_basis:
                near_lo = jnp.zeros_like(near_lo)
            else:
                near_lo = near_lo.at[0].set(False)
        if jnp.any(near_lo):
            n_snap = jnp.where(near_lo, 1, 0)
            _, fluxes_fd = self.apply_monodromy(moduli_fd, fluxes_fd, n_snap)
            # Set Re(z) exactly to hi
            moduli_fd = jnp.where(near_lo, hi + 1j * moduli_fd.imag, moduli_fd)

        return moduli_fd, tau_fd, fluxes_fd


    def apply_monodromy(self, moduli: Array, fluxes: Array, n) -> Tuple[Array, Array]:
        r"""
        **Description:**
        Apply a monodromy shift :math:`z^a \to z^a + n^a` to the moduli and
        transform the fluxes accordingly via the monodromy matrix :math:`M(n)`.

        The RR-fluxes :math:`f` and NSNS-fluxes :math:`h` both transform as
        :math:`f \to M(n) \cdot f`, :math:`h \to M(n) \cdot h`.

        Args:
            moduli (Array): Complex structure moduli.
            fluxes (Array): Flux vector ``[f | h]``.
            n (array-like): Integer shift vector of shape ``(h12,)``.

        Returns:
            Tuple[Array, Array]: ``(moduli_shifted, fluxes_shifted)``
        """
        M = self.monodromy_matrix(n)
        F_new = jnp.matmul(M, fluxes[:self.n_fluxes])
        H_new = jnp.matmul(M, fluxes[self.n_fluxes:])
        moduli_new = moduli + jnp.array(n, dtype=moduli.dtype)
        fluxes_new = jnp.append(F_new, H_new).astype(int)
        return moduli_new, fluxes_new


    @partial(jit, static_argnums = ())
    def _split_fluxes(
                    self,
                    fluxes: Array
                    ) -> Tuple[Array,Array,Array,Array]:
        r"""
        **Description:**
        Split flux vector into individual components.
        
        Args:
            fluxes (Array): Flux vector.
        
        Returns:
            Array: Flux vector :math:`f_1`.
            Array: Flux vector :math:`f_2`.
            Array: Flux vector :math:`h_1`.
            Array: Flux vector :math:`h_2`.
        
        """

        f = fluxes[:self.n_fluxes] # Extract the first part of the flux vector, which contains the RR-fluxes.
        h = fluxes[self.n_fluxes:] # Extract the second part of the flux vector, which contains the NSNS-fluxes.
        f1 = f[:self.dimension_H3] # Assign the first set of fluxes (f1) from the RR-fluxes, limited to the dimension of H3.
        f2 = f[self.dimension_H3:] # Assign the second set of fluxes (f2) from the RR-fluxes, starting from the dimension of H3.
        h1 = h[:self.dimension_H3] # Assign the first set of fluxes (h1) from the NSNS-fluxes, limited to the dimension of H3.
        h2 = h[self.dimension_H3:] # Assign the second set of fluxes (h2) from the NSNS-fluxes, starting from the dimension of H3.

        return f1,f2,h1,h2 # Return the split flux components as a tuple.



    @partial(jit, static_argnums = ())
    def _convert_real_to_complex(
        self, x: Array
        ) -> float:
        r"""
        
        **Description:**
        Converts the real field components to complex values of the complex structure moduli :math:`z^{i}`
        and the axio-dilaton :math:`\tau`.

        .. note::
            The input shape should be

            .. math:
                \bigl (\mathrm{Re}(z^{1}),\mathrm{Im}(z^{1}),\ldots , \mathrm{Re}(z^{h^{1,2}}),
                \mathrm{Im}(z^{h^{1,2}}), \mathrm{Re}(\tau),\mathrm{Im}(\tau) \bigl )\, .
        
        
        Args:
            x (Array): Array of shape (:math:`2(h^{1,2}+1)`, ) containing real and imaginary parts of
                the complex structure moduli :math:`z^i` and axio-dilaton :math:`\tau`.
        
        Returns:
            Array: Complex structure moduli values.
            Array: Complex conjugate values for complex structure moduli.
            complex: Axio-dilaton value.
            complex: Complex conjugate value for axio-dilaton.
        
        """
        moduli = x[0:-2:2]+1.j*x[1:-2:2]
        moduli_c = x[0:-2:2]-1.j*x[1:-2:2]
        tau = x[-2]+1.j*x[-1]
        tau_c = x[-2]-1.j*x[-1]

        return moduli,moduli_c,tau,tau_c

    @partial(jit, static_argnums = ())
    def _convert_complex_to_real(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex
        ) -> float:
        r"""
        
        **Description:**
        Converts the complex values of the complex structure moduli :math:`z^{i}`
        and the axio-dilaton :math:`\tau` to real components.

        .. note::
            The output shape is 

            .. math:
                \bigl (\mathrm{Re}(z^{1}),\mathrm{Im}(z^{1}),\ldots , \mathrm{Re}(z^{h^{1,2}}),
                \mathrm{Im}(z^{h^{1,2}}), \mathrm{Re}(\tau),\mathrm{Im}(\tau) \bigl )\, .
        
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
        
        Returns:
            Array: Array of shape (:math:`2(h^{1,2}+1)`, ) containing real and imaginary parts of
                the complex structure moduli :math:`z^i` and axio-dilaton :math:`\tau`.
            
        
        """
        real_z = jnp.real((moduli+moduli_c)/2.)
        imag_z = jnp.real(-1j*(moduli-moduli_c)/2.)
        real_tau = jnp.real((tau+tau_c)/2.)
        imag_tau = jnp.real(-1j*(tau-tau_c)/2.)

        x = jnp.append(jnp.array([real_z]),jnp.array([imag_z]),axis=0).T.flatten()

        return jnp.append(x,jnp.array([real_tau,imag_tau])).astype(jnp.float_)

    @partial(jit, static_argnums = ())
    def _convert_complex_to_real_nondif(
        self, moduli: Array, tau: complex
        ) -> float:
        r"""

        **Description:**
        Converts the complex values of the complex structure moduli :math:`z^{i}`
        and the axio-dilaton :math:`\tau` to real components.

        .. note::
            The output shape is

            .. math:
                \bigl (\mathrm{Re}(z^{1}),\mathrm{Im}(z^{1}),\ldots , \mathrm{Re}(z^{h^{1,2}}),
                \mathrm{Im}(z^{h^{1,2}}), \mathrm{Re}(\tau),\mathrm{Im}(\tau) \bigl )\, .


        Args:
            moduli (Array): Complex structure moduli values.
            tau (complex): Axio-dilaton value.

        Returns:
            Array: Array of shape (:math:`2(h^{1,2}+1)`, ) containing real and imaginary parts of
                the complex structure moduli :math:`z^i` and axio-dilaton :math:`\tau`.


        """
        moduli_c = jnp.conj(moduli)
        tau_c = jnp.conj(tau)
        real_z = jnp.real((moduli+moduli_c)/2.)
        imag_z = jnp.real(-1j*(moduli-moduli_c)/2.)
        real_tau = jnp.real((tau+tau_c)/2.)
        imag_tau = jnp.real(-1j*(tau-tau_c)/2.)

        x = jnp.append(jnp.array([real_z]),jnp.array([imag_z]),axis=0).T.flatten()

        return jnp.append(x,jnp.array([real_tau,imag_tau])).astype(jnp.float_)


    @partial(jit, static_argnums = ())
    def tadpole(
        self, fluxes: Array
        ) -> float:
        r"""
        
        **Description:**
        Calculates the D3-charge for given fluxes.
        
        .. admonition:: Details
            :class: dropdown
        
            The D3-charge induced by fluxes is defined as

            .. math::
                    N_{\mathrm{flux}} = \vec{f}\Sigma \vec{h}\, ,

            where :math`\vec{f}, \vec{h}` are the NSNS- and RR-fluxes respectively.
            The symplectic pairing :math:`\Sigma` is implemented in the periods class :func:`jaxvacua.periods.periods`.

        Args:
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
        
        Returns:
            int: D3-charge induced by 3-form fluxes.
        
        """
        
        FF = fluxes[:self.n_fluxes]
        HF = fluxes[self.n_fluxes:]
        
        return jnp.matmul(FF, jnp.matmul(self.periods.sigma, HF))
            

    ###################################################################################################################
    ############################################ GVW SUPERPOTENTIAL ###################################################
    ###################################################################################################################

    
    @partial(jit, static_argnums = (4,5,))
    def superpotential(
        self, moduli: Array, tau: complex, fluxes: Array, conj: bool = False, normalise: bool = False
        ) -> complex:
        r"""
        
        **Description:**
        Calculates the value of the superpotential for given flux, moduli and axio-dilaton.
        
        
        .. admonition:: Details
            :class: dropdown
        
            For a Calabi-Yau manifold :math:`X`, the Gukov-Vafa-Witten superpotential `hep-th/9906070 <https://inspirehep.net/literature/501505>`_ is defined in terms of the complex 3-form flux  :math:`G_3=F_3-\tau H_3` and the holomorphic 3-form  :math:`\Omega_3` as
        
            .. math::
                    W(Z,\tau) = \int_{X} \, G_3\wedge \Omega_3 = (\vec{f}-\tau\vec{h})^T\cdot\Sigma\cdot\Pi(Z)
        
        
            following the conventions of eq. (4) in `1912.10047 <https://inspirehep.net/literature/1772253>`_.
            The period vector :math:`\Pi` and the symplectic pairing :math:`\Sigma` are defined in the periods class :func:`jaxvacua.periods.periods`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            tau (complex): Axio-dilaton value.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            normalise (bool, optional): If ``True``, rescales superpotential by :math:`\sqrt{2/\pi}`. Defaults to ``False``.
        
        Returns:
            complex: Value of the superpotential.
        
        """

        FFlux = fluxes[:self.n_fluxes]
        HFlux = fluxes[self.n_fluxes:]

        W0 = jnp.matmul((FFlux-tau*HFlux),jnp.matmul(self.periods.sigma, self.period_vector(moduli,conj=conj)))

        if normalise:
            W0 = jnp.sqrt(2./jnp.pi)*W0
            
        W0 += self.lcs_tree.Wnp

        return W0

        
    
    W = superpotential
    
    @partial(jit, static_argnums = (4,5,))
    def superpotential_gauge_invariant(
        self, moduli: Array, tau: complex, fluxes: Array, conj: bool = False, normalise: bool = True
        ) -> complex:
        r"""
        
        **Description:**
        Calculates the value of the gauge invariant version of the flux superpotential.
        
        
        .. admonition:: Details
            :class: dropdown
        
            Following the conventions of `1912.10047 <https://inspirehep.net/literature/1772253>`_,
            we define the gauge invariant version of the superpotential as
        
            .. math::
                    \tilde{W} = \sqrt{\dfrac{2}{\pi}}\, \mathrm{e}^{K/2}\, W

            Its absolute value is proportional to the gravitino mass.
            The normalisation is taken from `1908.04788 <https://inspirehep.net/literature/1749542>`_.

        
        Args:
            moduli (Array): Complex structure moduli values.
            tau (complex): Axio-dilaton value.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            normalise (bool, optional): If ``True``, rescales superpotential by :math:`\sqrt{2/\pi}`. Defaults to ``True``.
        
        Returns:
            complex: Value of the gauge invariant superpotential.
            

        See also: :func:`superpotential`
    
        See also: :func:`kahler_potential`
        """
        if conj:
            KP = self.kahler_potential(jnp.conj(moduli),moduli,jnp.conj(tau),tau)
        else:
            KP = self.kahler_potential(moduli,jnp.conj(moduli),tau,jnp.conj(tau))

        W = self.superpotential(moduli,tau,fluxes,conj=conj,normalise=normalise)
        
        return jnp.exp(KP.real/2.)*W
    
    W0 = superpotential_gauge_invariant
        
    @partial(jit, static_argnums = (4,))
    def dW_z(
        self, moduli: Array, tau: complex, fluxes: Array, conj: bool = False
        ) -> Array:
        r"""

        **Description:**
        Calculates the holomorphic derivative :math:`W_i=\partial_{z^i}W`
        of the superpotential :math:`W` with respect to the complex structure
        moduli :math:`z^{i}`.

        .. admonition:: Details
            :class: dropdown

            The Gukov-Vafa-Witten superpotential is :math:`W = \int G_3 \wedge \Omega`,
            where :math:`\Omega` is the holomorphic 3-form. Its holomorphic derivative
            with respect to the complex structure moduli is computed via ``jax.grad``
            of :func:`superpotential` with ``holomorphic=True``.

            .. math::
                W_i = \partial_{z^i} W = \int G_3 \wedge \chi_i

            where :math:`\chi_i = \partial_{z^i}\Omega` are the Kodaira-Spencer forms.

        Args:
            moduli (Array): Complex structure moduli values.
            tau (complex): Value of the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of the holomorphic derivative of the superpotential with respect to  
                the complex structure moduli :math:`z^{i}`.
        
        """
        
        # Gradient of superpotential w.r.t the moduli and the axio dilaton
        return jax.grad(self.superpotential,argnums=0,holomorphic=True)(moduli,tau,fluxes,conj=conj)

    
    @partial(jit, static_argnums = (4,))
    def dW_tau(
        self, moduli: Array, tau: complex, fluxes: Array, conj: bool = False
        ) -> complex:
        r"""

        **Description:**
        Calculates the holomorphic derivative :math:`W_\tau=\partial_{\tau}W`
        of the superpotential :math:`W` with respect to the axio-dilaton
        :math:`\tau`.

        .. admonition:: Details
            :class: dropdown

            Since :math:`W = f^T\Pi - \tau\,h^T\Pi`, the axio-dilaton derivative is

            .. math::
                \partial_\tau W = -h^T\Pi = -\int H_3 \wedge \Omega

        Args:
            moduli (Array): Complex structure moduli values.
            tau (complex): Value of the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            complex: Value of the holomorphic gradient of the superpotential with respect to 
                the axio-dilaton :math:`\tau`.
        
        """
        
        return jax.grad(self.superpotential,argnums=1,holomorphic=True)(moduli,tau,fluxes,conj=conj)


    @partial(jit, static_argnums = (4,))
    def dW(
        self, moduli: Array, tau: complex, fluxes: Array, conj: bool = False
        ) -> Array:
        r"""

        **Description:**
        Calculates the holomorphic derivative :math:`W_I` of the superpotential
        :math:`W` with respect to all moduli :math:`(z^i, \tau)`.

        .. admonition:: Details
            :class: dropdown

            Combines :func:`dW_z` and :func:`dW_tau` into a single array via
            ``jax.grad(W, argnums=(0,1), holomorphic=True)``:

            .. math::
                W_I = (W_{z^1}, \ldots, W_{z^{h^{1,2}}}, W_\tau)

            Shape: ``(h12 + 1,)``.

        Args:
            moduli (Array): Complex structure moduli values.
            tau (complex): Value of the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of the holomorphic derivative of the superpotential with respect to 
                the complex structure moduli :math:`z^{i}` and the axio-dilaton :math:`\tau`.
        
        """
        
        #dWz = self.dW_z(moduli,tau,fluxes,conj=conj)
        #dWtau = self.dW_tau(moduli,tau,fluxes,conj=conj)
        
        dWz, dWtau= jax.grad(self.superpotential,argnums=(0,1),holomorphic=True)(moduli,tau,fluxes,conj=conj)

        return jnp.append(dWz,dWtau)


    @partial(jit, static_argnums = (4,))
    def ddW_z_z(
        self, moduli: Array, tau: complex, fluxes: Array, conj: bool = False
        ) -> Array:
        r"""

        **Description:**
        Calculates the second holomorphic derivatives
        :math:`W_{ij}=\partial_{z^i}\partial_{z^j}W` of the superpotential.

        .. admonition:: Details
            :class: dropdown

            Computed via ``jacrev`` of :func:`dW_z`. Shape ``(h12, h12)``.

            .. math::
                W_{ij} = \partial_{z^i}\partial_{z^j}\int G_3 \wedge \Omega

        Args:
            moduli (Array): Complex structure moduli values.
            tau (complex): Value of the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            complex: Value of :math:`W_{i j}=\partial_{z^i}\partial_{z^j}W`.
        
        """
        
        return jax.jacrev(self.dW_z,argnums=0,holomorphic=True)(moduli,tau,fluxes,conj=conj)

        
        
    
    @partial(jit, static_argnums = (4,))
    def ddW_z_tau(
        self, moduli: Array, tau: complex, fluxes: Array, conj: bool = False
        ) -> Array:
        r"""

        **Description:**
        Calculates the mixed second derivative
        :math:`W_{i\tau}=\partial_{z^i}\partial_{\tau}W` of the superpotential.

        .. admonition:: Details
            :class: dropdown

            Computed via ``jacrev(dW_z, argnums=1)``. Shape ``(h12,)``.

            .. math::
                W_{i\tau} = \partial_{z^i}\partial_\tau W = -\partial_{z^i}(h^T\Pi)

        Args:
            moduli (Array): Complex structure moduli values.
            tau (complex): Value of the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`W_{i \tau}=\partial_{z^i}\partial_{\tau}W`.
        
        """
        return jax.jacrev(self.dW_z,argnums=1,holomorphic=True)(moduli,tau,fluxes,conj=conj)

    @partial(jit, static_argnums = (4,))
    def ddW_tau_tau(
        self, moduli: Array, tau: complex, fluxes: Array, conj: bool = False
        ) -> Array:
        r"""

        **Description:**
        Calculates :math:`W_{\tau\tau}=\partial_\tau^2 W`.

        .. admonition:: Details
            :class: dropdown

            Since :math:`W` is linear in :math:`\tau`, this vanishes identically:
            :math:`W_{\tau\tau} = 0`. Nevertheless, it is computed via ``jacrev``
            for consistency with the autodiff framework.

        Args:
            moduli (Array): Complex structure moduli values.
            tau (complex): Value of the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`W_{\tau \tau}=\partial_{\tau}\partial_{\tau}W`.
        
        """
        return jax.jacrev(self.dW_tau,argnums=1,holomorphic=True)(moduli,tau,fluxes,conj=conj)


    @partial(jit, static_argnums = (4,))
    def ddW(
        self, moduli: Array, tau: complex, fluxes: Array, conj: bool = False
        ) -> Array:
        r"""

        **Description:**
        Calculates the full second holomorphic derivatives
        :math:`W_{IJ}=\partial_I\partial_J W` of the superpotential.

        .. admonition:: Details
            :class: dropdown

            Assembles the :math:`(h^{1,2}+1)\times(h^{1,2}+1)` matrix from the
            blocks :func:`ddW_z_z`, :func:`ddW_z_tau`, and :func:`ddW_tau_tau`:

            .. math::
                W_{IJ} = \begin{pmatrix}
                W_{z^i z^j} & W_{z^i \tau} \\
                W_{\tau z^j} & W_{\tau\tau}
                \end{pmatrix}
                
        
        Args:
            moduli (Array): Complex structure moduli values.
            tau (complex): Value of the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`W_{IJ}=\partial_{I}\partial_{J}W`.
        
        """

        ddW_z_z = self.ddW_z_z(moduli, tau, fluxes, conj=conj)
        ddW_z_tau = self.ddW_z_tau(moduli, tau, fluxes, conj=conj)
        ddW_tau_tau = self.ddW_tau_tau(moduli, tau, fluxes, conj=conj)
        
        block1 = jnp.append(ddW_z_z,ddW_z_tau.reshape(1,-1),axis=0)
        block2 = jnp.append(ddW_z_tau,ddW_tau_tau)
        
        return jnp.append(block1,block2.reshape(-1,1),axis=1)
        
    ###################################################################################################################
    ######################################### F-TERM CONDITIONS ETC.  #################################################
    ###################################################################################################################
    
    @partial(jit, static_argnums = (6,))
    def DW_z(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Calculates the Kähler covariant derivative of the superpotential :math:`W` 
        with respect to the complex structure moduli :math:`z^{i}`.
        
        .. admonition:: Details
            :class: dropdown
        
            The :math:`F`-term conditions for the complex structure moduli :math:`z^{i}` are given by
            
            .. math::
                D_i W = \partial_{z^i}W+(\partial_{z^i}K)\, W = W_i +K_i\, W \, .

            They appear prominently in the :math:`F`-term scalar potential :func:`scalar_potential`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Values of the :math:`F`-term conditions for the complex structure moduli :math:`z^{i}`.

        See also: :func:`superpotential`
    
        See also: :func:`dW_z`
    
        See also: :func:`dK_z`

        See also: :func:`dK_cz`
        
        """
        
        # F-Terms as the Kähler covariant derivative of W w.r.t cs moduli and axio dilaton
        if not conj:
            return self.dW_z(moduli,tau,fluxes)+self.dK_z(moduli,moduli_c,tau,tau_c)*self.superpotential(moduli,tau,fluxes)
        else:
            return self.dW_z(moduli_c,tau_c,fluxes,conj=True)+self.dK_cz(moduli,moduli_c,tau,tau_c)*self.superpotential(moduli_c, tau_c,fluxes,conj=True)
        
    
    
    @partial(jit, static_argnums = (6,))
    def DW_tau(
        self, moduli: Array, 
        moduli_c: Array, 
        tau: complex, 
        tau_c: complex, 
        fluxes: Array, 
        conj: bool = False
        ) -> complex:
        r"""
        
        **Description:**
        Calculates the Kähler covariant derivative of the superpotential with respect to the axio-dilaton :math:`\tau`.
        
        .. admonition:: Details
            :class: dropdown
        
            The :math:`F`-term conditions for the axio-dilaton :math:`\tau` are given by

            .. math::
                D_{\tau} W=\partial_{\tau}W+(\partial_{\tau}K)\, W = W_\tau+K_\tau\, W

            They appear prominently in the :math:`F`-term scalar potential :func:`scalar_potential`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            complex: Values of the :math:`F`-term conditions for the axio-dilaton :math:`\tau`.
            

        See also: :func:`superpotential`
    
        See also: :func:`dW_tau`
    
        See also: :func:`dK_tau`

        See also: :func:`dK_ctau`
        
        """
        if not conj:
            return self.dW_tau(moduli,tau,fluxes)+self.dK_tau(moduli,moduli_c,tau,tau_c)*self.superpotential(moduli,tau,fluxes)
        else:
            return self.dW_tau(moduli_c,tau_c,fluxes,conj=True)+self.dK_ctau(moduli,moduli_c,tau,tau_c)*self.superpotential(moduli_c,tau_c,fluxes,conj=True)
    
    
    @partial(jit, static_argnums = (6,))
    def DW(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic Kähler covariant derivatives of the superpotential with respect to
        the complex structure moduli :math:`z^{i}` and the axio-dilaton :math:`\tau`.
        
        .. admonition:: Details
            :class: dropdown
        
            This function combines the outputs of the functions :func:`DW_z` and :func:`DW_tau`

            .. math::
                D_I W = \bigl(D_{1} W,\ldots, D_{h^{1,2}} W,D_{\tau} W\bigl)\, .

            This output is later used in the :math:`F`-term scalar potential :func:`scalar_potential`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Values of the :math:`F`-term conditions for the complex structure moduli :math:`z^{i}`
                and the axio-dilaton :math:`\tau`.

        See also: :func:`DW_z`
    
        See also: :func:`DW_tau`
        
        """
        
        FTM = self.DW_z(moduli, moduli_c, tau, tau_c, fluxes, conj=conj)
        FTT = self.DW_tau(moduli, moduli_c, tau, tau_c, fluxes, conj=conj)
        
        return jnp.append(FTM, FTT)
         
        
    @partial(jit, static_argnums = (6,))
    def canonical_fterms(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Tuple[Array,Array]:
        r"""
        
        **Description:**
        Returns the canonically normalised :math:`F`-term conditions :math:`F_I,\, F^I`.
        
        .. admonition:: Details
            :class: dropdown

            The canonically normalised :math:`F`-term conditions are defined as

            .. math::
                F_I = \mathrm{e}^{K/2}\, D_IW\, ,\quad F^I = \mathrm{e}^{K/2}\, K^{I\bar{J}}\, D_{\bar{J}}\overline{W}\, .

            They parametrise the contribution to the flux scalar potential :func:`scalar_potential`

            .. math::
                F^I F_I = \mathrm{e}^{K}\, K^{I\bar{J}}\, D_IW\, D_{\bar{J}} \overline{W} \, .

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Canonically normalised :math:`F`-term conditions :math:`F_I`.
            Array: Canonically normalised :math:`F`-term conditions :math:`F^I`.
        
        """
        
        FT = self.DW(moduli, moduli_c, tau, tau_c, fluxes)
        cFT = self.DW(moduli, moduli_c, tau, tau_c, fluxes,conj=True)
        
        Inv_KM=self.inverse_kahler_metric(moduli, moduli_c, tau, tau_c)
        
        KP=self.kahler_potential(moduli, moduli_c, tau, tau_c)
        
        FT *= jnp.exp(KP/2.)
        cFT *= jnp.exp(KP/2.)
        
        if conj:
            return cFT, jnp.matmul(Inv_KM,FT)
        else:
            return FT, jnp.matmul(cFT,Inv_KM)


    @partial(jit, static_argnums = ())
    def DW_x(
        self, x: Array, fluxes: Array
        ) -> Array:
        r"""
        
        **Description:**
        Calculates the real and imaginary parts of the :math:`F`-term conditions.
        
        .. admonition:: Details
            :class: dropdown
        
            This function is used for finding roots of the :math:`F`-term conditions.
            It takes as input a real array of values for the complex structure moduli :math:`z^i`
            and the axio-dilaton :math:`\tau`

            .. math::
                \bigl (\mathrm{Re}(z^1),\mathrm{Im}(z^1),\ldots,\mathrm{Re}(z^{h^{1,2}}),
                \mathrm{Im}(z^{h^{1,2}}),\mathrm{Re}(\tau),\mathrm{Im}(\tau)\bigl )\, .

            It then assembles the output of :func:`DW` 
            in a :math:`(2h^{1,2}+2)`-dimensional real vector
        
            .. math::
                    \bigl (\text{Re}(D_1 W), \text{Im}(D_1 W),\ldots, \text{Re}(D_{h^{1,2}} W),
                    \text{Im}(D_{h^{1,2}} W),\text{Re}(D_{\tau} W), \text{Im}(D_{\tau} W)\bigl )\, .
        
        Args:
            x (Array): Array of shape (:math:`2(h^{1,2}+1)`, ) containing real and imaginary parts of
                the complex structure moduli :math:`z^i` and axio-dilaton :math:`\tau`.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
        
        Returns:
            Array: Vector of shape (:math:`2(h^{1,2}+1)`, ) containing real and imaginary parts 
                of the :math:`F`-term conditions in alternating order.       

        See also: :func:`DW`

        """

        moduli,moduli_c,tau,tau_c = self._convert_real_to_complex(x)
    
        FT = self.DW(moduli,moduli_c,tau,tau_c,fluxes)
    
        return jnp.column_stack((jnp.real(FT), jnp.imag(FT))).flatten()
    
    @partial(jit, static_argnums = ())
    def dDW_x(
        self, x: Array, fluxes: Array
        ) -> Array:
        r"""
        
        **Description:**
        Returns the first derivatives of the F-term conditions by differentiating with respect to the real fields.
        
        Args:
            x (Array): Array of shape (:math:`2(h^{1,2}+1)`, ) containing real and imaginary parts of
                the complex structure moduli :math:`z^i` and axio-dilaton :math:`\tau`.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
        
        Returns:
            Array: First derivatives of the real :math:`F`-term conditions with respect to the 
                real and imaginary parts of the complex structure moduli :math:`z^i` and axio-dilaton :math:`\tau`.

        See also: :func:`DW_x`
        
        """
        
        return jax.jacrev(self.DW_x,argnums=0,holomorphic=False)(x,fluxes)

    @partial(jit, static_argnums = ())
    def ddDW_x(
        self, x: Array, fluxes: Array
        ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives of the F-term conditions by differentiating with respect to the real fields.
        
        Args:
            x (Array): Array of shape (:math:`2(h^{1,2}+1)`, ) containing real and imaginary parts of
                the complex structure moduli :math:`z^i` and axio-dilaton :math:`\tau`.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
        
        Returns:
            Array: Second derivatives of the real :math:`F`-term conditions with respect to the 
                real and imaginary parts of the complex structure moduli :math:`z^i` and axio-dilaton :math:`\tau`.

        See also: :func:`DW_x`
        
        """
        
        return jax.jacrev(self.dDW_x,argnums=0,holomorphic=False)(x,fluxes)
        
    ###################################################################################################################
    ############################## 2nd holomorphic derivatives of F-term CONDITIONS ###################################
    ###################################################################################################################

    @partial(jit, static_argnums = (6,))
    def dDW_z_z(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{z^i}D_{z^j}W` of the :math:`F`-term
        :math:`D_{z^j}W` with respect to the complex structure moduli :math:`z^i`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`\partial_{z^i}D_{z^j}W`.
        
        """
        
        if conj:
            return jax.jacrev(self.DW_z,argnums=1,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        else:
            return jax.jacrev(self.DW_z,argnums=0,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

    
    @partial(jit, static_argnums = (6,))
    def dDW_tau_tau(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> complex:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\tau}D_{\tau}W` of the :math:`F`-term
        :math:`D_{\tau}W` with respect to the axio-dilaton :math:`\tau`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`\partial_{\tau}D_{\tau}W`.
        
        """

        if conj:
            return jax.grad(self.DW_tau,argnums=3,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        else:
            return jax.grad(self.DW_tau,argnums=2,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        
    @partial(jit, static_argnums = (6,))
    def dDW_z_tau(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\tau}D_{z^j}W` of the :math:`F`-term
        :math:`D_{z^j}W` with respect to the axio-dilaton :math:`\tau`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`\partial_{\tau}D_{z^j}W`.
        
        """
        if conj:
            return jax.jacrev(self.DW_z,argnums=3,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        else:
            return jax.jacrev(self.DW_z,argnums=2,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        
    @partial(jit, static_argnums = (6,))
    def dDW_tau_z(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{z^i}D_{\tau}W` of the :math:`F`-term
        :math:`D_{\tau}W` with respect to the complex structure moduli :math:`z^i`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`\partial_{z^i}D_{\tau}W`.
        
        """

        if conj:
            return jax.grad(self.DW_tau,argnums=1,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        else:
            return jax.grad(self.DW_tau,argnums=0,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

    @partial(jit, static_argnums = (6,))
    def dDW_z(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{z^i}D_{J}W` of the :math:`F`-terms
        :math:`D_{J}W` with respect to the complex structure moduli :math:`z^i`.
        
        .. note::
            The output shape is such that ``output[J-1][i-1]`` corresponds to :math:`\partial_{i}D_{J}W`
            with :math:`J=1,\ldots, h^{1,2}+1` and :math:`i=1,\ldots, h^{1,2}`.

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`\partial_{I}D_{z^j}W`.
        
        """

        if conj:
            return jax.jacrev(self.DW,argnums=1,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        else:
            return jax.jacrev(self.DW,argnums=0,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)


    @partial(jit, static_argnums = (6,))
    def dDW_tau(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\tau}D_{J}W` of the :math:`F`-terms
        :math:`D_{J}W` with respect to the the axio-dilaton :math:`\tau`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`\partial_{\tau}D_{J}W`.
        
        """
        if conj:
            return jax.jacrev(self.DW,argnums=3,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        else:
            return jax.jacrev(self.DW,argnums=2,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

    @partial(jit, static_argnums = (6,))
    def dDW(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{I}D_{J}W` of the :math:`F`-terms
        :math:`D_{J}W` with respect to the complex structure moduli :math:`z^i` and  the axio-dilaton :math:`\tau`.

        .. note::
            The output shape is such that ``output[J-1][I-1]`` corresponds to :math:`\partial_{I}D_{J}W`
            with :math:`I,J=1,\ldots, h^{1,2}+1`.

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`\partial_{I}D_{J}W`.
        
        """

        dDWz = self.dDW_z(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        dDWtau = self.dDW_tau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        
        return jnp.append(dDWz,jnp.array([dDWtau]).T,axis=1)

    ###################################################################################################################
    ############################ 2nd anti-holomorphic derivatives of F-term CONDITIONS ################################
    ###################################################################################################################


    @partial(jit, static_argnums = (6,))
    def dDW_z_cz(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\overline{z}^i}D_{z^j}W` of the :math:`F`-term
        :math:`D_{z^j}W` with respect to the complex conjugate complex structure moduli :math:`\overline{z}^i`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`\partial_{\overline{z}^i}D_{z^j}W`.
        
        """

        km = self.ddK_z_cz(moduli, moduli_c, tau, tau_c)

        if conj:
            return km.T*self.superpotential(moduli_c,tau_c,fluxes,conj=conj)
        else:
            return km*self.superpotential(moduli,tau,fluxes,conj=conj)

    
    @partial(jit, static_argnums = (6,))
    def dDW_tau_ctau(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\overline{\tau}}D_{\tau}W` of the :math:`F`-term
        :math:`D_{\overline{\tau}}W` with respect to the complex conjugate axio-dilaton :math:`\overline{\tau}`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`\partial_{\overline{\tau}}D_{\tau}W`.
        
        """

        # The Käher metric component is real and a scalar...
        km = self.ddK_tau_ctau(moduli, moduli_c, tau, tau_c)

        if conj:
            return km*self.superpotential(moduli_c,tau_c,fluxes,conj=conj)
        else:
            return km*self.superpotential(moduli,tau,fluxes,conj=conj)
        
    
    @partial(jit, static_argnums = (6,))
    def dDW_z_ctau(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\overline{\tau}}D_{z^j}W` of the :math:`F`-term
        :math:`D_{z^j}W` with respect to the complex conjugate axio-dilaton :math:`\overline{\tau}`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`\partial_{\overline{\tau}}D_{z^j}W`.
        
        """

        if conj:
            km = self.ddK_cz_tau(moduli, moduli_c, tau, tau_c)
            return km*self.superpotential(moduli_c,tau_c,fluxes,conj=conj)
        else:
            km = self.ddK_z_ctau(moduli, moduli_c, tau, tau_c)
            return km*self.superpotential(moduli,tau,fluxes,conj=conj)
        
    @partial(jit, static_argnums = (6,))
    def dDW_tau_cz(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\overline{z}^i}D_{\tau}W` of the :math:`F`-term
        :math:`D_{\tau}W` with respect to the complex conjugate complex structure moduli :math:`\overline{z}^i`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`\partial_{\overline{z}^i}D_{\tau}W`.
        
        """

        if conj:
            km = self.ddK_z_ctau(moduli, moduli_c, tau, tau_c)
            return km*self.superpotential(moduli_c,tau_c,fluxes,conj=conj)
        else:
            km = self.ddK_cz_tau(moduli, moduli_c, tau, tau_c)
            return km*self.superpotential(moduli,tau,fluxes,conj=conj)


    @partial(jit, static_argnums = (6,))
    def dDW_cz(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\overline{z}^i}D_{J}W` of the :math:`F`-term
        :math:`D_{J}W` with respect to the complex conjugate complex structure moduli :math:`\overline{z}^i`.
        
        .. note::
            The output shape is such that ``output[J-1][i-1]`` corresponds to :math:`\partial_{\bar{\imath}}D_{J}W`
            with :math:`J=1,\ldots, h^{1,2}+1` and :math:`i=1,\ldots, h^{1,2}`.

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`\partial_{\overline{z}^i}D_{J}W`.

        See also: :func:`dDW_z`
        
        """
        if conj:
            return jax.jacrev(self.DW,argnums=0,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        else:
            return jax.jacrev(self.DW,argnums=1,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

    @partial(jit, static_argnums = (6,))
    def dDW_ctau(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\overline{\tau}}D_{J}W` of the :math:`F`-term
        :math:`D_{J}W` with respect to the complex conjugate axio-dilaton :math:`\overline{\tau}`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`\partial_{\overline{\tau}}D_{J}W`.

        See also: :func:`dDW_tau`
        
        """
        
        if conj:
            return jax.jacrev(self.DW,argnums=2,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        else:
            return jax.jacrev(self.DW,argnums=3,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

    @partial(jit, static_argnums = (6,))
    def dDW_c(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\overline{I}}D_{J}W` of the :math:`F`-terms
        :math:`D_{J}W` with respect to the complex conjugate complex structure moduli :math:`\overline{z}^i` 
        and the axio-dilaton :math:`\overline{\tau}`.

        .. note::
            The output shape is such that ``output[J-1][I-1]`` corresponds to :math:`\partial_{\overline{I}}D_{J}W`
            with :math:`I,J=1,\ldots, h^{1,2}+1`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`\partial_{\overline{I}}D_{J}W`.

        See also: :func:`dDW`
        
        """

        dDWcz = self.dDW_cz(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        dDWctau = self.dDW_ctau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        
        return jnp.append(dDWcz,jnp.array([dDWctau]).T,axis=1)

    @partial(jit, static_argnums = ())
    def dDW_real(
        self, moduli: Array, tau: complex, fluxes: Array
        ) -> Array:
        r"""

        **Description:**
        Returns the Jacobian :math:`\partial_{\phi^\alpha}(D_J W)` of the
        :math:`F`-terms with respect to the real scalar fields
        :math:`\phi^\alpha = (a^i, v^i, c_0, s)`.

        .. admonition:: Details
            :class: dropdown

            Constructs the real Jacobian from the holomorphic and
            antiholomorphic derivatives via:

            .. math::
                \frac{\partial}{\partial v^i} = \mathrm{i}\Bigl(
                \frac{\partial}{\partial z^i}
                - \frac{\partial}{\partial \bar{z}^i}\Bigr),\quad
                \frac{\partial}{\partial a^i} =
                \frac{\partial}{\partial z^i}
                + \frac{\partial}{\partial \bar{z}^i}

            Output shape: ``(2n, 2n)`` where the rows are ``[Re(D_J W), Im(D_J W)]``
            and the columns are ``[∂_v, ∂_a]`` derivatives.

        Args:
            moduli (Array): Complex structure moduli values.
            tau (complex): Axio-dilaton value.
            fluxes (Array): Array of fluxes.

        Returns:
            Array: Shape ``(2n, 2n)`` real Jacobian :math:`\partial_{\phi^\alpha}(D_J W)`.

        See also: :func:`dDW`, :func:`dDW_c`

        """

        dDW = self.dDW(moduli,jnp.conj(moduli),tau,jnp.conj(tau),fluxes)
        dcDW = self.dDW_c(moduli,jnp.conj(moduli),tau,jnp.conj(tau),fluxes)

        dt_DW = 1j*(dDW-dcDW)
        dphi_DW = dDW+dcDW

        return jnp.block([[jnp.real(dt_DW),jnp.real(dphi_DW)],[jnp.imag(dt_DW),jnp.imag(dphi_DW)]])
        
    

    ###################################################################################################################
    ############################ SECOND KÄHLER COVARIANT DERIVATIVES OF F-TERMS #######################################
    ###################################################################################################################
        
    ###################################################################################################################
    ######################################## Holomorphic derivatives  #################################################
    ###################################################################################################################

    @partial(jit, static_argnums = (6,))
    def DDW_z_z(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Kähler covariant derivative :math:`D_{z^i}D_{z^j}W` of the :math:`F`-term
        :math:`D_{z^j}W` with respect to the complex structure moduli :math:`z^i`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`D_{z^i}D_{z^j}W`.
        
        """
        

        DWz = self.DW_z(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        if conj:
            DWz_Kz = jnp.outer(DWz,self.dK_cz(moduli,moduli_c,tau,tau_c))
        else:
            DWz_Kz = jnp.outer(DWz,self.dK_z(moduli,moduli_c,tau,tau_c))

        return self.dDW_z_z(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+DWz_Kz
    
    @partial(jit, static_argnums = (6,))
    def DDW_tau_tau(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Kähler covariant derivative :math:`D_{\tau}D_{\tau}W` of the :math:`F`-term
        :math:`D_{\tau}W` with respect to the axio-dilaton :math:`\tau`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`D_{\tau}D_{\tau}W`.
        
        """
        
        if conj:
            Kt = self.dK_ctau(moduli,moduli_c,tau,tau_c)
        else:
            Kt = self.dK_tau(moduli,moduli_c,tau,tau_c)

        return self.dDW_tau_tau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+Kt*self.DW_tau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
            
    @partial(jit, static_argnums = (6,))
    def DDW_z_tau(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Kähler covariant derivative :math:`D_{\tau}D_{z^j}W` of the :math:`F`-term
        :math:`D_{z^j}W` with respect to the axio-dilaton :math:`\tau`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`D_{\tau}D_{z^j}W`.
        
        """
        
        
        if conj:
            Kt = self.dK_ctau(moduli,moduli_c,tau,tau_c)
        else:
            Kt = self.dK_tau(moduli,moduli_c,tau,tau_c)

        return self.dDW_z_tau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+Kt*self.DW_z(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
    
    @partial(jit, static_argnums = (6,))
    def DDW_tau_z(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Kähler covariant derivative :math:`D_{z^i}D_{\tau}W` of the :math:`F`-term
        :math:`D_{\tau}W` with respect to the complex structure moduli :math:`z^i`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`D_{z^i}D_{\tau}W`.
        
        """
        

        if conj:
            Kz = self.dK_cz(moduli,moduli_c,tau,tau_c)
        else:
            Kz = self.dK_z(moduli,moduli_c,tau,tau_c)

        return self.dDW_tau_z(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+Kz*self.DW_tau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        
    
    @partial(jit, static_argnums = (6,))
    def DDW_general(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the second Kähler covariant derivatives :math:`D_I D_J W` of the superpotential :math:`W`
        with respect to the axio-dilaton :math:`\tau` and the complex structure moduli :math:`z^{i}`.
        
        .. admonition:: Details
            :class: dropdown

            We compute the Kähler covariant derivatives of the :math:`F`-terms :math:`D_J W`
        
            .. math::
                D_I D_J W = \partial_I D_J W + (\partial_I K) D_J W

            using automatic differentiation. To do so, we combine the outputs of :func:`DDW_z_z`,
            :func:`DDW_tau_tau`, :func:`DDW_z_tau` and :func:`DDW_tau_z`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Second Kähler covariant derivatives :math:`D_I D_J W` of the superpotential :math:`W`
                with respect to the axio-dilaton :math:`\tau` and  the moduli :math:`z^{i}`.
        
        """

        DWval = self.DW(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

        if conj:
            DW_dK = jnp.outer(DWval,self.dK_c(moduli,moduli_c,tau,tau_c))
        else:
            DW_dK = jnp.outer(DWval,self.dK(moduli,moduli_c,tau,tau_c))

        return self.dDW(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+DW_dK
        

        
    ###################################################################################################################
    ##################################### Anti-holomorphic derivatives  ###############################################
    ###################################################################################################################

    @partial(jit, static_argnums = (6,))
    def DDW_z_cz(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Kähler covariant derivative :math:`D_{\overline{z}^i}D_{z^j}W` of the :math:`F`-term
        :math:`D_{z^j}W` with respect to the complex conjugate complex structure moduli :math:`\overline{z}^i`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`D_{\overline{z}^i}D_{z^j}W`.
        
        """

        DW = self.DW_z(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

        if conj:
            KDW = jnp.outer(DW,self.dK_z(moduli,moduli_c,tau,tau_c))
        else:
            KDW = jnp.outer(DW,self.dK_cz(moduli,moduli_c,tau,tau_c))
        
        return self.dDW_z_cz(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+KDW
    
    @partial(jit, static_argnums = (6,))
    def DDW_tau_ctau(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Kähler covariant derivative :math:`D_{\overline{\tau}}D_{\tau}W` of the :math:`F`-term
        :math:`D_{\tau}W` with respect to the complex conjugate axio-dilaton :math:`\overline{\tau}`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`D_{\overline{\tau}}D_{\tau}W`.
        
        """
        

        DW = self.DW_tau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

        if conj:
            KDW = self.dK_tau(moduli,moduli_c,tau,tau_c)*DW
        else:
            KDW = self.dK_ctau(moduli,moduli_c,tau,tau_c)*DW

        return self.dDW_tau_ctau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+KDW
            
    @partial(jit, static_argnums = (6,))
    def DDW_z_ctau(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Kähler covariant derivative :math:`D_{\overline{\tau}}D_{z^j}W` of the :math:`F`-term
        :math:`D_{z^j}W` with respect to the complex conjugate axio-dilaton :math:`\overline{\tau}`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`D_{\overline{\tau}}D_{z^j}W`.
        
        """
        

        DW = self.DW_z(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

        if conj:
            KDW = self.dK_tau(moduli,moduli_c,tau,tau_c)*DW
        else:
            KDW = self.dK_ctau(moduli,moduli_c,tau,tau_c)*DW

        return self.dDW_z_ctau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+KDW
    
    @partial(jit, static_argnums = (6,))
    def DDW_tau_cz(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Kähler covariant derivative :math:`D_{\overline{z}^i}D_{\tau}W` of the :math:`F`-term
        :math:`D_{\tau}W` with respect to the complex conjugate complex structure moduli :math:`\overline{z}^i`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`D_{\overline{z}^i}D_{\tau}W`.
        
        """

        DW = self.DW_tau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

        if conj:
            KDW = self.dK_z(moduli,moduli_c,tau,tau_c)*DW
        else:
            KDW = self.dK_cz(moduli,moduli_c,tau,tau_c)*DW

        return self.dDW_tau_cz(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+KDW


    
    @partial(jit, static_argnums = (6,))
    def DcDW(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns a matrix containing the mixed second Kähler derivatives :math:`D_{\overline{I}}D_{J}W`.
        
        .. admonition:: Details
            :class: dropdown

            More explicitly, we compute

            .. math::
                D_A D_{\overline{B}} W=K_{\overline{B}}D_{A}W+K_{A\overline{B}}W

            Generically, this function is *not* symmetric, but it is in the SUSY case where :math:`D_AW=0` since

            .. math::
                D_A D_{\overline{B}} W\bigl |_{D_AW=0}=K_{A\overline{B}}W\, .

            In particular, this is symmetric.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Values of :math:`D_{\overline{I}}D_{J}W`.
        
        """

        """
        # OLD VERSION
        DcDMod=self.DDW_z_cz(moduli,moduli_c,tau,tau_c,fluxes)
        DcDTau=self.DDW_tau_ctau(moduli,moduli_c,tau,tau_c,fluxes)
        DTaucDMod=self.DDW_z_ctau(moduli,moduli_c,tau,tau_c,fluxes)
        DModcDTau=self.DDW_tau_cz(moduli,moduli_c,tau,tau_c,fluxes)
    
        a = jnp.hstack((jnp.asarray(DcDMod), jnp.asarray(DTaucDMod).reshape(self.h12, 1)))

        b = jnp.hstack((jnp.asarray(DModcDTau), jnp.asarray(DcDTau)))

        return jnp.vstack((a, b))
        """

        DW = self.DW(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        
        if conj:
            KDW = jnp.outer(DW,self.dK(moduli,moduli_c,tau,tau_c))
        else:
            KDW = jnp.outer(DW,self.dK_c(moduli,moduli_c,tau,tau_c))

        return self.dDW_c(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+KDW
        
    @partial(jit, static_argnums = (6,7,))
    def DDW_SUSY(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, conj: bool = False, mode: str = "block diagonal"
        ) -> Array:
        r"""
        
        **Description:**
        Returns a matrix containing the second holomorphic Kähler derivative of the superpotential 
        assuming that the F-flatness conditions :math:`D_{i}W=D_{\tau}W=0` are satisfied.
        
        
        .. admonition:: Details
            :class: dropdown
        
            Returns a matrix containing the second holomorphic Kähler derivative of the superpotential 
            assuming that the F-flatness conditions :math:`D_{i}W=D_{\tau}W=0` are satisfied. 
            More explicitly, we compute

            .. math::
                D_I D_I W = \partial_I D_J W + K_I D_J W 
                          = \partial_I \partial_J W + (K_{IJ}-K_I K_J) W + K_I D_J W + K_J D_I W \, .
            
            At the SUSY minimum, this suggests

            .. math::
                D_I D_I W\bigl |_{D_I W=0} = \partial_I \partial_J W + (K_{IJ}-K_I K_J) W\, .

            For the axio-dilaton, we have :math:`W_{\tau\tau}=0` for the GVW superpotential and thus

            .. math::
                D_\tau D_\tau W\bigl |_{D_\tau W=0} =  (K_{\tau\tau} - K_\tau K_\tau) W\, .

            For the standard Kähler potential :math:`K\supset -\log(-\mathrm{i}(\tau-\bar{\tau}))`,
            this automatically implies

            .. math::
                D_\tau D_\tau W\bigl |_{D_\tau W=0} =  0\; , \quad D_\tau D_i W\bigl |_{D_I W=0} =  0\, .
        
        .. warning::
            We assume that there is a block-diagonal structure in the Kahler potential such that the mixed second derivatives vanish :math:`K_{i\tau}=K_{\tau i}=0`. This might not necessarily be true once further corrections to the Kahler potential are included.
        
        .. warning::
            Further, we use that :math:`W_{\tau\tau}=0` because the flux superpotential :func:`superpotential` is linear in :math:`\tau`. Again, this might not necessarily be true anymore once non-perturbative instanton effects are present in the superpotential!
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (string, optional): Whether to assume that the Kähler metric is block-diagonal. Defaults to ``"block diagonal"``.
        
        Returns:
            Array: Value of :math:`D_I D_J W` assuming :math:`D_IW=0`.
        
        
        """

        if conj:
            K_ZZ=self.ddK_cz_cz(moduli, moduli_c, tau, tau_c)
        
            if mode!="block diagonal":
                K_tau_tau=self.ddK_ctau_ctau(moduli, moduli_c, tau, tau_c)

            K_Z=self.dK_cz(moduli, moduli_c, tau, tau_c)
            K_tau=self.dK_ctau(moduli, moduli_c, tau, tau_c)
            
            W_mod_tau=self.ddW_z_tau(moduli_c,tau_c,fluxes,conj=conj)
            
            W_mod_mod=self.ddW_z_z(moduli_c,tau_c,fluxes,conj=conj)
            
            W_val=self.superpotential(moduli_c,tau_c,fluxes,conj=conj)
            
            
            if mode!="block diagonal":
                ddK_z_tau = self.ddK_cz_ctau(moduli, moduli_c, tau, tau_c)

        else:
            K_ZZ=self.ddK_z_z(moduli, moduli_c, tau, tau_c)
        
            if mode!="block diagonal":
                K_tau_tau=self.ddK_tau_tau(moduli, moduli_c, tau, tau_c)
                
            K_Z=self.dK_z(moduli, moduli_c, tau, tau_c)
            K_tau=self.dK_tau(moduli, moduli_c, tau, tau_c)
            
            W_mod_tau=self.ddW_z_tau(moduli,tau,fluxes)
            
            W_mod_mod=self.ddW_z_z(moduli,tau,fluxes)
            
            W_val=self.superpotential(moduli,tau,fluxes)
            
            if mode!="block diagonal":
                
                ddK_z_tau = self.ddK_z_tau(moduli, moduli_c, tau, tau_c)
                


        if mode=="block diagonal":
            # DDTau = 0 because K_tau_tau-K_tau*K_tau = 0 for the standard Kähler potential
            DDTau = 0.+1j*0.
            DTauDMod=W_mod_tau.reshape(self.h12,1)-jnp.outer(K_Z,K_tau)*W_val
            DModDTau=W_mod_tau-jnp.outer(K_tau,K_Z)[0]*W_val
        else:
            warnings.warn("This method is not tested! Use with caution!")
            DDTau=(K_tau_tau-K_tau*K_tau)*W_val
            DTauDMod=W_mod_tau.reshape(self.h12,1)+(ddK_z_tau-jnp.outer(K_Z,K_tau))*W_val
            DModDTau=W_mod_tau+(ddK_z_tau-jnp.outer(K_tau,K_Z)[0])*W_val
            
        DDMod=W_mod_mod+(K_ZZ-jnp.outer(K_Z,K_Z))*W_val

        a = jnp.hstack((jnp.asarray(DDMod), jnp.asarray(DTauDMod).reshape(self.h12, 1)))

        b = jnp.append(jnp.asarray(DModDTau), jnp.asarray([DDTau]),axis=0)

        return jnp.vstack((a, b))
        
    @partial(jit, static_argnums = (6,7,))
    def DDW(
            self, 
            moduli: Array, 
            moduli_c: Array, 
            tau: complex, 
            tau_c: complex, 
            fluxes: Array, 
            conj: bool = False, 
            mode: str = None
            ) -> Array:
        r"""
        
        **Description:**
        Returns a matrix the second holomorphic Kähler derivatives :math:`D_I D_J W` of the superpotential.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (string, optional): The mode to :math:`DDW`. Currently implemented modes are:
                                            * None: general expression from explicit second derivatives.
                                            * "SUSY": computes at a SUSY locus where DW=0 using standard SUGRA formulas.
        
        Returns:
            Array: Value of :math:`D_I D_J W`.
        
        """

        modes = [None, "SUSY"]
        if mode not in modes:
            raise ValueError(f"Mode must be one of {modes}, but is {mode}!")
        
        
        if mode is None:
            return self.DDW_general(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        elif mode == "SUSY":
            return self.DDW_SUSY(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)


    @partial(jit, static_argnums = (6,))
    def ddDW(
            self,
            moduli: Array,
            moduli_c: Array,
            tau: complex,
            tau_c: complex,
            fluxes: Array,
            conj: bool = False
            ) -> Array:
        r"""

        **Description:**
        Returns the second derivative of the :math:`F`-terms :math:`\partial_A(D_I W)`.

        .. admonition:: Details
            :class: dropdown

            Differentiates :math:`\partial_A(D_I W)` (computed by :func:`dDW`)
            once more with respect to either the holomorphic or anti-holomorphic
            field-space directions.

            For ``conj=False`` (holomorphic):

            .. math::
                (\texttt{ddDW})_{I,A,B} = \partial_B\,\partial_A(D_I W)

            For ``conj=True`` (mixed):

            .. math::
                (\texttt{ddDW})_{I,A,\bar{B}} = \partial_{\bar{B}}\,\partial_A(D_I W)

            These quantities appear in the 9-term product-rule expansion of
            :math:`\partial_A\partial_B S` and :math:`\partial_A\partial_{\bar{B}} S`
            (see :func:`_hessian_SUGRA`), where
            :math:`S = D_{\bar{I}}\bar{W}\,K^{I\bar{J}}\,D_J W`.

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``False``, returns :math:`\partial_B\partial_A(D_I W)`
                (holomorphic second derivative). If ``True``, returns
                :math:`\partial_{\bar{B}}\partial_A(D_I W)` (mixed second derivative).
                Defaults to ``False``.

        Returns:
            Array: Shape ``(n, n, n)`` with index ordering ``[I, A, B]``
                (or ``[I, A, B̄]`` when ``conj=True``), where ``n = h^{1,2}+1``.

        See also: :func:`dDW`, :func:`_hessian_SUGRA`

        """
        if conj:
            dz = jax.jacrev(lambda zc_: self.dDW(moduli, zc_, tau, tau_c, fluxes),
                            holomorphic=True)(moduli_c)
            dt = jax.jacrev(lambda tc_: self.dDW(moduli, moduli_c, tau, tc_, fluxes),
                            holomorphic=True)(tau_c)
        else:
            dz = jax.jacrev(lambda z_: self.dDW(z_, moduli_c, tau, tau_c, fluxes),
                            holomorphic=True)(moduli)
            dt = jax.jacrev(lambda t_: self.dDW(moduli, moduli_c, t_, tau_c, fluxes),
                            holomorphic=True)(tau)
        return jnp.concatenate([dz, dt[:, :, None]], axis=-1)


    @partial(jit, static_argnums = ())
    def DDW_matrix_SUSY(
                        self, 
                        moduli: Array, 
                        moduli_c: Array, 
                        tau: complex, 
                        tau_c: complex, 
                        fluxes: Array
                        ) -> Array:
        r"""
        
        **Description:**
        Returns the matrix of second Kähler derivatives of the superpotential assuming that the F-term conditions are satisfied for all fields.
        
        .. admonition:: Details
            :class: dropdown
        
            Explicitly, we compute
        
            .. math::
                M=\left (\begin{array}{cc}\partial_{A}\partial_{B} W + \left ( K_{A B} - K_{A} K_{B} \right ) W & K_{A\overline{B}} W\ K_{A\overline{B}} \overline{W} & \partial_{\overline{A}}\partial_{\overline{B}} \overline{W} + \left ( K_{\overline{A} \overline{B}} - K_{\overline{A}} K_{\overline{B}} \right ) \overline{W} \end{array}\right )
        
            which is equivalent to the matrix of second derivatives of :math:`W` (see :func:`massive_directions_general`) provided that the F-flatness conditions
        
            .. math::
                D_{Z^{i}} W = D_{\tau} W = 0
        
            are satisfied. In this case, it turns out that
        
            .. math::
                D_{A} D_{\overline{B}} W = K_{A\overline{B}} W
        
            as well as
        
            .. math::
                D_{A} D_{B} W = \partial_{A}\partial_{B} W + \left ( K_{A B} - K_{A} K_{B} \right ) W
        
            as computed via :func:`DDW_SUSY`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
        
        Returns:
            See also: :func:`massive_directions_general`
            See also: :func:`DDW_SUSY`
        
        """
        
        DDW_val=self.DDW(moduli, moduli_c, tau, tau_c,fluxes,mode="SUSY")
        W_val=self.superpotential(moduli, tau,fluxes)
        KM_val=jnp.array(self.kahler_metric(moduli,moduli_c,tau,tau_c))
        
        a = jnp.hstack((DDW_val, KM_val * W_val))

        b = jnp.hstack((jnp.conj(KM_val * W_val), jnp.conj(DDW_val)))

        return jnp.vstack((a, b))
        
    @partial(jit, static_argnums = ())
    def DDW_matrix_general(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, fluxes: Array
        ) -> Array:
        r"""
        **Description:**
        Returns the matrix of second Kähler derivatives of the superpotential.
        
        .. admonition:: Details
            :class: dropdown
        
            Explicitly, we compute
        
            .. math::
                M=\left (\begin{array}{cc} D_{A}D_{B} W & D_{A}D_{\overline{B}} W \ D_{\overline{A}}D_{B} \overline{W} & D_{\overline{A}}D_{\overline{B}} \overline{W} \end{array}\right )
        
            which is encodes the mass terms for chiral fermions.
        
            In this case, it turns out that
        
            .. math::
                D_{A} D_{\overline{B}} W =K_{\overline{B}}W_{A}+\left ( K_{A\overline{B}}+K_{A}K_{\overline{B}}\right ) W
        
            This matrix encodes the mass terms for chiral fermions in
            the 4D effective theory (cf. Eq. (2.5) of `1312.5659
            <https://arxiv.org/abs/1312.5659>`_).

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.

        Returns:
            Array: Shape ``(2n, 2n)`` matrix with blocks
                ``[[DDW, DcDW], [conj(DcDW), conj(DDW)]]``.

        """

        DDW_val=self.DDW(moduli,moduli_c,tau,tau_c,fluxes)
        DcDW_val=self.DcDW(moduli,moduli_c,tau,tau_c,fluxes)
        
        a = jnp.hstack((DDW_val, DcDW_val))

        b = jnp.hstack((jnp.conj(DcDW_val), jnp.conj(DDW_val)))

        return jnp.vstack((a, b))
        
    @partial(jit, static_argnums = (6))
    def DDW_matrix(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, fluxes: Array, mode: str = "SUSY"
        ) -> Array:
        r"""
        
        **Description:**
        Returns the matrix of masses for chiral fermions.
        
        
        .. admonition:: Details
            :class: dropdown
        
            The diagonal entries of this matrix (evaluated at a minimum) are proportional to the mass matrix of chiral fermions, see equation (2.5) in `1312.5659 <https://arxiv.org/pdf/1312.5659.pdf>`_. By abuse of terminology, we call this matrix the ''fermionic mass matrix''.
        
        .. note::
        
            To speed up the computation, it can be advantageous to compute this function using the ``mode="SUSY"`` option provided that :math:`DW=0` for *all* fields.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (string, optional): Whether or not the point at which the mass matrix is evaluated corresponds to a minimum with :math:`DW=0` for *all* fields. Default is ``mode="SUSY"``.
        
        Returns:
            (Array):
        
        """

        modes=[None, "SUSY"]
        if mode not in modes:
            raise ValueError(f"Mode must be one of {modes}, but is {mode}!")
        
        if mode=="SUSY":
            return self.DDW_matrix_SUSY(moduli, moduli_c, tau, tau_c, fluxes)
        elif mode is None:
            return self.DDW_matrix_general(moduli, moduli_c, tau, tau_c, fluxes)

    ###################################################################################################################
    ######################################### F-TERM SCALAR POTENTIAL #################################################
    ###################################################################################################################
    

    @partial(jit, static_argnums = (6,7,))
    def scalar_potential(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, noscale: bool = True, normalise: bool = False
        ) -> complex:
        r"""
        
        **Description:**
        Returns the value of the :math:`F`-term scalar potential.
        
        .. admonition:: Details
            :class: dropdown
        
            The :math:`F`-term scalar potential in 4D :math:`\mathcal{N}=1` supergravity is given by

            .. math::
                V_F=\mathrm{e}^{K}\,\bigl (K^{I\bar{J}}\, D_IW\, D_{\bar{J}} \overline{W}-3|W|^2\bigl )\, .

            Here, the indices :math:`I = (1,\ldots,h^{1,2}+1)` run only over the axio-dilaton 
            and the complex structure moduli ignoring the Kähler moduli.
            Due to the no-scale structure of the classical Kähler potential, the term :math:`-3|W|^2`
            is cancelled by the Kähler moduli sector giving rise to the no-scale flux scalar potential

            .. math::
                V_F=\mathrm{e}^{K}\, K^{I\bar{J}}\, D_IW\, D_{\bar{J}} \overline{W}\, .

            In more detail, we have at the level of the classical :math:`\mathcal{N}=2` Kähler potential
            for the axio-dilaton :math:`\tau` and the complex structure moduli :math:`z^{i}`

            .. math::
                V_F=\mathrm{e}^{K}\, \bigl (K^{i\bar{\jmath}}\, D_iW\, D_{\bar{\jmath}} \overline{W}
                +K^{\tau\bar{\tau}}\, D_\tau W\, D_{\bar{\tau}} \overline{W}\bigl )\, .
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            normalise (bool, optional): If ``True``, rescales superpotential by :math:`\sqrt{2/\pi}`. Defaults to ``False``.

        Returns:
            complex: Value of the :math:`F`-term scalar potential induced by 3-form fluxes.

        Aliases:
            :func:`V`, :func:`scalar_potential`

        See also: :func:`DW`
    
        See also: :func:`inverse_kahler_metric`
    
        See also: :func:`kahler_potential`
        
        """
        
        """
        # ALTERNATIVE:
        FF = fluxes[:self.n_fluxes]
        HF = fluxes[self.n_fluxes:]
        
        G = FF-tau*HF
        cG = FF-tau_c*HF
        
        tau2 = -1j*(tau-tau_c)/2.
        
        NFlux = self.tadpole(fluxes)
        
        M = self.ISD_matrix(moduli,moduli_c)
        # Why do we have to take the inverse here?
        print("TODO: Why do we have to take the inverse here? Because we defined the ISD matrix as in https://arxiv.org/pdf/2310.06040, \
            but we use the inverse in our convention due to reordering of the periods!!")
        M = jnp.linalg.inv(M)
        
        #return -jnp.matmul(cG,jnp.matmul(M,G))/tau2,2.*NFlux,1j*jnp.matmul(cG,jnp.matmul(self.lcs_tree.sigma,G))/tau2
        return -jnp.matmul(cG,jnp.matmul(M,G))/tau2+2.*NFlux
        """

        FT = self.DW(moduli, moduli_c, tau, tau_c, fluxes)
        cFT = self.DW(moduli, moduli_c, tau, tau_c, fluxes,conj=True)
    
        Inv_KM = self.inverse_kahler_metric(moduli, moduli_c, tau, tau_c)
        
        KP = self.kahler_potential(moduli, moduli_c, tau, tau_c)

        if noscale:
            gravitino_mass_square=0.
        else:
            W=self.superpotential(moduli,tau,fluxes)
            cW=self.superpotential(moduli_c,tau_c,fluxes,conj=True)
            gravitino_mass_square=jnp.exp(KP) * W * cW

        V0 = jnp.exp(KP) * jnp.matmul(cFT, jnp.matmul(Inv_KM, FT))-3.*gravitino_mass_square

        # Normalise from normalisation of W!
        if normalise:
            V0 = 2./jnp.pi*V0
        
        return V0

    V = scalar_potential


    @partial(jit, static_argnums = (6,7,))
    def dV_z(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex,
        fluxes: Array, noscale: bool = True, conj: bool = False
        ) -> Array:
        r"""

        **Description:**
        Returns the first holomorphic derivative :math:`\partial_{z^i}V` of the
        :math:`F`-term scalar potential :math:`V` with respect to the complex
        structure moduli :math:`z^i`.

        .. admonition:: Details
            :class: dropdown

            Computed via ``jax.jacrev`` of :func:`scalar_potential` with respect to
            the holomorphic moduli (``argnums=0``). For ``conj=True``, differentiates
            with respect to the antiholomorphic moduli (``argnums=1``):

            .. math::
                (\texttt{dV\_z})_i = \partial_{z^i} V
                = \mathrm{e}^K \bigl[K_i\,(S - \lambda G) + \partial_i S - \lambda\,\partial_i G\bigr]

            where :math:`S = D_{\bar I}\bar W\,K^{I\bar J}\,D_J W` and :math:`G = |W|^2`.

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            conj (bool, optional): If ``True``, computes :math:`\partial_{\bar{z}^i}V`. Defaults to ``False``.

        Returns:
            Array: Value of :math:`\partial_{z^i}V`, shape ``(h12,)``.

        """
        if conj:
            return jax.jacrev(self.V, argnums = 1, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale)
        else:
            return jax.jacrev(self.V, argnums = 0, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale)

    @partial(jit, static_argnums = (6,7,))
    def dV_tau(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex,
        fluxes: Array, noscale: bool = True, conj: bool = False
        ) -> Array:
        r"""

        **Description:**
        Returns the first holomorphic derivative :math:`\partial_{\tau}V` of the
        :math:`F`-term scalar potential :math:`V` with respect to the axio-dilaton
        :math:`\tau`.

        .. admonition:: Details
            :class: dropdown

            Computed via ``jax.jacrev`` of :func:`scalar_potential` with respect to
            :math:`\tau` (``argnums=2``). For ``conj=True``, differentiates with
            respect to :math:`\bar\tau` (``argnums=3``):

            .. math::
                \partial_\tau V
                = \mathrm{e}^K \bigl[K_\tau\,(S - \lambda G) + \partial_\tau S
                  - \lambda\,\partial_\tau G\bigr]

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            conj (bool, optional): If ``True``, computes :math:`\partial_{\bar\tau}V`. Defaults to ``False``.

        Returns:
            complex: Value of :math:`\partial_{\tau}V` (scalar).

        """
        if conj:
            return jax.jacrev(self.V, argnums = 3, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale)
        else:
            return jax.jacrev(self.V, argnums = 2, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale)
        

    @partial(jit, static_argnums = (6,7,8,))
    def dV(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, noscale: bool = True, conj: bool = False, mode="complex"
        ) -> Array:
        r"""
        
        **Description:**
        Returns the first holomorphic derivative :math:`\partial_{I}V` of the 
        :math:`F`-term scalar potential :math:`V`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``. Becomes void if ``mode="real"``.
            mode (str, optional): String specifying whether to return complex or real derivatives. Defaults to ``"complex"``.
        
        Returns:
            Array: Value of :math:`\partial_{I}V`.

        
        """

        modes = ["complex","real"]
        if mode not in modes:
            raise ValueError(f"Unknown value for `mode`. Should be one of {modes}, but is {mode}!")

        if mode=="complex":

            dV_z = self.dV_z(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=conj)

            dV_tau = self.dV_tau(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=conj)

            return jnp.append(dV_z,dV_tau)

        else:

            dV = self.dV(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=False)
            
            dV_c = self.dV(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=True)

            dV_t = 1j*(dV-dV_c)
            
            dV_phi = dV+dV_c

            return jnp.append(dV_t,dV_phi)+1j*0.


    


    
    ###################################################################################################################
    ############################### SCALAR POTENTIAL AND DERIVATIVES WITH REAL INPUTS #################################
    ###################################################################################################################

    
    @partial(jit, static_argnums = (3))
    def V_x(
        self, x: Array, fluxes: Array, noscale: bool = True
        ) -> float:
        r"""
        
        **Description:**
        Returns the value of the real part of the F-term scalar potential :math:`V` for input of the real scalar fields.
        
        .. note::
            This function is a wrapper which is used to find non-SUSY minima by looking at gradients of the scalar potential directly,
            see for example :func:`_newton_method_flux_vacua_real`.
        
        Args:
            x (Array): JAX array of shape (:math:`2(h^{1,2}+1)`,) containing the moduli and axio-dilaton as real and imaginary parts.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
        
        Returns:
            float: Value of :math:`V`.
            

        See also: :func:`scalar_potential`
        """
        
        moduli,moduli_c,tau,tau_c = self._convert_real_to_complex(x)
        
        return self.V(moduli,moduli_c,tau,tau_c,fluxes,noscale=noscale).real
        
    @partial(jit, static_argnums = (3))
    def dV_x(
        self, x: Array, fluxes: Array, noscale: bool = True
        ) -> Array:
        r"""
        
        **Description:**
        Returns the gradients of the F-term scalar potential :math:`V` with respect to the real scalar fields for real inputs.
        
        .. admonition:: Details
            :class: dropdown
        
            Let us denote the complex scalar fields as

            .. math::
                z^i = a^i + \text{i} v^i\, ,\quad \tau = c_0 + \text{i} s\, .

            Then this function computes

            .. math::
                \nabla V = (\partial_{a^1}V, \partial_{v^1}V,\ldots ,\partial_{a^{h^{1,2}}}V, \partial_{v^{h^{1,2}}}V,\partial_{c_0}V, \partial_{s}V)\, .
        
        
        Args:
            x (Array): JAX array of shape (:math:`2(h^{1,2}+1)`,) containing the moduli and axio-dilaton as real and imaginary parts.
            fluxes (Array): Flux vector of shape (:math:`4(h^{1,2}+1)`,)
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
        
        Returns:
            Array: Value of :math:`(\partial_{a^1}V, \partial_{v^1}V,\ldots ,\partial_{a^{h^{1,2}}}V, \partial_{v^{h^{1,2}}}V,\partial_{c_0}V, \partial_{s}V)`.


        See also: :func:`V_x`
        """

        g = jax.grad(self.V_x, argnums = 0,holomorphic=False)(x,fluxes,noscale=noscale)
        
        #return jnp.real(g).flatten()
        
        return g
        
    
    @partial(jit, static_argnums = (3))
    def ddV_x(
        self, x: Array, fluxes: Array, noscale: bool = True
        ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives of the F-term scalar potential :math:`V` with respect to the real scalar fields for real inputs.
        
        .. admonition:: Details
            :class: dropdown

            Let us denote the complex scalar fields as

            .. math::
                z^i = a^i + \text{i} v^i\, ,\quad \tau = c_0 + \text{i} s\, .

            We introduce the fields

            .. math::
                \phi^\alpha = (a^1,v^1,\ldots ,a^{h^{1,2}},v^{h^{1,2}},c_0,s)\, .
            
            Then this function computes

            .. math::
                (\nabla \nabla V)_{\alpha\beta} = \partial_{\phi^\alpha}\partial_{\phi^\beta}V\, .

            Up to permutations, this function is equivalent to :func:`_hessian_real`, but slightly faster.
        
        Args:
            x (Array): JAX array of shape (:math:`2(h^{1,2}+1)`,) containing the moduli and axio-dilaton as real and imaginary parts.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
        
        Returns:
            Array: Value of :math:`\partial_{\phi^\alpha}\partial_{\phi^\beta}V`.
            

        See also: :func:`V_x`
        """
        
        return jax.jacrev(self.dV_x,argnums=0,holomorphic=False)(x,fluxes,noscale=noscale)

    
    ###################################################################################################################
    ##################################### HESSIAN AND CANONICAL NORMALISATION #########################################
    ###################################################################################################################


    @partial(jit, static_argnums = (6,7,))
    def ddV_z(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex,
        fluxes: Array, noscale: bool = True, conj: bool = False
        ) -> Array:
        r"""

        **Description:**
        Returns the holomorphic second derivatives :math:`\partial_{I}\partial_{z^j}V`
        of the :math:`F`-term scalar potential.

        .. admonition:: Details
            :class: dropdown

            Computed as ``jacrev(dV, argnums=0)`` for ``conj=False`` (holomorphic–holomorphic)
            or ``jacrev(dV, argnums=1)`` for ``conj=True`` (antiholomorphic–antiholomorphic):

            .. math::
                (\texttt{ddV\_z})_{I,j} = \partial_I \partial_{z^j} V

            where :math:`I` runs over all :math:`h^{1,2}+1` fields :math:`(z^i, \tau)`.

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            conj (bool, optional): If ``True``, computes :math:`\partial_{\bar I}\partial_{\bar z^j}V`. Defaults to ``False``.

        Returns:
            Array: Shape ``(n, h12)`` with :math:`\partial_{I}\partial_{z^j}V`.

        """

        if conj:

            return jax.jacrev(self.dV, argnums = 1, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=True)

        else:

            return jax.jacrev(self.dV, argnums = 0, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=False)


    @partial(jit, static_argnums = (6,7,))
    def ddV_cz(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex,
        fluxes: Array, noscale: bool = True, conj: bool = False
        ) -> Array:
        r"""

        **Description:**
        Returns the mixed second derivatives :math:`\partial_{I}\partial_{\bar{z}^j}V`
        of the :math:`F`-term scalar potential.

        .. admonition:: Details
            :class: dropdown

            Computed as ``jacrev(dV, argnums=1)`` for ``conj=False`` (holomorphic–antiholomorphic)
            or ``jacrev(dV, argnums=0)`` for ``conj=True``:

            .. math::
                (\texttt{ddV\_cz})_{I,\bar j} = \partial_I \partial_{\bar{z}^j} V

            This is the mixed block of the Hessian that contains the Riemann
            curvature contribution (see :func:`_hessian_SUGRA`).

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            conj (bool, optional): If ``True``, computes :math:`\partial_{\bar I}\partial_{z^j}V`. Defaults to ``False``.

        Returns:
            Array: Shape ``(n, h12)`` with :math:`\partial_{I}\partial_{\bar{z}^j}V`.

        """

        if conj:

            return jax.jacrev(self.dV, argnums = 0, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=True)

        else:

            return jax.jacrev(self.dV, argnums = 1, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=False)

    @partial(jit, static_argnums = (6,7,))
    def ddV_tau(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex,
        fluxes: Array, noscale: bool = True, conj: bool = False
        ) -> Array:
        r"""

        **Description:**
        Returns the second derivatives :math:`\partial_{I}\partial_{\tau}V` of the
        :math:`F`-term scalar potential.

        .. admonition:: Details
            :class: dropdown

            Computed as ``jacrev(dV, argnums=2)`` for ``conj=False`` or
            ``jacrev(dV, argnums=3)`` for ``conj=True``:

            .. math::
                (\texttt{ddV\_tau})_{I} = \partial_I \partial_\tau V

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            conj (bool, optional): If ``True``, computes :math:`\partial_{\bar I}\partial_{\bar\tau}V`. Defaults to ``False``.

        Returns:
            Array: Shape ``(n,)`` with :math:`\partial_{I}\partial_{\tau}V`.

        """

        if conj:

            return jax.jacrev(self.dV, argnums = 3, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=True)

        else:

            return jax.jacrev(self.dV, argnums = 2, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=False)


    @partial(jit, static_argnums = (6,7,))
    def ddV_ctau(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex,
        fluxes: Array, noscale: bool = True, conj: bool = False
        ) -> Array:
        r"""

        **Description:**
        Returns the mixed second derivatives :math:`\partial_{I}\partial_{\bar\tau}V`
        of the :math:`F`-term scalar potential.

        .. admonition:: Details
            :class: dropdown

            Computed as ``jacrev(dV, argnums=3)`` for ``conj=False``
            (holomorphic–antiholomorphic) or ``jacrev(dV, argnums=2)`` for
            ``conj=True``:

            .. math::
                (\texttt{ddV\_ctau})_{I} = \partial_I \partial_{\bar\tau} V

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            conj (bool, optional): If ``True``, computes :math:`\partial_{\bar I}\partial_\tau V`. Defaults to ``False``.
        
        Returns:
            Array: Value of :math:`\partial_{I}\partial_{\overline{\tau}}V`.

        """

        if conj:

            return jax.jacrev(self.dV, argnums = 2, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=True)

        else:

            return jax.jacrev(self.dV, argnums = 3, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=False)

    


    @partial(jit, static_argnums = (6,7,8,))
    def ddV(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex,
        fluxes: Array, noscale: bool = True, conj: bool = False, mode="complex"
        ) -> Array:
        r"""

        **Description:**
        Returns the second derivatives
        :math:`(\partial_I\partial_J V,\,\partial_I\partial_{\bar J}V)` of the
        :math:`F`-term scalar potential.

        .. admonition:: Details
            :class: dropdown

            For ``mode="complex"`` (default), returns the concatenation of the
            holomorphic block :math:`\partial_I\partial_J V` and the mixed block
            :math:`\partial_I\partial_{\bar J}V` as a :math:`n\times 2n` array.

            For ``mode="real"``, returns the real-coordinate Hessian
            :math:`\partial_{\phi^\alpha}\partial_{\phi^\beta}V` (equivalent to
            :func:`ddV_x`).

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``. Becomes void if ``mode="real"``.
            mode (str, optional): String specifying whether to return complex or real derivatives. Defaults to ``"complex"``.
        
        Returns:
            Array: Value of :math:`(\partial_{I}\partial_{J}V,\partial_{I}\partial_{\overline{J}}V)`.

        """

        modes = ["complex","real"]

        if mode not in modes:
            raise ValueError(f"Unknown value for `mode`. Should be one of {modes}, but is {mode}!")

        if mode=="complex":

            ddV_z = self.ddV_z(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=conj)

            ddV_cz = self.ddV_cz(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=conj)

            ddV_tau = self.ddV_tau(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=conj)

            ddV_ctau = self.ddV_ctau(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=conj)

            ddV = jnp.append(ddV_z,ddV_tau.reshape(-1,1),axis=1)
            ddV_mixed = jnp.append(ddV_cz,ddV_ctau.reshape(-1,1),axis=1)

            return jnp.hstack((ddV,ddV_mixed))

        else:

            ddV = self.ddV(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=conj)

            ddV_c = ddV[:,self.h12+1:]
            ddV = ddV[:,:self.h12+1]

            dt_dV = 1j*(ddV-ddV_c)
            dphi_dV = ddV+ddV_c

            return jnp.block([[jnp.real(dt_dV),jnp.real(dphi_dV)],[jnp.imag(dt_dV),jnp.imag(dphi_dV)]])


    

    
    
    @partial(jit, static_argnums = (6,))
    def _hessian_general(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, noscale: bool = True
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Hessian of the scalar potential at generic points for the complex structure moduli
        :math:`z^{i}` and the axio-dilaton :math:`\tau`.

        .. warning::
            This function is generically slower than the equivalent function :func:`ddV_x` which can also be accessed via
            :func:`hessian` with ``mode="real"``.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
        
        Returns:
            Array: Second derivatives of the Hessian.

        See also: :func:`scalar_potential`
        
        """

        h = jax.jacrev(self.dV, argnums = (0,1,2,3), holomorphic = True)

        hessian_val = h(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale)

        ddV_tz_z = hessian_val[0]
        ddV_tz_cz = hessian_val[1]
        ddV_tz_tau = hessian_val[2]
        ddV_tz_ctau = hessian_val[3]

        hess_mixed=jnp.hstack((ddV_tz_cz,ddV_tz_ctau.reshape(self.h12+1, 1)))
        hess_holom=jnp.hstack((ddV_tz_z,ddV_tz_tau.reshape(self.h12+1, 1)))
        
        e = jnp.hstack((jnp.asarray(hess_mixed), jnp.asarray(hess_holom)))

        f = jnp.hstack((jnp.asarray(jnp.conj(hess_holom)), jnp.asarray(jnp.conj(hess_mixed))))

        return jnp.vstack((e, f))
        
        
    
    
    @partial(jit, static_argnums = (6))
    def _hessian_SUSY(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, noscale: bool = True
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Hessian at SUSY loci where :math:`D_A W=0` for the complex structure moduli :math:`z^{i}`
        and the axio-dilaton :math:`\tau`.
        
        .. admonition:: Details
            :class: dropdown
        
            This implementation of the Hessian assumes that the input of complex structure moduli :math:`z^{i}`,
            axio-dilaton :math:`\tau` and fluxes satisfies the F-term conditions :math:`D_AW=0`. We use 
            standard supergravity formulas as spelled out in e.g. `hep-th/0404116 <https://inspirehep.net/literature/648462>`_. 
            That is, we compute the mixed second derivatives
        
            .. math::
                \partial_{A}\partial_{\overline{B}} V|_{D_A W=0} 
                        = \mathrm{e}^{K} \bigl (M_{AC} K^{C\overline{D}} \overline{M_{BD}}
                        -(\lambda -1) K_{A\overline{B}} |W|^2 \bigl )

            and the holomorphic second derivatives

            .. math::
                \partial_{A}\partial_{B} V|_{D_A W=0} =-(\lambda - 2) \mathrm{e}^{K} \,\overline{W}\, M_{AB}

            where we introduced

            .. math::
                M_{AB} = D_AD_BW\, , \quad \lambda = \begin{cases} 3 & \text{full potential}\\ 
                                                                0 & \text{no-scale potential}\end{cases}\; \, .

        .. warning::
            When using this function, one needs to ensure that F-term conditions :math:`D_AW=0` are satisfied
            with sufficiently small numerical tolerance. In general, this function only provides an approximation
            for the Hessian at a minimum where F-term conditions :math:`|D_A W|<\epsilon`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
        
        Returns:
            Array: Hessian matrix with entries :math:`\partial_{A}\partial_{\bar{B}} V`
                and :math:`\partial_{A}\partial_{B} V` assuming :math:`D_A W=0`.
        
        """
        
        DDW_val=self.DDW(moduli, moduli_c, tau, tau_c,fluxes,mode="SUSY")
        W_val=self.superpotential(moduli, tau,fluxes)
        KM_val=jnp.array(self.kahler_metric(moduli,moduli_c,tau,tau_c))
        KP=self.kahler_potential(moduli, moduli_c, tau, tau_c)
        Inv_KM=self.inverse_kahler_metric(moduli, moduli_c, tau, tau_c)
        
        if noscale:
            lam=0.
        else:
            lam=3.
        
        term1 = jnp.matmul(DDW_val, jnp.matmul(Inv_KM.T,jnp.conj(DDW_val)))
        term2 = -(lam-1.)*W_val*jnp.conj(W_val)*KM_val
        V_I_barJ = jnp.exp(KP) * (term1+term2)
        V_I_J = jnp.exp(KP) * (2.-lam)*DDW_val*jnp.conj(W_val)

        hess_mixed,hess_holom=V_I_barJ,V_I_J
            
        a = jnp.hstack((jnp.asarray(hess_mixed), jnp.asarray(hess_holom)))

        b = jnp.hstack((jnp.asarray(jnp.conj(hess_holom)), jnp.asarray(jnp.conj(hess_mixed))))

        return jnp.vstack((a, b))
    
    
    @partial(jit, static_argnums = (6,))
    def _hessian_SUGRA(
        self,
        moduli: Array,
        moduli_c: Array,
        tau: complex,
        tau_c: complex,
        fluxes: Array,
        noscale: bool = True
        ) -> Array:
        r"""

        **Description:**
        Returns the Hessian of the scalar potential for the complex structure moduli
        :math:`z^{i}` and the axio-dilaton :math:`\tau`, computed from explicit
        :math:`\mathcal{N}=1` supergravity building blocks.

        .. admonition:: Details
            :class: dropdown

            Decomposes the :math:`F`-term scalar potential as
            :math:`V = \mathrm{e}^K \cdot (S - \lambda\, G)` where

            .. math::
                S = D_{\bar{I}}\bar{W}\, K^{I\bar{J}}\, D_J W\,,\quad
                G = |W|^2\,,\quad
                \lambda = \begin{cases} 3 & \text{full potential}\\ 0 & \text{no-scale}\end{cases}\,.

            The Hessian is assembled via the product rule applied to :math:`V = \mathrm{e}^K\,F`:

            .. math::
                \partial_A\partial_{\bar{B}} V = \mathrm{e}^K \bigl[
                K_A K_{\bar{B}}\, F + K_{\bar{B}}\,\partial_A F
                + K_{A\bar{B}}\, F + K_A\,\partial_{\bar{B}} F
                + \partial_A\partial_{\bar{B}} F \bigr]

            .. math::
                \partial_A\partial_B V = \mathrm{e}^K \bigl[
                K_A K_B\, F + K_B\,\partial_A F
                + K_{AB}\, F + K_A\,\partial_B F
                + \partial_A\partial_B F \bigr]

            The first and second derivatives of :math:`S` are expanded into 9 product-rule
            terms each, using the Christoffel symbols :math:`\Gamma^E_{AC}` (from
            :func:`christoffel_symbols`), the Riemann tensor :math:`R_{i\bar{\jmath}k\bar{l}}`
            (from :func:`riemann_tensor`), the Kähler metric and its inverse, and the
            :math:`F`-terms :math:`D_I W`.

            The derivatives of :math:`G = W\bar{W}` are straightforward:

            .. math::
                \partial_A G = (\partial_A W)\,\bar{W}\,,\quad
                \partial_{\bar{B}} G = W\,(\partial_{\bar{B}}\bar{W})\,,\quad
                \partial_A\partial_{\bar{B}} G = (\partial_A W)\,(\partial_{\bar{B}}\bar{W})\,,\quad
                \partial_A\partial_B G = (\partial_A\partial_B W)\,\bar{W}\,.

            The 9 mixed terms of :math:`\partial_A\partial_{\bar{B}} S` arise from differentiating
            the three factors in :math:`S = \underbrace{D_{\bar{I}}\bar{W}}_{X}\,
            \underbrace{K^{I\bar{J}}}_{Y}\,\underbrace{D_J W}_{Z}`. Each factor has known holomorphic
            and anti-holomorphic derivatives:

            - :math:`\partial_A X_I = K_{A\bar{I}}\,\bar{W}`, :math:`\partial_{\bar{B}} X_I` via :func:`dDW` with ``conj=True``
            - :math:`\partial_A Y = -\Gamma^I_{AC}\,K^{C\bar{J}}`, :math:`\partial_{\bar{B}} Y` via :func:`dIKM_c`
            - :math:`\partial_A Z_J = \texttt{dDW}[J,A]`, :math:`\partial_{\bar{B}} Z_J = K_{J\bar{B}}\,W`
            - :math:`\partial_A\partial_{\bar{B}} Y` via :func:`ddIKM` (uses Riemann tensor explicitly)

        .. warning::
            This function does not call :func:`scalar_potential` or :func:`dV`.
            All derivatives are of the supergravity building blocks
            :math:`D_I W`, :math:`K^{I\bar{J}}`, and :math:`W`.

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.

        Returns:
            Array: Hessian matrix with entries :math:`\partial_{A}\partial_{\bar{B}} V`
                and :math:`\partial_{A}\partial_{B} V`.

        See also: :func:`_hessian_general`, :func:`_hessian_SUSY`, :func:`scalar_potential`

        """
        n = self.h12 + 1
        lam = 0. if noscale else 3.

        # ── 0th order SUGRA quantities ──
        DW   = self.DW(moduli, moduli_c, tau, tau_c, fluxes)
        DWc  = self.DW(moduli, moduli_c, tau, tau_c, fluxes, conj=True)
        W    = self.superpotential(moduli, tau, fluxes)
        Wc   = self.superpotential(moduli_c, tau_c, fluxes, conj=True)
        KM   = jnp.array(self.kahler_metric(moduli, moduli_c, tau, tau_c))
        IKM  = self.inverse_kahler_metric(moduli, moduli_c, tau, tau_c)
        eK   = jnp.exp(self.kahler_potential(moduli, moduli_c, tau, tau_c))
        dK   = self.dK(moduli, moduli_c, tau, tau_c)
        dKc  = self.dK_c(moduli, moduli_c, tau, tau_c)
        dDW_val = self.dDW(moduli, moduli_c, tau, tau_c, fluxes)
        Gamma = self.christoffel_symbols(moduli, moduli_c, tau, tau_c)

        # Kähler geometry tensors
        K3a       = self.dddK_c(moduli, moduli_c, tau, tau_c)
        dIKM_bar  = self.dIKM_c(moduli, moduli_c, tau, tau_c)
        dDWc_Bb   = self.dDW(moduli, moduli_c, tau, tau_c, fluxes, conj=True)
        d2DW_ABb  = self.ddDW(moduli, moduli_c, tau, tau_c, fluxes, conj=True)
        d2IKM_ABb = self.ddIKM(moduli, moduli_c, tau, tau_c)

        dWc_Bb = self.dW(moduli_c, tau_c, fluxes, conj=True)

        v = IKM @ DW

        # ── K_{AB} (holomorphic second derivative of K) ──
        KAB = jnp.block([
            [self.ddK_z_z(moduli, moduli_c, tau, tau_c),
             self.ddK_z_tau(moduli, moduli_c, tau, tau_c).reshape(-1, 1)],
            [self.ddK_z_tau(moduli, moduli_c, tau, tau_c).reshape(1, -1),
             jnp.atleast_2d(self.ddK_tau_tau(moduli, moduli_c, tau, tau_c))]
        ])

        # ══════════════════════════════════════════════════════════
        # ∂_A S — holomorphic first derivative (3 product-rule terms)
        # ══════════════════════════════════════════════════════════
        dS_A = (Wc * DW
                - jnp.einsum('i,iac,c->a', DWc, Gamma, v)
                + (DWc @ IKM) @ dDW_val)

        # ∂_{B̄} S — antiholomorphic first derivative
        dS_Bc = (W * DWc
                 + jnp.einsum('i,ijb,j->b', DWc, dIKM_bar, DW)
                 + jnp.einsum('ib,ij,j->b', dDWc_Bb, IKM, DW))

        # ══════════════════════════════════════════════════════════
        # ∂_A ∂_{B̄} S — 9 product-rule terms (mixed)
        # ══════════════════════════════════════════════════════════
        T1m = jnp.einsum('aib,i->ab', K3a, v) * Wc + jnp.outer(DW, dWc_Bb)
        T2m = Wc * jnp.einsum('ai,ijb,j->ab', KM, dIKM_bar, DW)
        T3m = Wc * W * KM
        T4m = -jnp.einsum('ib,ia->ab', dDWc_Bb, jnp.einsum('iac,c->ia', Gamma, v))
        T5m = jnp.einsum('i,ijba,j->ab', DWc, d2IKM_ABb, DW)
        T6m = -W * jnp.einsum('i,iab->ab', DWc, Gamma)
        T7m = jnp.einsum('ib,ij,ja->ab', dDWc_Bb, IKM, dDW_val)
        T8m = jnp.einsum('i,ijb,ja->ab', DWc, dIKM_bar, dDW_val)
        T9m = jnp.einsum('i,ij,jab->ab', DWc, IKM, d2DW_ABb)

        d2S_ABc = T1m + T2m + T3m + T4m + T5m + T6m + T7m + T8m + T9m

        # ══════════════════════════════════════════════════════════
        # ∂_A ∂_B S — 9 product-rule terms (holomorphic)
        # ══════════════════════════════════════════════════════════
        K3h = self.dddK(moduli, moduli_c, tau, tau_c)
        dGamma_hol = self.dGamma(moduli, moduli_c, tau, tau_c)
        d2DW_hol = self.ddDW(moduli, moduli_c, tau, tau_c, fluxes, conj=False)

        d2IKM_hol = (-jnp.einsum('iacb,cj->ijab', dGamma_hol, IKM)
                     + jnp.einsum('iac,cbd,dj->ijab', Gamma, Gamma, IKM))

        T1h = jnp.einsum('aib,ij,j->ab', K3h, IKM, DW) * Wc
        T2h = -Wc * jnp.einsum('ai,ibc,c->ab', KM, Gamma, v)
        T3h = Wc * dDW_val
        T4h = -Wc * jnp.einsum('bi,iac,c->ab', KM, Gamma, v)
        T5h = jnp.einsum('i,ijab,j->ab', DWc, d2IKM_hol, DW)
        T6h = -jnp.einsum('i,iac,cj,jb->ab', DWc, Gamma, IKM, dDW_val)
        T7h = Wc * dDW_val.T
        T8h = -jnp.einsum('i,ibc,cj,ja->ab', DWc, Gamma, IKM, dDW_val)
        T9h = jnp.einsum('i,ij,jab->ab', DWc, IKM, d2DW_hol)

        d2S_AB = T1h + T2h + T3h + T4h + T5h + T6h + T7h + T8h + T9h

        # ══════════════════════════════════════════════════════════
        # G = W W̄  derivatives
        # ══════════════════════════════════════════════════════════
        dW_A  = self.dW(moduli, tau, fluxes)
        ddW_  = self.ddW(moduli, tau, fluxes)

        dG_A    = dW_A * Wc
        dG_Bc   = W * dWc_Bb
        d2G_ABc = jnp.outer(dW_A, dWc_Bb)
        d2G_AB  = ddW_ * Wc

        # ══════════════════════════════════════════════════════════
        # Assemble F = S - λG and its derivatives
        # ═══════════════════════���════════════════════════════════��═
        S_val   = DWc @ IKM @ DW
        F_val   = S_val - lam * W * Wc
        dF_A    = dS_A    - lam * dG_A
        dF_Bc   = dS_Bc   - lam * dG_Bc
        d2F_ABc = d2S_ABc - lam * d2G_ABc
        d2F_AB  = d2S_AB  - lam * d2G_AB

        # ══════════════════════════════════════════════════════════
        # Product rule: V = e^K F
        # ══════════════════════════════════════════════════════════
        V_mixed = eK * (jnp.outer(dK, dKc) * F_val + jnp.outer(dF_A, dKc)
                        + KM * F_val + jnp.outer(dK, dF_Bc) + d2F_ABc)

        V_holom = eK * (jnp.outer(dK, dK) * F_val + jnp.outer(dF_A, dK)
                        + KAB * F_val + jnp.outer(dK, dF_A) + d2F_AB)

        top = jnp.hstack((V_mixed, V_holom))
        bot = jnp.hstack((jnp.conj(V_holom), jnp.conj(V_mixed)))
        return jnp.vstack((top, bot))
            
    
    @partial(jit, static_argnums = (6,7,))
    def hessian(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, 
        fluxes: Array, noscale: bool = True, mode: str = None
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Hessian of the scalar potential.
        
        .. admonition:: Details
            :class: dropdown
        
            Explicitly, we compute the following second derivatives of the scalar potential,
        
            .. math::
                V_{A\overline{B}}=\partial_{A}\partial_{\bar{B}} V \; ,\quad V_{AB}=\partial_{A}\partial_{B} V\, .
        
            We return the Hessian matrix
        
            .. math::
                H = \left (\begin{array}{cc}V_{A\overline{B}} & V_{AC}\ V_{\overline{D}\overline{B}} &  V_{\overline{D}C} \end{array}\right )
        
            This matrix is used to compute the mass spectrum from :func:`mass_matrix`.
        
        .. note::
            We provide two computational modes which can be set via the optional argument ``mode``.
            If we compute the Hessian at generic points in moduli space,
            then we should use ``mode=None`` which is also the default.
            At SUSY minimum where :math:`D_I W=0` for all fields,
            we can use a simplified version of the Hessian which is faster to evaluate.
            To do so, we have to use ``mode="SUSY"`` instead.

        .. warning::
            When using ``mode="SUSY"``, one should be cautious regarding numerical errors and noise.
            
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            mode (string, optional): The mode to compute the Hessian. For ``mode=None``, returns the
                                    general Hessian from explicit second derivatives of the scalar potential.
                                    For ``"SUSY"``, computes Hessian at a SUSY locus assuming
                                    :math:`D_I W=0` using standard SUGRA formulas for the Hessian.
                                    For ``"SUGRA"``, computes the Hessian from explicit SUGRA building blocks
                                    (:math:`D_I W`, :math:`K^{I\bar{J}}`, :math:`W`, :math:`\Gamma`, :math:`R`)
                                    at generic points. See :func:`_hessian_SUGRA`.
                                    For ``"real"``, computes Hessian for the real and imaginary components
                                    of the complex scalar fields.
                                    Defaults to ``None``.
        
        
        Returns:
            Array: Hessian matrix with entries :math:`\partial_{A}\partial_{\bar{B}} V` and :math:`\partial_{A}\partial_{B} V`.

        Aliases:
            :func:`H`, :func:`hessian`
            
        
        """
        modes=[None,"SUSY","SUGRA","real"]
        if mode not in modes:
            raise ValueError(f"Cannot determine `mode` to compute Hessian!\
                    `mode` should be one of {modes}, but got {mode}.")

        if mode is None:
            return self._hessian_general(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale)
        elif mode=="SUSY":
            return self._hessian_SUSY(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale)
        elif mode=="SUGRA":
            return self._hessian_SUGRA(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale)
        elif mode=="real":
            x = self._convert_complex_to_real(moduli, moduli_c, tau, tau_c)
            return self.ddV_x(x,fluxes,noscale=noscale)
        
    H = hessian

    @partial(jit, static_argnums = (6,7,8,))
    def mass_matrix(
        self, moduli: Array, moduli_c: Array, tau: complex, tau_c: complex, fluxes: Array,
        mode: str = None, noscale: bool = True, use_real_derivatives: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the mass matrix after canonical normalisation of the kinetic terms.
        
        .. admonition:: Details
            :class: dropdown

            In the action, we expand the scalar potential :math:`V(\phi^A,\overline{\phi}^A)` in terms of the fields :math:`\phi^A =(z^i, \tau)`
            around a reference point :math:`\phi^A_0=(z^i_0,\tau_0)`
        
            .. math::
                V(\phi^A_0+\phi^A , \overline{\phi}^A_0+\overline{\phi}^A )=V_0+ (V_{B})_0 \phi^B+(V_{AB})_0 \phi^A\phi^B
                    +(V_{\overline{A}B})_0 \phi^\overline{A}\phi^B+\text{c.c.} +\ldots
        
            where, by abuse of notation, we call the fluctuations collectively :math:`\phi^A\in\lbrace z^i,\tau\rbrace`,
            :math:`A=1,\ldots, h^{1,2}+1`. Further, we defined
        
            .. math::
                V_0=V(\phi^A_0 , \overline{\phi}^A_0 )\; ,\quad (V_{B})_0 = \partial_{B}V(\phi^A_0 , \overline{\phi}^A_0 )\; ,\quad \ldots
            
            We are primarily interested in evaluating this function for vaccum solutions :math:`\phi^A_0=(z^i_0,\tau_0)` 
            in which case
        
            .. math::
                (\partial_{B}V)_0 = 0 \; ,\quad (\partial_{\overline{B}}V)_0 = 0 \, .
        
            Further, in the case of vacua satisfying the :math:`F`-term conditions :math:`D_A W=0`, we compute the vacuum energy
        
            .. math::
                V_0= \begin{cases} -3\mathrm{e}^{K}\, |W|^2 & \text{full potential}\\
                                                      0 & \text{no-scale potential}\, .
                                        \end{cases}
        
            In particular, for no-scale models like for type IIB flux vacua, we find a vanishing cosmological constant.
            In this case, by introducing :math:`\Phi^{\Lambda}=(\phi^A,\overline{\phi}^{\overline{B}})`, the above becomes
        
            .. math::
                V(\phi^A_0+\phi^A , \overline{\phi}^A_0+\overline{\phi}^A )=\Phi^{\Lambda}H_{\Lambda\overline{\Delta}}(\phi^A_0 , \overline{\phi}^A_0 )\overline{\Phi}^{\overline{\Delta}} +\ldots
        
            where :math:`H_{\Lambda\overline{\Delta}}(\phi^A_0 , \overline{\phi}^A_0 )` is the Hessian evaluated at the minimum
            which is determined as (see :func:`hessian`)

            .. math::
                H_{\Lambda\overline{\Delta}}(\phi^A_0,\overline{\phi}^A_0) = \left (\begin{array}{cc}V_{A\overline{B}} \;&\; V_{AC}\\[0.3em] 
                V_{\overline{D}\overline{B}} \;&\;  V_{\overline{D}C} \end{array}\right ) \, .

            To obtain the mass matrix, we first perform a change of basis by diagonalising :math:`K_{A\overline{B}}`.
            That is, we compute the matrix :math:`\tilde{U}` of normalised eigenvectors of :math:`K_{A\overline{B}}`.
            Then, in order to define a canonically normalised basis of fields :math:`\psi^A`, we define
        
            .. math::
                U=\text{diag}(\sqrt{\lambda_1},\ldots ,\sqrt{\lambda_{h^{1,2}+1}})\, \tilde{U}

            so that

            .. math::
                \phi^A = U^{A}\,_{B} \psi^B

            and thus

            .. math::
                K_{A\overline{B}}\partial \phi^A\partial \overline{\phi}^B = \delta_{A\overline{B}}\partial \psi^A\partial \overline{\psi}^B \, .
        
            .. warning::
                Notice that by definition :math:`U` is clearly not a unitary matrix anymore because its determinant 
                does not necessarily satisfy :math:`|\text{det}(U)|=1`.


            Then, the mass matrix for the canonically normalised fields is defined as
        
            .. math::
                M = \mathbb{U} H \mathbb{U}^{\dagger}\, , \quad \mathbb{U} = \left (\begin{array}{cc}U^{A\overline{B}} \;&\; 0\[0.3em] 0 \;&\;  U^{\overline{D}C} \end{array}\right ) 

            or, in components,
        
            .. math::
                M = \left (\begin{array}{cc}U_{E}\,^{A} V_{A\overline{B}}U^{\overline{B}}\,_{\overline{F}} \quad&\quad U_{E}\,^{A}V_{AC}U^{C}\,_{F}\\[0.8em]
                    U_{\overline{E}}\,^{\overline{D}}V_{\overline{D}\overline{B}}U^{\overline{B}}\,_{\overline{F}} \quad&\quad  U_{\overline{E}}\,^{\overline{D}}V_{\overline{D}C}U^{C}\,_{F} \end{array}\right ) \, .
        
        
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (string, optional): The mode to compute the mass matrix. For ``mode=None``, uses the 
                                    general Hessian from explicit second derivatives of the scalar potential.
                                    For ``mode="SUSY"``, computes Hessian at a SUSY locus assuming 
                                    :math:`D_I W=0` using standard SUGRA formulas for the Hessian.
                                    For ``"real"``, calculates Hessian for the real and imaginary components 
                                    of the complex scalar fields.
                                    Defaults to ``None``.
            
        
        Returns:
            Array: Mass matrix of the canonically normalised fields.
        
        """
        
        modes=[None, "SUSY", "real"]
        if mode not in modes:
            raise ValueError(f"Cannot determine `mode` to compute Hessian!\
                    `mode` should be one of {modes}, but got {mode}.")
        
        # Alternative method:
        # THIS IS OFF BY FACTOR OF 2. WHY? -> Normalisation of complex vs. real scalar:
        # Complex scalar: phi = (a+i*b)/sqrt(2)
        # This implies: m^2 phi*phi^dagger = 1/2 m^2 * (a^2+b^2).
        if mode=="real":
            KM = self.kahler_metric(moduli, moduli_c, tau, tau_c)
            diagonalizer = jnp.linalg.cholesky(jnp.linalg.inv(KM))
            SP_hessian = self.hessian(moduli, moduli_c, tau, tau_c,fluxes,noscale=noscale,mode="real")

            return 1/2*jnp.block([[diagonalizer.T@SP_hessian[:self.dimension_H3,:self.dimension_H3]@diagonalizer,diagonalizer.T@SP_hessian[:self.dimension_H3,self.dimension_H3:]@diagonalizer],
                             [diagonalizer.T@SP_hessian[self.dimension_H3:,:self.dimension_H3]@diagonalizer,diagonalizer.T@SP_hessian[self.dimension_H3:,self.dimension_H3:]@diagonalizer]])

            
        
        KM = self.kahler_metric(moduli, moduli_c, tau, tau_c)
        
        eigvals, eigvecs = jnp.linalg.eigh(KM)
        
        # For canonical normalisation, we have to also absorb the eigenvalue in the basis change
        eigenvec_mat = jnp.matmul(eigvecs,jnp.sqrt(eigvals)*jnp.identity(self.dimension_H3))

        # The new normalised eigenvectors are the columns of `eigenvec_mat`: eigenvec_mat[:,i]

        # eigvecs is in fact a unitary matrix, but `eigenvec_mat` is not anymore because of the change of normalisation!
        eigenvec_mat_inv = jnp.linalg.inv(eigenvec_mat)

        # Stack the transformation matrix together with its conjugate -> we work with a field vector Phi = (z^i, \overline{z}^i)
        eigenvec_mat_stacked1 = jnp.hstack( ( eigenvec_mat_inv.T, jnp.zeros((self.dimension_H3,self.dimension_H3)) ) )
        eigenvec_mat_stacked2=jnp.hstack( ( jnp.zeros((self.dimension_H3,self.dimension_H3)) , jnp.conj(eigenvec_mat_inv.T) ) )
        eigenvec_mat_stacked=jnp.vstack( ( eigenvec_mat_stacked1, eigenvec_mat_stacked2 ) )


        # Evaluate the Hessian
        SP_hessian = self.hessian(moduli, moduli_c, tau, tau_c, fluxes,mode=mode,noscale=noscale)
        
        #Rotate the Scalar potential hessian using the normalised eigenvectors of the Kähler Metric
        return jnp.matmul( eigenvec_mat_stacked.T , jnp.matmul(SP_hessian, eigenvec_mat_stacked))
    
    @partial(jit,static_argnums=())
    def compute_tau_vev(self,moduli,flux):
        r"""
        **Description:**
        Computes the value of the axio-dilaton :math:`\tau` from the fluxes and complex structure moduli as a solution to the corresponding :math:`F`-term condition :math:`D_{\tau}W=0`.
        
        .. admonition:: Details
            :class: dropdown

            The axio-dilaton :math:`\tau` can be computed from the fluxes and complex structure moduli as a solution to the :math:`F`-term condition
            
            .. math::
                D_{\tau}W = \partial_{\tau}W + W \partial_{\tau}K = 0\, .
                
            This condition can be written as
            
            .. math::
                D_{\tau}W = -\frac{1}{\tau-\overline{\tau}} (f - \overline{\tau} h)\cdot \Sigma \cdot \overline{\Pi}(z^i) = 0\, ,
            
            and the corresponding solution is given by

            .. math::
                \tau = \frac{f\cdot \Sigma \cdot \overline{\Pi}(z^i)}{h\cdot \Sigma \cdot \overline{\Pi}(z^i)}\, ,

            where :math:`f=(f_1,f_2)` and :math:`h=(h_1,h_2)` are the RR- and NSNS-fluxes respectively with :math:`f_i,h_i \in \mathbb{Z}^{h^{1,2}+1}`.
            Further, :math:`\Pi(z^i)` is the period vector depending on the complex structure moduli and :math:`\Sigma` is the symplectic matrix defined in :func:`period_matrix`.
            
        .. note::
        
            Notice that this equation only has a solution if the denominator is non-zero, i.e. if

            .. math::
                h\cdot \Sigma \cdot \overline{\Pi}(z^i) \neq 0\, .

            If this condition is not satisfied, then this function raises a ``ZeroDivisionError``.
            
        Args:
            moduli (Array): Complex structure moduli values.
            flux (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            
        Returns:
            complex: Value of the axio-dilaton :math:`\tau`.

        """
        
        f = flux[:self.n_fluxes]
        h = flux[self.n_fluxes:]
            
        Pi = self.period_vector(jnp.conj(moduli),conj=True)
        
        SigmaPi = jnp.matmul(self.periods.sigma,Pi)
        
        return jnp.matmul(f,SigmaPi)/jnp.matmul(h,SigmaPi)

    
    @partial(jit, static_argnums = (6,))
    def ISD_condition(self,moduli,moduli_c,tau,tau_c,fluxes, mode="complex"):
        r"""
        **Description:**
        Checks whether the fluxes satisfy the ISD condition :math:`\star G_3=\text{i}G_3`.

        .. admonition:: Details
            :class: dropdown

            If the 3-form fluxes are ISD, then the RR-fluxes :math:`f=(f_1,f_2)` and NSNS-fluxes :math:`h=(h_1,h_2)` 
            with :math:`f_i,h_i \in \mathbb{Z}^{h^{1,2}+1}` satisfy

            .. math::
                f_1-\tau h_1=\overline{\mathcal{N}}(z^i,\overline{z}^i)\, (f_2-\tau h_2)\, ,

            where the matrix :math:`\mathcal{N}` corresponding to the gauge kinetic function is computed using :func:`gauge_kinetic_matrix`. 
            Alternatively, this condition can also be written as

            .. math::
                f=(s\,  M(z^i,\overline{z}^i)\Sigma + c_0)\, h\; ,\quad \tau=c_0 + \text{i} s \, .
            
            where the matrix :math:`M(z^i,\overline{z}^i)` (also called ISD-matrix) is computed via :func:`ISD_matrix`.
            
            This function computes the difference between left and right hand side of either of these equations depending on the input.
            If ``mode="complex"``, then it returns

            .. math::
                f_1-\tau h_1 - \overline{\mathcal{N}}(z^i,\overline{z}^i)\, (f_2-\tau h_2)\, ,
                
            and for ``mode="real"``
            
            .. math::
                f-(s \,  M(z^i,\overline{z}^i)\Sigma +  c_0)\, h\, .

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): String specifying which version of the ISD equation to use. Options are :c:var`"complex"` or :c:var`"real"`. Defaults to :c:var`mode = "complex"`.

        Returns:
            Array: Difference between the left and right hand side of the ISD equation.

        """
            
        modes = ["complex","real"]
        if mode not in modes:
            raise ValueError(f"Unknown value for `mode`. Should be one of {modes}, but is {mode}!")
        
        FF = fluxes[:self.n_fluxes]
        HF = fluxes[self.n_fluxes:]
        
        if mode == "real":
            MMatVal=self.ISD_matrix(moduli,moduli_c)
            c0 = (tau+tau_c)/2
            s = -1j*(tau-tau_c)/2

            # Initialise residual as F-flux
            res = FF

            # Subtract LHS of ISD condition
            res = res - (jnp.matmul(MMatVal,jnp.matmul(self.periods.sigma,HF))*s+c0*HF)

            return res
        else:
            # Compute matrix
            cNMatVal=self.gauge_kinetic_matrix(moduli,moduli_c,conj=True)

            return FF[:self.dimension_H3]-tau*HF[:self.dimension_H3]-jnp.matmul(cNMatVal,(FF[self.dimension_H3:]-tau*HF[self.dimension_H3:]))    


    @partial(jit, static_argnums = ())
    def _get_hf_from_N(
                      self,
                      N: Array,
                      tau: complex
                      ) -> Tuple[Array,Array]:
        r"""
        **Description:**
        Dummy function to get NSNS- and RR-fluxes from value of array ``N``
        and axio-dilaton values :math:`\tau`.
        
        Args:
            N (Array): Array.
            tau (complex): Value of the axio-dilaton.
        
        Returns:
            Array: NSNS-flux.
            Array: RR-flux.
        """

        h = -jnp.imag(N)/jnp.imag(tau)

        f = jnp.real(N)+jnp.real(tau)*h

        return h,f

    @partial(jit, static_argnums = ())
    def _projection_fluxes_0_3(
                              self,
                              moduli: Array,
                              tau: complex,
                              fluxes: Array,
                              Kcs: complex,
                              cpi_vec: Array):
        r"""
        **Description:**
        Computes the :math:`(0,3)`-component of the 3-form flux.
        
        Args:
            moduli (Array): Values for the complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            Kcs (complex): Value of the complex structure Kähler potential :math:`K_{\mathrm{cs}}`.
            pi_vec (Array): Complex conjugate period vector :math:`\overline{\Pi}`.
        
        Returns:
            Array: :math:`(0,3)`-component of the 3-form flux.
        """
        W0 = self.superpotential(moduli,tau,fluxes)
        N = -1j*jnp.exp(Kcs)*W0*cpi_vec
        
        h,f = self._get_hf_from_N(N,tau)
        
        return jnp.append(f,h)

    @partial(jit, static_argnums = ())
    def _projection_fluxes_2_1(
                              self,
                              moduli: Array,
                              tau: complex,
                              fluxes: Array,
                              Kcs: complex,
                              pi_vec: Array,
                              IKM: Array
                              ):
        r"""
        **Description:**
        Computes the :math:`(2,1)`-component of the 3-form flux.
        
        Args:
            moduli (Array): Values for the complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            Kcs (complex): Value of the complex structure Kähler potential :math:`K_{\mathrm{cs}}`.
            pi_vec (Array): Period vector :math:`\Pi`.
            IKM (Array): Inverse Kähler metric :math:`K^{i\bar{\jmath}}`.
        
        Returns:
            Array: :math:`(2,1)`-component of the 3-form flux.

        """
        DDW = self.DDW_general(moduli,jnp.conj(moduli),tau,jnp.conj(tau),fluxes)
        cDDW = jnp.conj(DDW)
        
        DPi = self.periods.D_period_vector_per(self.moduli_to_periods(moduli),self.moduli_to_periods(jnp.conj(moduli),conj=True),conj=False)
        DPi = DPi[:,1:].T

        N = -1j*jnp.exp(Kcs)*jnp.matmul(jnp.matmul(jnp.conj(IKM),cDDW)[:-1].T,DPi)[-1]*(tau-jnp.conj(tau))
        h,f = self._get_hf_from_N(N,tau)
        
        return jnp.append(f,h)

    @partial(jit, static_argnums = ())
    def _projection_fluxes_3_0(
                              self,
                              moduli: Array,
                              tau: complex,
                              fluxes: Array,
                              Kcs: complex,
                              pi_vec: Array):
        r"""
        **Description:**
        Computes the :math:`(3,0)`-component of the 3-form flux.
        
        Args:
            moduli (Array): Values for the complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            Kcs (complex): Value of the complex structure Kähler potential :math:`K_{\mathrm{cs}}`.
            pi_vec (Array): Period vector :math:`\Pi`.
        
        Returns:
            Array: :math:`(3,0)`-component of the 3-form flux.

        """

        Ftau = self.DW_tau(moduli,jnp.conj(moduli),tau,jnp.conj(tau),fluxes)
        
        N = 1j*jnp.exp(Kcs)*jnp.conj(Ftau)*(tau-jnp.conj(tau))*pi_vec
        
        h,f = self._get_hf_from_N(N,tau)
        
        return jnp.append(f,h)

    @partial(jit, static_argnums = ())
    def _projection_fluxes_1_2(
                              self,
                              moduli: Array,
                              tau: complex,
                              fluxes: Array,
                              Kcs: complex,
                              IKM: Array,
                              pi_vec: Array
                              ):
        r"""
        **Description:**
        Computes the :math:`(1,2)`-component of the 3-form flux.
        
        Args:
            moduli (Array): Values for the complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            Kcs (complex): Value of the complex structure Kähler potential :math:`K_{\mathrm{cs}}`.
            IKM (Array): Inverse Kähler metric :math:`K^{i\bar{\jmath}}`.
            pi_vec (Array): Period vector :math:`\Pi`.
        
        Returns:
            Array: :math:`(1,2)`-component of the 3-form flux.

        """
        fterms_mod = self.DW_z(moduli,jnp.conj(moduli),tau,jnp.conj(tau),fluxes)
        
        Fi_bar = jnp.matmul(IKM[:-1,:-1],fterms_mod)

        DPi = self.periods.D_period_vector_per(self.moduli_to_periods(moduli),self.moduli_to_periods(jnp.conj(moduli),conj=True),conj=False)
        DPi = DPi[:,1:].T

        cDPi = jnp.conj(DPi)
        
        N = 1j*jnp.exp(Kcs)*jnp.matmul(Fi_bar,cDPi)
        h,f = self._get_hf_from_N(N,tau)
        
        return jnp.append(f,h)

        
    @partial(jit, static_argnums = (4,))
    def projection_fluxes(
                          self,
                          moduli: Array,
                          tau: complex,
                          fluxes: Array,
                          mode: str = None
                          ):
        r"""

        **Description:**
        Computes the Hodge decomposition of the 3-form flux :math:`G_3 = F_3 - \tau H_3`
        into its :math:`(p,q)`-components on the Calabi-Yau threefold.

        .. admonition:: Details
            :class: dropdown

            The complexified 3-form flux decomposes as

            .. math::
                G_3 = G_3^{(3,0)} + G_3^{(2,1)} + G_3^{(1,2)} + G_3^{(0,3)}

            The ISD condition (imaginary self-dual) requires :math:`G_3^{(3,0)} = G_3^{(1,2)} = 0`.
            At a SUSY minimum, additionally :math:`G_3^{(0,3)} = 0`, leaving only the
            :math:`(2,1)`-component.

            For ``mode="SUSY"``, returns only the :math:`(2,1)` and :math:`(0,3)` components.

        Args:
            moduli (Array): Values for the complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): If ``mode="SUSY"``, returns only :math:`(2,1)` and :math:`(0,3)` components.

        Returns:
            Array: :math:`(3,0)`-component of the 3-form flux.
            Array: :math:`(2,1)`-component of the 3-form flux.
            Array: :math:`(1,2)`-component of the 3-form flux.
            Array: :math:`(0,3)`-component of the 3-form flux.

        """
        pi_vec = self.period_vector(moduli)
        cpi_vec = jnp.conj(pi_vec)
        
        Kcs = -jnp.log(-1j*jnp.matmul(cpi_vec,jnp.matmul(self.periods.sigma,pi_vec)))
        
        IKM = self.inverse_kahler_metric(moduli,jnp.conj(moduli),tau,jnp.conj(tau))
        
        N_0_3 = self._projection_fluxes_0_3(moduli,tau,fluxes,Kcs,cpi_vec)
        N_2_1 = self._projection_fluxes_2_1(moduli,tau,fluxes,Kcs,pi_vec,IKM)
        
        if mode=="SUSY":
            return N_2_1,N_0_3
        else:
            
            N_3_0 = self._projection_fluxes_3_0(moduli,tau,fluxes,Kcs,pi_vec)
            N_1_2 = self._projection_fluxes_1_2(moduli,tau,fluxes,Kcs,IKM,pi_vec)
            
            return N_3_0,N_2_1,N_1_2,N_0_3



"""

# Bind conifold utility methods (defined in conifold_utils.py with explicit self) to FluxEFT
from .conifold_utils import (
    W1_tilde as _W1_tilde_func,
    compute_zcf as _compute_zcf_func,
    zcf_handling as _zcf_handling_func,
    #DW_x_bulk as _DW_x_bulk_func,
    #dDW_x_bulk as _dDW_x_bulk_func,
    DWbulk_x as _DWbulk_x_func,
    dDWbulk_x as _dDWbulk_x_func,
    DWbulk as _DWbulk_func,
    dDWbulk as _dDWbulk_func,
    W_bulk as _W_bulk_func,
    conifold_fluxes as _conifold_fluxes_func,
    compute_zcf_correction as _compute_zcf_correction_func,
    compute_zcf_explicit as _compute_zcf_explicit_func,
    compute_zcf_compact as _compute_zcf_compact_func,
)
FluxEFT.W1_tilde = _W1_tilde_func
FluxEFT.compute_zcf = _compute_zcf_func
FluxEFT.zcf_handling = _zcf_handling_func
#FluxEFT.DW_x_bulk = _DW_x_bulk_func
#FluxEFT.dDW_x_bulk = _dDW_x_bulk_func
FluxEFT.DWbulk_x = _DWbulk_x_func
FluxEFT.dDWbulk_x = _dDWbulk_x_func
FluxEFT.DWbulk = _DWbulk_func
FluxEFT.dDWbulk = _dDWbulk_func
FluxEFT.W_bulk = _W_bulk_func
FluxEFT.conifold_fluxes = _conifold_fluxes_func
FluxEFT.compute_zcf_correction = _compute_zcf_correction_func
FluxEFT.compute_zcf_explicit = _compute_zcf_explicit_func
FluxEFT.compute_zcf_compact = _compute_zcf_compact_func
"""


# Bind flux utility methods (defined in flux_utils.py with explicit self) to FluxEFT
from .flux_utils import (
    pfv_to_flux as _pfv_to_flux_func,
    pfv_to_moduli as _pfv_to_moduli_func,
    flux_to_pfv as _flux_to_pfv_func,
)
FluxEFT.pfv_to_flux = _pfv_to_flux_func
FluxEFT.pfv_to_moduli = _pfv_to_moduli_func
FluxEFT.flux_to_pfv = _flux_to_pfv_func


# Conifold methods are attached via the ``_ConifoldGated`` descriptor so they
# are only surfaced when ``self.periods.limit ∈ {coniLCS, coniLCS_series,
# coniLCS_bulk}``.  The list of method names lives in :mod:`jaxvacua.conifold`
# (single source of truth — append there to add a new attached method, no
# edit needed here).
from jaxvacua import conifold as _cf

for _name in _cf._FLUXEFT_METHODS:
    setattr(FluxEFT, _name, _cf._ConifoldGated(getattr(_cf, _name)))

unflatten_func = lambda aux_data, children: unflatten_func_class(aux_data, children, FluxEFT)

register_pytree_node(FluxEFT, flatten_func, unflatten_func)

