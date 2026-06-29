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

r"""Reduced EFT interfaces for integrating out heavy moduli.

Purpose
-------
Provide abstractions for solving heavy-field equations of motion and
evaluating a reduced flux EFT on the remaining light fields.

Main public API
---------------
- ``Freezer``: abstract base class defining the reduced-EFT interface,
  including heavy/light index bookkeeping, reconstruction and light-field
  derivatives.
- ``ConifoldFreezer``: concrete implementation for freezing the conifold
  modulus ``z_cf`` in coniLCS models.
- ``Freezer.light_mass_spectrum`` / ``LightSpectrum``: the reduced light-field
  mass spectrum, obtained by integrating out the heavy moduli and solving the
  generalised eigenproblem :math:`H_{\rm eff}\,v = \lambda\,K_{\rm eff}\,v` in
  the real interleaved basis -- which avoids the ill-conditioning of the full
  ``FluxEFT.mass_matrix`` near a conifold.

Design notes
------------
Freezers wrap an existing flux model.  They do not own the underlying
geometry; instead they solve heavy fields as functions of light moduli,
axio-dilaton and fluxes, then reuse the model's superpotential and derivative
methods on the reconstructed full field point.

The reduced light-field theory exposes three Hessian-reduction schemes (the
``reduction`` argument of :meth:`Freezer.ddV_x_light` and
:meth:`Freezer.light_mass_spectrum`): ``"frozen"`` (the bare selection block,
no back-reaction -- a diagnostic), ``"schur"`` (the exact Schur complement of
the heavy block, exact at a vacuum) and ``"autodiff"`` (the Hessian of the
scalar potential differentiated through the on-shell heavy solve).  Masses are
formed in the real interleaved basis from a generalised eigenproblem rather than
through ``mass_matrix``.  The mass-spectrum entry points are eager host-side
helpers (NumPy/SciPy at the eigensolve), i.e. not ``jit``/``vmap``-able; batch
over vacua with a Python loop.
"""

import warnings
from functools import partial
from dataclasses import dataclass
from typing import Tuple, Any, Dict, Optional
from abc import ABC, abstractmethod

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular
from scipy.linalg import eigh as _geigh

__all__ = ["Freezer", "ConifoldFreezer", "LightSpectrum"]


# ----------------------------------------------------------------------------
# Real <-> complex Kähler-metric helpers (interleaved layout)
# ----------------------------------------------------------------------------

def _G_from_real_hessian(H_K: Array) -> Array:
    r"""
    **Description:**
    Extract the complex Hermitian Kähler metric :math:`G_{A\bar B}` from the
    real symmetric Hessian of a real Kähler potential in the interleaved
    ``(Re_0, Im_0, Re_1, Im_1, ...)`` layout (modulus :math:`z = a + \mathrm{i} b`):

    .. math::
        \mathrm{Re}\,G_{AB} = \tfrac14\bigl(H[2A,2B] + H[2A+1,2B+1]\bigr),\quad
        \mathrm{Im}\,G_{AB} = \tfrac14\bigl(H[2A,2B+1] - H[2A+1,2B]\bigr).

    The holomorphic-holomorphic :math:`K_{AB}` pieces cancel, so the result is
    Hermitian by construction whenever ``H_K`` is symmetric.

    Args:
        H_K (Array): Real symmetric Hessian of a real scalar, of shape
            ``(2N, 2N)`` in the interleaved layout.

    Returns:
        Array: Complex Hermitian metric of shape ``(N, N)``.
    """
    ReG = 0.25 * (H_K[0::2, 0::2] + H_K[1::2, 1::2])
    ImG = 0.25 * (H_K[0::2, 1::2] - H_K[1::2, 0::2])
    G = ReG + 1j * ImG
    return 0.5 * (G + jnp.conj(G.T))


def _kahler_metric_real_interleaved(G: Array) -> Array:
    r"""
    **Description:**
    Rebuild the real ``(2N, 2N)`` Kähler metric in the interleaved basis
    ``(Re_0, Im_0, ..., Re_{N-1}, Im_{N-1})`` from a complex Hermitian metric
    :math:`G_{A\bar B}`.

    For :math:`\phi = (a + \mathrm{i} b)/\sqrt2` the kinetic term carries the 2x2
    block :math:`\left[\begin{smallmatrix}\mathrm{Re}\,G & -\mathrm{Im}\,G\\
    \mathrm{Im}\,G & \mathrm{Re}\,G\end{smallmatrix}\right]` for each
    :math:`(A, B)` pair.

    Args:
        G (Array): Complex Hermitian metric of shape ``(N, N)``.

    Returns:
        Array: Real metric of shape ``(2N, 2N)``.
    """
    ReG = jnp.real(G)
    ImG = jnp.imag(G)
    n = G.shape[0]
    Kr = jnp.zeros((2 * n, 2 * n), dtype=ReG.dtype)
    Kr = Kr.at[0::2, 0::2].set(ReG)
    Kr = Kr.at[1::2, 1::2].set(ReG)
    Kr = Kr.at[0::2, 1::2].set(-ImG)
    Kr = Kr.at[1::2, 0::2].set(ImG)
    return Kr


@dataclass
class LightSpectrum:
    r"""
    **Description:**
    Container for the light (reduced) mass spectrum returned by
    :meth:`Freezer.light_mass_spectrum`.

    Attributes:
        masses (np.ndarray): Signed light masses
            :math:`\mathrm{sign}(m^2)\sqrt{|m^2|}`, sorted ascending.
        eigenvalues (np.ndarray): Raw generalised eigenvalues
            :math:`\lambda = 2 m^2` of
            :math:`H_{\rm eff}\,v = \lambda\,K_{\rm eff}\,v`.
        m2_min (float): Smallest light mass squared, :math:`0.5\,\min\lambda`.
        stable (bool): ``True`` if ``m2_min >= -rel_tol * max|m^2|`` (flat
            directions allowed, tachyons rejected).  Note the tolerance is
            scaled by the *largest* mass: a tachyon smaller than
            ``rel_tol * max|m^2|`` is reported stable, so consult ``m2_min`` and
            ``info["m2_dynamic_range"]`` directly for a strongly hierarchical
            spectrum.
        dw_residual (float): On-shell screen value ``max|DW_x|`` of the *full*
            F-terms (heavy component included) at the evaluation point.  When
            ``x_full`` is supplied this is the stored-vacuum residual; otherwise
            it is the analytic-reconstructed point's residual (the heavy solve's
            accuracy).
        reduction (str): Reduction scheme used
            (``"frozen"`` / ``"schur"`` / ``"autodiff"``).
        info (Dict[str, Any]): Auxiliary diagnostics -- the reduced-metric
            condition number ``cond_Keff``, the reduced-spectrum dynamic range
            ``m2_dynamic_range`` (``max|m^2| / min|m^2|``, flat directions
            excluded; large values mean the lightest masses are precision-limited
            relative to the heaviest), the mode count ``n_modes``, and -- on an
            early return -- a ``reason`` string.
    """
    masses: np.ndarray
    eigenvalues: np.ndarray
    m2_min: float
    stable: bool
    dw_residual: float
    reduction: str
    info: Dict[str, Any]


class Freezer(ABC):
    r"""
    **Description:**
    Abstract base class for a reduced effective field theory obtained by
    integrating out a set of heavy moduli.

    Given a full model with moduli :math:`(z_{\text{heavy}}, z_{\text{light}}, \tau)`
    and fluxes, the reduced EFT expresses the heavy moduli as functions of the
    light fields via their leading-order EOM:

    .. math::
        z_{\text{heavy}} = z_{\text{heavy}}(z_{\text{light}}, \tau, \text{fluxes})

    and provides the superpotential, its derivatives, etc. evaluated on this
    solution.

    Subclasses must implement:
        - ``heavy_indices``: which moduli are frozen out
        - ``solve_heavy``: solve for heavy moduli given light fields
        - ``_real_light_to_full``: convert real light-field coordinates to full array
    """

    def __init__(self, model: Any) -> None:
        r"""
        **Description:**
        Initialise the Freezer base class.

        Args:
            model: A flux EFT model object providing ``superpotential``,
                ``DW``, ``DW_x``, ``dDW_x``, ``_convert_real_to_complex``,
                and period data via ``lcs_tree``.
        """
        self.model = model
        self.lcs_tree = model.lcs_tree

    @property
    @abstractmethod
    def heavy_indices(self) -> Tuple[int, ...]:
        r"""
        Description:
        Indices of the heavy moduli within the full moduli array."""
        ...

    @property
    def light_indices(self) -> Tuple[int, ...]:
        r"""
        Description:
        Indices of the light moduli (complement of ``heavy_indices``)."""
        all_idx = set(range(self.model.h12))
        return tuple(sorted(all_idx - set(self.heavy_indices)))

    @property
    def n_heavy(self) -> int:
        r"""
        Description:
        Number of heavy moduli."""
        return len(self.heavy_indices)

    @property
    def n_light(self) -> int:
        r"""
        Description:
        Number of light moduli."""
        return len(self.light_indices)

    @abstractmethod
    def solve_heavy(
        self,
        z_light: Array,
        tau: complex,
        fluxes: Array,
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Solve the leading-order EOM for the heavy moduli as functions of
        the light moduli, axio-dilaton, and fluxes.

        Args:
            z_light (Array): Values of the light complex structure moduli.
            tau (complex): Axio-dilaton value.
            fluxes (Array): Full flux vector.

        Returns:
            Array: Values of the heavy moduli.
        """
        ...

    def reconstruct_full_moduli(
        self,
        z_light: Array,
        tau: complex,
        fluxes: Array,
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Reconstruct the full moduli array by solving for the heavy moduli
        and inserting them at the correct positions.

        Args:
            z_light (Array): Light moduli values.
            tau (complex): Axio-dilaton value.
            fluxes (Array): Full flux vector.

        Returns:
            Array: Full moduli array of length ``h12``.
        """
        z_heavy = self.solve_heavy(z_light, tau, fluxes, **kwargs)
        z_full = jnp.zeros(self.model.h12, dtype=complex)
        z_full = z_full.at[jnp.array(self.light_indices)].set(z_light)
        z_full = z_full.at[jnp.array(self.heavy_indices)].set(z_heavy)
        return z_full

    def superpotential(
        self,
        z_light: Array,
        tau: complex,
        fluxes: Array,
        **kwargs,
    ) -> complex:
        r"""
        **Description:**
        Superpotential of the reduced theory.

        Args:
            z_light (Array): Light moduli values.
            tau (complex): Axio-dilaton.
            fluxes (Array): Full flux vector.

        Returns:
            complex: :math:`W(z_{\text{light}}, \tau)` with heavy moduli on-shell.
        """
        z_full = self.reconstruct_full_moduli(z_light, tau, fluxes, **kwargs)
        return self.model.superpotential(z_full, tau, fluxes)

    def DW_light(
        self,
        z_light: Array,
        z_light_c: Array,
        tau: complex,
        tau_c: complex,
        fluxes: Array,
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Covariant derivatives :math:`D_i W` with respect to the light moduli,
        with heavy moduli on-shell.

        Args:
            z_light (Array): Light moduli values.
            z_light_c (Array): Conjugate light moduli values.
            tau (complex): Axio-dilaton.
            tau_c (complex): Conjugate axio-dilaton.
            fluxes (Array): Full flux vector.

        Returns:
            Array: :math:`D_i W` for the light moduli and :math:`D_\tau W`.
        """
        z_full = self.reconstruct_full_moduli(z_light, tau, fluxes, **kwargs)
        z_full_c = self.reconstruct_full_moduli(
            z_light_c, tau_c, fluxes, **kwargs
        )
        DW_full = self.model.DW(z_full, z_full_c, tau, tau_c, fluxes)
        # Extract DW for light moduli + tau
        light_idx = jnp.array(self.light_indices)
        DW_z_light = DW_full[light_idx]
        DW_tau = DW_full[-1]
        return jnp.append(DW_z_light, DW_tau)

    def DW_x_light(
        self,
        x_light: Array,
        fluxes: Array,
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Gradient of the superpotential :math:`\partial_{x^a} W` in
        real coordinates for the light moduli, with heavy moduli on-shell.

        This is the analogue of ``model.DW_x`` but restricted to the light
        degrees of freedom.

        Args:
            x_light (Array): Real variables for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.

        Returns:
            Array: Real gradient restricted to light directions.
        """
        x_full = self._real_light_to_full(x_light, fluxes, **kwargs)
        DW_x_full = self.model.DW_x(x_full, fluxes)
        return self._real_light_jacobian.T @ DW_x_full

    def dDW_x_light(
        self,
        x_light: Array,
        fluxes: Array,
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Hessian :math:`\partial_{x^a}\partial_{x^b} W` in real coordinates
        for the light moduli.

        Args:
            x_light (Array): Real variables for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.

        Returns:
            Array: Hessian restricted to light directions.
        """
        x_full = self._real_light_to_full(x_light, fluxes, **kwargs)
        dDW_x_full = self.model.dDW_x(x_full, fluxes)
        J = self._real_light_jacobian
        return J.T @ dDW_x_full @ J

    def V_x_light(
        self,
        x_light: Array,
        fluxes: Array,
        noscale: bool = True,
        **kwargs,
    ) -> float:
        r"""
        **Description:**
        Scalar potential :math:`V` evaluated at the light-field coordinates,
        with heavy moduli on-shell.

        Args:
            x_light (Array): Real coordinates for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.
            noscale (bool, optional): If ``True``, uses the no-scale scalar
                potential :math:`V = e^K K^{I\bar J} D_I W D_{\bar J}\bar W`.
                Defaults to ``True``.

        Returns:
            float: Value of :math:`V` with heavy moduli at their on-shell values.
        """
        x_full = self._real_light_to_full(x_light, fluxes, **kwargs)
        return self.model.V_x(x_full, fluxes, noscale=noscale)

    def dV_x_light(
        self,
        x_light: Array,
        fluxes: Array,
        noscale: bool = True,
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Gradient of the scalar potential :math:`\nabla_\phi V` with respect to
        the real light-field coordinates, with heavy moduli on-shell.

        .. admonition:: Details
            :class: dropdown

            Let :math:`\phi^\alpha = (a^1, v^1, \ldots, a^{n_{\rm light}},
            v^{n_{\rm light}}, c_0, s)` denote the real light-field coordinates,
            where :math:`z^i = a^i + \mathrm{i}\,v^i` and
            :math:`\tau = c_0 + \mathrm{i}\,s`. This function returns the
            restriction

            .. math::
                \nabla_\phi V \big|_{\phi^\alpha}
                = \partial_{\phi^\alpha} V(x_{\rm full}(\phi))

            where :math:`x_{\rm full}(\phi)` substitutes the on-shell heavy
            moduli via :meth:`_real_light_to_full`.

        Args:
            x_light (Array): Real coordinates for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.
            noscale (bool, optional): If ``True``, uses the no-scale scalar
                potential. Defaults to ``True``.

        Returns:
            Array: Gradient :math:`\partial_{\phi^\alpha} V`, restricted to
            light directions, of shape ``(2 * n_light + 2,)``.
        """
        x_full = self._real_light_to_full(x_light, fluxes, **kwargs)
        dV_full = self.model.dV_x(x_full, fluxes, noscale=noscale)
        return self._real_light_jacobian.T @ dV_full

    def ddV_x_light(
        self,
        x_light: Array,
        fluxes: Array,
        noscale: bool = True,
        reduction: str = "frozen",
        x_full: Optional[Array] = None,
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Reduced Hessian of the scalar potential
        :math:`\partial_{\phi^\alpha}\partial_{\phi^\beta} V` with respect to
        the real light-field coordinates, with the heavy moduli on-shell.

        .. admonition:: Details
            :class: dropdown

            The real light-field coordinates are
            :math:`\phi^\alpha = (a^1, v^1, \ldots, a^{n_{\rm light}},
            v^{n_{\rm light}}, c_0, s)` (with :math:`z^i = a^i + \mathrm{i}\,v^i`
            and :math:`\tau = c_0 + \mathrm{i}\,s`).  Three reduction schemes are
            provided via ``reduction``:

            - ``"frozen"`` (default): the *selection* block
              :math:`J^T (\nabla\nabla V) J`, with the heavy moduli held at their
              on-shell values but WITHOUT the back-reaction of the light fields
              on the heavy solution.

            - ``"schur"``: the exact Schur complement on the heavy block,

              .. math::
                  H_{\rm eff} = H_{\ell\ell}
                      - H_{\ell h} H_{hh}^{-1} H_{h\ell}\, ,

              obtained by extremising the quadratic form over the heavy
              directions.  This is exact only where the heavy direction is
              on-shell, :math:`\partial_{z_{\rm heavy}} V = 0`; it must therefore
              be evaluated at the genuine vacuum.  Pass ``x_full`` (the stored
              full point) to evaluate there -- otherwise the heavy field is
              reconstructed from the *analytic* solve, which is on-shell only deep
              in the throat, and ``schur`` can return a spurious tachyon at
              moderate throats.  Only the heavy block
              :math:`H_{hh}` is inverted, so it is well conditioned at a genuine
              vacuum, but the back-reaction is a difference of large nearly
              cancelling terms once the heavy/light mass hierarchy approaches
              ``1/eps`` (deep in a conifold throat); there it loses precision.

            - ``"autodiff"``: the Hessian of :math:`V(x_{\rm full}(\phi))`
              differentiated directly through the on-shell heavy solve, so the
              back-reaction enters automatically via the chain rule.  This builds
              a fresh ``jax.hessian`` trace on each call (the inner kernels are
              cached, the outer transform is not), so jit/loop accordingly for
              repeated use.

        .. warning::
            ``reduction="frozen"`` omits the integrate-out back-reaction
            :math:`-H_{\ell h} H_{hh}^{-1} H_{h\ell}`, which can dominate the
            lightest light mass deep in a conifold throat.  Use
            ``reduction="schur"`` or ``reduction="autodiff"`` for physical light
            masses.  Note :meth:`light_mass_spectrum` pairs the frozen Hessian
            with the *substituted* reduced metric, so the resulting eigenvalues
            are a hybrid (a no-back-reaction Hessian against a with-back-reaction
            metric), not the naive frozen masses.

        Args:
            x_light (Array): Real coordinates for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.
            noscale (bool, optional): If ``True``, uses the no-scale scalar
                potential. Defaults to ``True``.
            reduction (str, optional): Reduction scheme, one of
                ``{"frozen", "schur", "autodiff"}``. Defaults to ``"frozen"``
                (the backwards-compatible selection block).  Distinct from the
                ``mode`` keyword (forwarded via ``**kwargs`` to the heavy solve).
            x_full (Array, optional): Full real point at which to evaluate the
                Hessian for ``"frozen"``/``"schur"`` (e.g. the stored vacuum,
                with the heavy field on-shell).  If ``None`` (default) the heavy
                field is reconstructed from the analytic solve via
                :meth:`_real_light_to_full`.  Ignored by ``"autodiff"`` (which
                differentiates through the solve).

        Returns:
            Array: Reduced Hessian restricted to light directions, of shape
            ``(2 * n_light + 2, 2 * n_light + 2)``.
        """
        if reduction == "frozen":
            xf = self._real_light_to_full(x_light, fluxes, **kwargs) \
                if x_full is None else x_full
            ddV_full = self.model.ddV_x(xf, fluxes, noscale=noscale)
            J = self._real_light_jacobian
            return J.T @ ddV_full @ J

        if reduction == "autodiff":
            # autodiff differentiates *through* the heavy solve, so it always
            # re-solves z_heavy(phi); a supplied x_full cannot be used here.
            def _V_light(xl: Array) -> float:
                xf = self._real_light_to_full(xl, fluxes, **kwargs)
                return self.model.V_x(xf, fluxes, noscale=noscale)
            return jax.hessian(_V_light)(x_light)

        if reduction == "schur":
            xf = self._real_light_to_full(x_light, fluxes, **kwargs) \
                if x_full is None else x_full
            ddV_full = self.model.ddV_x(xf, fluxes, noscale=noscale)
            J_l = self._real_light_jacobian
            J_h = self._real_heavy_jacobian
            H_ll = J_l.T @ ddV_full @ J_l
            H_hh = J_h.T @ ddV_full @ J_h
            H_lh = J_l.T @ ddV_full @ J_h
            return H_ll - H_lh @ jnp.linalg.solve(H_hh, H_lh.T)

        raise ValueError(
            "`reduction` must be one of {'frozen', 'schur', 'autodiff'}, "
            f"got {reduction!r}."
        )

    def K_x_light(
        self,
        x_light: Array,
        fluxes: Array,
        **kwargs,
    ) -> float:
        r"""
        **Description:**
        Real Kähler potential :math:`\mathrm{Re}\,K` evaluated at the
        light-field coordinates, with the heavy moduli integrated out on-shell.

        .. admonition:: Details
            :class: dropdown

            The heavy field is substituted at the level of the *potential*: the
            light real coordinates are mapped to the full point via
            :meth:`_real_light_to_full` (heavy moduli on-shell), converted to
            complex moduli, and inserted into the model's Kähler potential.
            Using the *actual* conjugate (rather than an independent
            :math:`\bar\phi`) keeps the result a genuinely real scalar, so its
            real Hessian — the reduced Kähler metric of
            :meth:`G_x_light` — is symmetric and the extracted
            metric Hermitian by construction.

        Args:
            x_light (Array): Real coordinates for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.

        Returns:
            float: :math:`\mathrm{Re}\,K(z_{\rm heavy}^\ast(\phi), \phi)`.
        """
        x_full = self._real_light_to_full(x_light, fluxes, **kwargs)
        moduli, moduli_c, tau, tau_c = self.model._convert_real_to_complex(x_full)
        return jnp.real(self.model.kahler_potential(moduli, moduli_c, tau, tau_c))

    def G_x_light(
        self,
        x_light: Array,
        fluxes: Array,
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Reduced Kähler metric of the light fields in the real interleaved basis,
        obtained by integrating out the heavy moduli at the level of the Kähler
        potential.

        .. admonition:: Details
            :class: dropdown

            The reduced metric is the mixed second derivative of the
            *substituted* Kähler potential
            :math:`K(z_{\rm heavy}^\ast(\phi), \phi)`, NOT the light submatrix of
            the full metric.  The substitution couples the light moduli through
            the heavy solution, so the reduced metric carries chain-rule terms
            (e.g. complex-structure–dilaton mixing) absent from the bare bulk
            block.

            Concretely, the real symmetric Hessian
            :math:`H_K = \partial_{\phi^\alpha}\partial_{\phi^\beta}
            \mathrm{Re}\,K` of :meth:`K_x_light` is taken by
            automatic differentiation, the complex Hermitian metric
            :math:`G_{A\bar B}` is extracted via :func:`_G_from_real_hessian`,
            and the real metric is rebuilt with
            :func:`_kahler_metric_real_interleaved`.

        Args:
            x_light (Array): Real coordinates for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.

        Returns:
            Array: Reduced Kähler metric in the real interleaved basis, of shape
            ``(2 * n_light + 2, 2 * n_light + 2)``.
        """
        H_K = jax.hessian(self.K_x_light)(x_light, fluxes, **kwargs)
        G = _G_from_real_hessian(H_K)
        return _kahler_metric_real_interleaved(G)

    def light_mass_spectrum(
        self,
        x_light: Array,
        fluxes: Array,
        reduction: str = "schur",
        noscale: bool = True,
        dw_tol: float = 1e-4,
        rel_tol: float = 1e-8,
        eig_backend: str = "scipy",
        warn_dynamic_range: float = 1e14,
        x_full: Optional[Array] = None,
        **kwargs,
    ) -> LightSpectrum:
        r"""
        **Description:**
        Mass spectrum of the light fields with the heavy moduli integrated out.

        Solves the generalised eigenvalue problem
        :math:`H_{\rm eff}\,v = \lambda\,K_{\rm eff}\,v`, where
        :math:`H_{\rm eff}` is the reduced Hessian (:meth:`ddV_x_light`) and
        :math:`K_{\rm eff}` the reduced Kähler metric (:meth:`G_x_light`).
        Routing the masses through a generalised eigenproblem in the real basis
        avoids the ill-conditioning and basis artefacts of the full
        :meth:`mass_matrix`.

        .. admonition:: Details
            :class: dropdown

            The light masses follow the supergravity real-field normalisation
            :math:`\phi = (a + \mathrm{i} b)/\sqrt2`, so
            :math:`m^2 = \tfrac12\,\lambda`.

            **On-shell evaluation.** The reduced Hessian (``schur``/``frozen``)
            is exact only where the heavy direction is on-shell.  Pass ``x_full``
            (the stored full vacuum point) to evaluate the Hessian there;
            otherwise the heavy field is reconstructed from the analytic
            :meth:`compute_zcf` solve, which is on-shell only deep in the throat.
            The point is screened on the **full** F-term residual
            ``max|DW_x(x_full)| <= dw_tol`` (the heavy component included), so a
            point whose heavy direction is off-shell -- an imperfect analytic
            seed at a moderate throat, or a wrong ``apply_correction``/``mode`` --
            is rejected with ``reason="off-shell"`` rather than silently returning
            a spurious tachyon.  The reduced metric always uses the analytic
            substituted potential (it differentiates through the solve).  For
            coniLCS the ``apply_correction=True`` z_cf-solve default (set by
            :class:`ConifoldFreezer`) gives a usable analytic seed; it belongs to
            the *z_cf solve* and is unrelated to ``reduction="autodiff"``.

        .. note::
            Eager host-side helper (NumPy/SciPy at the eigensolve): **not**
            ``jit``/``vmap``-able; batch over vacua with a Python loop.

        Args:
            x_light (Array): Real coordinates for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.
            reduction (str, optional): Hessian reduction scheme, one of
                ``{"frozen", "schur", "autodiff"}``. Defaults to ``"schur"``
                (exact at a vacuum). ``"frozen"`` omits the back-reaction (and is
                paired with the *substituted* reduced metric, so its eigenvalues
                are a hybrid -- see :meth:`ddV_x_light`); it is diagnostic only.
                Orthogonal to the z_cf-solve ``mode`` forwarded via ``**kwargs``.
            noscale (bool, optional): If ``True``, uses the no-scale scalar
                potential. Defaults to ``True``.
            dw_tol (float, optional): On-shell tolerance on the full residual
                ``max|DW_x(x_full)|`` (heavy component included). Defaults to
                ``1e-4``.
            rel_tol (float, optional): Relative tolerance for the stability flag
                and the flat-direction floor of the dynamic-range diagnostic.
                Defaults to ``1e-8``.
            eig_backend (str, optional): ``"scipy"`` (default, generalised
                ``scipy.linalg.eigh``; tolerates an indefinite ``H_eff`` but
                requires a positive-definite ``K_eff``) or ``"jax"`` (Cholesky
                whitening, also requires a positive-definite ``K_eff``). Defaults
                to ``"scipy"``.
            warn_dynamic_range (float, optional): Warn when the reduced
                spectrum's dynamic range ``max|m^2|/min|m^2|`` (flat directions
                excluded) exceeds this -- a large range means the lightest masses
                are precision-limited relative to the heaviest (float64
                ``1/eps ~ 4.5e15``). Defaults to ``1e14``.
            x_full (Array, optional): Full real point (the stored vacuum, heavy
                field on-shell) at which to evaluate the reduced Hessian and the
                on-shell screen.  If ``None`` the heavy field is reconstructed
                from the analytic solve via ``**kwargs``.

        Returns:
            LightSpectrum: The reduced spectrum and stability diagnostics.
        """
        if reduction not in ("frozen", "schur", "autodiff"):
            raise ValueError(
                "`reduction` must be one of {'frozen', 'schur', 'autodiff'}, "
                f"got {reduction!r}."
            )
        if eig_backend not in ("scipy", "jax"):
            raise ValueError(
                f"`eig_backend` must be one of {{'scipy', 'jax'}}, got {eig_backend!r}."
            )
        f_j = jnp.asarray(fluxes)
        xf = (jnp.asarray(x_full) if x_full is not None
              else self._real_light_to_full(x_light, f_j, **kwargs))

        # Screen the FULL F-term residual (heavy component included): the
        # light-projected DW_x_light is blind to a wrong heavy solve and would
        # let an off-shell heavy direction return a spurious tachyon.
        dw = float(jnp.max(jnp.abs(self.model.DW_x(xf, f_j))))
        if not np.isfinite(dw) or dw > dw_tol:
            return LightSpectrum(np.array([]), np.array([]), np.nan, False, dw,
                                 reduction, {"reason": "off-shell"})

        H_eff = np.asarray(self.ddV_x_light(
            x_light, f_j, noscale=noscale, reduction=reduction, x_full=xf, **kwargs))
        K_eff = np.asarray(self.G_x_light(x_light, f_j, **kwargs))
        H_eff = 0.5 * (H_eff + H_eff.T)
        K_eff = 0.5 * (K_eff + K_eff.T)
        if not (np.all(np.isfinite(H_eff)) and np.all(np.isfinite(K_eff))):
            return LightSpectrum(np.array([]), np.array([]), np.nan, False, dw,
                                 reduction, {"reason": "nan-hessian"})

        try:
            lam = self._generalised_eigvals(H_eff, K_eff, eig_backend)
        except np.linalg.LinAlgError as exc:   # non-PD / singular reduced metric
            return LightSpectrum(
                np.array([]), np.array([]), np.nan, False, dw, reduction,
                {"reason": f"eig-failed: {type(exc).__name__}"})
        if not np.all(np.isfinite(lam)):       # e.g. jax Cholesky on a non-PD K
            return LightSpectrum(np.array([]), np.array([]), np.nan, False, dw,
                                 reduction, {"reason": "nan-eig"})

        m2 = 0.5 * lam
        masses = np.where(m2 > 0, np.sqrt(np.abs(m2)), -np.sqrt(np.abs(m2)))
        m2_min = float(np.min(m2))
        absm2 = np.abs(m2)
        scale = max(1.0, float(np.max(absm2)))
        stable = bool(m2_min >= -rel_tol * scale)
        # Dynamic range of the REDUCED (light) spectrum, flat directions (|m2|
        # below the roundoff floor) excluded.  This is a light-sector property,
        # not the heavy/light precision wall (the heavy mode is integrated out);
        # a large range means the lightest masses are precision-limited.
        big = absm2[absm2 > rel_tol * scale]
        dyn_range = float(np.max(absm2) / np.min(big)) if big.size else float("inf")
        if np.isfinite(dyn_range) and dyn_range > warn_dynamic_range:
            warnings.warn(
                f"light_mass_spectrum: reduced-spectrum dynamic range "
                f"{dyn_range:.2e} exceeds {warn_dynamic_range:.0e}; the lightest "
                "masses are precision-limited relative to the heaviest.",
                RuntimeWarning, stacklevel=2,
            )
        return LightSpectrum(
            np.sort(masses), lam, m2_min, stable, dw, reduction,
            {"cond_Keff": float(np.linalg.cond(K_eff)),
             "m2_dynamic_range": dyn_range, "n_modes": int(len(lam))},
        )

    @staticmethod
    def _generalised_eigvals(
        H: np.ndarray,
        K: np.ndarray,
        backend: str = "scipy",
    ) -> np.ndarray:
        r"""
        **Description:**
        Solve the generalised symmetric eigenvalue problem
        :math:`H\,v = \lambda\,K\,v` and return the eigenvalues.

        Args:
            H (np.ndarray): Symmetric reduced Hessian.
            K (np.ndarray): Symmetric reduced (kinetic) metric.
            backend (str, optional): ``"scipy"`` uses ``scipy.linalg.eigh``
                (tolerates an indefinite ``H`` but requires ``K`` positive
                definite -- it raises ``LinAlgError`` otherwise); ``"jax"`` uses a
                Cholesky whitening ``L = chol(K)``,
                :math:`\lambda = \mathrm{eigvalsh}(L^{-1} H L^{-T})` and requires
                ``K`` positive-definite. Defaults to ``"scipy"``.

        Returns:
            np.ndarray: The generalised eigenvalues.
        """
        if backend == "scipy":
            return _geigh(H, K, eigvals_only=True)
        if backend == "jax":
            L = jnp.linalg.cholesky(jnp.asarray(K))
            # cholesky returns NaN (does not raise) on a non-PD metric; surface
            # it as a LinAlgError so callers handle it like the scipy backend.
            if not bool(jnp.all(jnp.isfinite(L))):
                raise np.linalg.LinAlgError(
                    "reduced metric is not positive-definite (Cholesky failed)."
                )
            Y = solve_triangular(L, jnp.asarray(H), lower=True)   # L^{-1} H
            M = solve_triangular(L, Y.T, lower=True).T            # L^{-1} H L^{-T}
            return np.asarray(jnp.linalg.eigvalsh(0.5 * (M + M.T)))
        raise ValueError(
            f"`eig_backend` must be one of {{'scipy', 'jax'}}, got {backend!r}."
        )

    @property
    def _real_light_slice(self) -> Array:
        r"""
        **Description:**
        Indices into the real coordinate array ``x`` corresponding to the
        light moduli. The real array is ordered as
        ``[Re(z_0), Im(z_0), ..., Re(z_{h-1}), Im(z_{h-1}), Re(tau), Im(tau)]``,
        so modulus ``i`` maps to real indices ``2*i`` and ``2*i+1``.
        """
        light = list(self.light_indices)
        real_idx = []
        for i in light:
            real_idx.extend([2 * i, 2 * i + 1])
        # tau is always the last two entries
        real_idx.extend([2 * self.model.h12, 2 * self.model.h12 + 1])
        return jnp.array(real_idx)

    @property
    def _real_light_jacobian(self) -> Array:
        r"""
        **Description:**
        Real Jacobian :math:`J = \partial x_{\rm full}/\partial x_{\rm light}`
        (heavy moduli held fixed) mapping the light real coordinates into the
        full real array.  The light real gradient is :math:`J^T\cdot(\text{full
        gradient})` and the Hessian block :math:`J^T\cdot(\text{full
        Hessian})\cdot J`.

        For an axis-aligned light/heavy split this is the selection matrix that
        picks :attr:`_real_light_slice` (so :math:`J^T v = v[\text{slice}]` and
        :math:`J^T M J = M[\text{slice},\text{slice}]`, bit-identical to plain
        slicing).  Subclasses whose light directions are *not* coordinate axes
        — e.g. a conifold modulus that is a generic charge combination
        (``conifold_basis=False``) — override this with the corresponding
        embedding Jacobian.
        """
        dim_full = 2 * (self.model.h12 + 1)
        return jnp.eye(dim_full)[:, self._real_light_slice]

    @property
    def _real_heavy_slice(self) -> Array:
        r"""
        **Description:**
        Indices into the real coordinate array ``x`` corresponding to the heavy
        moduli (``2*i`` and ``2*i+1`` for each ``i`` in :attr:`heavy_indices`).

        Unlike :attr:`_real_light_slice` this contains no axio-dilaton entry,
        since :math:`\tau` is always a light field.
        """
        real_idx = []
        for i in self.heavy_indices:
            real_idx.extend([2 * i, 2 * i + 1])
        return jnp.array(real_idx)

    @property
    def _real_heavy_jacobian(self) -> Array:
        r"""
        **Description:**
        Real Jacobian :math:`J_h = \partial x_{\rm full}/\partial x_{\rm heavy}`
        selecting the heavy directions — the complement of
        :attr:`_real_light_jacobian`.  Together ``[J_h | J_l]`` form an
        invertible change of basis on the full real field space, so the Schur
        complement in :meth:`ddV_x_light` (``reduction="schur"``) is the exact
        on-shell reduced Hessian expressed in the light coordinates.

        For an axis-aligned heavy/light split this is the selection matrix that
        picks :attr:`_real_heavy_slice`.  Subclasses whose heavy direction is
        *not* a coordinate axis — e.g. a conifold modulus that is a generic
        charge combination (``conifold_basis=False``) — override this with the
        corresponding embedding Jacobian.
        """
        dim_full = 2 * (self.model.h12 + 1)
        return jnp.eye(dim_full)[:, self._real_heavy_slice]

    @abstractmethod
    def _real_light_to_full(
        self,
        x_light: Array,
        fluxes: Array,
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Convert real light-field coordinates to the full real coordinate
        array by solving for and inserting the heavy moduli.

        Args:
            x_light (Array): Real coordinates for light moduli + tau.
            fluxes (Array): Full flux vector.

        Returns:
            Array: Full real coordinate array.
        """
        ...


class ConifoldFreezer(Freezer):
    r"""
    **Description:**
    Integrates out the conifold modulus :math:`z_{\text{cf}}` (index 0) in
    coniLCS models.

    Near the conifold locus, the conifold modulus acquires a parametrically
    large mass from the flux superpotential. Its leading-order EOM gives

    .. math::
        z_{\text{cf}} = -\frac{1}{2\pi i}
            \exp\!\Bigl(-\frac{2\pi i\,\widetilde{W}_1}{n_{\text{cf}}(M_1 - \tau H_1)}\Bigr)

    where :math:`\widetilde{W}_1` is the effective superpotential contribution
    from the bulk moduli and :math:`n_{\text{cf}}` is the conifold degree.

    Args:
        model: A flux EFT model with ``"coniLCS"`` in ``model.periods.limit``.
        conifold_index (int, optional): Index of the conifold modulus in the
            moduli array. Defaults to ``0``.
        ncf (int, optional): Conifold degree. Defaults to ``2``.
    """

    def __init__(
        self,
        model: Any,
        conifold_index: int = 0,
    ) -> None:
        r"""
        **Description:**
        Initialise the ConifoldFreezer.

        Args:
            model: A flux EFT model with ``"coniLCS"`` in ``model.periods.limit``.
            conifold_index (int, optional): Index of the conifold modulus in the
                moduli array. Defaults to ``0``.

        Attributes:
            _conifold_index (int): Stored index of the conifold modulus.
        """
        super().__init__(model)
        self._conifold_index = conifold_index

    @property
    def heavy_indices(self) -> Tuple[int, ...]:
        r"""
        Description:
        Indices of the heavy (conifold) modulus; always a length-1 tuple."""
        return (self._conifold_index,)

    @property
    def ncf(self) -> int:
        r"""
        Description:
        Conifold degree :math:`n_{\text{cf}}`, sourced from
        ``self.model.lcs_tree.conifold.ncf`` (single source of truth)."""
        return int(self.model.lcs_tree.conifold.ncf)

    # ------------------------------------------------------------------ #
    # conifold_basis=False reconstruction.
    #
    # When the geometry is NOT rotated into the conifold-aligned frame the
    # conifold modulus is the charge combination z_cf = q·z, not a coordinate
    # axis, and the light (bulk) directions span ker(q).  The light↔full map is
    # then z_full = z_cf·e_q + bulk_embedding·z_light (instead of an index
    # scatter), and the light F-terms / Jacobian project through ``bulk_embedding``
    # (= Λ[1:]ᵀ).  In the aligned basis (``conifold_basis=True``) every method
    # below defers to the base-class index-based implementation, bit-identical.
    # ------------------------------------------------------------------ #

    def reconstruct_full_moduli(self, z_light, tau, fluxes, **kwargs):
        r"""
        **Description:**
        Reconstruct the full modulus vector from the light (bulk) moduli with
        the conifold modulus on-shell.  Aligned: index scatter (base class).
        General: :math:`z_{\rm full} = z_{\rm cf}\,e_q + \text{bulk\_embedding}\,z_{\rm light}`.
        """
        if self.model.lcs_tree.conifold_basis:
            return super().reconstruct_full_moduli(z_light, tau, fluxes, **kwargs)
        z_cf = self.solve_heavy(z_light, tau, fluxes, **kwargs)[0]
        coni = self.model.lcs_tree.conifold
        e_q = jnp.asarray(coni.embedding,      dtype=z_light.dtype)
        be  = jnp.asarray(coni.bulk_embedding, dtype=z_light.dtype)
        return z_cf * e_q + be @ z_light

    def DW_light(self, z_light, z_light_c, tau, tau_c, fluxes, **kwargs):
        r"""
        **Description:**
        Covariant derivatives :math:`D_i W` for the light moduli (+ :math:`D_\tau W`)
        with the conifold modulus on-shell.  General basis: project the full
        :math:`D_i W` onto the bulk directions, :math:`D_a W = D_i W\,
        \text{bulk\_embedding}^{i}{}_{a}` (the conifold component
        :math:`D_i W\,e_q^i = \partial_{z_{\rm cf}}W \approx 0` on-shell).
        """
        if self.model.lcs_tree.conifold_basis:
            return super().DW_light(z_light, z_light_c, tau, tau_c, fluxes, **kwargs)
        z_full   = self.reconstruct_full_moduli(z_light,   tau,   fluxes, **kwargs)
        z_full_c = self.reconstruct_full_moduli(z_light_c, tau_c, fluxes, **kwargs)
        DW_full = self.model.DW(z_full, z_full_c, tau, tau_c, fluxes)
        be = jnp.asarray(self.model.lcs_tree.conifold.bulk_embedding, dtype=DW_full.dtype)
        DW_z_light = DW_full[:self.model.h12] @ be
        DW_tau = DW_full[-1]
        return jnp.append(DW_z_light, DW_tau)

    @property
    def _real_light_jacobian(self) -> Array:
        r"""
        **Description:**
        Real Jacobian :math:`\partial x_{\rm full}/\partial x_{\rm light}` for the
        ConifoldFreezer.  General basis: the bulk embedding lifted to real
        coordinates, :math:`\text{bulk\_embedding}\otimes \mathbb{1}_2` on the
        moduli block plus :math:`\mathbb{1}_2` for :math:`\tau`.  Aligned basis:
        the selection matrix (base class).
        """
        if self.model.lcs_tree.conifold_basis:
            return super()._real_light_jacobian
        be = jnp.asarray(self.model.lcs_tree.conifold.bulk_embedding)
        J_mod = jnp.kron(be, jnp.eye(2, dtype=be.dtype))   # (2*h12, 2*(h12-1))
        nm, nl = J_mod.shape
        J = jnp.zeros((nm + 2, nl + 2), dtype=be.dtype)
        J = J.at[:nm, :nl].set(J_mod)
        J = J.at[nm:, nl:].set(jnp.eye(2, dtype=be.dtype))   # tau block
        return J

    @property
    def _real_heavy_jacobian(self) -> Array:
        r"""
        **Description:**
        Real Jacobian selecting the heavy (conifold) direction — the complement
        of :attr:`_real_light_jacobian`.  General basis (``conifold_basis=False``):
        the conifold charge direction lifted to real coordinates,
        :math:`e_q\otimes\mathbb{1}_2` on the moduli block (no :math:`\tau`
        entry, as :math:`\tau` is light).  Aligned basis: the selection matrix
        (base class).
        """
        if self.model.lcs_tree.conifold_basis:
            return super()._real_heavy_jacobian
        e_q = jnp.asarray(self.model.lcs_tree.conifold.embedding, dtype=jnp.float_)
        J_mod = jnp.kron(e_q.reshape(-1, 1), jnp.eye(2, dtype=e_q.dtype))   # (2*h12, 2)
        nm = J_mod.shape[0]
        J = jnp.zeros((nm + 2, 2), dtype=e_q.dtype)
        J = J.at[:nm, :].set(J_mod)
        return J

    def light_mass_spectrum(
        self,
        x_light: Array,
        fluxes: Array,
        **kwargs,
    ) -> LightSpectrum:
        r"""
        **Description:**
        Conifold-aware override of :meth:`Freezer.light_mass_spectrum`: defaults
        the z_cf solve to ``apply_correction=True`` (the Kähler-covariant
        correction needed for the analytic seed to reproduce the stored vacuum),
        then defers to the base implementation.  All arguments are as there.
        """
        kwargs.setdefault("apply_correction", True)
        return super().light_mass_spectrum(x_light, fluxes, **kwargs)

    def bulk_mass_spectrum(
        self,
        x_light: Array,
        fluxes: Array,
        **kwargs,
    ) -> LightSpectrum:
        r"""
        **Description:**
        Bulk mass spectrum of a coniLCS vacuum with the conifold modulus
        integrated out.  Identical to :meth:`light_mass_spectrum` (here the
        "bulk" fields *are* the base-class "light" fields); the alias provides
        the conifold/throat vocabulary used in the literature.  See
        :meth:`Freezer.light_mass_spectrum` for arguments (``reduction``,
        ``dw_tol``, ``x_full``, ``eig_backend``, ...) and the returned
        :class:`LightSpectrum`.  As there, the on-shell
        ``apply_correction=True`` z_cf-solve default is applied.
        """
        return self.light_mass_spectrum(x_light, fluxes, **kwargs)

    def solve_heavy(
        self,
        z_light: Array,
        tau: complex,
        fluxes: Array,
        conj: bool = False,
        mode: str = "manual",
        apply_correction: bool = False,
    ) -> Array:
        r"""
        **Description:**
        Solve for :math:`z_{\text{cf}}` from its leading-order EOM by
        delegating to :func:`jaxvacua.conifold.zcf_solver.compute_zcf` (the unified
        complex-coord dispatcher attached to the model).

        Args:
            z_light (Array): Bulk (light) moduli values.
            tau (complex): Axio-dilaton.
            fluxes (Array): Full flux vector.
            conj (bool, optional): Conjugate conventions. Defaults to ``False``.
            mode (str, optional): One of ``{"manual", "autodiff", "pfv"}``.
                Routes through ``model.W_log_coeff(..., mode=mode)``.
                Defaults to ``"manual"`` (closed-form ``kappa`` /
                ``a_matrix`` / ``b_vector`` + ``Li`` assembly).
            apply_correction (bool, optional): If ``True``, add the
                Kähler-covariant correction ``log_coeff_K_corr`` to the log
                coefficient before exponentiating. Defaults to ``False``.

        Returns:
            Array: Value of :math:`z_{\text{cf}}` (length-1 array).
        """
        cz_light = jnp.conj(z_light)
        ctau = jnp.conj(tau)
        zcf = self.model.compute_zcf(
            z_light, cz_light, tau, ctau, fluxes,
            mode=mode, apply_correction=apply_correction, conj=conj,
        )
        return jnp.array([zcf])

    def _real_light_to_full(
        self,
        x_light: Array,
        fluxes: Array,
        mode: str = "manual",
        conj: bool = False,
        apply_correction: bool = False,
    ) -> Array:
        r"""
        **Description:**
        Convert real light-field coordinates to the full real array by solving
        for :math:`z_{\text{cf}}` and prepending it.  Delegates to
        :func:`jaxvacua.conifold.zcf_solver.zcf_handling`, which expects ``x_light``
        to already be the bulk-only real vector (length ``2 * h12``, no
        conifold direction).

        Args:
            x_light (Array): Real coordinates for bulk moduli + tau
                (length ``2 * n_light + 2``).
            fluxes (Array): Full flux vector.
            mode (str, optional): Solving mode forwarded to ``zcf_handling``.
                Defaults to ``"manual"``.
            conj (bool, optional): Conjugate conventions, forwarded to
                ``zcf_handling``.
            apply_correction (bool, optional): If ``True``, include the
                Kähler-covariant correction in the z_cf solve.

        Returns:
            Array: Full real coordinate array.
        """
        return self.model.zcf_handling(
            x_light, fluxes,
            mode=mode, apply_correction=apply_correction, conj=conj,
        )
