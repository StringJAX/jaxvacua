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

"""Reduced EFT interfaces for integrating out heavy moduli.

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

Design notes
------------
Freezers wrap an existing flux model.  They do not own the underlying
geometry; instead they solve heavy fields as functions of light moduli,
axio-dilaton and fluxes, then reuse the model's superpotential and derivative
methods on the reconstructed full field point.
"""

import jax
import jax.numpy as jnp
from jax import Array
from functools import partial
from typing import Tuple, Any
from abc import ABC, abstractmethod


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
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Hessian of the scalar potential
        :math:`\partial_{\phi^\alpha}\partial_{\phi^\beta} V` with respect to
        the real light-field coordinates, with heavy moduli on-shell.

        .. admonition:: Details
            :class: dropdown

            Using the same real light-field coordinates
            :math:`\phi^\alpha = (a^1, v^1, \ldots, a^{n_{\rm light}},
            v^{n_{\rm light}}, c_0, s)` as in :meth:`dV_x_light`, this
            function returns the block of the full Hessian restricted to the
            light directions:

            .. math::
                (\nabla\nabla V)_{\alpha\beta}
                = \partial_{\phi^\alpha}\partial_{\phi^\beta} V

            with heavy moduli substituted by their on-shell values.

        Args:
            x_light (Array): Real coordinates for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.
            noscale (bool, optional): If ``True``, uses the no-scale scalar
                potential. Defaults to ``True``.

        Returns:
            Array: Hessian block restricted to light directions, of shape
            ``(2 * n_light + 2, 2 * n_light + 2)``.
        """
        x_full = self._real_light_to_full(x_light, fluxes, **kwargs)
        ddV_full = self.model.ddV_x(x_full, fluxes, noscale=noscale)
        J = self._real_light_jacobian
        return J.T @ ddV_full @ J

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
