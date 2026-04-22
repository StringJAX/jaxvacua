# ==============================================================================
# freezer — freeze out heavy moduli from the flux EFT.
#
# When a subset of moduli is parametrically heavier than the rest, one can
# solve their leading-order equations of motion as functions of the light
# fields and substitute back to obtain a reduced EFT with fewer degrees of
# freedom.
#
# This module provides:
#   - Freezer: abstract base class defining the interface
#   - ConifoldFreezer: concrete implementation for freezing out the
#     conifold modulus z_cf in coniLCS models
# ==============================================================================

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
        return DW_x_full[self._real_light_slice]

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
        idx = self._real_light_slice
        return dDW_x_full[jnp.ix_(idx, idx)]

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
        return dV_full[self._real_light_slice]

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
        idx = self._real_light_slice
        return ddV_full[jnp.ix_(idx, idx)]

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
        ncf: int = 2,
    ) -> None:
        r"""
        **Description:**
        Initialise the ConifoldFreezer.

        Args:
            model: A flux EFT model with ``"coniLCS"`` in ``model.periods.limit``.
            conifold_index (int, optional): Index of the conifold modulus in the
                moduli array. Defaults to ``0``.
            ncf (int, optional): Conifold degree. Defaults to ``2``.

        Attributes:
            _conifold_index (int): Stored index of the conifold modulus.
            _ncf (int): Stored conifold degree.
        """
        super().__init__(model)
        self._conifold_index = conifold_index
        self._ncf = ncf

    @property
    def heavy_indices(self) -> Tuple[int, ...]:
        r"""
        Description:
        Indices of the heavy (conifold) modulus; always a length-1 tuple."""
        return (self._conifold_index,)

    def _W1_tilde(
        self,
        zbulk: Array,
        tau: complex,
        f2: Array,
        h2: Array,
        P1: complex,
        K1: complex,
        conj: bool = False,
    ) -> complex:
        r"""
        **Description:**
        Compute :math:`\widetilde{W}_1`, the effective superpotential
        contribution from the bulk fields that sources the conifold modulus.

        Args:
            zbulk (Array): Bulk moduli values (excluding conifold).
            tau (complex): Axio-dilaton.
            f2 (Array): RR flux component :math:`f_2`.
            h2 (Array): NSNS flux component :math:`h_2`.
            P1 (complex): Flux :math:`P^1`.
            K1 (complex): Flux :math:`K^1`.
            conj (bool, optional): If ``True``, use conjugate conventions.

        Returns:
            complex: Value of :math:`\widetilde{W}_1`.
        """
        M0, H0 = f2[0], h2[0]
        M1, H1 = f2[1], h2[1]
        Malpha, Halpha = f2[2:], h2[2:]

        kappa = self.lcs_tree.intnums

        coeff = 2 * jnp.pi * 1j
        if conj:
            coeff = -coeff

        ncf = self._ncf

        F1b = (
            (kappa[0, 1:, 1:] @ zbulk) @ zbulk / 2
            + self.lcs_tree.b_vector[0]
            - jnp.pi**2 / 6 * ncf / coeff**2
        )
        F2b = -(kappa[0, 0, 1:] @ zbulk) + self.lcs_tree.a_matrix[0, 0]
        dF1b = -(kappa[0, 1:, 1:] @ zbulk) + self.lcs_tree.a_matrix[0, 1:]

        tmp = (M0 - tau * H0) * F1b
        tmp += (M1 - tau * H1) * F2b
        tmp += (Malpha - tau * Halpha) @ dF1b
        tmp += (-1) * (P1 - tau * K1)

        return tmp

    def solve_heavy(
        self,
        z_light: Array,
        tau: complex,
        fluxes: Array,
        conj: bool = False,
        mode: str | None = None,
    ) -> Array:
        r"""
        **Description:**
        Solve for :math:`z_{\text{cf}}` from its leading-order EOM.

        Args:
            z_light (Array): Bulk (light) moduli values.
            tau (complex): Axio-dilaton.
            fluxes (Array): Full flux vector.
            conj (bool, optional): Conjugate conventions. Defaults to ``False``.
            mode (str, optional): If ``"pfv"``, use the PFV approximation.
                If ``None``, use the full expression. Defaults to ``None``.

        Returns:
            Array: Value of :math:`z_{\text{cf}}` (length-1 array).
        """
        from .flux_utils import split_fluxes

        f1, f2, h1, h2 = split_fluxes(fluxes, self.lcs_tree)

        coeff = 2 * jnp.pi * 1j
        if conj:
            coeff = -coeff

        ncf = self._ncf

        if mode == "pfv":
            Mvec = f2[1:]
            Kvec = h1[1:]
            M = Mvec[0]
            P1 = f1[1]
            N = self.lcs_tree.intnums @ Mvec
            pvec = jnp.linalg.inv(N[1:, 1:]) @ Kvec[1:]
            Kprime = Kvec[0] - N[0, 1:] @ pvec

            # Extract tau components from light moduli real repr
            # In PFV mode, tau is encoded in the last two entries of x_light
            c0 = tau.real
            gs = 1.0 / tau.imag

            phase_comb = -1j * (
                self.lcs_tree.a_matrix[0] @ Mvec - P1 + c0 * Kprime
            )
            if conj:
                phase_comb = -phase_comb

            radial = Kprime / gs  # gs = 1/s, so Kprime/gs = Kprime*s
            exponent = 2 * jnp.pi / ncf / M * (phase_comb + radial)

        else:
            zbulk = z_light
            M1, H1 = f2[1], h2[1]
            P1, K1 = f1[1], h1[1]

            W1 = self._W1_tilde(zbulk, tau, f2, h2, P1, K1, conj=conj)
            exponent = -coeff * W1 / ncf / (M1 - tau * H1)

        zcf = (-1) / coeff * jnp.exp(exponent)
        return jnp.array([zcf])

    def reconstruct_full_moduli(
        self,
        z_light: Array,
        tau: complex,
        fluxes: Array,
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Reconstruct the full moduli array by prepending the solved conifold modulus
        :math:`z_{\text{cf}}` to the bulk (light) moduli.

        Args:
            z_light (Array): Bulk moduli values (excluding the conifold modulus).
            tau (complex): Axio-dilaton value.
            fluxes (Array): Full flux vector.

        Returns:
            Array: Full moduli array of length ``h12``, with :math:`z_{\text{cf}}` at index 0.
        """
        z_heavy = self.solve_heavy(z_light, tau, fluxes, **kwargs)
        return jnp.append(z_heavy, z_light)

    def _real_light_to_full(
        self,
        x_light: Array,
        fluxes: Array,
        mode: str | None = None,
        conj: bool = False,
    ) -> Array:
        r"""
        **Description:**
        Convert real light-field coordinates to the full real array by
        solving for :math:`z_{\text{cf}}` and prepending it.

        Args:
            x_light (Array): Real coordinates for bulk moduli + tau
                (length ``2 * n_light + 2``).
            fluxes (Array): Full flux vector.
            mode (str, optional): Solving mode for ``solve_heavy``.
            conj (bool, optional): Conjugate conventions.

        Returns:
            Array: Full real coordinate array.
        """
        # Extract tau from x_light (last two entries)
        x_out = jnp.append(jnp.ones(2), x_light)
        z, _, tau, _ = self.model._convert_real_to_complex(x_out)
        z_light = z[1:]

        zcf = self.solve_heavy(z_light, tau, fluxes, conj=conj, mode=mode)
        xcz = jnp.array([zcf[0].real, zcf[0].imag])
        return jnp.append(xcz, x_light)
