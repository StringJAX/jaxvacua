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

"""Flux-bounding algorithm for finite-region vacuum enumeration.

Purpose
-------
Implement the flux-bounding strategy of arXiv:2501.03984 for systematic
enumeration of Type IIB flux candidates in bounded regions of moduli space.

Main public API
---------------
- Module-level JIT kernels for processing NSNS-flux batches and ISD-completed
  flux candidates without recompiling per ``bounded_fluxes`` instance.
- ``bounded_fluxes``: computes local/global eigenvalue bounds, enumerates or
  samples admissible fluxes, applies tadpole and dilaton constraints, and
  refines candidates with the vacuum solver.
- Cluster/export helpers for splitting large enumeration jobs and merging
  results.

Design notes
------------
The implementation is performance-oriented.  Expensive kernels are kept at
module scope with static shape arguments so models with matching dimensions
can share compiled XLA code.
"""

import math
import time
import warnings
from functools import partial
from typing import Any, Callable, Iterator, List, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, vmap, Array
# =========================================================================
#  Module-level JIT-compiled functions (shared across all bounded_fluxes
#  instances with the same n_fluxes / dimension_H3, avoiding per-instance
#  recompilation).
# =========================================================================

@partial(jit, static_argnums=(2, 3))
def _process_h_at_modulus_jit(
    h_chunk: Array,
    M0_sigma_j: Array,
    n_fluxes: int,
    dim_H3: int,
    s_j: Array,
    c0_j: Array,
    evs_j: Tuple,
    tau_j: Array,
    sigma: Array,
    lambda_max_gl: float,
    mu_min_gl: float,
    mu_max_gl: float,
    tilde_mu_min_gl: float,
    tilde_mu_max_gl: float,
    dil_min: float,
    dil_max: float,
    Nmax: float,
) -> Tuple[Array, Array]:
    r"""

    **Description:**
    Module-level JIT-compiled batch processing of NSNS-flux candidates
    :math:`h` at a **single** moduli point :math:`z_j`.

    This function is independent of the :class:`bounded_fluxes` instance, so
    its compiled XLA kernel is shared across all models with the same
    ``n_fluxes`` and ``dim_H3`` (e.g. all :math:`h^{1,2}=1` models),
    avoiding per-instance recompilation overhead.

    .. admonition:: Details
        :class: dropdown

        For each candidate :math:`h` in the batch, the function performs two
        steps:

        **Step 1 — ISD completion** (Eq. 21 of arXiv:2501.03984).
        Given :math:`\tau_j = c_0 + \mathrm{i}s` and the pre-computed product
        :math:`\mathcal{M}_j\,\Sigma`, the continuous ISD-projected RR-flux is

        .. math::
            f = s\,(\mathcal{M}_j\,\Sigma)\,h + c_0\,h\,,

        which is then rounded to the nearest integer.

        **Step 2 — Bounds checking.**
        The full flux :math:`[f \mid h]` is checked against:

        - the D3-tadpole constraint :math:`0 < N_{\rm flux} \leq N_{\max}`;
        - all local eigenvalue inequalities (for :math:`h_1, h_2, f_1, f_2, s, f`)
          evaluated at :math:`(z_j, \tau_j)`;
        - all global eigenvalue inequalities evaluated against the
          pre-computed extrema :math:`\lambda_{\max}^{\rm gl}`,
          :math:`\mu_{\min}^{\rm gl}`, :math:`\tilde\mu_{\min}^{\rm gl}`.

    Args:
        h_chunk (Array): NSNS-flux candidates, shape ``(n_h, n_fluxes)``.
        M0_sigma_j (Array): Pre-computed :math:`\mathcal{M}(z_j)\,\Sigma`,
            shape ``(n_fluxes, n_fluxes)``.
        n_fluxes (int): Length of a half-flux vector :math:`2(h^{1,2}+1)`.
        dim_H3 (int): Dimension of the :math:`H^3_+` sector,
            :math:`h^{1,2}+1`.
        s_j (Array): Dilaton :math:`\mathrm{Im}(\tau_j)` at modulus :math:`z_j`.
        c0_j (Array): Axion :math:`\mathrm{Re}(\tau_j)` at modulus :math:`z_j`.
        evs_j (Tuple): Pre-computed eigenvalue 5-tuple
            :math:`(\lambda_{\max}, \mu_{\min}, \mu_{\max}, \tilde\mu_{\min},
            \tilde\mu_{\max})` at :math:`z_j`.
        tau_j (Array): Complex axio-dilaton :math:`\tau_j` (for bound
            evaluation).
        sigma (Array): Symplectic matrix :math:`\Sigma`, shape
            ``(n_fluxes, n_fluxes)``.
        lambda_max_gl (float): Global maximum eigenvalue of :math:`\mathcal{M}`.
        mu_min_gl (float): Global minimum eigenvalue of
            :math:`-\operatorname{Im}(\mathcal{N})`.
        mu_max_gl (float): Global maximum eigenvalue of
            :math:`-\operatorname{Im}(\mathcal{N})`.
        tilde_mu_min_gl (float): Global minimum eigenvalue of
            :math:`\operatorname{Im}(\mathcal{N}^{-1})`.
        tilde_mu_max_gl (float): Global maximum eigenvalue of
            :math:`\operatorname{Im}(\mathcal{N}^{-1})`.
        dil_min (float): Lower bound on :math:`s = \operatorname{Im}(\tau)`.
        dil_max (float): Upper bound on :math:`s`.
        Nmax (float): Maximum D3-tadpole :math:`N_{\max}`.

    Returns:
        Tuple[Array, Array]:
            - **flux_full** — ISD-completed flux vectors :math:`[f \mid h]`,
              shape ``(n_h, 2*n_fluxes)``.
            - **valid** — boolean mask, shape ``(n_h,)``; ``True`` where all
              bounds (tadpole + local + global) are satisfied.
    """
    # ISD completion: f = s * M0 @ (Sigma @ h) + c0 * h
    # Matches sampling.py _ISD_sampling_FH mode="H":
    #   SigmaFlux = sigma @ h;  f = s * M0 @ SigmaFlux + c0 * h
    # Batched: (M0_sigma_j @ h_chunk.T).T applies M0 sigma column-wise to each h.
    f = s_j * (M0_sigma_j @ h_chunk.T).T + c0_j * h_chunk
    f_int = jnp.round(f)

    # Build full flux [f | h]
    flux_full = jnp.concatenate([f_int, h_chunk], axis=-1)

    # Tadpole: N_flux = f @ sigma @ h  (no model dependency)
    def _tadpole(flux):
        """Compute tadpole N_flux for a single flux vector."""
        return jnp.abs(
            jnp.dot(flux[:n_fluxes], jnp.dot(sigma, flux[n_fluxes:])).real
        )

    tad = vmap(_tadpole)(flux_full)
    tad_ok = (tad > 0) & (tad <= Nmax)

    # Bounds check (inlined, no self dependency)
    def _check_single(lm, mmin, mmax, tmmin, tmmax, tau, flux, Nflux):
        """Check bounding-box constraints for a single flux candidate."""
        c0 = jnp.real(tau)
        s  = jnp.imag(tau)

        f = flux[:n_fluxes].real
        h = flux[n_fluxes:].real
        h1, h2 = h[:dim_H3], h[dim_H3:]
        f1, f2 = f[:dim_H3], f[dim_H3:]

        hnorm  = jnp.dot(h, h)
        h1norm = jnp.dot(h1, h1)
        h2norm = jnp.dot(h2, h2)
        fnorm  = jnp.dot(f, f)
        f1norm = jnp.dot(f1, f1)
        f2norm = jnp.dot(f2, f2)

        f1tilde = f1 - c0 * h1
        f2tilde = f2 - c0 * h2
        f1tilde_norm = jnp.dot(f1tilde, f1tilde)
        f2tilde_norm = jnp.dot(f2tilde, f2tilde)

        sqrt_f1h1 = jnp.sqrt(f1norm * h1norm)
        sqrt_f2h2 = jnp.sqrt(f2norm * h2norm)

        sN = s * Nflux
        c0_abs = jnp.abs(c0)
        c0sq = c0 ** 2

        # h bounds
        b_h1_local  = s * tmmin * h1norm <= Nflux
        b_h1_global = dil_min * tilde_mu_min_gl * h1norm <= Nmax
        b_h2_local  = s * mmin * h2norm <= Nflux
        b_h2_global = dil_min * mu_min_gl * h2norm <= Nmax
        b_h_local   = hnorm <= Nflux * lm / s
        b_h_global  = hnorm <= Nmax * lambda_max_gl / dil_min

        # f1 bounds
        b_f1_l1 = tmmin * f1tilde_norm <= sN
        b_f1_l2 = (tmmin * (f1norm + c0sq * h1norm)
                   - 2.0 * c0_abs * tmmax * sqrt_f1h1 <= sN)
        b_f1_l5 = (tmmin * (f1norm + (c0sq + 0.75) * h1norm)
                   - 2.0 * c0_abs * tmmax * sqrt_f1h1 <= sN)
        b_f1_l6 = tmmin * (f1tilde_norm + (c0sq + 0.75) * h1norm) <= sN
        b_f1_g3 = (tilde_mu_min_gl * (f1norm + 0.75 * h1norm)
                   - tilde_mu_max_gl * sqrt_f1h1 <= dil_max * Nmax)
        b_f1_g4 = (tilde_mu_min_gl * (f1tilde_norm + 0.75 * h1norm)
                   <= dil_max * Nmax)

        # f2 bounds
        b_f2_l1 = mmin * f2tilde_norm <= sN
        b_f2_l2 = (mmin * (f2norm + c0sq * h2norm)
                   - 2.0 * c0_abs * mmax * sqrt_f2h2 <= sN)
        b_f2_l5 = (mmin * (f2norm + (c0sq + 0.75) * h2norm)
                   - mmax * c0_abs * sqrt_f2h2 <= sN)
        b_f2_l6 = mmin * (f2tilde_norm + (c0sq + 0.75) * h2norm) <= sN
        b_f2_g3 = (mu_min_gl * (f2norm + 0.75 * h2norm)
                   - mu_max_gl * sqrt_f2h2 <= dil_max * Nmax)
        b_f2_g4 = (mu_min_gl * (f2tilde_norm + 0.75 * h2norm)
                   <= dil_max * Nmax)

        # s bounds
        b_s_l1 = s >= dil_min
        b_s_l2 = s <= lm * Nflux
        b_s_l3 = jnp.where(
            hnorm > 0,
            s <= lm * Nflux / hnorm + hnorm / (4.0 * lm),
            True,
        )
        b_s_g1 = s >= dil_min
        b_s_g2 = s <= lambda_max_gl * Nmax

        # f bounds
        b_f_l2 = fnorm >= sN / lm
        b_f_l3 = jnp.where(
            hnorm > 0,
            fnorm <= lm ** 2 * Nflux ** 2 / hnorm * (1.0 + c0sq / s ** 2),
            True,
        )
        b_f_g1 = fnorm >= dil_min * Nmax / lambda_max_gl
        b_f_g3 = fnorm <= lambda_max_gl ** 2 * Nmax ** 2 * 4.0 / 3.0
        b_f_g4 = jnp.where(
            hnorm > 0,
            fnorm <= lambda_max_gl ** 2 * Nflux ** 2 / hnorm + hnorm / 4.0,
            True,
        )

        return (
            b_h1_local & b_h1_global & b_h2_local & b_h2_global
            & b_h_local & b_h_global
            & b_f1_l1 & b_f1_l2 & b_f1_l5 & b_f1_l6 & b_f1_g3 & b_f1_g4
            & b_f2_l1 & b_f2_l2 & b_f2_l5 & b_f2_l6 & b_f2_g3 & b_f2_g4
            & b_s_l1 & b_s_l2 & b_s_l3 & b_s_g1 & b_s_g2
            & b_f_l2 & b_f_l3 & b_f_g1 & b_f_g3 & b_f_g4
        )

    n_h = h_chunk.shape[0]
    evs_rep = tuple(jnp.broadcast_to(e, (n_h,)) for e in evs_j)
    tau_rep = jnp.broadcast_to(tau_j, (n_h,))

    bounds_ok = vmap(_check_single)(
        evs_rep[0], evs_rep[1], evs_rep[2], evs_rep[3], evs_rep[4],
        tau_rep, flux_full, tad,
    )

    return flux_full, tad_ok & bounds_ok


@partial(jit, static_argnums=(2, 3))
def _process_h_all_moduli_jit(
    h_chunk: Array,
    M0_sigma_all: Array,
    n_fluxes: int,
    dim_H3: int,
    s_vec: Array,
    c0_vec: Array,
    evs_all: Tuple,
    tau_vec: Array,
    sigma: Array,
    lambda_max_gl: float,
    mu_min_gl: float,
    mu_max_gl: float,
    tilde_mu_min_gl: float,
    tilde_mu_max_gl: float,
    dil_min: float,
    dil_max: float,
    Nmax: float,
) -> Tuple[Array, Array]:
    r"""

    **Description:**
    Module-level JIT-compiled batch processing of NSNS-flux candidates
    :math:`h` across **all** moduli points simultaneously via ``vmap``.

    Vmaps :func:`_process_h_at_modulus_jit` over the moduli dimension and
    takes the logical OR across moduli to identify :math:`h`-candidates
    that pass bounds at *any* sampled moduli point.  Returns the flux
    vectors from the *first* valid moduli point per :math:`h`-candidate.

    .. admonition:: Details
        :class: dropdown

        The function applies :func:`_process_h_at_modulus_jit` at each of
        the ``n_mod`` sampled moduli points using ``vmap``, obtaining
        validity masks of shape ``(n_mod, n_h)``.  Across moduli it computes

        .. math::
            \mathrm{any\_valid}[i] = \bigvee_{j=1}^{n_{\rm mod}}
            \mathrm{valid}_j[i]\,,

        and for each :math:`h`-candidate :math:`i` selects the ISD-completed
        flux :math:`f_j^*` from the first valid modulus :math:`j^*`:

        .. math::
            j^* = \arg\min_j \{\mathrm{valid}_j[i] = \mathrm{True}\}\,.

        This multi-moduli strategy increases the chance of finding the
        correct ISD completion when the continuous solution
        :math:`f = s\mathcal{M}\Sigma h + c_0 h` rounds to a valid integer
        vector at only a subset of moduli points, or when vacua exist in
        distinct regions of moduli space.

        The single vmapped JIT call avoids the Python overhead of looping
        over moduli points and amortises XLA compilation cost.

    Args:
        h_chunk (Array): NSNS-flux candidates, shape ``(n_h, n_fluxes)``.
        M0_sigma_all (Array): Pre-computed :math:`\mathcal{M}(z_j)\,\Sigma`
            for all moduli points, shape ``(n_mod, n_fluxes, n_fluxes)``.
        n_fluxes (int): Length of a half-flux vector.
        dim_H3 (int): Dimension of the :math:`H^3_+` sector.
        s_vec (Array): :math:`\operatorname{Im}(\tau_j)` for each modulus,
            shape ``(n_mod,)``.
        c0_vec (Array): :math:`\operatorname{Re}(\tau_j)` for each modulus,
            shape ``(n_mod,)``.
        evs_all (Tuple): Pre-computed eigenvalue 5-tuples, each component
            shape ``(n_mod,)``.
        tau_vec (Array): Complex axio-dilaton per modulus, shape ``(n_mod,)``.
        sigma (Array): Symplectic matrix :math:`\Sigma`, shape
            ``(n_fluxes, n_fluxes)``.
        lambda_max_gl (float): Global maximum eigenvalue of :math:`\mathcal{M}`.
        mu_min_gl (float): Global minimum eigenvalue of
            :math:`-\operatorname{Im}(\mathcal{N})`.
        mu_max_gl (float): Global maximum eigenvalue of
            :math:`-\operatorname{Im}(\mathcal{N})`.
        tilde_mu_min_gl (float): Global minimum eigenvalue of
            :math:`\operatorname{Im}(\mathcal{N}^{-1})`.
        tilde_mu_max_gl (float): Global maximum eigenvalue of
            :math:`\operatorname{Im}(\mathcal{N}^{-1})`.
        dil_min (float): Lower bound on :math:`s`.
        dil_max (float): Upper bound on :math:`s`.
        Nmax (float): Maximum D3-tadpole :math:`N_{\max}`.

    Returns:
        Tuple[Array, Array, Array]:
            - **best_flux** — ISD-completed flux :math:`[f \mid h]` from the
              first valid modulus per candidate; shape ``(n_h, 2*n_fluxes)``.
              Rows for which no modulus is valid contain zeros.
            - **any_valid** — boolean mask, shape ``(n_h,)``; ``True`` where
              at least one modulus point yields a valid ISD completion.
            - **best_mod_idx** — index of the first valid modulus per
              candidate, shape ``(n_h,)``; value is ``0`` when no valid modulus
              exists (checked by ``any_valid``).
    """
    # vmap _process_h_at_modulus_jit over moduli dimension
    # Each call returns (flux_full, valid) with shapes (n_h, 2*n_fl), (n_h,)
    def _single_modulus(M0_sigma_j, s_j, c0_j, evs_j, tau_j):
        return _process_h_at_modulus_jit(
            h_chunk, M0_sigma_j,
            n_fluxes, dim_H3,
            s_j, c0_j, evs_j, tau_j, sigma,
            lambda_max_gl, mu_min_gl, mu_max_gl,
            tilde_mu_min_gl, tilde_mu_max_gl,
            dil_min, dil_max, Nmax,
        )

    # evs_all is a tuple of 5 arrays each of shape (n_mod,)
    # vmap over the leading dimension of each
    all_flux, all_valid = vmap(_single_modulus)(
        M0_sigma_all, s_vec, c0_vec,
        evs_all, tau_vec,
    )
    # all_flux shape: (n_mod, n_h, 2*n_fl)
    # all_valid shape: (n_mod, n_h)

    # any_valid: True if valid at any moduli point
    any_valid = jnp.any(all_valid, axis=0)   # (n_h,)

    # For each h-candidate, pick the flux from the FIRST valid moduli point.
    # argmax on bool array returns the first True index (or 0 if none).
    first_valid_idx = jnp.argmax(all_valid, axis=0)  # (n_h,)

    # Gather: best_flux[i] = all_flux[first_valid_idx[i], i, :]
    best_flux = all_flux[first_valid_idx, jnp.arange(h_chunk.shape[0]), :]

    return best_flux, any_valid, first_valid_idx


class bounded_fluxes:
    r"""

    **Description:**
    Implements the flux-bounding algorithm of arXiv:2501.03984 for
    systematic enumeration of Type IIB flux vacua in finite regions of moduli
    space.  Integrates with a model class (:class:`jaxvacua.flux_eft.FluxEFT`)
    and optionally a sampler (:class:`jaxvacua.sampling.data_sampler`).

    .. admonition:: Details
        :class: dropdown

        **Physical setup.**
        In Type IIB compactifications on orientifolds of CY threefolds, the
        Gukov-Vafa-Witten superpotential is

        .. math::
            W = \int_X G_3\wedge\Omega_3
              = (f - \tau h)^T \Sigma\,\Pi(z)\,,

        where :math:`G_3 = F_3 - \tau H_3`, :math:`f = (f_1, f_2)` are the
        integer RR-flux quanta, :math:`h = (h_1, h_2)` are the integer NSNS-flux
        quanta, :math:`\tau = c_0 + \mathrm{i}s` is the axio-dilaton, and
        :math:`\Pi(z)` is the period vector.

        **Flux and symplectic conventions.**
        The full flux vector is ordered as

        .. math::
            \mathtt{flux} = [f_1,\, f_2 \mid h_1,\, h_2]\,,

        each sub-vector of length :math:`d = h^{1,2}+1 = \texttt{dimension\_H3}`.
        The symplectic matrix acts as

        .. math::
            \Sigma = \begin{pmatrix} 0 & \mathbf{1} \\ -\mathbf{1} & 0 \end{pmatrix}\,,

        and the D3-tadpole contribution from the fluxes is (Eq. 10 of
        arXiv:2501.03984)

        .. math::
            N_{\rm flux} = f^T \Sigma h\,.

        **ISD condition and the ISD matrix.**
        A flux configuration is imaginary self-dual (ISD) when
        :math:`\star_6 G_3 = \mathrm{i}G_3`, which in components reads
        (Eq. 21 of arXiv:2501.03984)

        .. math::
            f = (s\,\Sigma\,\mathcal{M} + c_0\,\mathbf{1})\,h\,,

        where the ISD matrix :math:`\mathcal{M}` is constructed from the gauge
        kinetic matrix :math:`\mathcal{N} = \mathcal{R} + \mathrm{i}\mathcal{I}`
        as (Eq. 22)

        .. math::
            \mathcal{M} = \begin{pmatrix}
                -\mathcal{I}^{-1} & \mathcal{I}^{-1}\mathcal{R} \\
                \mathcal{R}\mathcal{I}^{-1} & -(\mathcal{I} + \mathcal{R}\mathcal{I}^{-1}\mathcal{R})
            \end{pmatrix}.

        **Bounding box derivation.**
        For any vacuum satisfying :math:`0 < N_{\rm flux} \leq N_{\max}` and
        :math:`s \geq s_{\min}`, the flux quanta are constrained (Eqs. 27,
        31a-b of arXiv:2501.03984)

        .. math::
            \|h_1\|^2 &\leq \frac{N_{\max}}{s_{\min}\,\tilde\mu_{\min}}\,,\quad
            \|h_2\|^2 \leq \frac{N_{\max}}{s_{\min}\,\mu_{\min}}\,,\quad
            \|h\|^2 \leq \frac{2\,\lambda_{\max}\,N_{\max}}{s_{\min}}\,.

        Here the eigenvalue quantities are:

        - :math:`\lambda_{\max}`: largest eigenvalue of :math:`\mathcal{M}`.
        - :math:`\mu_{\min/\max}`: extreme eigenvalues of :math:`-\operatorname{Im}(\mathcal{N})`.
        - :math:`\tilde\mu_{\min/\max}`: extreme eigenvalues of :math:`\operatorname{Im}(\mathcal{N}^{-1})`.

        These radii define bounding boxes that, combined with a pre-filter on
        the dilaton upper bound

        .. math::
            s_{\max}(h) = \frac{\lambda_{\max}N_{\max}}{\|h\|^2}
                        + \frac{\|h\|^2}{4\,\lambda_{\max}}\,,

        allow systematic exhaustive enumeration (Algorithm 1 of
        arXiv:2501.03984) via :func:`enumerate_fluxes` or stochastic search
        via :func:`sample_bounded_fluxes`.

    .. note::
        All eigenvalue bounds are computed globally over the sampled moduli
        region; tighter *local* bounds (evaluated at each modulus point
        individually) are also checked inside the JIT-compiled kernel
        :func:`_process_h_all_moduli_jit`.
    """

    def __init__(
        self,
        model: Any,
        sampler: Any | None = None,
        Nmax: int | None = None,
        dil_min: float | None = None,
        safety_lambda: float = 10.0,
        safety_mu: float = 1.5,
        map_to_fd: bool = False,
    ) -> None:
        r"""
        **Description:**
        Initialises the bounded-fluxes class.

        Args:
            model: A model object (e.g. :class:`jaxvacua.flux_eft.FluxEFT`) providing
                ``gauge_kinetic_matrix``, ``ISD_matrix``, ``tadpole``, ``n_fluxes``,
                ``dimension_H3``, and ``D3_tadpole``.
            sampler (data_sampler, optional): A
                :class:`jaxvacua.sampling.data_sampler` instance used for
                moduli sampling in :func:`enumerate_fluxes`.  If ``None``,
                moduli must be supplied explicitly to
                :func:`compute_bounding_box`.
            Nmax (int, optional): Maximum allowed D3-tadpole charge.
                Defaults to ``model.D3_tadpole``.
            dil_min (float, optional): Minimum value of
                :math:`s = \operatorname{Im}(\tau)`.  Defaults to
                :math:`\sqrt{3}/2`, the boundary of the
                :math:`\mathrm{SL}(2,\mathbb{Z})` fundamental domain.
            safety_lambda (float, optional): Additive safety margin on
                :math:`\lambda_{\max}`.  Defaults to ``10.0``.
                Set to ``0`` to disable.
            safety_mu (float, optional): Divisive safety margin on
                :math:`\mu_{\min}` and :math:`\tilde\mu_{\min}`.
                Defaults to ``1.5``.  Set to ``1`` to disable.
            map_to_fd (bool, optional): If ``True`` and *model* has
                ``map_to_fd``, map each vacuum to the fundamental domain
                (monodromy + SL(2,Z)) before deduplicating and returning.
                Defaults to ``False``.

        Attributes:
            model: The compactification model.
            sampler: Optional moduli sampler.
            Nmax (int): Maximum allowed D3-tadpole.
            dil_min (float): Lower bound on :math:`s`.
            dil_max (float | None): Upper bound on :math:`s`
                (set after :func:`compute_bounding_box`).
            n_fluxes (int): Length of half the full flux vector,
                :math:`2(h^{1,2}+1)`.
            dimension_H3 (int): Dimension of the :math:`A`-cycle flux sector,
                :math:`h^{1,2}+1`.
            lambda_max_gl (float): Global maximum eigenvalue of
                :math:`\mathcal{M}`.
            mu_min_gl (float): Global minimum eigenvalue of
                :math:`-\operatorname{Im}(\mathcal{N})`.
            mu_max_gl (float): Global maximum eigenvalue of
                :math:`-\operatorname{Im}(\mathcal{N})`.
            tilde_mu_min_gl (float): Global minimum eigenvalue of
                :math:`\operatorname{Im}(\mathcal{N}^{-1})`.
            tilde_mu_max_gl (float): Global maximum eigenvalue of
                :math:`\operatorname{Im}(\mathcal{N}^{-1})`.
        """
        self.model   = model
        self.sampler = sampler
        self._map_to_fd = map_to_fd and hasattr(model, 'map_to_fd')

        # Flux-vector dimensions
        self.n_fluxes     = model.n_fluxes       # = 2*(h12+1)
        self.dimension_H3 = model.dimension_H3   # = h12+1

        # D3-tadpole bound
        self.Nmax = int(model.D3_tadpole) if Nmax is None else int(Nmax)

        # Dilaton lower bound (SL(2,Z) fundamental domain).
        # Tighten to the sampler's actual lower bound when available:
        # the default √3/2 is the SL(2,Z) floor, but if the sampler
        # restricts s ≥ s_lower > √3/2 (e.g. dilaton_bounds=(2,10)),
        # using the larger value gives tighter bounding boxes.
        self.dil_min = float(jnp.sqrt(3.0) / 2.0) if dil_min is None else float(dil_min)
        if sampler is not None and hasattr(sampler, 's_lower'):
            self.dil_min = max(self.dil_min, float(sampler.s_lower))
        self.dil_max = None  # set by compute_bounding_box

        # Safety margins for bounding box (see compute_bounding_box).
        # Set to 0 to disable.
        self.safety_lambda = float(safety_lambda)  # additive on λ_max
        self.safety_mu     = float(safety_mu)       # divisive on μ_min, tμ_min

        # Global eigenvalue extrema (initialised to safe sentinels)
        self.lambda_max_gl   = 0.0
        self.mu_min_gl       = np.inf
        self.mu_max_gl       = 0.0
        self.tilde_mu_min_gl = np.inf
        self.tilde_mu_max_gl = 0.0

        # Local state — populated by update_local / update_evs
        # Initialised to NaN so arithmetic is always well-typed;
        # call update_local() before invoking any bound_* method.
        _nan = float("nan")
        self.lambda_max:   float = _nan
        self.mu_min:       float = _nan
        self.mu_max:       float = _nan
        self.tilde_mu_min: float = _nan
        self.tilde_mu_max: float = _nan
        self.c0:           float = _nan
        self.s:            float = _nan
        self.Nflux:        float = _nan

        # Bounding-box cache — set by compute_bounding_box
        self._h1_box = None
        self._h2_box = None
        self._h_box  = None

        # h2-pool cache — built lazily by _build_h2_pool, invalidated when
        # compute_bounding_box is called again.
        self._h2_pool_cache: Tuple[np.ndarray, np.ndarray] | None = None

        # ISD kernel cache — built once by _get_isd_kernels, keyed by
        # (Q, id(constraints_fn)).  Avoids re-creating JIT/vmap closures
        # on every call to isd_refine_batch.
        self._isd_kernel_cache: dict = {}

    def __repr__(self) -> str:
        r"""
        **Description:**
        Returns a string representation of the bounded-fluxes class.

        Returns:
            str: Description of the class.
        """
        return (
            f"bounded_fluxes(Nmax={self.Nmax}, dil_min={self.dil_min:.4f}, "
            f"model={self.model!r})"
        )

    # =========================================================================
    #  Fundamental domain mapping
    # =========================================================================

    def _map_result_to_fd(self, flux, moduli, tau):
        r"""Map a single (flux, moduli, tau) to the fundamental domain.

        If ``_map_to_fd`` is ``False``, returns the inputs unchanged.
        Otherwise calls ``self.model.map_to_fd`` to apply monodromy and
        SL(2,Z) transformations.

        Returns:
            Tuple: ``(flux_fd, moduli_fd, tau_fd)``
        """
        if not self._map_to_fd:
            return flux, moduli, tau
        moduli_fd, tau_fd, fluxes_fd = self.model.map_to_fd(
            jnp.asarray(moduli), complex(tau), jnp.asarray(flux),
        )
        return np.asarray(fluxes_fd, dtype=np.int32), np.asarray(moduli_fd), complex(tau_fd)

    # =========================================================================
    #  Global eigenvalue bounds
    # =========================================================================

    @property
    def bounds_initialized(self) -> bool:
        r"""
        Description:
        ``True`` if :func:`compute_eigenvalue_bounds` (or
        :func:`compute_bounding_box`) has been called at least once."""
        return self._h1_box is not None and self.lambda_max_gl > 0

    def compute_eigenvalue_bounds(
        self,
        n_sample: int = 100_000,
        rns_key=None,
        verbose: bool = True,
    ) -> Tuple[float, float, float]:
        r"""
        **Description:**
        Sample moduli from :attr:`sampler` and compute global eigenvalue
        bounds.  Results are stored as class attributes and reused by
        subsequent calls to :func:`enumerate_fluxes` and
        :func:`sample_bounded_fluxes`, which will skip re-computation.

        Call this method once (with a large ``n_sample``, e.g. 1M) before
        running any scans.  The eigenvalue extrema accumulate over repeated
        calls (monotone min/max), so calling this multiple times only
        tightens the bounds.

        Args:
            n_sample (int): Number of moduli points to sample.
                Defaults to ``100_000``.
            rns_key (Any, optional): JAX PRNG key for reproducible sampling.
            verbose (bool): Print progress.  Defaults to ``True``.

        Returns:
            Tuple[float, float, float]: ``(h1_box, h2_box, h_box)`` — the
            bounding box dimensions for :math:`h`.
        """
        if self.sampler is None:
            raise ValueError(
                "No sampler provided. Pass sampler=<data_sampler> to "
                "bounded_fluxes() to use compute_eigenvalue_bounds()."
            )

        if verbose:
            print(
                f"[compute_eigenvalue_bounds] Sampling {n_sample:,} moduli "
                f"points ...",
                flush=True,
            )

        t0 = time.perf_counter()
        #moduli = self.sampler.get_complex_moduli(n_sample, rns_key=rns_key)
        #tau = self.sampler.get_complex_tau(n_sample, rns_key=rns_key)
        moduli, tau = self.sampler.initial_guesses(n_sample,
                                                   filter_moduli=True,
                                                   include_fluxes=False, 
                                                   rns_key=rns_key)

        h1, h2, h = self.compute_bounding_box(moduli, tau)

        if verbose:
            dt = time.perf_counter() - t0
            print(
                f"[compute_eigenvalue_bounds] "
                f"λ_max={self.lambda_max_gl:.4f}, "
                f"μ_min={self.mu_min_gl:.4f}, "
                f"μ̃_min={self.tilde_mu_min_gl:.6f}  "
                f"[{dt:.1f}s]",
                flush=True,
            )
            print(
                f"[compute_eigenvalue_bounds] "
                f"Bounding box: h1_box={h1:.2f}, h2_box={h2:.2f}, "
                f"h_box={h:.2f}",
                flush=True,
            )

        return h1, h2, h

    def reset_eigenvalue_bounds(self) -> None:
        r"""
        **Description:**
        Reset global eigenvalue bounds to their initial sentinels.  After
        calling this, :func:`enumerate_fluxes` and
        :func:`sample_bounded_fluxes` will recompute bounds from scratch.
        """
        self.lambda_max_gl   = 0.0
        self.mu_min_gl       = np.inf
        self.mu_max_gl       = 0.0
        self.tilde_mu_min_gl = np.inf
        self.tilde_mu_max_gl = 0.0
        self.dil_max = None
        self._h1_box = None
        self._h2_box = None
        self._h_box  = None
        self._h2_pool_cache = None
        self._evs_cache = None
        self._M_cache = None
        self._isd_kernel_cache = {}

    # =========================================================================
    #  Eigenvalue computation (per-point)
    # =========================================================================

    def _compute_evs_raw(self, moduli: Array) -> Tuple:
        r"""
        **Description:**
        Core eigenvalue computation — not JIT-compiled so it can be vmapped.

        Args:
            moduli (Array): Complex structure moduli, shape ``(h^{1,2},)``.

        Returns:
            Tuple: ``(lambda_max, mu_min, mu_max, tilde_mu_min, tilde_mu_max)``.
        """
        evs, _ = self._compute_evs_and_M_raw(moduli)
        return evs

    def _compute_evs_and_M_raw(self, moduli: Array) -> Tuple:
        r"""

        **Description:**
        Combined eigenvalue and ISD-matrix computation at a single moduli
        point.  Not JIT-compiled so it can be vmapped inside
        :func:`_compute_evs_and_M_vmap`.

        .. admonition:: Details
            :class: dropdown

            Computes the following quantities from the gauge kinetic matrix
            :math:`\mathcal{N}(z) = \mathcal{R} + \mathrm{i}\mathcal{I}`:

            - :math:`\mu_{\min/\max}`: extreme eigenvalues of
              :math:`-\operatorname{Im}(\mathcal{N}) = -\mathcal{I}`.
            - :math:`\tilde\mu_{\min/\max}`: extreme eigenvalues of
              :math:`\operatorname{Im}(\mathcal{N}^{-1})`.
            - :math:`\lambda_{\max}`: largest eigenvalue of the ISD matrix
              :math:`\mathcal{M}(z)` (Eq. 22 of arXiv:2501.03984).
            - :math:`\mathcal{M}(z)` itself (returned for ISD completion).

            All eigenvalue decompositions use
            :func:`jnp.linalg.eigvalsh` (symmetric matrices → real
            eigenvalues, more numerically stable than ``eigvals``).

        Args:
            moduli (Array): Complex structure moduli, shape ``(h^{1,2},)``.

        Returns:
            Tuple:
                - **evs** — 5-tuple :math:`(\lambda_{\max}, \mu_{\min},
                  \mu_{\max}, \tilde\mu_{\min}, \tilde\mu_{\max})`.
                - **M** — ISD matrix :math:`\mathcal{M}(z)`, shape
                  ``(n_fluxes, n_fluxes)``.
        """
        moduli_c = jnp.conj(moduli)

        # Gauge kinetic matrix N; eigenvalues of -Im(N) and Im(N^{-1})
        N      = self.model.gauge_kinetic_matrix(moduli, moduli_c)
        mu_evs = jnp.linalg.eigvalsh(-N.imag)
        mu_min = jnp.min(mu_evs)
        mu_max = jnp.max(mu_evs)

        tmu_evs      = jnp.linalg.eigvalsh(jnp.linalg.inv(N).imag)
        tilde_mu_min = jnp.min(tmu_evs)
        tilde_mu_max = jnp.max(tmu_evs)

        # ISD matrix M; largest eigenvalue AND full matrix
        M          = self.model.ISD_matrix(moduli, moduli_c)
        M_evs      = jnp.linalg.eigvalsh(M)
        lambda_max = jnp.max(M_evs)

        evs = (lambda_max, mu_min, mu_max, tilde_mu_min, tilde_mu_max)
        return evs, M

    @partial(jit, static_argnums=(0,))
    def _compute_evs_and_M_vmap(self, moduli_batch: Array) -> Tuple:
        r"""

        **Description:**
        JIT-compiled, vmapped version of :func:`_compute_evs_and_M_raw`.

        Compiles the eigenvalue and ISD-matrix kernel once and reuses it for
        any batch size.  Results are also cached internally after a
        :func:`compute_bounding_box` call so that subsequent ISD completion
        steps can reuse the :math:`\mathcal{M}(z_j)` matrices without
        recomputation.

        Args:
            moduli_batch (Array): Complex structure moduli, shape
                ``(N, h^{1,2})``.

        Returns:
            Tuple:
                - **evs_all** — 5-tuple of eigenvalue arrays, each shape
                  ``(N,)``:
                  :math:`(\lambda_{\max}, \mu_{\min}, \mu_{\max},
                  \tilde\mu_{\min}, \tilde\mu_{\max})`.
                - **M_all** — ISD matrices :math:`\mathcal{M}(z_j)`, shape
                  ``(N, \texttt{n\_fluxes}, \texttt{n\_fluxes})``.
        """
        return vmap(self._compute_evs_and_M_raw)(moduli_batch)

    @partial(jit, static_argnums=(0,))
    def compute_evs(self, moduli: Array) -> Tuple:
        r"""
        **Description:**
        Computes the eigenvalue quantities at a single moduli point.

        Args:
            moduli (Array): Complex structure moduli, shape ``(h^{1,2},)``.

        Returns:
            Tuple: ``(lambda_max, mu_min, mu_max, tilde_mu_min, tilde_mu_max)``
            where the entries are the largest eigenvalue of :math:`\mathcal{M}`,
            the extreme eigenvalues of :math:`-\operatorname{Im}(\mathcal{N})`,
            and the extreme eigenvalues of
            :math:`\operatorname{Im}(\mathcal{N}^{-1})`.
        """
        return self._compute_evs_raw(moduli)

    @partial(jit, static_argnums=(0,))
    def compute_tadpole_batch(self, flux_batch: Array) -> Array:
        r"""
        **Description:**
        Batched tadpole computation via ``vmap``.

        Args:
            flux_batch (Array): Batch of flux vectors, shape ``(N, 2*n_fluxes)``.

        Returns:
            Array: Tadpole values, shape ``(N,)``.
        """
        return vmap(self.model.tadpole)(flux_batch)

    @partial(jit, static_argnums=(0,))
    def compute_evs_vmap(self, moduli_batch: Array) -> Tuple:
        r"""
        **Description:**
        Batched version of :func:`compute_evs` via ``vmap``.

        Args:
            moduli_batch (Array): Batch of moduli, shape ``(N, h^{1,2})``.

        Returns:
            Tuple: Each element is an array of shape ``(N,)`` corresponding to
            one of the five eigenvalue quantities.
        """
        return vmap(self._compute_evs_raw)(moduli_batch)

    # =================================================================
    #  Precomputed ISD init — compute M₀, dM, dM_c once per moduli set
    # =================================================================

    @partial(jit, static_argnums=(0,))
    def precompute_isd_data(
        self, moduli_pts: Array,
    ) -> Tuple[Array, Array, Array]:
        r"""
        **Description:**
        Precompute ISD-matrix and its derivatives at a batch of moduli
        starting points.  These quantities depend **only on moduli**,
        not on the h-flux, so they can be computed once and reused for
        all h-vectors in the init step.

        Args:
            moduli_pts (Array): Complex structure moduli, shape
                ``(n_pts, h12)``.

        Returns:
            Tuple[Array, Array, Array]:
                - **M0_all** — ISD matrices, shape
                  ``(n_pts, n_fl, n_fl)``.
                - **dM_all** — :math:`\partial_z \mathcal{M}`, shape
                  ``(n_pts, n_fl, n_fl, h12)``.
                - **dM_c_all** — :math:`\partial_{\bar z} \mathcal{M}`,
                  shape ``(n_pts, n_fl, n_fl, h12)``.
        """
        def _per_pt(z):
            """Compute ISD matrix and derivatives at a single moduli point."""
            zc = jnp.conj(z)
            M0 = self.model.ISD_matrix(z, zc)
            dM_val = self.model.dM(z, zc)
            dM_c_val = self.model.dM_c(z, zc)
            return M0, dM_val, dM_c_val

        return vmap(_per_pt)(moduli_pts)

    def _build_isd_apply_init(self, Q: int, constraints_fn):
        r"""
        **Description:**
        Build a JIT-compiled init kernel that uses precomputed
        ``(M0, dM, dM_c)`` for the ISD-matrix and its derivatives.

        The returned function performs only the h-dependent part
        (ISD completion, rounding, linearised shift solve, flag check)
        and is ~2.5× faster than the full ``linearised_shifts`` call
        for large h-batches, since the expensive period computation
        is done once in :func:`precompute_isd_data`.

        Returns:
            Callable: ``apply_init_sub(h_sub, M0_all, dM_all, dM_c_all,
            s_all, c0_all, moduli_pts, tau_pts)`` →
            ``(m_new, t_new, f_new, flags)`` each
            ``(n_h_sub, n_pts, ...)``.
        """
        model = self.model
        sigma = self.model.periods.sigma
        n_fl = self.n_fluxes

        @jit
        def apply_init_sub(
            h_sub, M0_all, dM_all, dM_c_all, s_all, c0_all,
            moduli_pts, tau_pts,
        ):
            """Apply ISD init kernel over a sub-batch of h-vectors."""
            def _per_h(h):
                """Process a single h-vector across all moduli points."""
                SigmaH = sigma @ h

                def _per_pt(M0, dM_v, dM_c_v, s, c0, z_start, tau_start):
                    """Perform ISD completion and linearised shift at one moduli point."""
                    M0_SigmaH = M0 @ SigmaH
                    f_cont = jnp.real(M0_SigmaH * s + h * c0)
                    f = jnp.around(f_cont, 0)
                    b = jnp.real(f - f_cont)
                    RHS = s * SigmaH

                    # dM/dM_c shapes: (n_fl, n_fl, h12)
                    Theta_k = jnp.einsum('ijk,j->ki', dM_v, RHS)
                    Theta_bk = jnp.einsum('ijk,j->ki', dM_c_v, RHS)

                    re_ak = jnp.real(Theta_k) + jnp.real(Theta_bk)
                    re_vk = -jnp.imag(Theta_k) + jnp.imag(Theta_bk)

                    A = jnp.concatenate([
                        re_ak, re_vk,
                        h[None, :],
                        jnp.real(M0_SigmaH)[None, :],
                    ], axis=0).T

                    shift = jnp.linalg.solve(A, b)
                    h12 = dM_v.shape[2]
                    moduli_new = z_start + shift[:h12] + 1j * shift[h12:2*h12]
                    tau_new = tau_start + shift[-2] + 1j * shift[-1]
                    flux_new = jnp.concatenate([f, h])

                    # Flag checks (matching linearised_shifts return_flag)
                    hyperplane_dist = model.lcs_tree.hyperplanes @ jnp.imag(moduli_new)
                    flag = jnp.all(hyperplane_dist >= 0.0)
                    NFlux = model.tadpole(flux_new)
                    flag &= (NFlux > 0)
                    flag &= (NFlux <= Q)
                    if constraints_fn is not None:
                        flag &= constraints_fn(moduli_new, tau_new, flux_new)

                    return moduli_new, tau_new, flux_new, flag

                return vmap(_per_pt)(
                    M0_all, dM_all, dM_c_all, s_all, c0_all,
                    moduli_pts, tau_pts,
                )

            return vmap(_per_h)(h_sub)

        return apply_init_sub

    # =================================================================
    #  ISD refinement kernels (init + step with flags)
    # =================================================================

    def _get_isd_kernels(
        self, Q: int, constraints_fn: Callable | None,
    ) -> Tuple[Callable, Callable]:
        r"""
        **Description:**
        Return cached ``(apply_init_sub, step_sub)`` kernels.

        Builds the JIT/vmap closures on first call for a given
        ``(Q, constraints_fn)`` pair and caches them on the instance.
        Subsequent calls with the same arguments return the cached
        kernels immediately — no re-tracing.

        Args:
            Q (int): Tadpole bound :math:`N_{\max}`.
            constraints_fn (Callable | None): Extra constraint function.

        Returns:
            Tuple[Callable, Callable]:
                - **apply_init_sub** — precomputed init kernel.
                - **step_sub** — iteration step kernel.
        """
        key = (Q, id(constraints_fn))
        if key not in self._isd_kernel_cache:
            _, step_sub = self._build_isd_kernels(Q, constraints_fn)
            apply_init_sub = self._build_isd_apply_init(Q, constraints_fn)
            self._isd_kernel_cache[key] = (apply_init_sub, step_sub)
        return self._isd_kernel_cache[key]

    def _build_isd_kernels(self, Q: int, constraints_fn):
        r"""
        **Description:**
        Build JIT-compiled init / step kernels for the ISD refinement
        loop, following the vmap structure of ``run_hscan_34.py``.

        Both kernels call :func:`model.linearised_shifts` with
        ``mode="Hflux"``, ``return_flag=True``, and the given
        ``constraints_fn``.  Flags encode validity (Kähler cone,
        tadpole ≤ Q, user constraints) so that hopeless candidates can
        be detected and the iteration stopped early.

        The returned functions are JIT-compiled and cached internally
        (subsequent calls with the same ``Q`` and ``constraints_fn``
        return the same objects).

        Returns:
            Tuple[Callable, Callable]:
                - **init_sub** ``(h_sub, moduli_pts, tau_pts)``
                  → ``(m, t, f, flags)`` each ``(n_h_sub, n_pts, ...)``.
                - **step_sub** ``(m, t, f)``
                  → ``(m, t, f, flags)`` each ``(n_h_sub, n_pts, ...)``.
        """
        model = self.model
        n_fl = self.n_fluxes

        # ---- per-h init: vmap over moduli starting points ----
        @jit
        def _init_per_h(h, moduli_pts, tau_pts):
            """Run ISD init for one h-vector across all moduli starting points."""
            flux_init = jnp.concatenate([jnp.zeros(n_fl), h])

            def _per_pt(m, t):
                """Apply linearised shifts at a single moduli/tau starting point."""
                return model.linearised_shifts(
                    m, t, flux_init,
                    Q=Q, mode="Hflux",
                    return_flag=True, constraints=constraints_fn,
                    remove_NANs=True,
                )

            # in_axes=(0, 0): vmap over (moduli, tau), flux broadcast
            return vmap(_per_pt)(moduli_pts, tau_pts)

        # ---- per-h step: vmap over per-point (m, t, f) ----
        @jit
        def _step_per_h(ms, ts, fs):
            """Run one ISD refinement step for a single h-vector."""
            def _per_pt(m, t, f):
                """Apply linearised shifts at a single (moduli, tau, flux) triple."""
                return model.linearised_shifts(
                    m, t, f,
                    Q=Q, mode="Hflux",
                    return_flag=True, constraints=constraints_fn,
                    remove_NANs=True,
                )

            # in_axes=(0, 0, 0): each starting point has its own (m, t, f)
            return vmap(_per_pt)(ms, ts, fs)

        # ---- sub-batch wrappers: vmap over h-vectors ----
        @jit
        def init_sub(h_sub, moduli_pts, tau_pts):
            """Vmap _init_per_h over a sub-batch of h-vectors."""
            return vmap(
                lambda h: _init_per_h(h, moduli_pts, tau_pts)
            )(h_sub)

        @jit
        def step_sub(ms, ts, fs):
            """Vmap _step_per_h over a sub-batch of h-vectors."""
            return vmap(_step_per_h)(ms, ts, fs)

        return init_sub, step_sub

    def isd_refine_batch(
        self,
        h_batch: Array,
        moduli_pts: Array,
        tau_pts: Array,
        n_iters: int = 10,
        h_sub_batch: int = 200,
        constraints=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        **Description:**
        ISD-refinement for a batch of NSNS-flux candidates via iterated
        :meth:`~jaxvacua.flux_vacua_finder.FluxVacuaFinder.linearised_shifts`
        calls with flag-based early stopping.

        .. admonition:: Details
            :class: dropdown

            Follows the algorithm of ``run_hscan_34.py``
            (arXiv:2501.03984):

            **Init step** — for each :math:`h`, construct :math:`[0 \mid h]`
            and evaluate :func:`linearised_shifts` (mode ``"Hflux"``,
            ``return_flag=True``) at every moduli starting point.  The
            inner vmap runs over ``n_pts`` starting points; the outer
            vmap runs over ``h_sub_batch`` h-vectors.

            **Iteration** — apply :func:`linearised_shifts` to the
            per-point ``(m, t, f)`` triples from the previous step.
            After each step, the returned ``flags`` boolean array
            ``(h_sub_batch, n_pts)`` indicates which pairs still
            satisfy the Kähler-cone, tadpole, and user constraints.

            **Early stopping** — if after ≥ 5 iterations **no** flag
            is ``True`` across the entire sub-batch, iteration stops
            (no candidate is converging).

            Peak memory is ``O(h_sub_batch × n_pts × matrix_size)``
            per sub-batch.

        Args:
            h_batch (Array): NSNS-flux candidates, ``(n_h, n_fluxes)``.
            moduli_pts (Array): Starting moduli, ``(n_pts, h12)``.
            tau_pts (Array): Starting axio-dilatons, ``(n_pts,)``.
            n_iters (int): Maximum linearised-shifts iterations.
                Defaults to ``10``.
            h_sub_batch (int): h-vectors per vmapped sub-batch.
                Defaults to ``200``.
            constraints (Callable, optional): Extra constraint function
                ``(moduli, tau, flux) → bool``.  Passed to
                :func:`linearised_shifts`.  Defaults to ``None``
                (only hyperplane + tadpole checks).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - **mod_out** — shape ``(n_h, n_pts, h12)``.
                - **tau_out** — shape ``(n_h, n_pts)``.
                - **flux_out** — shape ``(n_h, n_pts, 2*n_fluxes)``.
        """
        Q = int(self.Nmax)
        apply_init_sub, step_sub = self._get_isd_kernels(Q, constraints)

        # Precompute M₀, dM, dM_c once for all h-vectors (moduli-only)
        M0_all, dM_all, dM_c_all = self.precompute_isd_data(moduli_pts)
        s_all = jnp.imag(tau_pts)
        c0_all = jnp.real(tau_pts)

        n_h = len(h_batch)
        n_fl = h_batch.shape[1]
        mod_list, tau_list, flux_list = [], [], []

        for start in range(0, n_h, h_sub_batch):
            end = min(start + h_sub_batch, n_h)
            h_sub = h_batch[start:end]

            # Pad last sub-batch to fixed size → no JIT recompilation
            pad_n = h_sub_batch - len(h_sub)
            if pad_n > 0:
                h_sub_padded = jnp.concatenate(
                    [h_sub, jnp.zeros((pad_n, n_fl), dtype=h_sub.dtype)],
                    axis=0,
                )
            else:
                h_sub_padded = h_sub

            # Init: precomputed M₀/dM/dM_c → fast ISD completion + shift
            m, t, f, flags = apply_init_sub(
                h_sub_padded, M0_all, dM_all, dM_c_all, s_all, c0_all,
                moduli_pts, tau_pts,
            )

            # Iterate with early stopping (steps use full linearised_shifts
            # since moduli change each iteration)
            for it in range(n_iters - 1):
                m, t, f, flags = step_sub(m, t, f)
                if it >= 4 and not jnp.any(flags):
                    break

            # Unpad and collect
            if jnp.any(flags):
                """
                actual = end - start
                mod_list.append(np.asarray(m[:actual]))
                tau_list.append(np.asarray(t[:actual]))
                flux_list.append(np.asarray(f[:actual]))
                print("F SHAPE: ",f.shape)
                print("F FLAG SHAPE: ",f[flags].shape)
                print("M SHAPE: ",m.shape)
                print("M FLAG SHAPE: ",m[flags].shape)
                print("t SHAPE: ",t.shape)
                print("t FLAG SHAPE: ",t[flags].shape)
                """
                
                mod_list.append(np.asarray(m[flags]))
                tau_list.append(np.asarray(t[flags]))
                flux_list.append(np.asarray(f[flags]))

        if len(mod_list)==0:
            return (
                np.array([]),
                np.array([]),
                np.array([]),
            )
            
        return (
            np.concatenate(mod_list, axis=0),
            np.concatenate(tau_list, axis=0),
            np.concatenate(flux_list, axis=0),
        )

    def _extract_refined_results(
        self,
        mod_out: np.ndarray,
        tau_out: np.ndarray,
        flux_out: np.ndarray,
    ) -> Tuple[list, list, list]:
        r"""
        **Description:**
        Extracts valid ``(flux, moduli, tau)`` triples from the output of
        :func:`isd_refine_batch`.

        Validity requires: (1) non-zero tadpole
        :math:`0 < N_{\rm flux} \leq N_{\max}`, and (2) dilaton
        :math:`s = \operatorname{Im}(\tau) \geq s_{\min}`.

        Args:
            mod_out (np.ndarray): Refined moduli, shape
                ``(n_h, n_pts, h^{1,2})``.
            tau_out (np.ndarray): Refined axio-dilaton, shape
                ``(n_h, n_pts)`` (complex).
            flux_out (np.ndarray): Refined flux :math:`[f \mid h]`, shape
                ``(n_h, n_pts, 2*n_fluxes)`` (real part used).

        Returns:
            Tuple[list, list, list]:
                Lists of valid ``(flux, moduli, tau)`` entries, one per
                valid :math:`h`-candidate.  For each valid :math:`h` the
                entry from the **first** valid starting point is used.
        """
        
        
        """
        sigma_np = np.asarray(self.model.periods.sigma)
        flux_r = flux_out.real
        f_part = flux_r[:, :, :self.n_fluxes]
        h_part = flux_r[:, :, self.n_fluxes:]
        sigma_h = np.einsum('ij,klj->kli', sigma_np, h_part)
        tad_np = np.abs(np.einsum('kli,kli->kl', f_part, sigma_h))
        s_np = tau_out.imag

        valid_mat = (
            (tad_np > 0)
            & (tad_np <= float(self.Nmax))
            & (s_np >= self.dil_min)
        )

        # Hyperplane / Kähler-cone check on Im(moduli)
        hyperplanes = np.asarray(self.model.lcs_tree.hyperplanes)
        im_mod = mod_out.imag  # (n_h, n_pts, h12)
        # hyperplanes: (n_hyp, h12), im_mod: (n_h, n_pts, h12)
        hp_dist = np.einsum('ij,klj->kli', hyperplanes, im_mod)  # (n_h, n_pts, n_hyp)
        _stretch = float(getattr(self.sampler, "stretching", 0.0)) if self.sampler is not None else 0.0
        _excl    = getattr(self.sampler, "exclude_walls", None) if self.sampler is not None else None
        _cone_opted_in = bool(getattr(self.sampler, "cone_opted_in", False)) \
                         if self.sampler is not None else False
        # Bare cone membership (hp_dist >= 0) is always required — the
        # refined moduli must at least be inside the Kähler cone.
        hp_ok = np.all(hp_dist >= 0.0, axis=2)
        # Additional per-wall stretching check only when the user opted
        # into cone constraints.
        if _cone_opted_in:
            if _excl is not None and _excl.shape == (hp_dist.shape[-1],):
                _thr = np.where(np.asarray(_excl), 0.0, _stretch)
                hp_ok &= np.all(hp_dist >= _thr[None, None, :], axis=2)
            else:
                hp_ok &= np.all(hp_dist >= _stretch, axis=2)
        valid_mat &= hp_ok
        # Moduli patch check (sampler bounds on Im(z)).  Broadcasts for
        # both scalar (legacy) and length-`h12` array forms.
        if self.sampler is not None:
            _mod_lo = getattr(self.sampler, 'moduli_lower', None)
            _mod_hi = getattr(self.sampler, 'moduli_upper', None)
            if _mod_lo is not None:
                valid_mat &= np.all(im_mod >= np.asarray(_mod_lo), axis=2)
            if _mod_hi is not None:
                valid_mat &= np.all(im_mod <= np.asarray(_mod_hi), axis=2)
            # Cut-off only applies when cone constraints are active.
            if _cone_opted_in:
                _cutoff = float(getattr(self.sampler, "cone_cutoff", np.inf))
                if np.isfinite(_cutoff):
                    norms = np.linalg.norm(im_mod, axis=2)  # (n_h, n_pts)
                    valid_mat &= (norms <= _cutoff)
        any_valid = np.any(valid_mat, axis=1)       # (n_h,)
        first_pt  = np.argmax(valid_mat, axis=1)    # (n_h,)

        out_fluxes: list = []
        out_moduli: list = []
        out_taus:   list = []
        if np.any(any_valid):
            valid_idx = np.where(any_valid)[0]
            best_flux = flux_r [valid_idx, first_pt[valid_idx]]  # (n_v, 2*n_fl)
            best_mod  = mod_out[valid_idx, first_pt[valid_idx]]  # (n_v, h12)
            best_tau  = tau_out[valid_idx, first_pt[valid_idx]]  # (n_v,)
            out_fluxes.extend(best_flux)
            out_moduli.extend(best_mod)
            out_taus.extend(best_tau.tolist())
        return out_fluxes, out_moduli, out_taus
        """
        
        # Guard: if isd_refine_batch returned empty or 1D arrays, return empty
        if flux_out.ndim < 2 or len(flux_out) == 0:
            return [], [], []

        # ------------------------------------------------------------------
        # All downstream math uses ``jnp.*`` primitives so this function
        # runs on GPU when the inputs live on-device.  Only the final
        # ``device_get`` + ``.extend(...)`` step crosses back to CPU (to
        # produce the Python-list return contract callers expect).
        # ------------------------------------------------------------------
        sigma_j = jnp.asarray(self.model.periods.sigma)
        flux_j  = jnp.asarray(flux_out)
        mod_j   = jnp.asarray(mod_out)
        tau_j   = jnp.asarray(tau_out)

        flux_r = flux_j.real
        f_part = flux_r[:, :self.n_fluxes]
        h_part = flux_r[:, self.n_fluxes:]
        sigma_h = jnp.einsum('ij,lj->li', sigma_j, h_part)
        tad     = jnp.abs(jnp.einsum('li,li->l', f_part, sigma_h))
        s       = tau_j.imag

        valid_mat = (
            (tad > 0)
            & (tad <= float(self.Nmax))
            & (s   >= self.dil_min)
        )

        # Hyperplane / Kähler-cone check on Im(moduli)
        hyperplanes = jnp.asarray(self.model.lcs_tree.hyperplanes)
        im_mod = mod_j.imag  # (n_pts, h12)
        hp_dist = jnp.einsum('ij,lj->li', hyperplanes, im_mod)  # (n_pts, n_hyp)
        _stretch = float(getattr(self.sampler, "stretching", 0.0)) if self.sampler is not None else 0.0
        _excl    = getattr(self.sampler, "exclude_walls", None)    if self.sampler is not None else None
        _cone_opted_in = bool(getattr(self.sampler, "cone_opted_in", False)) \
                         if self.sampler is not None else False
        # Bare cone membership is always required — the refined moduli
        # must at least be inside the Kähler cone.
        hp_ok = jnp.all(hp_dist >= 0.0, axis=1)
        if _cone_opted_in:
            if _excl is not None and _excl.shape == (hp_dist.shape[-1],):
                _thr  = jnp.where(jnp.asarray(_excl), 0.0, _stretch)
                hp_ok = hp_ok & jnp.all(hp_dist >= _thr[None, :], axis=1)
            else:
                hp_ok = hp_ok & jnp.all(hp_dist >= _stretch, axis=1)
        valid_mat = valid_mat & hp_ok

        # Moduli patch check (sampler bounds on Im(z))
        if self.sampler is not None:
            _mod_lo = getattr(self.sampler, 'moduli_lower', None)
            _mod_hi = getattr(self.sampler, 'moduli_upper', None)
            if _mod_lo is not None:
                valid_mat = valid_mat & jnp.all(
                    im_mod >= jnp.asarray(_mod_lo), axis=1,
                )
            if _mod_hi is not None:
                valid_mat = valid_mat & jnp.all(
                    im_mod <= jnp.asarray(_mod_hi), axis=1,
                )
            if _cone_opted_in:
                _cutoff = float(getattr(self.sampler, "cone_cutoff", jnp.inf))
                if jnp.isfinite(_cutoff):
                    valid_mat = valid_mat & (
                        jnp.linalg.norm(im_mod, axis=1) <= _cutoff
                    )

        # Boundary to Python-lists.  Pull the boolean mask back to the
        # host exactly once; from there `.extend` on small ranked arrays
        # is cheap.
        valid_host = np.asarray(valid_mat)
        out_fluxes: list = []
        out_moduli: list = []
        out_taus:   list = []
        if valid_host.any():
            # Boolean indexing is supported on jnp arrays via host mask.
            flux_sel = jnp.asarray(flux_r)[valid_host]
            mod_sel  = mod_j[valid_host]
            tau_sel  = tau_j[valid_host]
            out_fluxes.extend(flux_sel)
            out_moduli.extend(mod_sel)
            out_taus.extend(tau_sel)

        return out_fluxes, out_moduli, out_taus

    def _check_bounds_jax_raw(
        self,
        evs: Tuple,
        tau: Array,
        flux: Array,
        lambda_max_gl: float,
        mu_min_gl: float,
        mu_max_gl: float,
        tilde_mu_min_gl: float,
        tilde_mu_max_gl: float,
        dil_min: float,
        dil_max: float,
        Nmax: float,
    ) -> Array:
        r"""

        **Description:**
        Pure-JAX bound checking for a single :math:`(\mathrm{evs}, \tau,
        \mathrm{flux})` triple.  Returns a scalar boolean (``True`` = all
        bounds satisfied).

        Not JIT-compiled directly so that it can be vmapped over batches
        inside :func:`check_bounds_batch`.

        .. admonition:: Details
            :class: dropdown

            Implements the full hierarchy of inequalities derived in
            arXiv:2501.03984 from the ISD condition.  Bounds are split
            into *local* (evaluated at the current moduli point) and
            *global* (evaluated against the worst-case eigenvalue extrema).
            Let :math:`s = \operatorname{Im}(\tau)`,
            :math:`c_0 = \operatorname{Re}(\tau)`,
            :math:`\tilde f_1 = f_1 - c_0 h_1`,
            :math:`\tilde f_2 = f_2 - c_0 h_2`.

            **h bounds:**

            .. math::
                s\,\tilde\mu_{\min}\,\|h_1\|^2 &\leq N_{\rm flux}\,,\\
                s\,\mu_{\min}\,\|h_2\|^2 &\leq N_{\rm flux}\,,\\
                \|h\|^2 &\leq N_{\rm flux}\,\lambda_{\max}/s
                    \quad\text{(local)}\,,\\
                s_{\min}\,\tilde\mu_{\min}^{\rm gl}\,\|h_1\|^2 &\leq N_{\max}\,,\\
                s_{\min}\,\mu_{\min}^{\rm gl}\,\|h_2\|^2 &\leq N_{\max}\,,\\
                \|h\|^2 &\leq N_{\max}\,\lambda_{\max}^{\rm gl}/s_{\min}
                    \quad\text{(global)}\,.

            **f1 and f2 bounds** (local, 4 inequalities each involving
            :math:`\tilde\mu_{\min/\max}` and :math:`\mu_{\min/\max}`
            respectively; global, 2 inequalities each):

            .. math::
                \tilde\mu_{\min}\,\|\tilde f_1\|^2 &\leq s\,N_{\rm flux}\,,\\
                \tilde\mu_{\min}^{\rm gl}\,
                    (\|\tilde f_1\|^2 + 0.75\|h_1\|^2)
                    &\leq s_{\max}\,N_{\max}\quad\text{(global)}\,,

            and analogous inequalities with :math:`\mu_{\min/\max}` for the
            :math:`f_2` sector.

            **s bounds:**

            .. math::
                s_{\min} \leq s \leq \lambda_{\max}\,N_{\rm flux}\,,\quad
                s \leq \frac{\lambda_{\max}\,N_{\rm flux}}{\|h\|^2}
                        + \frac{\|h\|^2}{4\,\lambda_{\max}}\,.

            **f bounds:**

            .. math::
                \frac{s\,N_{\rm flux}}{\lambda_{\max}} \leq \|f\|^2 \leq
                \frac{\lambda_{\max}^2\,N_{\rm flux}^2}{\|h\|^2}
                \Bigl(1 + \frac{c_0^2}{s^2}\Bigr)\,.

        Args:
            evs (Tuple): Pre-computed eigenvalue 5-tuple
                :math:`(\lambda_{\max}, \mu_{\min}, \mu_{\max},
                \tilde\mu_{\min}, \tilde\mu_{\max})` at the current modulus.
            tau (Array): Axio-dilaton (complex scalar).
            flux (Array): Full flux vector :math:`[f \mid h]`,
                shape ``(2*n_fluxes,)``.
            lambda_max_gl (float): Global maximum eigenvalue of
                :math:`\mathcal{M}`.
            mu_min_gl (float): Global minimum eigenvalue of
                :math:`-\operatorname{Im}(\mathcal{N})`.
            mu_max_gl (float): Global maximum eigenvalue of
                :math:`-\operatorname{Im}(\mathcal{N})`.
            tilde_mu_min_gl (float): Global minimum eigenvalue of
                :math:`\operatorname{Im}(\mathcal{N}^{-1})`.
            tilde_mu_max_gl (float): Global maximum eigenvalue of
                :math:`\operatorname{Im}(\mathcal{N}^{-1})`.
            dil_min (float): Lower bound :math:`s_{\min}` on
                :math:`\operatorname{Im}(\tau)`.
            dil_max (float): Upper bound :math:`s_{\max}` on
                :math:`\operatorname{Im}(\tau)`.
            Nmax (float): Maximum tadpole :math:`N_{\max}`.

        Returns:
            Array: Scalar boolean; ``True`` iff all flux bounds are satisfied.
        """
        dim_H3 = self.dimension_H3
        n_fl   = self.n_fluxes

        # Pre-computed eigenvalues at this moduli point
        lambda_max, mu_min, mu_max, tmu_min, tmu_max = evs

        # Decompose flux and tau
        c0 = jnp.real(tau)
        s  = jnp.imag(tau)
        Nflux = jnp.abs(self.model.tadpole(flux).real)

        f = flux[:n_fl].real
        h = flux[n_fl:].real
        h1, h2 = h[:dim_H3], h[dim_H3:]
        f1, f2 = f[:dim_H3], f[dim_H3:]

        hnorm  = jnp.dot(h, h)
        h1norm = jnp.dot(h1, h1)
        h2norm = jnp.dot(h2, h2)
        fnorm  = jnp.dot(f, f)
        f1norm = jnp.dot(f1, f1)
        f2norm = jnp.dot(f2, f2)

        f1tilde = f1 - c0 * h1
        f2tilde = f2 - c0 * h2
        f1tilde_norm = jnp.dot(f1tilde, f1tilde)
        f2tilde_norm = jnp.dot(f2tilde, f2tilde)

        sqrt_f1h1 = jnp.sqrt(f1norm * h1norm)
        sqrt_f2h2 = jnp.sqrt(f2norm * h2norm)

        sN = s * Nflux
        c0_abs = jnp.abs(c0)
        c0sq = c0 ** 2

        # --- h bounds ---
        b_h1_local  = s * tmu_min * h1norm <= Nflux
        b_h1_global = dil_min * tilde_mu_min_gl * h1norm <= Nmax
        b_h2_local  = s * mu_min * h2norm <= Nflux
        b_h2_global = dil_min * mu_min_gl * h2norm <= Nmax
        b_h_local   = hnorm <= Nflux * lambda_max / s
        b_h_global  = hnorm <= Nmax * lambda_max_gl / dil_min

        # --- f1 bounds ---
        b_f1_l1 = tmu_min * f1tilde_norm <= sN
        b_f1_l2 = (tmu_min * (f1norm + c0sq * h1norm)
                   - 2.0 * c0_abs * tmu_max * sqrt_f1h1 <= sN)
        b_f1_l5 = (tmu_min * (f1norm + (c0sq + 0.75) * h1norm)
                   - 2.0 * c0_abs * tmu_max * sqrt_f1h1 <= sN)
        b_f1_l6 = tmu_min * (f1tilde_norm + (c0sq + 0.75) * h1norm) <= sN
        b_f1_g3 = (tilde_mu_min_gl * (f1norm + 0.75 * h1norm)
                   - tilde_mu_max_gl * sqrt_f1h1 <= dil_max * Nmax)
        b_f1_g4 = (tilde_mu_min_gl * (f1tilde_norm + 0.75 * h1norm)
                   <= dil_max * Nmax)

        # --- f2 bounds ---
        b_f2_l1 = mu_min * f2tilde_norm <= sN
        b_f2_l2 = (mu_min * (f2norm + c0sq * h2norm)
                   - 2.0 * c0_abs * mu_max * sqrt_f2h2 <= sN)
        b_f2_l5 = (mu_min * (f2norm + (c0sq + 0.75) * h2norm)
                   - mu_max * c0_abs * sqrt_f2h2 <= sN)
        b_f2_l6 = mu_min * (f2tilde_norm + (c0sq + 0.75) * h2norm) <= sN
        b_f2_g3 = (mu_min_gl * (f2norm + 0.75 * h2norm)
                   - mu_max_gl * sqrt_f2h2 <= dil_max * Nmax)
        b_f2_g4 = (mu_min_gl * (f2tilde_norm + 0.75 * h2norm)
                   <= dil_max * Nmax)

        # --- s bounds ---
        b_s_l1 = s >= dil_min
        b_s_l2 = s <= lambda_max * Nflux
        b_s_l3 = jnp.where(
            hnorm > 0,
            s <= lambda_max * Nflux / hnorm + hnorm / (4.0 * lambda_max),
            True,
        )
        b_s_g1 = s >= dil_min
        b_s_g2 = s <= lambda_max_gl * Nmax

        # --- f bounds ---
        b_f_l2 = fnorm >= sN / lambda_max
        b_f_l3 = jnp.where(
            hnorm > 0,
            fnorm <= lambda_max ** 2 * Nflux ** 2 / hnorm * (1.0 + c0sq / s ** 2),
            True,
        )
        b_f_g1 = fnorm >= dil_min * Nmax / lambda_max_gl
        b_f_g3 = fnorm <= lambda_max_gl ** 2 * Nmax ** 2 * 4.0 / 3.0
        b_f_g4 = jnp.where(
            hnorm > 0,
            fnorm <= lambda_max_gl ** 2 * Nflux ** 2 / hnorm + hnorm / 4.0,
            True,
        )

        all_pass = (
            b_h1_local & b_h1_global & b_h2_local & b_h2_global
            & b_h_local & b_h_global
            & b_f1_l1 & b_f1_l2 & b_f1_l5 & b_f1_l6 & b_f1_g3 & b_f1_g4
            & b_f2_l1 & b_f2_l2 & b_f2_l5 & b_f2_l6 & b_f2_g3 & b_f2_g4
            & b_s_l1 & b_s_l2 & b_s_l3 & b_s_g1 & b_s_g2
            & b_f_l2 & b_f_l3 & b_f_g1 & b_f_g3 & b_f_g4
        )
        return all_pass

    @partial(jit, static_argnums=(0,))
    def check_bounds_batch(
        self,
        evs_batch: Tuple,
        tau_batch: Array,
        flux_batch: Array,
        lambda_max_gl: float,
        mu_min_gl: float,
        mu_max_gl: float,
        tilde_mu_min_gl: float,
        tilde_mu_max_gl: float,
        dil_min: float,
        dil_max: float,
        Nmax: float,
    ) -> Array:
        r"""

        **Description:**
        JIT-compiled, vmapped bound checking for a batch of flux candidates.
        Replaces the per-candidate Python loop over :func:`update_local` +
        :func:`check_bounds_flat` with a single vectorized JAX call.

        Eigenvalues must be pre-computed via :func:`compute_evs_vmap` and
        passed as ``evs_batch`` to avoid redundant recomputation.  All global
        parameters are passed explicitly so the function is a pure module-level
        JIT kernel.

        Args:
            evs_batch (Tuple): Pre-computed eigenvalue 5-tuple, each component
                of shape ``(N,)``:
                :math:`(\lambda_{\max}, \mu_{\min}, \mu_{\max},
                \tilde\mu_{\min}, \tilde\mu_{\max})`.
            tau_batch (Array): Axio-dilaton values, shape ``(N,)``.
            flux_batch (Array): Full flux vectors :math:`[f \mid h]`,
                shape ``(N, 2 \times \texttt{n\_fluxes})``.
            lambda_max_gl (float): Global maximum eigenvalue of
                :math:`\mathcal{M}`.
            mu_min_gl (float): Global minimum eigenvalue of
                :math:`-\operatorname{Im}(\mathcal{N})`.
            mu_max_gl (float): Global maximum eigenvalue of
                :math:`-\operatorname{Im}(\mathcal{N})`.
            tilde_mu_min_gl (float): Global minimum eigenvalue of
                :math:`\operatorname{Im}(\mathcal{N}^{-1})`.
            tilde_mu_max_gl (float): Global maximum eigenvalue of
                :math:`\operatorname{Im}(\mathcal{N}^{-1})`.
            dil_min (float): Lower bound :math:`s_{\min}`.
            dil_max (float): Upper bound :math:`s_{\max}`.
            Nmax (float): Maximum tadpole :math:`N_{\max}`.

        Returns:
            Array: Boolean array of shape ``(N,)``; ``True`` where all bounds
            pass.
        """
        # evs_batch is a tuple of 5 arrays each of shape (N,)
        # We need to vmap over the batch dimension of each
        def _single(lm, mmin, mmax, tmmin, tmmax, tau, flux):
            """Check bounds for a single (eigenvalues, tau, flux) entry."""
            return self._check_bounds_jax_raw(
                (lm, mmin, mmax, tmmin, tmmax),
                tau, flux,
                lambda_max_gl, mu_min_gl, mu_max_gl,
                tilde_mu_min_gl, tilde_mu_max_gl,
                dil_min, dil_max, Nmax,
            )

        lm, mmin, mmax, tmmin, tmmax = evs_batch
        return vmap(_single)(lm, mmin, mmax, tmmin, tmmax, tau_batch, flux_batch)

    # =========================================================================
    #  Newton refinement
    # =========================================================================

    def _newton_refine_single(
        self, moduli: Array, tau: Array, flux: Array,
        step_size: float, tol: float, max_iters: int,
    ) -> Tuple[Array, Array, Array]:
        r"""
        **Description:**
        Newton-refine a single ``(moduli, tau, flux)`` triple to find the
        exact SUSY vacuum :math:`D_I W = 0`.

        Not JIT-compiled directly so it can be vmapped over batches.

        Args:
            moduli (Array): Starting complex structure moduli, shape ``(h^{1,2},)``.
            tau (Array): Starting axio-dilaton (complex scalar).
            flux (Array): Flux vector, shape ``(2*n_fluxes,)``.
            step_size (float): Newton step size. Defaults to ``1.0``.
            tol (float): Convergence tolerance.
            max_iters (int): Maximum number of Newton iterations.

        Returns:
            Tuple[Array, Array, Array]: ``(moduli_out, tau_out, residual)`` — converged
            moduli, axio-dilaton, and final residual :math:`\sum |D_I W|`.
        """
        _, moduli_out, tau_out, res = (
            self.model._newton_method_flux_vacua_complex(
                moduli, tau, flux, mode="SUSY",
                step_size_Newton=step_size, tol=tol, max_iters=max_iters,
            )
        )
        return moduli_out, tau_out, res

    @partial(jit, static_argnums=(0, 4, 5, 6))
    def newton_refine_batch(
        self,
        moduli_batch: Array,
        tau_batch: Array,
        flux_batch: Array,
        step_size: float = 1e-1,
        tol: float = 1e-10,
        max_iters: int = 100,
    ) -> Tuple[Array, Array, Array]:
        r"""

        **Description:**
        JIT-compiled, vmapped Newton refinement for a batch of flux
        candidates.  Solves :math:`D_I W = 0` simultaneously for all
        ``(moduli, tau, flux)`` triples.

        .. admonition:: Details
            :class: dropdown

            Newton's method minimises :math:`\sum_I |D_I W|^2` by stepping in
            the direction of the F-term residual.  The implementation wraps
            :func:`FluxEFT._newton_method_flux_vacua_complex` in
            ``mode="SUSY"``, which uses the full Jacobian of the F-term system.

            With ``step_size=1.0`` (the default) each step is a full Newton
            step giving locally quadratic convergence near a SUSY minimum.
            Reduce ``step_size`` (e.g. to ``0.1``) for non-SUSY or unstable
            cases.

            The residual :math:`\sum |D_I W|` is used for convergence
            checking: a candidate is considered converged when
            ``residual < tol``.

        Args:
            moduli_batch (Array): Starting complex structure moduli,
                shape ``(N, h^{1,2})``.
            tau_batch (Array): Starting axio-dilaton, shape ``(N,)``.
            flux_batch (Array): Full flux vectors :math:`[f \mid h]`,
                shape ``(N, 2 \times \texttt{n\_fluxes})``.
            step_size (float): Newton step size.  ``1.0`` gives quadratic
                convergence near a SUSY vacuum; use ``0.1`` for non-SUSY.
                Defaults to ``0.1``.
            tol (float): Convergence tolerance on :math:`\sum |D_I W|`.
                Defaults to ``1e-10``.
            max_iters (int): Maximum number of Newton iterations.
                Defaults to ``100``.

        Returns:
            Tuple[Array, Array, Array]:
                - **moduli_out** — converged moduli, shape ``(N, h^{1,2})``.
                - **tau_out** — converged axio-dilaton, shape ``(N,)``.
                - **residuals** — final :math:`\sum |D_I W|`, shape ``(N,)``;
                  compare against ``tol`` to identify converged vacua.
        """
        def _single(moduli, tau, flux):
            """Newton-refine a single (moduli, tau, flux) candidate."""
            return self._newton_refine_single(
                moduli, tau, flux, step_size, tol, max_iters,
            )

        return vmap(_single)(moduli_batch, tau_batch, flux_batch)

    def _in_patch(
        self,
        moduli: Array,
        tau: Array,
    ) -> Array:
        r"""

        **Description:**
        Checks whether converged ``(moduli, tau)`` lie inside the
        sampler's moduli patch.  Pure JAX, vmappable.

        The patch is the **intersection** of three constraints (any
        one of which may be trivial depending on the sampler's
        configuration):

        * Per-component box:
          :math:`\operatorname{Im}(z_i) \in
          [\texttt{moduli\_lower}[i], \texttt{moduli\_upper}[i]]`.
        * Stretched Kähler cone: for every hyperplane row
          :math:`H_k`, :math:`(H_k \cdot \operatorname{Im}(z)) \geq
          \tau_k`, where :math:`\tau_k = 0` for excluded walls
          (:attr:`sampler.exclude_walls`) and
          :math:`\tau_k = \texttt{stretching}` otherwise.
        * Scalar L² cut-off:
          :math:`\|\operatorname{Im}(z)\|_2 \leq
          \texttt{cone\_cutoff}`.

        For a pure box-mode user (default ``stretching=0``,
        ``exclude_walls=None``, ``cone_cutoff`` auto-resolved from
        ``moduli_upper``), the cone and cut-off checks pass trivially
        for any point inside the Kähler cone — so behaviour is
        identical to the pre-refactor per-component check.  For a
        coniLCS user with ``exclude_walls=[conifold_hp_idx]``, the
        check permits moduli arbitrarily close to the conifold wall
        while still enforcing distance from all others.

        Args:
            moduli (Array): Complex structure moduli, shape ``(h^{1,2},)``.
            tau (Array): Axio-dilaton :math:`\tau = c_0 + \mathrm{i}s`
                (complex scalar).

        Returns:
            Array: Scalar boolean; ``True`` if ``(moduli, tau)`` lies
            inside the sampler's patch.
        """
        s = jnp.imag(tau)
        c0 = jnp.real(tau)

        in_s = (s >= self.sampler.s_lower) & (s <= self.sampler.s_upper)
        in_c0 = (c0 >= self.sampler.axion_lower) & (c0 <= self.sampler.axion_upper)

        im_moduli = jnp.imag(moduli)
        # Per-component box check — works for both scalar (broadcast)
        # and per-direction array forms of moduli_lower / moduli_upper.
        in_box = jnp.all(
            (im_moduli >= self.sampler.moduli_lower)
            & (im_moduli <= self.sampler.moduli_upper)
        )

        # Stretched-cone and L² cut-off checks fire **only** when the
        # user explicitly opted into cone constraints (non-zero
        # stretching or any excluded wall).  Pure box-mode sampling
        # does not guarantee Kähler-cone membership, so applying those
        # checks unconditionally would reject valid box-mode samples.
        #
        # ``cone_opted_in`` is a static Python bool cached on the
        # sampler at construction time, so reading it inside this
        # vmapped/jitted function does not trigger
        # ``TracerBoolConversionError``.
        _cone_opted_in = bool(getattr(self.sampler, "cone_opted_in", False))

        if _cone_opted_in and self.sampler._hyperplanes is not None:
            stretching = float(getattr(self.sampler, "stretching", 0.0))
            mask = getattr(self.sampler, "exclude_walls", None)
            H = jnp.asarray(self.sampler._hyperplanes)
            Hdot = H @ im_moduli                               # (n_hp,)
            if mask is not None and mask.shape == (Hdot.shape[0],):
                in_cone = jnp.all(
                    jnp.where(mask, Hdot > 0.0, Hdot >= stretching)
                )
            else:
                in_cone = jnp.all(Hdot >= stretching)
            cutoff = float(getattr(self.sampler, "cone_cutoff", jnp.inf))
            in_radius = jnp.linalg.norm(im_moduli) <= cutoff
        else:
            in_cone = jnp.asarray(True)
            in_radius = jnp.asarray(True)

        return in_s & in_c0 & in_box & in_cone & in_radius

    @partial(jit, static_argnums=(0,))
    def in_patch_batch(
        self,
        moduli_batch: Array,
        tau_batch: Array,
    ) -> Array:
        r"""
        **Description:**
        Vmapped patch-membership check.

        Args:
            moduli_batch (Array): Shape ``(N, h^{1,2})``.
            tau_batch (Array): Shape ``(N,)``.

        Returns:
            Array: Boolean array of shape ``(N,)``.
        """
        return vmap(self._in_patch)(moduli_batch, tau_batch)

    # =========================================================================
    #  Global / local state management
    # =========================================================================

    def update_global(self) -> None:
        r"""

        **Description:**
        Updates the global eigenvalue extrema from the current local values.
        Should be called after :func:`update_evs` or :func:`update_local`.

        Maintains running maximum/minimum of
        :math:`\lambda_{\max}`, :math:`\mu_{\min/\max}`, and
        :math:`\tilde\mu_{\min/\max}` across all moduli points seen so far.
        These global extrema are used to compute the bounding box radii in
        :func:`compute_bounding_box`.
        """
        if not math.isnan(self.lambda_max)   and self.lambda_max   > self.lambda_max_gl:
            self.lambda_max_gl   = self.lambda_max
        if not math.isnan(self.mu_max)       and self.mu_max       > self.mu_max_gl:
            self.mu_max_gl       = self.mu_max
        if not math.isnan(self.tilde_mu_max) and self.tilde_mu_max > self.tilde_mu_max_gl:
            self.tilde_mu_max_gl = self.tilde_mu_max
        if not math.isnan(self.mu_min)       and self.mu_min       < self.mu_min_gl:
            self.mu_min_gl       = self.mu_min
        if not math.isnan(self.tilde_mu_min) and self.tilde_mu_min < self.tilde_mu_min_gl:
            self.tilde_mu_min_gl = self.tilde_mu_min

    def update_evs(self, moduli: Array) -> None:
        r"""
        **Description:**
        Computes eigenvalues at *moduli*, sets the local eigenvalue attributes,
        and updates the global extrema via :func:`update_global`.

        Args:
            moduli (Array): Complex structure moduli, shape ``(h^{1,2},)``.
        """
        lm, mu_min, mu_max, tmu_min, tmu_max = self.compute_evs(moduli)
        self.lambda_max   = float(lm)
        self.mu_min       = float(mu_min)
        self.mu_max       = float(mu_max)
        self.tilde_mu_min = float(tmu_min)
        self.tilde_mu_max = float(tmu_max)
        self.update_global()

    def update_local(self, moduli: Array, tau: complex, flux: Array) -> None:
        r"""
        **Description:**
        Updates the full local state — eigenvalues, axio-dilaton components,
        and flux norms — for a given point ``(moduli, tau, flux)``.  Also
        calls :func:`update_global`.

        Args:
            moduli (Array): Complex structure moduli, shape ``(h^{1,2},)``.
            tau (complex): Axio-dilaton :math:`\tau = c_0 + \mathrm{i}\,s`.
            flux (Array): Full flux vector ``[f \mid h]`` of length
                ``2 * n_fluxes``.
        """
        self.update_evs(moduli)
        self.c0    = float(jnp.real(tau))
        self.s     = float(jnp.imag(tau))
        self.Nflux = float(self.model.tadpole(flux).real)

        self.f, self.h   = self.get_fh(flux)
        self.h1, self.h2 = self.get_flux_split(self.h)
        self.f1, self.f2 = self.get_flux_split(self.f)

        self.hnorm  = self.compute_norm(self.h)
        self.fnorm  = self.compute_norm(self.f)
        self.h1norm = self.compute_norm(self.h1)
        self.h2norm = self.compute_norm(self.h2)
        self.f1norm = self.compute_norm(self.f1)
        self.f2norm = self.compute_norm(self.f2)

        self.f2tilde      = self.f2 - self.c0 * self.h2
        self.f2tilde_norm = self.compute_norm(self.f2tilde)
        self.f1tilde      = self.f1 - self.c0 * self.h1
        self.f1tilde_norm = self.compute_norm(self.f1tilde)

    def _dump_eigenvalue_diagnostic(
        self,
        bad_arrays,
        moduli_batch,
        lm_np,
        mu_np,
        tmu_np,
    ) -> None:
        r"""Print a platform/state summary when eigenvalues are NaN or non-positive.

        Intended for debugging a platform-specific divergence (e.g.
        GHA Linux py3.12 producing NaN where macOS py3.13 produces
        finite values). Dumps JAX config, backend/platform, per-array
        NaN/non-positive counts, and the offending moduli + eigenvalue
        values for the first ~5 bad samples. Does not raise on its own
        — the caller decides whether to raise.
        """
        import sys
        try:
            import jax
            x64 = bool(jax.config.jax_enable_x64)
            backend = str(jax.default_backend())
            devices = [str(d) for d in jax.devices()]
            jax_ver = jax.__version__
        except Exception as exc:
            x64 = f"(jax introspect failed: {exc})"
            backend, devices, jax_ver = "?", [], "?"

        try:
            import numpy as _np
            np_cfg = _np.show_config(mode="dicts") if hasattr(_np, "show_config") else None
            if isinstance(np_cfg, dict):
                bd = np_cfg.get("Build Dependencies", {})
                blas = bd.get("blas", {}).get("name", "?")
                lapack = bd.get("lapack", {}).get("name", "?")
                np_backend = f"blas={blas} lapack={lapack}"
            else:
                np_backend = "(show_config unavailable)"
            np_ver = _np.__version__
        except Exception as exc:
            np_backend = f"(numpy show_config failed: {exc})"
            np_ver = "?"

        mb = np.asarray(moduli_batch)
        bad_mask = (
            np.isnan(lm_np)  | (lm_np  <= 0)
            | np.isnan(mu_np) | (mu_np <= 0)
            | np.isnan(tmu_np) | (tmu_np <= 0)
        )
        bad_idx = np.where(bad_mask)[0]
        first_bad = bad_idx[:5]

        print("=" * 72, flush=True)
        print("[compute_bounding_box] eigenvalue diagnostic", flush=True)
        print("-" * 72, flush=True)
        print(f"  python           : {sys.version.split()[0]} on {sys.platform}", flush=True)
        print(f"  numpy            : {np_ver}", flush=True)
        print(f"  jax              : {jax_ver}   x64={x64}   backend={backend}", flush=True)
        print(f"  jax devices      : {devices}", flush=True)
        print(f"  numpy BLAS/LAPACK: {np_backend}", flush=True)
        print(f"  n_samples        : {mb.shape[0]}    n_bad={len(bad_idx)}", flush=True)
        for name, arr, n_nan, n_nonpos in bad_arrays:
            finite_min = float(np.nanmin(arr)) if not np.all(np.isnan(arr)) else float("nan")
            finite_max = float(np.nanmax(arr)) if not np.all(np.isnan(arr)) else float("nan")
            print(
                f"  {name:>14s}: #nan={n_nan:>3d}  #nonpos={n_nonpos:>3d}  "
                f"finite_range=[{finite_min:.4e}, {finite_max:.4e}]",
                flush=True,
            )
        print(f"  first bad idx    : {list(map(int, first_bad))}", flush=True)
        for idx in first_bad:
            print(
                f"    sample[{int(idx):3d}]: lambda={float(lm_np[idx]):+.6e}  "
                f"mu={float(mu_np[idx]):+.6e}  tmu={float(tmu_np[idx]):+.6e}  "
                f"z={mb[idx]}",
                flush=True,
            )
        print("=" * 72, flush=True)

    # =========================================================================
    #  Bounding box computation
    # =========================================================================

    def compute_bounding_box(
        self,
        moduli_sample: Array,
        tau_sample: Array | None = None,
    ) -> Tuple[float, float, float]:
        r"""

        **Description:**
        Computes global eigenvalue bounds over a sample of moduli points and
        returns the bounding box dimensions for the NSNS-flux vector
        :math:`h = (h_1, h_2)`.

        .. admonition:: Details
            :class: dropdown

            The bounding box radii are derived from the global versions of the
            eigenvalue inequalities of arXiv:2501.03984 (Eqs. 27 and 31a-b).
            For a flux vacuum with
            :math:`0 < N_{\rm flux} \leq N_{\max}` and
            :math:`s \geq s_{\min}`, each sub-vector of the NSNS-flux satisfies

            .. math::
                \|h_1\|^2 &\leq \frac{N_{\max}}{s_{\min}\,\tilde\mu_{\min}^{\rm gl}}
                    \quad\Longrightarrow\quad
                    h_{1,\rm box} = \sqrt{\frac{N_{\max}}{s_{\min}\,\tilde\mu_{\min}^{\rm gl}}}\,,\\
                \|h_2\|^2 &\leq \frac{N_{\max}}{s_{\min}\,\mu_{\min}^{\rm gl}}
                    \quad\Longrightarrow\quad
                    h_{2,\rm box} = \sqrt{\frac{N_{\max}}{s_{\min}\,\mu_{\min}^{\rm gl}}}\,,\\
                \|h\|^2 &\leq \frac{2\,\lambda_{\max}^{\rm gl}\,N_{\max}}{s_{\min}}
                    \quad\Longrightarrow\quad
                    h_{\rm box} = \sqrt{\frac{2\,\lambda_{\max}^{\rm gl}\,N_{\max}}{s_{\min}}}\,.

            The three radii implement a combined ellipsoidal constraint: each
            candidate :math:`h = (h_1, h_2)` must satisfy all three
            simultaneously.  Enumeration of integer lattice points inside this
            region is performed by :func:`get_h_candidates` (small boxes) or
            the streaming generator :func:`_iter_h_chunks_streaming` (large
            boxes).

            The dilaton upper bound is also set:

            .. math::
                s_{\max} = \lambda_{\max}^{\rm gl}\,N_{\max}\,.

            Eigenvalues are computed using the JIT-compiled vmapped kernel
            :func:`_compute_evs_and_M_vmap`, which also caches the full ISD
            matrices :math:`\mathcal{M}(z_j)` for subsequent use in the ISD
            completion step of :func:`enumerate_fluxes`.

        .. note::
            Call :func:`compute_bounding_box_converged` to automatically
            iterate until the running maximum :math:`\lambda_{\max}` has
            stabilised to a relative tolerance, giving a tighter box.

        Args:
            moduli_sample (Array): Complex structure moduli sample, shape
                ``(N, h^{1,2})``.
            tau_sample (Array, optional): Axio-dilaton sample, shape ``(N,)``.
                Currently unused; reserved for future s-dependent refinement.

        Returns:
            Tuple[float, float, float]: ``(h1_box, h2_box, h_box)`` — the
            maximum :math:`L^2` norms allowed for :math:`h_1`, :math:`h_2`,
            and :math:`h` respectively.

        Raises:
            ValueError: If all sampled eigenvalues are NaN (model not
                implemented for the chosen limit or moduli point).
        """
        moduli_batch = jnp.array(moduli_sample, dtype=complex)
        evs_all, M_all = self._compute_evs_and_M_vmap(moduli_batch)
        lm, mu_min, mu_max, tmu_min, tmu_max = evs_all

        # Cache per-point eigenvalues and ISD matrices for enumerate_fluxes
        self._evs_cache = evs_all
        self._M_cache = M_all

        # Use np.nan* reductions so that NaN eigenvalues (e.g. from a model
        # that fails for certain moduli points) don't silently leave the
        # sentinels unchanged, producing a degenerate h_box=0 with no error.
        lm_np    = np.asarray(lm)
        mu_np    = np.asarray(mu_min)
        mumax_np = np.asarray(mu_max)
        tmu_np   = np.asarray(tmu_min)
        tmumax_np = np.asarray(tmu_max)

        # The eigenvalue arrays must be strictly positive for the bounding-box
        # formulas below to produce a real, finite box. Bad samples (NaN, or
        # slightly-negative due to numerical noise on near-singular gauge-
        # kinetic matrices near the Kähler-cone boundary) otherwise propagate
        # via nanmin → mu_min_gl → sqrt(negative) → NaN bounding box →
        # "cannot convert float NaN to integer" downstream.
        #
        # Strategy: build a per-sample validity mask, filter the eigenvalue
        # arrays to the valid subset, then compute min/max on the clean
        # subset. Only raise when every sample in the batch is bad (the
        # sampler is producing nothing physical); a handful of bad samples
        # in an otherwise healthy batch is just numerical noise and gets
        # silently dropped — but still dumped via the diagnostic helper so
        # the CI log records them.
        valid = (
            np.isfinite(lm_np)   & (lm_np   > 0.0) &
            np.isfinite(mu_np)   & (mu_np   > 0.0) &
            np.isfinite(tmu_np)  & (tmu_np  > 0.0)
        )
        n_bad = int((~valid).sum())

        if n_bad > 0:
            bad_arrays = []
            for name, arr in [("lambda_max", lm_np),
                              ("mu_min",     mu_np),
                              ("tilde_mu_min", tmu_np)]:
                n_nan    = int(np.sum(np.isnan(arr)))
                n_nonpos = int(np.sum((~np.isnan(arr)) & (arr <= 0.0)))
                if n_nan > 0 or n_nonpos > 0:
                    bad_arrays.append((name, arr, n_nan, n_nonpos))
            self._dump_eigenvalue_diagnostic(
                bad_arrays, moduli_batch, lm_np, mu_np, tmu_np,
            )
            if not valid.any():
                raise ValueError(
                    "compute_bounding_box: every moduli sample in this "
                    "batch produced a non-positive or NaN eigenvalue. "
                    "The sampler is likely producing points outside the "
                    "Kähler cone or the gauge-kinetic matrix is singular "
                    "throughout the sampled region. See the diagnostic "
                    "block above for per-sample values."
                )

        # Update global extrema from the valid subset only.
        lm_ok     = lm_np[valid]
        mu_ok     = mu_np[valid]
        mumax_ok  = mumax_np[valid]
        tmu_ok    = tmu_np[valid]
        tmumax_ok = tmumax_np[valid]

        self.lambda_max_gl   = max(self.lambda_max_gl,   float(lm_ok.max()))
        self.mu_min_gl       = min(self.mu_min_gl,       float(mu_ok.min()))
        self.mu_max_gl       = max(self.mu_max_gl,       float(mumax_ok.max()))
        self.tilde_mu_min_gl = min(self.tilde_mu_min_gl, float(tmu_ok.min()))
        self.tilde_mu_max_gl = max(self.tilde_mu_max_gl, float(tmumax_ok.max()))

        # Apply optional safety margins to avoid missing flux vectors near
        # the boundary due to finite moduli sampling.  The margins ensure
        # the bounding box is slightly larger than strictly necessary.
        # Set safety_lambda=0 and safety_mu=1 at construction to disable.
        lm_safe   = self.lambda_max_gl + self.safety_lambda
        mu_safe   = self.mu_min_gl / self.safety_mu
        tmu_safe  = self.tilde_mu_min_gl / self.safety_mu

        self.dil_max = float(lm_safe * self.Nmax)

        # Bounding box with safety margins.
        # h1_box, h2_box: derived from individual eigenvalue bounds.
        # h_box: global norm bound from the ISD matrix eigenvalue λ_max.
        #   The factor 2/√3 is the maximum of s/(s²+c₀²) over the SL(2,Z)
        #   fundamental domain (attained at τ = e^{iπ/3}).  This is tighter
        #   than the naive 2/s_min when s_min = √3/2.
        Nmax_eff = float(self.Nmax + 1)  # +1 for integer rounding margin
        h1_box = float(jnp.sqrt(Nmax_eff / (self.dil_min * tmu_safe)))
        h2_box = float(jnp.sqrt(Nmax_eff / (self.dil_min * mu_safe)))
        h_box  = float(jnp.sqrt(2.0 / jnp.sqrt(3.0) * lm_safe * self.Nmax))
        self._h1_box, self._h2_box, self._h_box = h1_box, h2_box, h_box
        # Invalidate cached h2 pool — box dimensions have changed
        self._h2_pool_cache = None

        return h1_box, h2_box, h_box

    def compute_bounding_box_converged(
        self,
        batch_size: int = 100,
        max_batches: int = 500,
        tol: float = 1e-3,
        min_batches: int = 10,
        verbose: bool = True,
    ) -> Tuple[float, float, float]:
        r"""
        **Description:**
        Iterative version of :func:`compute_bounding_box` that keeps sampling
        moduli batches until the running maximum eigenvalue
        :math:`\lambda_{\rm max}` has converged to relative tolerance ``tol``.

        Warm-starts from any prior call: if :attr:`lambda_max_gl` is already
        non-zero (i.e. a previous call already established an estimate), the
        first batch updates from that starting point.

        The sampler attached to this :class:`bounded_fluxes` instance
        (:attr:`sampler`) is used to draw moduli samples.

        Args:
            batch_size (int): Number of moduli points sampled per batch.
                Defaults to ``100``.
            max_batches (int): Stop even if not converged after this many
                batches.  A warning is issued. Defaults to ``500``.
            tol (float): Relative convergence threshold on
                :math:`\lambda_{\rm max}`.  Defaults to ``1e-3``.
            min_batches (int): Always run at least this many batches before
                checking convergence. Defaults to ``10``.
            verbose (bool): Print per-batch progress. Defaults to ``True``.

        Returns:
            Tuple[float, float, float]: ``(h1_box, h2_box, h_box)`` — same
            as :func:`compute_bounding_box`.

        Raises:
            ValueError: If no :attr:`sampler` has been set.
        """
        if self.sampler is None:
            raise ValueError(
                "No sampler set — provide sampler= when constructing "
                "bounded_fluxes() to use compute_bounding_box_converged()."
            )

        lm_prev = self.lambda_max_gl  # nan if fresh; existing value if warm-starting

        for batch_idx in range(max_batches):
            #mod_batch = self.sampler.get_complex_moduli(batch_size)
            #tau_batch = self.sampler.get_complex_tau(batch_size)
            mod_batch, tau_batch = self.sampler.initial_guesses(batch_size,
                                            filter_moduli=True,
                                            include_fluxes=False)
            h1_box, h2_box, h_box = self.compute_bounding_box(mod_batch, tau_batch)

            lm_new = self.lambda_max_gl

            if batch_idx >= min_batches - 1 and not math.isnan(lm_prev):
                rel_change = abs(lm_new - lm_prev) / max(abs(lm_prev), 1e-30)
                if verbose:
                    print(
                        f"[compute_bounding_box_converged] "
                        f"Batch {batch_idx + 1}/{max_batches} — "
                        f"λ_max={lm_new:.6f}  "
                        f"Δrel={rel_change:.2e}  "
                        f"(tol={tol:.0e})  "
                        f"h_box={h_box:.2f}",
                        flush=True,
                    )
                if rel_change < tol:
                    if verbose:
                        print(
                            f"[compute_bounding_box_converged] "
                            f"Converged after {batch_idx + 1} batches "
                            f"({(batch_idx + 1) * batch_size} moduli points).  "
                            f"λ_max={lm_new:.6f}, h_box={h_box:.2f}",
                            flush=True,
                        )
                    return h1_box, h2_box, h_box
            elif verbose and batch_idx % 10 == 0:
                print(
                    f"[compute_bounding_box_converged] "
                    f"Batch {batch_idx + 1}/{max_batches} — "
                    f"λ_max={lm_new:.6f}  h_box={h_box:.2f}",
                    flush=True,
                )

            lm_prev = lm_new

        warnings.warn(
            f"compute_bounding_box_converged: did not converge after "
            f"{max_batches} batches ({max_batches * batch_size} moduli points). "
            f"Final λ_max={self.lambda_max_gl:.6f}.  "
            f"Consider increasing max_batches or batch_size.",
            stacklevel=2,
        )
        return h1_box, h2_box, h_box

    # =========================================================================
    #  h-vector enumeration
    # =========================================================================

    def get_h_box(self) -> Tuple[float, float, float]:
        r"""
        **Description:**
        Returns the cached bounding box dimensions
        ``(h1_box, h2_box, h_box)`` set by :func:`compute_bounding_box`.

        Returns:
            Tuple[float, float, float]: Bounding box :math:`L^2` radii for
            :math:`h_1`, :math:`h_2`, and :math:`h`.

        Raises:
            RuntimeError: If :func:`compute_bounding_box` has not been called.
        """
        if self._h1_box is None:
            raise RuntimeError("Call compute_bounding_box() before get_h_box().")
        return float(self._h1_box), float(self._h2_box), float(self._h_box)  # type: ignore[arg-type]

    def get_h_candidates(
        self,
        max_candidates: int | None = 1_000_000,
    ) -> np.ndarray:
        r"""

        **Description:**
        Enumerates **all** integer NSNS-flux vectors :math:`h = (h_1, h_2)`
        inside the bounding box computed by :func:`compute_bounding_box`,
        pre-filtered by the :math:`L^2`-norm constraints on :math:`h_1` and
        :math:`h_2` separately.

        .. note::
            This method materialises the full candidate array in memory.
            For large boxes (many millions of candidates) use the streaming
            path in :func:`enumerate_fluxes` (activated automatically when
            the estimated count exceeds ``max_h_candidates``), or use
            :func:`_iter_h_chunks_streaming` directly.

        Args:
            max_candidates (int, optional): Emit a warning if the unfiltered
                box contains more than this many candidate vectors.

        Returns:
            np.ndarray: Integer array of shape ``(N_candidates, n_fluxes)``,
            where each row is one candidate :math:`h = [h_1 \mid h_2]` with
            :math:`\|h_1\|^2 \leq h_{1,\rm box}^2` and
            :math:`\|h_2\|^2 \leq h_{2,\rm box}^2`.

        Raises:
            RuntimeError: If :func:`compute_bounding_box` has not been called.
        """
        if self._h1_box is None:
            raise RuntimeError(
                "Call compute_bounding_box() before get_h_candidates()."
            )

        dim    = self.dimension_H3
        h1_max = int(np.ceil(self._h1_box))
        h2_max = int(np.ceil(self._h2_box))

        n_box = (2 * h1_max + 1) ** dim * (2 * h2_max + 1) ** dim
        if max_candidates is not None and n_box > max_candidates:
            warnings.warn(
                f"Bounding box contains up to {n_box:,} candidate h vectors "
                f"(> max_candidates={max_candidates:,}). "
                "Consider tightening the bounds or reducing Nmax.",
                stacklevel=2,
            )

        h1_sq_max = self._h1_box ** 2
        h2_sq_max = self._h2_box ** 2

        # Vectorised h1 enumeration (integer dtype — norms don't need floats)
        h1_range = np.arange(-h1_max, h1_max + 1, dtype=np.int32)
        h1_grids = np.meshgrid(*([h1_range] * dim), indexing="ij")
        h1_all = np.stack([g.ravel() for g in h1_grids], axis=-1)  # (M1, dim)
        h1_norms = np.einsum('ij,ij->i', h1_all, h1_all)
        h1_all = h1_all[h1_norms <= h1_sq_max]

        # Vectorised h2 enumeration
        h2_range = np.arange(-h2_max, h2_max + 1, dtype=np.int32)
        h2_grids = np.meshgrid(*([h2_range] * dim), indexing="ij")
        h2_all = np.stack([g.ravel() for g in h2_grids], axis=-1)  # (M2, dim)
        h2_norms = np.einsum('ij,ij->i', h2_all, h2_all)
        h2_all = h2_all[h2_norms <= h2_sq_max]

        # Combine via outer product
        n1, n2 = len(h1_all), len(h2_all)
        h1_rep = np.repeat(h1_all, n2, axis=0)  # (n1*n2, dim)
        h2_rep = np.tile(h2_all, (n1, 1))        # (n1*n2, dim)
        candidates = np.concatenate([h1_rep, h2_rep], axis=-1)  # (n1*n2, 2*dim)

        return candidates

    def _build_h2_pool(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""

        **Description:**
        Builds and caches the NSNS :math:`h_2`-flux pool used as the outer
        loop of the streaming enumeration in :func:`_iter_h_chunks_streaming`.

        The pool contains all integer vectors :math:`h_2 \in \mathbb{Z}^d`
        satisfying :math:`\|h_2\|^2 \leq h_{2,\rm box}^2`, where
        :math:`d = \texttt{dimension\_H3}`.  Because :math:`h_{2,\rm box}` is
        typically much smaller than :math:`h_{1,\rm box}` (the gauge kinetic
        matrix eigenvalue :math:`\mu_{\min}` is generally larger than
        :math:`\tilde\mu_{\min}`), the pool fits comfortably in memory
        (~hundreds to low thousands of entries) even when the :math:`h_1`
        sub-space is very large.

        The result is cached in :attr:`_h2_pool_cache` and is invalidated
        automatically when :func:`compute_bounding_box` is called again.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - **h2_pool** — shape ``(N_h2, d)`` int32 array of :math:`h_2`
                  candidates satisfying the norm bound.
                - **h2_norms_sq** — shape ``(N_h2,)`` int64 array of
                  :math:`\|h_2\|^2` values.

        Raises:
            RuntimeError: If :func:`compute_bounding_box` has not been called.
        """
        if self._h2_box is None:
            raise RuntimeError(
                "Call compute_bounding_box() before _build_h2_pool()."
            )
        if self._h2_pool_cache is not None:
            return self._h2_pool_cache

        dim = self.dimension_H3
        h2_max = int(np.ceil(self._h2_box))
        h2_sq_max = self._h2_box ** 2

        h2_range = np.arange(-h2_max, h2_max + 1, dtype=np.int32)
        h2_grids = np.meshgrid(*([h2_range] * dim), indexing="ij")
        h2_all = np.stack([g.ravel() for g in h2_grids], axis=-1)
        h2_norms_sq = np.einsum('ij,ij->i', h2_all, h2_all)
        mask = h2_norms_sq <= h2_sq_max
        h2_all = h2_all[mask]
        h2_norms_sq = h2_norms_sq[mask]

        self._h2_pool_cache = (h2_all, h2_norms_sq)
        return self._h2_pool_cache

    def _iter_h_chunks_streaming(
        self,
        h_box_sq: float,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        r"""

        **Description:**
        Generator that yields ``(h_chunk, h_norms_sq)`` pairs using
        **h2-outer streaming**, avoiding materialisation of the full
        :math:`h_1 \times h_2` Cartesian product in memory.

        .. admonition:: Details
            :class: dropdown

            For each :math:`h_2` in the pre-built pool (see
            :func:`_build_h2_pool`), the per-:math:`h_2` constraint
            on :math:`h_1` is

            .. math::
                \|h_1\|^2 \leq \min\!\bigl(h_{1,\rm box}^2,\;
                h_{\rm box}^2 - \|h_2\|^2\bigr)\,.

            All integer :math:`h_1 \in \mathbb{Z}^d` satisfying this
            cross-constraint are enumerated, then concatenated with the
            repeated :math:`h_2` to form a slice

            .. math::
                h_{\rm slice} = \begin{bmatrix} h_1 \mid h_2 \end{bmatrix}
                    \in \mathbb{Z}^{2d}\,.

            The norm :math:`\|h\|^2 = \|h_1\|^2 + \|h_2\|^2` is computed
            here for free and forwarded to the caller, so the s_max filter
            in :func:`enumerate_fluxes` does not re-compute norms (saving
            one O(n) einsum per chunk).

            The generator yields one slice per :math:`h_2` entry
            (no sub-chunking); uniform chunk sizes are enforced downstream
            by :func:`_stream_fixed_chunks` to prevent JIT recompilation.

            **Memory profile**: only one :math:`h_2`-slice is live at a time;
            peak memory is :math:`O(N_{h_1,\rm max} \times 2d \times 4)` bytes
            per slice, typically a few MB.

        Args:
            h_box_sq (float): Global :math:`\|h\|^2` upper bound
                :math:`h_{\rm box}^2`.

        Yields:
            Tuple[np.ndarray, np.ndarray]:
                - **h_slice** — shape ``(N, 2*d)`` int32 array of :math:`h`
                  rows for this :math:`h_2`.
                - **h_norms_sq** — shape ``(N,)`` int64 array of
                  :math:`\|h\|^2 = \|h_1\|^2 + \|h_2\|^2` values.
        """
        h2_pool, h2_norms_sq_pool = self._build_h2_pool()
        dim = self.dimension_H3
        h1_sq_max = self._h1_box ** 2

        # Optional: continuous tadpole data for early h2 rejection.
        # If _stream_M_inv is set (by enumerate_fluxes), use it to skip
        # h2 entries whose D-block tadpole alone exceeds Nmax/s_min.
        M_inv_arr = getattr(self, '_stream_M_inv', None)
        tad_budget = getattr(self, '_stream_tad_budget', None)

        for h2, h2_nsq in zip(h2_pool, h2_norms_sq_pool):
            r1_sq = h_box_sq - float(h2_nsq)
            if r1_sq < 0:
                continue

            # ── Tadpole-based h2 skip ──
            # For h=[h1|h2], the quadratic form decomposes as:
            #   h^T M_inv h = h1^T A h1 + 2 h1^T B h2 + h2^T D h2
            # If min_j(h2^T D_j h2) > budget at ALL moduli points,
            # then no h1 can bring the total below Nmax.
            if M_inv_arr is not None and tad_budget is not None:
                h2_f = h2.astype(np.float64)
                # D-block: M_inv[dim:, dim:]
                D_blocks = M_inv_arr[:, dim:, dim:]  # (n_mod, dim, dim)
                h2Dh2 = np.einsum('i,mij,j->m', h2_f, D_blocks, h2_f)  # (n_mod,)
                # If min contribution from h2 alone exceeds budget, skip
                if np.min(np.abs(h2Dh2)) > tad_budget:
                    continue

            r1_max = int(np.ceil(np.sqrt(r1_sq)))
            if r1_max == 0:
                # Only zero vector
                h1_zero = np.zeros((1, dim), dtype=np.int32)
                h_slice = np.concatenate(
                    [h1_zero, np.broadcast_to(h2, (1, dim))], axis=-1
                )
                yield h_slice, np.array([int(h2_nsq)], dtype=np.int64)
                continue

            h1_range = np.arange(-r1_max, r1_max + 1, dtype=np.int32)
            h1_grids = np.meshgrid(*([h1_range] * dim), indexing="ij")
            h1_all = np.stack([g.ravel() for g in h1_grids], axis=-1)
            h1_norms_sq = np.einsum('ij,ij->i', h1_all, h1_all)
            # Cross-constraint: h1_box² AND ||h1||²+||h2||² ≤ h_box²
            mask = (h1_norms_sq <= h1_sq_max) & (h1_norms_sq + h2_nsq <= h_box_sq)
            h1_all = h1_all[mask]
            if len(h1_all) == 0:
                continue

            h1_norms_sq_f = h1_norms_sq[mask]
            h_norms_sq_slice = h1_norms_sq_f + int(h2_nsq)

            h2_rep = np.broadcast_to(h2, (len(h1_all), dim))
            h_slice = np.concatenate([h1_all, h2_rep], axis=-1)
            yield h_slice, h_norms_sq_slice

    @staticmethod
    def _stream_fixed_chunks(
        gen: Iterator[Tuple[np.ndarray, np.ndarray]],
        chunk_size: int,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        r"""

        **Description:**
        Re-chunks a ``(h_slice, norms_sq)`` generator into fixed-size outputs,
        ensuring all but the final flush contain exactly ``chunk_size`` rows.

        .. admonition:: Details
            :class: dropdown

            JAX/XLA traces a new kernel for each unique array shape it
            encounters.  Without this buffering layer, each :math:`h_2` entry
            whose per-:math:`h_2` :math:`h_1`-slice is smaller than
            ``chunk_size`` would trigger a separate XLA recompilation (~10-40 s
            each), adding hours of overhead for large problems.

            This static method maintains a rolling buffer.  When the buffer
            accumulates at least ``chunk_size`` rows it emits a full chunk
            (fast path: single-array pass-through; merge path: concatenation
            from multiple smaller slices).  The remainder stays buffered.  On
            generator exhaustion the buffer is flushed as a single partial
            chunk — the only size that differs from ``chunk_size``, causing at
            most *one* additional XLA recompilation.

        Args:
            gen (Iterator): Generator of ``(h_slice, norms_sq)`` pairs as
                produced by :func:`_iter_h_chunks_streaming`.
            chunk_size (int): Target number of rows per yielded output chunk.

        Yields:
            Tuple[np.ndarray, np.ndarray]:
                - **h_chunk** — shape ``(chunk_size, 2*d)`` int32 (or smaller
                  for the final flush).
                - **h_norms_sq** — shape ``(chunk_size,)`` int64 (or smaller
                  for the final flush).
        """
        h_buf: List[np.ndarray] = []
        n_buf: List[np.ndarray] = []
        buf_count = 0

        for h_slice, norms_slice in gen:
            h_buf.append(h_slice)
            n_buf.append(norms_slice)
            buf_count += len(h_slice)

            while buf_count >= chunk_size:
                # Fast path: single array already the right size (most common)
                if len(h_buf) == 1 and len(h_buf[0]) == chunk_size:
                    yield h_buf[0], n_buf[0]
                    h_buf.clear()
                    n_buf.clear()
                    buf_count = 0
                    break
                # Merge path: partial chunk at h2 boundary (rare)
                combined_h = np.concatenate(h_buf, axis=0)
                combined_n = np.concatenate(n_buf, axis=0)
                yield combined_h[:chunk_size], combined_n[:chunk_size]
                remainder_h = combined_h[chunk_size:]
                remainder_n = combined_n[chunk_size:]
                h_buf = [remainder_h] if len(remainder_h) > 0 else []
                n_buf = [remainder_n] if len(remainder_n) > 0 else []
                buf_count = len(remainder_h)

        # Final flush (at most one partial chunk — only one JIT retrace)
        if buf_count > 0:
            final_h = np.concatenate(h_buf, axis=0) if len(h_buf) > 1 else h_buf[0]
            final_n = np.concatenate(n_buf, axis=0) if len(n_buf) > 1 else n_buf[0]
            yield final_h, final_n

    # =========================================================================
    #  Shared pipeline preparation
    # =========================================================================

    def _prepare_isd_pipeline(
        self,
        n_sample: int = 500,
        n_isd_per_h: int = 20,
        moduli_regions=None,
        use_linearised_shifts: bool = False,
        n_isd_iters: int = 5,
        constraints=None,
        rns_key=None,
        verbose: bool = True,
        label: str = "pipeline",
    ):
        r"""
        **Description:**
        Shared preparation for both :func:`enumerate_fluxes` and
        :func:`sample_bounded_fluxes`.  Computes eigenvalue bounds, prepares
        the moduli/tau slice for ISD completion, builds the s_max and continuous
        tadpole pre-filters, and returns all precomputed objects as a dict.

        Args:
            n_sample (int): Number of moduli points for eigenvalue bounds.
            n_isd_per_h (int): Moduli points per h-vector for ISD.
            moduli_regions: Optional list of (lo, hi) bands.
            use_linearised_shifts (bool): Use linearised_shifts pipeline.
            n_isd_iters (int): Iterations for linearised_shifts.
            constraints (Callable): Optional constraint function.
            rns_key: Random key for sampling.
            verbose (bool): Print progress.
            label (str): Label for progress messages.

        Returns:
            dict: Precomputed pipeline objects with keys:
                ``moduli_slice``, ``tau_slice``, ``moduli_slice_np``,
                ``tau_slice_np``, ``evs_batch``, ``M0_all``, ``M0_sigma``,
                ``s_vec``, ``c0_vec``, ``gl_params``, ``sigma``,
                ``M_inv_all_np``, ``h1_box``, ``h2_box``, ``h_box``,
                ``_smax_filter``, ``_continuous_tadpole_filter``,
                ``chunk_size``, ``t0``.
        """
        import time as _time
        t0 = _time.perf_counter()
        def _elapsed():
            s = _time.perf_counter() - t0
            if s < 120:
                return f"{s:.1f}s"
            elif s < 3600:
                return f"{s / 60:.1f}m"
            else:
                h = int(s // 3600)
                m = int((s % 3600) // 60)
                return f"{h}h {m}m"

        # ----------------------------------------------------------
        # Step 1: Global eigenvalue bounds
        # ----------------------------------------------------------
        if verbose:
            sb = self.sampler
            region_parts = []
            if hasattr(sb, 'moduli_bounds'):
                region_parts.append(f"Im(z) ∈ {sb.moduli_bounds}")
            if hasattr(sb, 'dilaton_bounds'):
                region_parts.append(f"s ∈ {sb.dilaton_bounds}")
            if hasattr(sb, 'axion_bounds'):
                region_parts.append(f"c₀ ∈ {sb.axion_bounds}")
            region_str = ", ".join(region_parts) if region_parts else "sampler-defined"
            print(f"[{label}] Moduli region: {region_str}", flush=True)

        _bounds_precomputed = self.bounds_initialized
        
        #moduli_sample = self.sampler.get_complex_moduli(n_sample, rns_key=rns_key)
        #tau_sample = self.sampler.get_complex_tau(n_sample, rns_key=rns_key)
        
        moduli_sample, tau_sample = self.sampler.initial_guesses(n_sample,filter_moduli=True,include_fluxes=False,rns_key=rns_key)
        
        if _bounds_precomputed:
            h1_box = self._h1_box
            h2_box = self._h2_box
            h_box  = self._h_box
            
            
            if verbose:
                print(f"[{label}] Step 1 — Using pre-computed eigenvalue bounds.", flush=True)
        else:
            if verbose:
                print(
                    f"[{label}] Step 1 — Sampling {n_sample} moduli points "
                    f"and computing eigenvalue bounds ...", flush=True)
            
            h1_box, h2_box, h_box = self.compute_bounding_box(moduli_sample, tau_sample)

        if verbose:
            print(
                f"[{label}]   λ_max={self.lambda_max_gl:.4f}, "
                f"μ_min={self.mu_min_gl:.4f}, "
                f"μ̃_min={self.tilde_mu_min_gl:.4f}  [{_elapsed()}]", flush=True)
            print(
                f"[{label}]   Bounding box: "
                f"h1_box={h1_box:.2f}, h2_box={h2_box:.2f}, "
                f"h_box={h_box:.2f}", flush=True)

        # ----------------------------------------------------------
        # Step 2: Prepare moduli/tau slice for ISD completion
        # ----------------------------------------------------------
        if verbose:
            print(
                f"[{label}] Step 2 — Preparing ISD kernels "
                f"({n_isd_per_h} moduli points) ...", flush=True)

        if moduli_regions is not None:
            from itertools import product as _iproduct
            _sb = self.sampler
            h12 = self.model.h12
            axion_lo, axion_hi = getattr(_sb, 'axion_bounds', (-0.5, 0.5))
            _rng = np.random.default_rng(getattr(_sb, 'seed', 0))
            combos = list(_iproduct(moduli_regions, repeat=h12))
            region_moduli = []
            region_tau = []
            for combo in combos:
                im_parts = np.column_stack([
                    _rng.uniform(lo, hi, n_isd_per_h) for lo, hi in combo
                ])
                re_parts = _rng.uniform(axion_lo, axion_hi, (n_isd_per_h, h12))
                region_moduli.append(re_parts + 1j * im_parts)
                region_tau.append(_sb.get_complex_tau(n_isd_per_h))
            moduli_slice_np_raw = np.concatenate(region_moduli, axis=0)
            tau_slice_np_raw = np.concatenate(region_tau, axis=0)

            if verbose:
                print(
                    f"[{label}]   Multi-start: {len(combos)} region combos "
                    f"({len(moduli_regions)} regions ^ h12={h12}) "
                    f"× {n_isd_per_h} points = "
                    f"{len(moduli_slice_np_raw)} ISD moduli points.", flush=True)

            moduli_slice = jnp.array(moduli_slice_np_raw, dtype=complex)
            tau_slice = jnp.array(tau_slice_np_raw)
            evs_all_r, M_all_r = self._compute_evs_and_M_vmap(moduli_slice)
            evs_batch = evs_all_r
            M0_all = M_all_r
        else:
            n_mod = min(n_isd_per_h, len(moduli_sample))
            moduli_slice = jnp.array(moduli_sample[:n_mod], dtype=complex)
            tau_slice = jnp.array(tau_sample[:n_mod])
            evs_batch = tuple(e[:n_mod] for e in self._evs_cache)
            M0_all = self._M_cache[:n_mod]

        # ----------------------------------------------------------
        # Step 3: Pre-compute JIT parameters and filters
        # ----------------------------------------------------------
        gl_params = (
            self.lambda_max_gl, self.mu_min_gl, self.mu_max_gl,
            self.tilde_mu_min_gl, self.tilde_mu_max_gl,
            self.dil_min, self.dil_max, float(self.Nmax),
        )

        sigma = self.model.periods.sigma
        M0_sigma = jnp.einsum('mij,jk->mik', M0_all, sigma)
        s_vec = jnp.imag(tau_slice)
        c0_vec = jnp.real(tau_slice)

        moduli_slice_np = np.asarray(moduli_slice)
        tau_slice_np = np.asarray(tau_slice)

        # M_inv for continuous tadpole filter
        M0_all_np = np.asarray(M0_all.real)
        M_inv_all_np = np.array([
            np.linalg.inv(M0_all_np[j]) for j in range(len(M0_all_np))
        ])
        _s_min_tad = float(self.dil_min)
        _Nmax = float(self.Nmax)
        _lm = self.lambda_max_gl

        # s_max filter closure
        def _smax_filter(h_chunk_np, h_norms_sq=None):
            if h_norms_sq is None:
                h_norms_sq = np.einsum('ij,ij->i', h_chunk_np, h_chunk_np).astype(np.float64)
            else:
                h_norms_sq = h_norms_sq.astype(np.float64)
            h_norms = np.sqrt(h_norms_sq)
            nonzero = h_norms > 0
            s_max = np.where(
                nonzero,
                _lm * _Nmax / np.where(nonzero, h_norms, 1.0)
                + np.where(nonzero, h_norms, 0.0) / (4.0 * _lm),
                np.inf)
            keep = s_max >= self.dil_min
            return h_chunk_np[keep], h_norms_sq[keep]

        # Continuous tadpole filter — JIT-compiled for speed.
        # Uses JAX einsum (~9× faster than numpy for large chunks).
        M_inv_jax = jnp.array(M_inv_all_np)

        @jax.jit
        def _tadpole_scores(h_arr):
            """Return min continuous tadpole across moduli for each h."""
            Minv_h = jnp.einsum('mij,hj->mhi', M_inv_jax, h_arr)
            hMinvh = jnp.einsum('hi,mhi->mh', h_arr, Minv_h)
            return jnp.min(jnp.abs(_s_min_tad * hMinvh), axis=0)

        def _continuous_tadpole_filter(h_chunk_np):
            if len(h_chunk_np) == 0:
                return h_chunk_np
            h_jax = jnp.array(h_chunk_np, dtype=jnp.float_)
            scores = np.asarray(_tadpole_scores(h_jax))
            return h_chunk_np[scores <= _Nmax]

        if verbose:
            print(f"[{label}]   {len(moduli_slice)} ISD moduli points.  [{_elapsed()}]", flush=True)

        return {
            'moduli_slice': moduli_slice,
            'tau_slice': tau_slice,
            'moduli_slice_np': moduli_slice_np,
            'tau_slice_np': tau_slice_np,
            'moduli_sample': moduli_sample,
            'tau_sample': tau_sample,
            'evs_batch': evs_batch,
            'M0_all': M0_all,
            'M0_sigma': M0_sigma,
            's_vec': s_vec,
            'c0_vec': c0_vec,
            'gl_params': gl_params,
            'sigma': sigma,
            'M_inv_all_np': M_inv_all_np,
            'h1_box': h1_box,
            'h2_box': h2_box,
            'h_box': h_box,
            '_smax_filter': _smax_filter,
            '_continuous_tadpole_filter': _continuous_tadpole_filter,
            't0': t0,
            '_elapsed': _elapsed,
        }

    def _process_h_chunk(
        self,
        h_chunk: Array,
        pipeline: dict,
        use_linearised_shifts: bool = False,
        n_isd_iters: int = 5,
        n_moduli_batches: int = 1,
        constraints=None,
        rns_key=None,
        n_sample: int = 200,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[complex]]:
        r"""
        **Description:**
        Process a single chunk of h-vectors through the ISD completion pipeline
        and return valid (flux, moduli, tau) triples.

        This is the shared inner loop used by both :func:`enumerate_fluxes`
        and :func:`sample_bounded_fluxes`.

        When ``use_linearised_shifts=True``, the ISD completion is repeated
        ``n_moduli_batches`` times, each time with fresh random moduli/tau
        starting points drawn from the sampler.  This improves coverage of the
        moduli space: a single batch may miss vacua whose moduli lie far from
        the sampled points, but multiple batches with different starting points
        increase the probability of finding them.

        Args:
            h_chunk (Array): JAX array of h-vectors, shape ``(N, n_fluxes)``.
            pipeline (dict): Precomputed objects from :func:`_prepare_isd_pipeline`.
            use_linearised_shifts (bool): Use iterative linearised_shifts.
            n_isd_iters (int): Number of linearised_shifts iterations per batch.
            n_moduli_batches (int): Number of fresh moduli/tau batches to try
                for each h-chunk (only used with ``use_linearised_shifts=True``).
                Defaults to ``1``.
            constraints (Callable): Optional constraint function.
            rns_key: Random key.
            n_sample (int): Number of moduli points per batch.

        Returns:
            Tuple of (new_fluxes, new_moduli, new_taus) lists.
        """
        moduli_slice    = pipeline['moduli_slice']
        tau_slice       = pipeline['tau_slice']
        moduli_slice_np = pipeline['moduli_slice_np']
        tau_slice_np    = pipeline['tau_slice_np']
        evs_batch       = pipeline['evs_batch']
        M0_sigma        = pipeline['M0_sigma']
        s_vec           = pipeline['s_vec']
        c0_vec          = pipeline['c0_vec']
        gl_params       = pipeline['gl_params']
        sigma           = pipeline['sigma']

        new_fluxes: List[np.ndarray] = []
        new_moduli: List[np.ndarray] = []
        new_taus: List[complex] = []

        n_passes = n_moduli_batches if use_linearised_shifts else 1
        for _ in range(n_passes):
            if use_linearised_shifts:
                # Resample moduli for each chunk to improve coverage
                n_mod = min(len(moduli_slice), n_sample)
                
                #moduli_slice0 = jnp.array(self.sampler.get_complex_moduli(n_mod, rns_key=rns_key)[:n_mod], dtype=complex)
                #tau_slice0 = jnp.array( self.sampler.get_complex_tau(n_mod, rns_key=rns_key)[:n_mod])
                    
                moduli_sample, tau_sample = self.sampler.initial_guesses(n_mod,filter_moduli=True,include_fluxes=False,rns_key=rns_key)
                
                moduli_slice0, tau_slice0 = jnp.array(moduli_sample[:n_mod], dtype=complex), jnp.array(tau_sample[:n_mod])

                mod_out, tau_out, flux_out = self.isd_refine_batch(
                    h_chunk, moduli_slice0, tau_slice0, n_isd_iters,
                    constraints=constraints,
                )
                new_f, new_m, new_t = self._extract_refined_results(
                    np.asarray(mod_out),
                    np.asarray(tau_out),
                    np.asarray(flux_out),
                )
                new_fluxes.extend(new_f)
                new_moduli.extend(new_m)
                new_taus.extend(new_t)
            else:
                best_flux, any_valid, first_mod_idx = _process_h_all_moduli_jit(
                    h_chunk, M0_sigma,
                    self.n_fluxes, self.dimension_H3,
                    s_vec, c0_vec, evs_batch, tau_slice, sigma,
                    *gl_params,
                )
                any_valid_np = np.asarray(any_valid)
                if np.any(any_valid_np):
                    valid_idx = np.where(any_valid_np)[0]
                    flux_np = np.asarray(best_flux)
                    mod_idx_np = np.asarray(first_mod_idx)
                    new_fluxes.extend(flux_np[valid_idx])
                    new_moduli.extend(moduli_slice_np[mod_idx_np[valid_idx]])
                    new_taus.extend(tau_slice_np[mod_idx_np[valid_idx]].tolist())
        if len(new_fluxes)>0:
            for _ in range(n_passes):
                if use_linearised_shifts:
                    # Resample moduli for each chunk to improve coverage
                    n_mod = min(len(moduli_slice), n_sample)
                    #moduli_slice0 = jnp.array( self.sampler.get_complex_moduli(n_mod, rns_key=rns_key)[:n_mod], dtype=complex)
                    #tau_slice0 = jnp.array(self.sampler.get_complex_tau(n_mod, rns_key=rns_key)[:n_mod])
                    
                    moduli_sample, tau_sample = self.sampler.initial_guesses(n_mod,filter_moduli=True,include_fluxes=False,rns_key=rns_key)
                
                    moduli_slice0, tau_slice0 = jnp.array(moduli_sample[:n_mod], dtype=complex), jnp.array(tau_sample[:n_mod])

                    mod_out, tau_out, flux_out = self.isd_refine_batch(
                        h_chunk, moduli_slice0, tau_slice0, n_isd_iters,
                        constraints=constraints,
                    )
                    new_f, new_m, new_t = self._extract_refined_results(
                        np.asarray(mod_out),
                        np.asarray(tau_out),
                        np.asarray(flux_out),
                    )
                    new_fluxes.extend(new_f)
                    new_moduli.extend(new_m)
                    new_taus.extend(new_t)
                else:
                    best_flux, any_valid, first_mod_idx = _process_h_all_moduli_jit(
                        h_chunk, M0_sigma,
                        self.n_fluxes, self.dimension_H3,
                        s_vec, c0_vec, evs_batch, tau_slice, sigma,
                        *gl_params,
                    )
                    any_valid_np = np.asarray(any_valid)
                    if np.any(any_valid_np):
                        valid_idx = np.where(any_valid_np)[0]
                        flux_np = np.asarray(best_flux)
                        mod_idx_np = np.asarray(first_mod_idx)
                        new_fluxes.extend(flux_np[valid_idx])
                        new_moduli.extend(moduli_slice_np[mod_idx_np[valid_idx]])
                        new_taus.extend(tau_slice_np[mod_idx_np[valid_idx]].tolist())
                    
        return new_fluxes, new_moduli, new_taus

    # =========================================================================
    #  Main enumeration algorithm
    # =========================================================================

    def enumerate_fluxes(
        self,
        n_sample: int = 500,
        n_isd_per_h: int = 20,
        max_h_candidates: int = 10_000_000,
        verbose: bool = True,
        rns_key: Any | None = None,
        refine: bool = False,
        return_moduli: bool = False,
        newton_tol: float = 1e-10,
        newton_max_iters: int = 100,
        newton_step_size: float = 1.0,
        confirm_streaming: bool = True,
        moduli_regions: List[Tuple[float, float]] | None = None,
        use_linearised_shifts: bool = False,
        n_isd_iters: int = 5,
        n_moduli_batches: int = 1,
        constraints: Callable | None = None,
        chunk_size: int | None = None,
    ) -> list:
        r"""

        **Description:**
        Main flux enumeration algorithm (Algorithm 1 of arXiv:2501.03984).

        Systematically constructs Type IIB flux vacua in a finite region of
        moduli space by exhaustively enumerating all integer NSNS-flux vectors
        :math:`h` inside the eigenvalue-based bounding box.

        .. admonition:: Details
            :class: dropdown

            The algorithm proceeds in up to 6 steps:

            **Step 1 — Bounding box.**
            Sample :math:`n_{\rm sample}` moduli points, compute global
            eigenvalue extrema :math:`(\lambda_{\max}, \mu_{\min},
            \tilde\mu_{\min})`, and derive bounding radii
            :math:`h_{1,\rm box}`, :math:`h_{2,\rm box}`, :math:`h_{\rm box}`
            (Eqs. 27, 31 of arXiv:2501.03984; see :func:`compute_bounding_box`).
            Also tightens :attr:`dil_min` to the sampler's lower dilaton bound.

            **Step 2 — Enumeration / streaming.**
            If the box contains at most ``max_h_candidates`` vectors, all
            integer :math:`h` are materialised via :func:`get_h_candidates`.
            For larger boxes, the h2-outer streaming generator
            :func:`_iter_h_chunks_streaming` is used, which keeps only one
            :math:`h_2`-slice in memory at a time.

            **Step 3 — s_max pre-filter.**
            Before ISD completion, each :math:`h` is checked against the
            per-:math:`h` dilaton ceiling

            .. math::
                s_{\max}(h) = \frac{\lambda_{\max}\,N_{\max}}{\|h\|^2}
                            + \frac{\|h\|^2}{4\,\lambda_{\max}}\,.

            Candidates with :math:`s_{\max}(h) < s_{\min}` cannot host any
            vacuum in the sampler's patch and are dropped immediately,
            typically reducing the candidate count by 40–80 %.

            **Steps 4–5 — ISD completion and bounds checking.**
            For each surviving :math:`h`, the RR-flux is completed as
            (Eq. 21 of arXiv:2501.03984)

            .. math::
                f = \mathrm{round}\!\bigl(
                    s\,\mathcal{M}(z_j)\,\Sigma\,h + c_0\,h
                \bigr)

            at up to ``n_isd_per_h`` sampled moduli points
            :math:`\{z_j, \tau_j\}` using the vmapped JIT kernel
            :func:`_process_h_all_moduli_jit`.  The full flux
            :math:`[f \mid h]` is checked against all local and global
            bounds simultaneously.  A candidate is kept if it passes at
            any of the ``n_isd_per_h`` moduli points.

            **Step 6 (optional) — Newton refinement.**
            When ``refine=True``, each candidate is Newton-refined to
            solve :math:`D_I W = 0` exactly (see :func:`newton_refine_batch`),
            then filtered for convergence and patch membership, and
            deduplicated by flux vector.

        .. note::
            With ``refine=False``, returned fluxes satisfy the ISD bounds
            and tadpole constraint but are *not* exact SUSY vacua: the
            continuous ISD completion is rounded to integers and
            :math:`D_I W \neq 0` in general.  Use ``refine=True`` for
            exact vacua.

        .. note::
            For problems where the full enumeration is infeasible (very large
            :math:`N_{\max}` or :math:`h^{1,2}`), use
            :func:`sample_bounded_fluxes` instead, which randomly samples
            :math:`h` vectors from the bounding box.

        Systematically constructs Type IIB flux vacua in a finite region of
        moduli space via the following steps:

        1. Sample :math:`n_{\rm sample}` moduli points from :attr:`sampler`
           and compute global eigenvalue bounds.
        2. Enumerate all integer :math:`h` vectors in the bounding box via
           :func:`get_h_candidates`.
        3. Pre-filter :math:`h` candidates by the global norm bound
           :math:`\|h\|^2 \leq h_{\rm box}^2`.
        4. For each surviving :math:`h`, compute the ISD-projected RR-flux

           .. math::
               f \approx \bigl(s\,\mathcal{M}(z,\bar z)\,\Sigma + c_0\bigr)\,h

           (rounded to integers) via
           :func:`jaxvacua.sampling.data_sampler.ISD_sampling` at up to
           ``n_isd_per_h`` sampled moduli points.
        5. Retain ``[f | h]`` pairs satisfying the D3-tadpole constraint and
           all local eigenvalue bounds (via :func:`check_bounds_batch`).
        6. (Optional, ``refine=True``) Newton-refine each candidate to solve
           :math:`D_I W = 0` exactly, then verify that the solution lies
           inside the sampler's moduli patch and deduplicate.

        Args:
            n_sample (int): Number of moduli points to sample for computing
                global bounds. Defaults to 500.
            n_isd_per_h (int): Maximum number of moduli sample points tried
                per :math:`h` candidate for ISD completion. Defaults to 20.
            max_h_candidates (int): Threshold above which the bounding box is
                considered "large" and streaming mode is activated
                automatically (h2-outer enumeration, no full materialisation).
                Defaults to ``10_000_000``.  Set to ``None`` to always use
                the standard (non-streaming) path.
            confirm_streaming (bool): When streaming mode is triggered,
                print runtime/memory estimates and ask for interactive
                confirmation before proceeding.  Set to ``False`` to skip
                the prompt (e.g. in scripts/cluster jobs).
                Defaults to ``True``.
            moduli_regions (List[Tuple[float, float]], optional): List of
                ``(lo, hi)`` intervals for the imaginary part of the complex
                structure moduli.  When provided, the ISD kernel samples
                ``n_isd_per_h`` points from each region in turn and the
                combined sample is used for ISD completion, increasing the
                chance of finding vacua spread across the moduli space.
                Example: ``[(1., 2.), (2., 3.), (3., 4.)]``.
                Defaults to ``None`` (use the main sampler's full range).
            verbose (bool): Print progress with timing. Defaults to ``True``.
            rns_key (Any, optional): Random number key for reproducible
                sampling. Passed to :func:`sampler.get_complex_moduli` and
                :func:`sampler.get_complex_tau`.  When ``use_jax=True`` on
                the sampler, this should be a JAX PRNG key.
            refine (bool): If ``True``, Newton-refine candidates to solve
                :math:`D_I W = 0` exactly and filter by convergence and
                patch membership. Defaults to ``False``.
            return_moduli (bool): If ``True``, return a ``List[dict]`` with
                keys ``"flux"``, ``"moduli"``, ``"tau"`` even when
                ``refine=False``.  Defaults to ``False``.
            newton_tol (float): Residual tolerance for Newton convergence.
                Defaults to ``1e-10``.
            newton_max_iters (int): Maximum Newton iterations.
                Defaults to ``100``.
            newton_step_size (float): Step size for Newton's method.
                Defaults to ``1.0`` (full Newton steps, quadratic convergence).
            use_linearised_shifts (bool): If ``True``, replaces the
                fixed-moduli ISD completion with iterated
                :func:`linearised_shifts_H` calls (see
                :func:`isd_refine_batch`).  The moduli are *moved* to be
                self-consistent with each :math:`h`, matching the algorithm
                of arXiv:2501.03984.  Requires :attr:`model` to be a
                :class:`~jaxvacua.flux_vacua_finder.FluxVacuaFinder`
                instance.  Defaults to ``False``.
            n_isd_iters (int): Number of :func:`linearised_shifts_H`
                iterations when ``use_linearised_shifts=True``.
                Defaults to ``5``.

        Returns:
            List[np.ndarray]: When ``refine=False`` and
            ``return_moduli=False``, valid flux vectors ``[f | h]`` of
            length ``2 * n_fluxes`` satisfying the tadpole constraint and
            all local eigenvalue bounds.

            List[dict]: When ``refine=True`` or ``return_moduli=True``,
            each entry is a dict with keys ``"flux"``, ``"moduli"``,
            ``"tau"`` (and ``"residual"`` when ``refine=True``) containing
            the integer flux vector, an associated moduli point
            :math:`z^*`, axio-dilaton :math:`\\tau^*`, and (if refined)
            the F-term residual.

        Raises:
            ValueError: If no :attr:`sampler` has been provided.
        """
        if self.sampler is None:
            raise ValueError(
                "No sampler provided.  Pass sampler=<data_sampler> to "
                "bounded_fluxes() to use enumerate_fluxes()."
            )

        if use_linearised_shifts and not hasattr(self.model, 'linearised_shifts_H'):
            raise ValueError(
                "use_linearised_shifts=True requires the model to expose "
                "'linearised_shifts_H'.  Use a FluxVacuaFinder instance."
            )

        # Tighten dil_min to the sampler's actual lower dilaton bound.
        # The default dil_min = sqrt(3)/2 is the SL(2,Z) fundamental domain
        # floor, but when the sampler restricts s >= s_lower > sqrt(3)/2
        # (e.g. dilaton_bounds=[2,10]), using the larger value gives smaller
        # (tighter) bounding boxes and higher candidate yield.
        if hasattr(self.sampler, 's_lower') and self.sampler.s_lower > self.dil_min:
            self.dil_min = float(self.sampler.s_lower)

        t0 = time.perf_counter()

        def _elapsed() -> str:
            """Return formatted elapsed time since t0."""
            s = time.perf_counter() - t0
            if s < 120:
                return f"{s:.1f}s"
            elif s < 3600:
                return f"{s / 60:.1f}m"
            else:
                h = int(s // 3600)
                m = int((s % 3600) // 60)
                return f"{h}h {m}m"

        # ------------------------------------------------------------------
        # Step 1: Global eigenvalue bounds
        #   - If compute_eigenvalue_bounds() was called beforehand, reuse the
        #     stored values (avoids expensive re-sampling with 1M+ points).
        #   - Otherwise, sample n_sample points and compute on the fly.
        # ------------------------------------------------------------------
        if verbose:
            sb = self.sampler
            region_parts = []
            if hasattr(sb, 'moduli_bounds'):
                region_parts.append(f"Im(z) ∈ {sb.moduli_bounds}")
            if hasattr(sb, 'dilaton_bounds'):
                region_parts.append(f"s ∈ {sb.dilaton_bounds}")
            if hasattr(sb, 'axion_bounds'):
                region_parts.append(f"c₀ ∈ {sb.axion_bounds}")
            region_str = ", ".join(region_parts) if region_parts else "sampler-defined"
            print(
                f"[enumerate_fluxes] Moduli region: {region_str}",
                flush=True,
            )

        _bounds_precomputed = self.bounds_initialized
        if _bounds_precomputed:
            h1_box = self._h1_box
            h2_box = self._h2_box
            h_box  = self._h_box
            if verbose:
                print(
                    f"[enumerate_fluxes] Step 1/5 — Using pre-computed "
                    f"eigenvalue bounds.",
                    flush=True,
                )
                print(
                    f"[enumerate_fluxes]   λ_max={self.lambda_max_gl:.4f}, "
                    f"μ_min={self.mu_min_gl:.4f}, "
                    f"μ̃_min={self.tilde_mu_min_gl:.4f}",
                    flush=True,
                )
                print(
                    f"[enumerate_fluxes]   Bounding box: "
                    f"h1_box={h1_box:.2f}, h2_box={h2_box:.2f}, "
                    f"h_box={h_box:.2f}",
                    flush=True,
                )
        else:
            if verbose:
                print(
                    f"[enumerate_fluxes] Step 1/5 — Sampling {n_sample} moduli "
                    f"points and computing eigenvalue bounds ...",
                    flush=True,
                )
            #moduli_sample = self.sampler.get_complex_moduli(n_sample, rns_key=rns_key)
            #tau_sample = self.sampler.get_complex_tau(n_sample, rns_key=rns_key)
            
            moduli_sample, tau_sample = self.sampler.initial_guesses(n_sample,filter_moduli=True,include_fluxes=False,rns_key=rns_key)
            
            h1_box, h2_box, h_box = self.compute_bounding_box(
                moduli_sample, tau_sample,
            )
            if verbose:
                print(
                    f"[enumerate_fluxes]   λ_max={self.lambda_max_gl:.4f}, "
                    f"μ_min={self.mu_min_gl:.4f}, "
                    f"μ̃_min={self.tilde_mu_min_gl:.4f}  "
                    f"[{_elapsed()}]",
                    flush=True,
                )
                print(
                    f"[enumerate_fluxes]   Bounding box: "
                    f"h1_box={h1_box:.2f}, h2_box={h2_box:.2f}, "
                    f"h_box={h_box:.2f}",
                    flush=True,
                )

        # ------------------------------------------------------------------
        # Feasibility check — abort before enumerating if box is too large
        # ------------------------------------------------------------------
        dim = self.dimension_H3
        h1_max = int(np.ceil(h1_box))
        h2_max = int(np.ceil(h2_box))
        n_box_est = (2 * h1_max + 1) ** dim * (2 * h2_max + 1) ** dim
        ram_est_gb = n_box_est * dim * 2 * 8 / 1e9  # float64 storage

        if verbose:
            if n_box_est < 1e5:
                feasibility = "fast (<1 s expected)"
            elif n_box_est < 1e6:
                feasibility = "moderate (seconds expected)"
            elif n_box_est < 1e7:
                feasibility = "slow (minutes) — consider sample_bounded_fluxes"
            elif max_h_candidates is None or n_box_est < max_h_candidates:
                feasibility = "slow (minutes) — consider sample_bounded_fluxes"
            else:
                feasibility = "large — switching to h2-outer streaming mode"
            print(
                f"[enumerate_fluxes]   Box estimate: {n_box_est:,} candidates "
                f"(≈{ram_est_gb:.2f} GB RAM) — {feasibility}",
                flush=True,
            )

        use_streaming = max_h_candidates is not None and n_box_est > max_h_candidates

        if use_streaming:
            # Build h2 pool now so we can show its size in the estimate
            h2_pool_preview, h2_norms_sq_preview = self._build_h2_pool()
            n_h2 = len(h2_pool_preview)

            # Better estimate: sum over h2 candidates of the h1 ball volume
            # using the per-h2 cross-constraint radius
            # r1(h2) = min(h1_box, sqrt(h_box² - ||h2||²))
            h_box_sq_est = h_box ** 2
            h1_sq_max_est = self._h1_box ** 2
            r1_sq_arr = np.minimum(
                h1_sq_max_est,
                np.maximum(0.0, h_box_sq_est - h2_norms_sq_preview.astype(float)),
            )
            # h1 ball volume in `dim` dimensions: V_dim(r) = π^(dim/2)/Γ(dim/2+1) * r^dim
            ball_const = math.pi ** (dim / 2) / math.gamma(dim / 2 + 1)
            n_h1_per_h2 = ball_const * r1_sq_arr ** (dim / 2)
            n_h_est = int(np.sum(n_h1_per_h2))
            n_chunks_est = max(1, n_h_est // 100_000)

            # Memory: one chunk of int32 h-vectors (2*dim components per vector)
            peak_mem_mb = 100_000 * (2 * dim) * 4 / 1e6

            # Rough timing: assume ~50 ms per JIT chunk on GPU (~500 ms on CPU)
            t_gpu_min = n_chunks_est * 0.05 / 60
            t_cpu_min = n_chunks_est * 0.5 / 60

            if verbose:
                print(
                    f"[enumerate_fluxes]   Box too large — switching to "
                    f"h2-outer streaming mode.",
                    flush=True,
                )
                print(
                    f"[enumerate_fluxes]   Estimates:"
                    f"\n                       h2 pool size    : {n_h2:,}"
                    f"\n                       total h (approx): {n_h_est:,}  (~{n_h_est/1e6:.1f}M)"
                    f"\n                       chunks (≈100K)  : {n_chunks_est:,}"
                    f"\n                       peak memory     : {peak_mem_mb:.1f} MB (one chunk at a time)"
                    f"\n                       time (GPU)      : ~{t_gpu_min:.0f} min"
                    f"\n                       time (CPU)      : ~{t_cpu_min:.0f} min",
                    flush=True,
                )

            if confirm_streaming:
                answer = input(
                    "[enumerate_fluxes]   Continue with streaming enumeration? [y/N] "
                ).strip().lower()
                if answer not in ("y", "yes"):
                    raise RuntimeError(
                        "[enumerate_fluxes] Streaming enumeration cancelled by user."
                    )

        # ------------------------------------------------------------------
        # Step 2 & 3: Enumerate and pre-filter h candidates
        #   (skipped in streaming mode — chunks are yielded on the fly)
        # ------------------------------------------------------------------
        if not use_streaming:
            if verbose:
                print(
                    f"[enumerate_fluxes] Step 2/5 — Enumerating h candidates ...",
                    flush=True,
                )

            h_candidates = self.get_h_candidates(max_candidates=None)

            # Vectorised global h-norm pre-filter
            h_norms_sq = np.einsum('ij,ij->i', h_candidates, h_candidates)
            h_candidates = h_candidates[h_norms_sq <= h_box ** 2]

            if verbose:
                print(
                    f"[enumerate_fluxes]   {len(h_candidates)} h candidates after "
                    f"L²-norm pre-filter  [{_elapsed()}]",
                    flush=True,
                )

        # ------------------------------------------------------------------
        # Step 3: Prepare JIT parameters
        # ------------------------------------------------------------------
        if verbose:
            print(
                f"[enumerate_fluxes] Step 3/5 — Preparing ISD kernels ...",
                flush=True,
            )

        valid_fluxes: List[np.ndarray] = []
        valid_moduli: List[np.ndarray] = []
        valid_taus: List[complex] = []

        # Build the moduli/tau slice used for ISD completion.
        # If moduli_regions is given, take the Cartesian product of regions
        # across all h12 moduli so that off-diagonal combinations are covered.
        # Example: moduli_regions=[(2,3),(3,4),(4,5)], h12=2 → 9 combos:
        #   (2,3)×(2,3), (2,3)×(3,4), (2,3)×(4,5), (3,4)×(2,3), ...
        # Without the product, a vacuum at Im(z1)=2.5, Im(z2)=4.5 is never
        # sampled as an ISD starting point when regions are applied uniformly.
        if moduli_regions is not None:
            from itertools import product as _iproduct
            _sb = self.sampler
            h12 = self.model.h12
            axion_lo, axion_hi = getattr(_sb, 'axion_bounds', (-0.5, 0.5))
            _rng = np.random.default_rng(getattr(_sb, 'seed', 0))
            combos = list(_iproduct(moduli_regions, repeat=h12))

            region_moduli = []
            region_tau = []
            for combo in combos:
                # combo: ((lo_0,hi_0), (lo_1,hi_1), ...) — one interval per modulus
                im_parts = np.column_stack([
                    _rng.uniform(lo, hi, n_isd_per_h) for lo, hi in combo
                ])
                re_parts = _rng.uniform(axion_lo, axion_hi, (n_isd_per_h, h12))
                region_moduli.append(re_parts + 1j * im_parts)
                region_tau.append(_sb.get_complex_tau(n_isd_per_h))

            moduli_slice_np_raw = np.concatenate(region_moduli, axis=0)
            tau_slice_np_raw = np.concatenate(region_tau, axis=0)

            if verbose:
                print(
                    f"[enumerate_fluxes]   Multi-start: {len(combos)} region combos "
                    f"({len(moduli_regions)} regions ^ h12={h12}) "
                    f"× {n_isd_per_h} points = "
                    f"{len(moduli_slice_np_raw)} ISD moduli points.",
                    flush=True,
                )

            moduli_slice = jnp.array(moduli_slice_np_raw, dtype=complex)
            tau_slice = jnp.array(tau_slice_np_raw)

            # Recompute eigenvalues for the region samples (not in _evs_cache)
            evs_all_r, M_all_r = self._compute_evs_and_M_vmap(moduli_slice)
            evs_batch = evs_all_r
            M0_all = M_all_r
        else:
            # No moduli_regions — use a sample from the full sampler range.
            # When bounds were pre-computed, moduli_sample may not exist;
            # sample a small set for ISD completion + compute their evs/M.
            if _bounds_precomputed:
                n_mod = n_isd_per_h
                #_isd_moduli = self.sampler.get_complex_moduli(n_mod, rns_key=rns_key, )
                #_isd_tau = self.sampler.get_complex_tau(n_mod, rns_key=rns_key)
                
                _isd_moduli, _isd_tau = self.sampler.initial_guesses(n_mod,filter_moduli=True,include_fluxes=False, rns_key=rns_key)
                
                moduli_slice = jnp.array(_isd_moduli, dtype=complex)
                tau_slice = jnp.array(_isd_tau)
                evs_all_r, M_all_r = self._compute_evs_and_M_vmap(moduli_slice)
                evs_batch = evs_all_r
                M0_all = M_all_r
            else:
                n_mod = min(n_isd_per_h, len(moduli_sample))
                moduli_slice = jnp.array(moduli_sample[:n_mod], dtype=complex)
                tau_slice = jnp.array(tau_sample[:n_mod])
                # Reuse eigenvalues from bounding box
                evs_batch = tuple(e[:n_mod] for e in self._evs_cache)
                M0_all = self._M_cache[:n_mod]
            if verbose:
                print(
                    f"[enumerate_fluxes]   {len(moduli_slice)} ISD moduli points.",
                    flush=True,
                )

        # Pre-compute global bound parameters for the JIT-compiled checker
        gl_params = (
            self.lambda_max_gl, self.mu_min_gl, self.mu_max_gl,
            self.tilde_mu_min_gl, self.tilde_mu_max_gl,
            self.dil_min, self.dil_max, float(self.Nmax),
        )

        # Pre-compute M0_sigma  (n_mod, n_fl, n_fl)
        sigma = self.model.periods.sigma
        M0_sigma = jnp.einsum(
            'mij,jk->mik', M0_all, sigma
        )

        s_vec = jnp.imag(tau_slice)
        c0_vec = jnp.real(tau_slice)

        # Pre-compute moduli/tau as numpy arrays for result gathering
        moduli_slice_np = np.asarray(moduli_slice)
        tau_slice_np = np.asarray(tau_slice)

        # Pre-compute constants for the s_max per-H pre-filter (Fix 3).
        # s_max(h) = λmax·Nmax/||h|| + ||h||/(4·λmax)
        # Keep only h where s_max ≥ dil_min.  Filters out h vectors whose
        # dilaton maximum is too low to host any vacuum in the sampler's patch.
        _lm = self.lambda_max_gl
        _Nmax = float(self.Nmax)
        _dil_min = self.dil_min

        def _smax_filter(
            h_chunk_np: np.ndarray,
            h_norms_sq: np.ndarray | None = None,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            **Description:**
            Return (h_rows, norms_sq) that pass the s_max filter.

            If ``h_norms_sq`` is provided (e.g. from the streaming generator),
            it is reused directly, avoiding an O(n) einsum per chunk.
            """
            if h_norms_sq is None:
                h_norms_sq = np.einsum(
                    'ij,ij->i', h_chunk_np, h_chunk_np
                ).astype(np.float64)
            else:
                h_norms_sq = h_norms_sq.astype(np.float64)
            h_norms = np.sqrt(h_norms_sq)
            # Zero-norm (h=0) has s_max=∞ — always keep
            nonzero = h_norms > 0
            s_max = np.where(
                nonzero,
                _lm * _Nmax / np.where(nonzero, h_norms, 1.0)
                + np.where(nonzero, h_norms, 0.0) / (4.0 * _lm),
                np.inf,
            )
            keep = s_max >= _dil_min
            return h_chunk_np[keep], h_norms_sq[keep]

        # ------------------------------------------------------------------
        # Continuous tadpole pre-filter.
        #
        # For continuous (unrounded) ISD fluxes, the tadpole is
        #     N_flux^cont(h, z) = s * h^T M^{-1}(z) h
        # where M is the ISD matrix.  A necessary condition for an integer
        # flux with N_flux ≤ Nmax is that N_flux^cont ≤ Nmax at SOME moduli
        # point in the search region.
        #
        # We evaluate at the sampled moduli points using s_min (the most
        # optimistic dilaton value).  An h-vector is kept if
        #     min_j [ s_min * h^T M_inv_j h ] ≤ Nmax
        # for any j = 1, ..., n_mod.
        #
        # Cost: ~10 μs per h (batched einsum) vs ~4 ms per h for full ISD
        # completion.  Benchmark shows ~99.97% rejection rate for the h12=2
        # test model while retaining 100% of known Dataset B solutions.
        # ------------------------------------------------------------------
        M0_all_np = np.asarray(M0_all.real)  # (n_mod, n_fl, n_fl)
        # M_inv at each moduli point — used for the chunk-level tadpole filter.
        # A larger sample is computed in the streaming branch for the h2-level
        # skip, but this set covers the ISD moduli points.
        M_inv_all_np = np.array([
            np.linalg.inv(M0_all_np[j]) for j in range(len(M0_all_np))
        ])  # (n_mod, n_fl, n_fl)
        _s_min_tadpole = float(self.dil_min)

        def _continuous_tadpole_filter(h_chunk_np: np.ndarray) -> np.ndarray:
            r"""
            **Description:**
            Return h-vectors that pass the continuous tadpole pre-filter:

            .. math::
                \min_j \bigl[ s_{\min} \cdot h^T M^{-1}(z_j) h \bigr] \leq N_{\max}

            This is a necessary condition for the integer-rounded ISD flux to
            satisfy the D3-tadpole constraint.  Applied AFTER the s_max filter
            but BEFORE the expensive ISD completion step.
            """
            if len(h_chunk_np) == 0:
                return h_chunk_np
            h_f = h_chunk_np.astype(np.float64)
            # Batched quadratic form: h^T M_inv_j h for all (j, h) pairs
            # M_inv_all_np: (n_mod, n_fl, n_fl), h_f: (n_h, n_fl)
            Minv_h = np.einsum('mij,hj->mhi', M_inv_all_np, h_f)  # (n_mod, n_h, n_fl)
            hMinvh = np.einsum('hi,mhi->mh', h_f, Minv_h)          # (n_mod, n_h)
            N_cont = np.abs(_s_min_tadpole * hMinvh)                # (n_mod, n_h)
            # Keep h if min over moduli points ≤ Nmax
            N_cont_min = np.min(N_cont, axis=0)                     # (n_h,)
            keep = N_cont_min <= _Nmax
            return h_chunk_np[keep]

        # ------------------------------------------------------------------
        # Steps 4 & 5: ISD completion and bound checking (batched)
        # ------------------------------------------------------------------
        # chunk_size: number of h-vectors per processing batch.
        #   - Non-linearised path: default 100K (single vmapped JIT call).
        #   - Linearised-shifts path: default 10K (Python loop over
        #     sub-batches of 200, so memory is independent of chunk_size;
        #     smaller chunks give more frequent progress updates).
        #   - Override via the chunk_size parameter.
        if chunk_size is None:
            chunk_size = 10_000 if use_linearised_shifts else 100_000

        if use_linearised_shifts:
            n_mod = len(moduli_slice)
            if verbose:
                print(
                    f"[enumerate_fluxes]   linearised_shifts mode: "
                    f"chunk_size={chunk_size:,}, n_mod={n_mod}"
                    f" (Python loop over h, vmap over moduli only)",
                    flush=True,
                )

        _interrupted = False

        if use_streaming:
            if verbose:
                print(
                    f"[enumerate_fluxes] Steps 4–5/5 — ISD completion + bound "
                    f"checking (streaming, Nmax={self.Nmax}) ...",
                    flush=True,
                )
            h_box_sq = h_box ** 2
            n_h_processed = 0
            n_h_filtered = 0
            chunk_idx = 0
            jit_compiled = False
            # Pass M_inv data to the streaming generator for h2-level filtering.
            # Use more moduli points than the ISD slice for better coverage,
            # but not too many (the filter cost scales linearly with n_filter_pts).
            # 30 points gives good coverage with ~5× faster filtering than 100.
            n_filter_pts = max(30, len(moduli_slice))
            if n_filter_pts > len(moduli_slice):
                #filter_moduli = self.sampler.get_complex_moduli(n_filter_pts)
                filter_moduli, _ = self.sampler.initial_guesses(n_filter_pts,
                                                                filter_moduli=True,
                                                                include_fluxes=False)
                filter_M_inv = []
                for j in range(n_filter_pts):
                    zj = jnp.array(filter_moduli[j], dtype=complex)
                    Mj = self.model.ISD_matrix(zj, jnp.conj(zj))
                    filter_M_inv.append(np.asarray(jnp.linalg.inv(Mj).real))
                self._stream_M_inv = np.array(filter_M_inv)
            else:
                self._stream_M_inv = M_inv_all_np
            self._stream_tad_budget = float(self.Nmax) / float(self.dil_min)
            if verbose:
                print(
                    f"[enumerate_fluxes]   Tadpole pre-filter: "
                    f"{len(self._stream_M_inv)} M⁻¹ sample points, "
                    f"budget = Nmax/s_min = {self._stream_tad_budget:.1f}",
                    flush=True,
                )

            t_stream_start = time.perf_counter()
            t_last_print = t_stream_start
            _print_interval = 10.0  # seconds between progress lines

            def _stream_eta(n_done: int, n_total: int, elapsed: float) -> str:
                """Return a human-readable ETA string."""
                if n_total <= 0 or n_done <= 0 or elapsed < 1.0:
                    return "—"
                frac = n_done / n_total
                eta_s = elapsed * (1.0 - frac) / frac
                if eta_s >= 7200:
                    return f"{eta_s / 3600:.1f}h"
                if eta_s >= 120:
                    return f"{eta_s / 60:.0f}m"
                return f"{eta_s:.0f}s"

            # Recompute n_chunks_est accounting for h2 tadpole skip.
            # Count how many h2 entries pass the D-block filter.
            h2_pool_pre, h2_nsq_pre = self._build_h2_pool()
            M_inv_filt = self._stream_M_inv
            tad_bgt = self._stream_tad_budget
            dim = self.dimension_H3
            D_blocks = M_inv_filt[:, dim:, dim:]
            n_h2_pass = 0
            n_h_est_filtered = 0
            h1_sq_max_pre = self._h1_box ** 2
            for h2_v, h2_n in zip(h2_pool_pre, h2_nsq_pre):
                h2_f = h2_v.astype(np.float64)
                h2Dh2 = np.einsum('i,mij,j->m', h2_f, D_blocks, h2_f)
                if np.min(np.abs(h2Dh2)) <= tad_bgt:
                    n_h2_pass += 1
                    r1_sq = h_box_sq - float(h2_n)
                    if r1_sq > 0:
                        r1 = min(int(np.ceil(np.sqrt(r1_sq))),
                                 int(np.ceil(np.sqrt(h1_sq_max_pre))))
                        n_h_est_filtered += (2 * r1 + 1) ** dim
            n_chunks_est = max(1, n_h_est_filtered // chunk_size)

            if verbose:
                print(
                    f"[enumerate_fluxes]   After h₂ tadpole skip: "
                    f"{n_h2_pass}/{len(h2_pool_pre)} h₂ entries pass "
                    f"→ ~{n_h_est_filtered:,} candidates "
                    f"(~{n_chunks_est} chunks × {chunk_size:,} rows)",
                    flush=True,
                )

            # _stream_fixed_chunks ensures every JIT call (except the final
            # flush) sees the same h_chunk shape → no repeated XLA recompilation
            # at h2 boundaries.  Norms are forwarded from the generator for free.
            raw_gen = self._iter_h_chunks_streaming(h_box_sq)
            try:
                for h_chunk_np, h_norms_sq_np in self._stream_fixed_chunks( raw_gen, chunk_size):
                    
                    n_h_processed += len(h_chunk_np)
                    # s_max pre-filter (cheap scalar bound) reuses pre-computed norms
                    h_chunk_np, _ = _smax_filter(h_chunk_np, h_norms_sq_np)
                    # Continuous tadpole pre-filter (quadratic form, ~400× cheaper than ISD)
                    h_chunk_np = _continuous_tadpole_filter(h_chunk_np)
                    n_h_filtered += len(h_chunk_np)
                    if len(h_chunk_np) == 0:
                        chunk_idx += 1
                        continue

                    h_chunk = jnp.array(h_chunk_np, dtype=float)

                    # Core ISD processing — shared with sample_bounded_fluxes
                    _pipeline_s = {
                        'moduli_slice': moduli_slice, 'tau_slice': tau_slice,
                        'moduli_slice_np': moduli_slice_np, 'tau_slice_np': tau_slice_np,
                        'evs_batch': evs_batch, 'M0_sigma': M0_sigma,
                        's_vec': s_vec, 'c0_vec': c0_vec,
                        'gl_params': gl_params, 'sigma': sigma,
                    }
                    new_f, new_m, new_t = self._process_h_chunk(
                        h_chunk, _pipeline_s,
                        use_linearised_shifts=use_linearised_shifts,
                        n_isd_iters=n_isd_iters,
                        n_moduli_batches=n_moduli_batches,
                        constraints=constraints,
                        rns_key=rns_key,
                        n_sample=n_isd_per_h,
                    )
                    valid_fluxes.extend(new_f)
                    valid_moduli.extend(new_m)
                    valid_taus.extend(new_t)

                    chunk_idx += 1
                    if verbose:
                        now = time.perf_counter()
                        first_done = not jit_compiled
                        if first_done:
                            jit_compiled = True
                            # Record JIT compilation time; reset the stream
                            # timer so ETA is based on post-JIT throughput only.
                            jit_time = now - t_stream_start
                            t_post_jit = now
                            print(
                                f"[enumerate_fluxes]   JIT compiled in {jit_time:.1f}s"
                                f" | chunk 1 processed {n_h_processed:,} candidates"
                                f" | {n_h_filtered:,} passed filters"
                                f" | {len(valid_fluxes)} valid  [{_elapsed()}]",
                                flush=True,
                            )
                            t_last_print = now
                        elif now - t_last_print >= _print_interval:
                            t_last_print = now
                            # Use post-JIT elapsed time for rate and ETA.
                            # ETA is based on CHUNK index progress (not candidate
                            # count), since the h2-level tadpole skip can eliminate
                            # entire h2 slices, making the actual candidate count
                            # much smaller than n_h_est.
                            elapsed_post_jit = now - t_post_jit
                            rate = (
                                n_h_processed / elapsed_post_jit
                                if elapsed_post_jit > 0.5 else 0.0
                            )
                            # Chunk-based ETA (more reliable than candidate-based)
                            if chunk_idx > 1 and elapsed_post_jit > 0.5:
                                secs_per_chunk = elapsed_post_jit / (chunk_idx - 1)
                                remaining_chunks = max(0, n_chunks_est - chunk_idx)
                                eta_s = secs_per_chunk * remaining_chunks
                            else:
                                eta_s = 0.
                            if eta_s >= 7200:
                                eta_str = f"{eta_s / 3600:.1f}h"
                            elif eta_s >= 120:
                                eta_str = f"{eta_s / 60:.0f}m"
                            elif eta_s > 0:
                                eta_str = f"{eta_s:.0f}s"
                            else:
                                eta_str = "—"
                            frac = n_h_processed / n_h_est if n_h_est > 0 else 0.0
                            smax_pct = (
                                100.0 * n_h_filtered / n_h_processed
                                if n_h_processed > 0 else 0.0
                            )
                            print(
                                f"[enumerate_fluxes]   chunk {chunk_idx:,}/{n_chunks_est:,}"
                                f" | {n_h_processed / 1e6:.2f}M"
                                f" / {n_h_est / 1e6:.1f}M"
                                f" | {rate / 1e3:.0f}K/s"
                                f" | pass: {smax_pct:.0f}%"
                                f" | found: {len(valid_fluxes)}"
                                f" | ETA: {eta_str}"
                                f"  [{_elapsed()}]",
                                flush=True,
                            )
            except KeyboardInterrupt:
                _interrupted = True
                print(
                    f"\n[enumerate_fluxes] Interrupted — returning "
                    f"{len(valid_fluxes)} candidates found so far.  "
                    f"[{_elapsed()}]",
                    flush=True,
                )
            if verbose and not _interrupted:
                elapsed_stream = time.perf_counter() - t_stream_start
                rate_final = (
                    n_h_processed / elapsed_stream if elapsed_stream > 0 else 0.0
                )
                print(
                    f"[enumerate_fluxes]   Streaming complete:"
                    f" {n_h_processed / 1e6:.2f}M candidates processed"
                    f" | {n_h_filtered:,} passed filters"
                    f" | {len(valid_fluxes)} valid fluxes"
                    f" | avg {rate_final / 1e3:.0f}K/s"
                    f"  [{_elapsed()}]",
                    flush=True,
                )
            # Clean up streaming state
            self._stream_M_inv = None
            self._stream_tad_budget = None
        else:
            if verbose:
                print(
                    f"[enumerate_fluxes] Steps 4–5/5 — ISD completion + bound "
                    f"checking for {len(h_candidates)} candidates "
                    f"(Nmax={self.Nmax}) ...",
                    flush=True,
                )

            # Apply s_max filter to the full candidate array upfront
            n_before_smax = len(h_candidates)
            h_candidates, _ = _smax_filter(h_candidates)
            if verbose and len(h_candidates) < n_before_smax:
                print(
                    f"[enumerate_fluxes]   s_max filter: "
                    f"{n_before_smax - len(h_candidates):,} h candidates removed "
                    f"({len(h_candidates):,} remaining)  [{_elapsed()}]",
                    flush=True,
                )

            # Apply continuous tadpole pre-filter
            n_before_tad = len(h_candidates)
            h_candidates = _continuous_tadpole_filter(h_candidates)
            if verbose and len(h_candidates) < n_before_tad:
                print(
                    f"[enumerate_fluxes]   continuous tadpole filter: "
                    f"{n_before_tad - len(h_candidates):,} h candidates removed "
                    f"({len(h_candidates):,} remaining)  [{_elapsed()}]",
                    flush=True,
                )

            n_h = len(h_candidates)
            # Target ~10 progress updates; min chunk 1, max chunk 100_000.
            n_updates = 10
            chunk_size = max(1, min(chunk_size, math.ceil(n_h / n_updates)))
            n_chunks = math.ceil(n_h / chunk_size)

            # Build pipeline dict for _process_h_chunk
            _pipeline = {
                'moduli_slice': moduli_slice, 'tau_slice': tau_slice,
                'moduli_slice_np': moduli_slice_np, 'tau_slice_np': tau_slice_np,
                'evs_batch': evs_batch, 'M0_sigma': M0_sigma,
                's_vec': s_vec, 'c0_vec': c0_vec,
                'gl_params': gl_params, 'sigma': sigma,
            }

            try:
              for chunk_idx, start in enumerate(range(0, n_h, chunk_size)):
                end = min(start + chunk_size, n_h)
                h_chunk = jnp.array(h_candidates[start:end], dtype=float)

                # Core ISD processing — shared with sample_bounded_fluxes
                new_f, new_m, new_t = self._process_h_chunk(
                    h_chunk, _pipeline,
                    use_linearised_shifts=use_linearised_shifts,
                    n_isd_iters=n_isd_iters,
                    n_moduli_batches=n_moduli_batches,
                    constraints=constraints,
                    rns_key=rns_key,
                    n_sample=n_isd_per_h,
                )
                valid_fluxes.extend(new_f)
                valid_moduli.extend(new_m)
                valid_taus.extend(new_t)

                if verbose:
                    print(
                        f"[enumerate_fluxes]   Chunk {chunk_idx + 1}/{n_chunks} "
                        f"| {end}/{n_h} h-candidates processed "
                        f"| {len(valid_fluxes)} valid  [{_elapsed()}]",
                        flush=True,
                    )
            except KeyboardInterrupt:
                _interrupted = True
                print(
                    f"\n[enumerate_fluxes] Interrupted — returning "
                    f"{len(valid_fluxes)} candidates found so far.  "
                    f"[{_elapsed()}]",
                    flush=True,
                )

        if verbose and not _interrupted:
            print(
                f"[enumerate_fluxes] Done: {len(valid_fluxes)} valid flux "
                f"candidates  [{_elapsed()}]",
                flush=True,
            )

        if not refine or _interrupted:
            # Deduplicate (and optionally map to FD)
            seen: set = set()
            deduped_fluxes: list = []
            deduped_moduli: list = []
            deduped_taus: list = []
            for fv, zm, tv in zip(valid_fluxes, valid_moduli, valid_taus):
                fv, zm, tv = self._map_result_to_fd(fv, zm, tv)
                flux_key = np.round(np.asarray(fv).real).astype(np.int32).tobytes()
                if flux_key not in seen:
                    seen.add(flux_key)
                    deduped_fluxes.append(fv)
                    deduped_moduli.append(zm)
                    deduped_taus.append(tv)

            n_removed = len(valid_fluxes) - len(deduped_fluxes)
            if verbose and n_removed > 0:
                print(
                    f"[enumerate_fluxes] Removed {n_removed} duplicates, "
                    f"{len(deduped_fluxes)} unique remain  [{_elapsed()}]",
                    flush=True,
                )

            if return_moduli or _interrupted:
                return [
                    {"flux": fv, "moduli": zm, "tau": tv}
                    for fv, zm, tv in zip(deduped_fluxes, deduped_moduli, deduped_taus)
                ]
            return deduped_fluxes

        if len(valid_fluxes) == 0:
            return []

        # ------------------------------------------------------------------
        # Step 6: Newton refinement
        # ------------------------------------------------------------------
        if verbose:
            print(
                f"[enumerate_fluxes] Step 6 — Newton-refining "
                f"{len(valid_fluxes)} candidates "
                f"(tol={newton_tol}, max_iters={newton_max_iters}) ...",
                flush=True,
            )

        flux_arr = jnp.array(valid_fluxes)
        moduli_arr = jnp.array(valid_moduli, dtype=complex)
        tau_arr = jnp.array(valid_taus, dtype=complex)

        moduli_out, tau_out, residuals = self.newton_refine_batch(
            moduli_arr, tau_arr, flux_arr,
            step_size=newton_step_size,
            tol=newton_tol,
            max_iters=newton_max_iters,
        )

        # Filter: converged AND inside patch
        converged = residuals < newton_tol
        in_patch = self.in_patch_batch(moduli_out, tau_out)
        keep = converged & in_patch

        n_converged = int(jnp.sum(converged))
        n_in_patch = int(jnp.sum(converged & in_patch))

        if verbose:
            print(
                f"[enumerate_fluxes]   {n_converged}/{len(valid_fluxes)} "
                f"converged, {n_in_patch} in patch  [{_elapsed()}]",
                flush=True,
            )

        # Deduplicate — optionally map to FD first so monodromy-equivalent
        # vacua hash to the same key.
        keep_idx = np.where(np.asarray(keep))[0]
        seen_fluxes: set = set()
        results: list = []
        for idx in keep_idx:
            fv = valid_fluxes[idx]
            mv = np.asarray(moduli_out[idx])
            tv = complex(tau_out[idx])
            fv, mv, tv = self._map_result_to_fd(fv, mv, tv)
            flux_key = np.round(fv).astype(np.int32).tobytes()
            if flux_key in seen_fluxes:
                continue
            seen_fluxes.add(flux_key)
            results.append({
                "flux": fv,
                "moduli": mv,
                "tau": tv,
                "residual": float(residuals[idx]),
            })

        if verbose:
            print(
                f"[enumerate_fluxes] {len(results)} unique vacua after "
                f"deduplication  [{_elapsed()}]",
                flush=True,
            )

        return results

    # =========================================================================
    #  Stochastic h-vector sampling
    # =========================================================================

    def _sample_h_vectors(
        self,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        r"""

        **Description:**
        Samples integer NSNS-flux vectors :math:`h = (h_1, h_2)` uniformly
        at random from the combined ellipsoidal bounding region defined by
        :func:`compute_bounding_box`.

        Uses rejection sampling: draws components uniformly from the enclosing
        integer box, then keeps only vectors satisfying all three simultaneous
        norm constraints

        .. math::
            \|h_1\|^2 \leq h_{1,\rm box}^2\,,\quad
            \|h_2\|^2 \leq h_{2,\rm box}^2\,,\quad
            \|h\|^2   \leq h_{\rm box}^2\,.

        Oversamples by a factor of ~3 per iteration to reduce the expected
        number of loop iterations.

        Args:
            n_samples (int): Number of valid :math:`h` vectors to return.
            rng (np.random.Generator, optional): NumPy random number generator.
                Defaults to ``np.random.default_rng()``.

        Returns:
            np.ndarray: Integer array of shape ``(n_samples, n_fluxes)``.

        Raises:
            RuntimeError: If :func:`compute_bounding_box` has not been called.
        """
        if self._h1_box is None:
            raise RuntimeError(
                "Call compute_bounding_box() before _sample_h_vectors()."
            )
        if rng is None:
            rng = np.random.default_rng()

        dim = self.dimension_H3
        h1_max = int(np.ceil(self._h1_box))
        h2_max = int(np.ceil(self._h2_box))
        h1_sq_max = self._h1_box ** 2
        h2_sq_max = self._h2_box ** 2
        h_sq_max = self._h_box ** 2

        results: list = []
        n_remaining = n_samples

        while n_remaining > 0:
            # Oversample by ~3x to reduce iterations
            n_try = min(n_remaining * 3 + 1024, 5_000_000)

            # Keep as int64 — norm filtering works on integers directly.
            h1 = rng.integers(-h1_max, h1_max + 1, size=(n_try, dim))
            h2 = rng.integers(-h2_max, h2_max + 1, size=(n_try, dim))

            # Filter by individual sub-vector norms
            h1_norms = np.einsum('ij,ij->i', h1, h1)
            h2_norms = np.einsum('ij,ij->i', h2, h2)
            mask = (h1_norms <= h1_sq_max) & (h2_norms <= h2_sq_max)
            h1, h2 = h1[mask], h2[mask]

            if len(h1) == 0:
                continue

            # Combine and filter by total norm
            h = np.concatenate([h1, h2], axis=-1)
            h_norms = np.einsum('ij,ij->i', h, h)
            h = h[h_norms <= h_sq_max]

            if len(h) > 0:
                take = min(len(h), n_remaining)
                results.append(h[:take])
                n_remaining -= take

        return np.concatenate(results, axis=0)

    def _sample_h_vectors_importance(
        self,
        n_samples: int,
        M_inv_all: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        r"""
        **Description:**
        Samples integer NSNS-flux vectors :math:`h` from a **Gaussian prior
        weighted by the ISD matrix**, concentrating samples where the continuous
        tadpole is small.

        .. admonition:: Details
            :class: dropdown

            Instead of sampling uniformly from the bounding box (which wastes
            most samples at large :math:`\|h\|`), we draw from

            .. math::
                h \sim \mathcal{N}\bigl(0,\;\sigma^2\,M(z)\bigr)

            where :math:`M` is the ISD matrix at a representative moduli point
            and :math:`\sigma^2 = N_{\max} / (\mathrm{tr}(M) \cdot s_{\min})`
            is chosen so that most samples satisfy the tadpole constraint
            :math:`s_{\min}\,h^T M^{-1} h \leq N_{\max}`.

            The :math:`M`-weighting ensures samples are concentrated along
            the eigendirections where :math:`M^{-1}` has *small* eigenvalues
            (low tadpole cost), matching the physical distribution of flux
            vacua.  After drawing continuous samples, they are rounded to
            the nearest integer lattice point.

        Args:
            n_samples (int): Number of valid h-vectors to return.
            M_inv_all (np.ndarray): Inverse ISD matrices at sample moduli points,
                shape ``(n_mod, n_fl, n_fl)``.
            rng (np.random.Generator, optional): Random generator.

        Returns:
            np.ndarray: Integer array of shape ``(n_samples, n_fluxes)``.
        """
        if rng is None:
            rng = np.random.default_rng()

        n_fl = self.n_fluxes

        # Use the best-conditioned M_inv to build the sampling distribution.
        # M = (M_inv)^{-1} → eigendecompose M_inv, invert eigenvalues.
        best_j = 0
        best_cond = np.inf
        for j in range(len(M_inv_all)):
            eigs = np.abs(np.linalg.eigvalsh(M_inv_all[j]))
            cond = eigs.max() / max(eigs.min(), 1e-30)
            if cond < best_cond:
                best_cond = cond
                best_j = j

        # M_inv eigendecomposition: M_inv = V diag(λ_inv) V^T
        # → M = V diag(1/λ_inv) V^T
        eigvals_inv, eigvecs = np.linalg.eigh(M_inv_all[best_j])
        eigvals_inv = np.abs(eigvals_inv)
        eigvals_M = 1.0 / np.maximum(eigvals_inv, 1e-30)

        # σ² chosen so most samples satisfy the tadpole:
        # For h ~ N(0, σ²M), E[s_min h^T M^{-1} h] = s_min σ² tr(I) = s_min σ² d
        # Want this ≈ Nmax → σ² = Nmax / (d × s_min)
        s_min = float(self.dil_min)
        sigma_sq = float(self.Nmax) / (n_fl * s_min)

        # Sampling scale per eigendirection: sqrt(σ² × λ_M_i)
        scales = np.sqrt(sigma_sq * eigvals_M)

        results: list = []
        n_remaining = n_samples

        while n_remaining > 0:
            n_try = min(n_remaining * 3 + 2048, 5_000_000)

            # Draw z ~ N(0, I), then h = V @ diag(scales) @ z
            z_std = rng.standard_normal((n_try, n_fl))
            h_continuous = (z_std * scales[None, :]) @ eigvecs.T
            h_int = np.round(h_continuous).astype(int)

            # Remove zero vectors
            h_norms = np.einsum('ij,ij->i', h_int, h_int)
            h_int = h_int[h_norms > 0]

            if len(h_int) > 0:
                take = min(len(h_int), n_remaining)
                results.append(h_int[:take])
                n_remaining -= take

        return np.concatenate(results, axis=0) if results else np.zeros((0, n_fl), dtype=int)

    # =========================================================================
    #  Stochastic flux search
    # =========================================================================

    def sample_bounded_fluxes(
        self,
        n_target: int = 1000,
        n_batch: int = 50_000,
        n_sample: int = 500,
        n_mod: int = 20,
        max_batches: int = 100,
        verbose: bool = True,
        rns_key: Any | None = None,
        seed: int | None = None,
        refine: bool = False,
        return_moduli: bool = False,
        newton_tol: float = 1e-10,
        newton_max_iters: int = 100,
        newton_step_size: float = 1.0,
        use_linearised_shifts: bool = False,
        n_isd_iters: int = 5,
        n_moduli_batches: int = 1,
        moduli_regions: List[Tuple[float, float]] | None = None,
        constraints: Callable | None = None,
    ) -> list:
        r"""
        **Description:**
        Stochastic flux search guided by the eigenvalue bounds of
        arXiv:2501.03984.

        Unlike :func:`enumerate_fluxes`, which exhaustively enumerates all
        integer :math:`h` vectors inside the bounding box, this method
        **randomly samples** :math:`h` vectors from within the box and
        ISD-completes them.  It scales to arbitrarily large :math:`N_{\max}`
        and higher :math:`h^{1,2}` where full enumeration is infeasible.

        **Algorithm:**

        1. Sample :math:`n_{\rm sample}` moduli points and compute global
           eigenvalue bounds (identical to :func:`enumerate_fluxes` Step 1).
        2. Loop over batches of :math:`n_{\rm batch}` randomly sampled
           :math:`h` vectors (drawn uniformly from the bounding ellipsoid).
        3. For each batch, ISD-complete at :math:`n_{\rm mod}` moduli points
           and check all bounds via the JIT-compiled kernel.
        4. Accumulate valid flux vectors until :math:`n_{\rm target}` are
           found or :math:`\text{max\_batches}` are exhausted.
        5. (Optional, ``refine=True``) Newton-refine each batch immediately,
           accumulate converged+in-patch vacua, and stop once
           :math:`n_{\rm target}` actual vacua have been found.

        .. note::
            Candidates returned with ``refine=False`` are **not** exact SUSY
            vacua: :math:`f` is the continuous ISD completion at a sampled
            modulus, rounded to the nearest integer.  :math:`D_I W \neq 0` at
            the returned modulus is therefore expected.  Use ``refine=True`` to
            Newton-solve :math:`D_I W = 0` and obtain actual vacua.

        Args:
            n_target (int): Target number of results to collect.
                With ``refine=False`` this counts raw flux candidates; with
                ``refine=True`` it counts converged, in-patch SUSY vacua.
                The search stops early once this many are found.
                Defaults to ``1000``.
            n_batch (int): Number of :math:`h` vectors to sample per batch.
                Larger batches amortise JIT overhead but use more memory.
                Defaults to ``50_000``.
            n_sample (int): Number of moduli points for computing global
                eigenvalue bounds.  Defaults to ``500``.
            n_mod (int): Number of moduli points for ISD completion per batch.
                Defaults to ``20``.
            max_batches (int): Maximum number of sampling batches before
                stopping.  Defaults to ``100``.
            verbose (bool): Print progress with timing.  Defaults to ``True``.
            rns_key (Any, optional): JAX PRNG key for moduli/tau sampling.
            seed (int, optional): NumPy seed for h-vector sampling.
                Defaults to ``None`` (non-reproducible).
            refine (bool): If ``True``, Newton-refine each batch's candidates
                immediately and count only converged+in-patch solutions toward
                ``n_target``.  Guarantees at least ``n_target`` true
                :math:`D_I W = 0` solutions (or as many as can be found within
                ``max_batches``).  Defaults to ``False``.
            return_moduli (bool): If ``True``, return a ``List[dict]`` with
                keys ``"flux"``, ``"moduli"``, ``"tau"`` even when
                ``refine=False``.  Defaults to ``False``.
            newton_tol (float): Newton convergence tolerance.
                Defaults to ``1e-10``.
            newton_max_iters (int): Maximum Newton iterations.
                Defaults to ``100``.
            newton_step_size (float): Newton step size.
                Defaults to ``1.0`` (full Newton steps, quadratic convergence).
            use_linearised_shifts (bool): If ``True``, replaces the
                fixed-moduli ISD completion with iterated
                :func:`linearised_shifts` calls (see
                :func:`isd_refine_batch`) with flag-based early stopping.
                Requires :attr:`model` to be a
                :class:`~jaxvacua.flux_vacua_finder.FluxVacuaFinder`
                instance.  Defaults to ``False``.
            n_isd_iters (int): Maximum :func:`linearised_shifts`
                iterations when ``use_linearised_shifts=True``.
                Defaults to ``5``.
            moduli_regions (List[Tuple[float, float]], optional): List of
                ``(lo, hi)`` intervals for Im(z).  The Cartesian product
                across all :math:`h^{1,2}` moduli dimensions is used to
                build the ISD starting-point set (same logic as
                :func:`enumerate_fluxes`).  Defaults to ``None``
                (use the sampler's full range).
            constraints (Callable, optional): Extra constraint function
                ``(moduli, tau, flux) → bool`` passed to
                :func:`linearised_shifts` when
                ``use_linearised_shifts=True``.  Defaults to ``None``.

        Returns:
            List[np.ndarray]: When ``refine=False`` and
            ``return_moduli=False``, valid flux vectors ``[f | h]`` of
            length ``2 * n_fluxes``.

            List[dict]: When ``refine=True`` or ``return_moduli=True``,
            each entry is a dict with keys ``"flux"``, ``"moduli"``,
            ``"tau"`` (and ``"residual"`` when ``refine=True``).

        Raises:
            ValueError: If no :attr:`sampler` has been provided.
        """
        if self.sampler is None:
            raise ValueError(
                "No sampler provided.  Pass sampler=<data_sampler> to "
                "bounded_fluxes() to use sample_bounded_fluxes()."
            )

        if use_linearised_shifts and not hasattr(self.model, 'linearised_shifts_H'):
            raise ValueError(
                "use_linearised_shifts=True requires the model to expose "
                "'linearised_shifts_H'.  Use a FluxVacuaFinder instance."
            )

        # Tighten dil_min to the sampler's actual lower dilaton bound.
        # The default dil_min = sqrt(3)/2 is the SL(2,Z) fundamental domain
        # floor, but when the sampler restricts s >= s_lower > sqrt(3)/2
        # (e.g. dilaton_bounds=[2,10]), using the larger value gives smaller
        # (tighter) bounding boxes and higher candidate yield.
        if hasattr(self.sampler, 's_lower') and self.sampler.s_lower > self.dil_min:
            self.dil_min = float(self.sampler.s_lower)

        t0 = time.perf_counter()

        def _elapsed() -> str:
            """Return formatted elapsed time since t0."""
            s = time.perf_counter() - t0
            if s < 120:
                return f"{s:.1f}s"
            elif s < 3600:
                return f"{s / 60:.1f}m"
            else:
                h = int(s // 3600)
                m = int((s % 3600) // 60)
                return f"{h}h {m}m"

        rng = np.random.default_rng(seed)

        # ------------------------------------------------------------------
        # Steps 1 & 2: Shared pipeline preparation (eigenvalue bounds,
        # moduli slice, JIT parameters, pre-filters).
        # Delegates to _prepare_isd_pipeline to avoid code duplication
        # with enumerate_fluxes.
        # ------------------------------------------------------------------
        P = self._prepare_isd_pipeline(
            n_sample=n_sample,
            n_isd_per_h=n_mod,
            moduli_regions=moduli_regions,
            use_linearised_shifts=use_linearised_shifts,
            n_isd_iters=n_isd_iters,
            constraints=constraints,
            rns_key=rns_key,
            verbose=verbose,
            label="sample_bounded",
        )

        # Unpack shared objects
        moduli_slice    = P['moduli_slice']
        tau_slice       = P['tau_slice']
        moduli_slice_np = P['moduli_slice_np']
        tau_slice_np    = P['tau_slice_np']
        evs_batch       = P['evs_batch']
        M0_sigma        = P['M0_sigma']
        s_vec           = P['s_vec']
        c0_vec          = P['c0_vec']
        gl_params       = P['gl_params']
        sigma           = P['sigma']
        _cont_tadpole_filter_sb = P['_continuous_tadpole_filter']
        _elapsed        = P['_elapsed']

        if verbose:
            print(
                f"[sample_bounded] Stochastic search: "
                f"target={n_target}, batch_size={n_batch}, "
                f"n_mod={len(moduli_slice)}, "
                f"max_batches={max_batches}  [{_elapsed()}]",
                flush=True,
            )

        # ------------------------------------------------------------------
        # Step 3: Stochastic sampling loop
        # When refine=False: stop after collecting n_target candidates.
        # When refine=True:  refine each batch immediately and stop after
        #                    collecting n_target converged+in-patch vacua.
        # ------------------------------------------------------------------
        valid_fluxes: List[np.ndarray] = []
        valid_moduli: List[np.ndarray] = []
        valid_taus: List[complex] = []
        seen_fluxes: set = set()
        refined_results: list = []

        n_tried = 0
        n_pass_filter = 0
        _interrupted = False

        try:
            for batch_idx in range(max_batches):
                n_found = len(refined_results) if refine else len(valid_fluxes)
                if n_found >= n_target:
                    break

                # Importance sampling: draw h from tadpole ellipsoid instead
                # of the full bounding box.  Falls back to uniform box sampling
                # if M_inv data is not available.
                M_inv_data = P.get('M_inv_all_np', None)
                if M_inv_data is not None and len(M_inv_data) > 0:
                    h_chunk_np = self._sample_h_vectors_importance(
                        n_batch, M_inv_data, rng=rng)
                else:
                    h_chunk_np = self._sample_h_vectors(n_batch, rng=rng)
                n_tried += len(h_chunk_np)
                # Apply continuous tadpole pre-filter (cheap double-check:
                # importance sampling targets the right region but rounding
                # to integers can push some vectors outside)
                h_chunk_np = _cont_tadpole_filter_sb(h_chunk_np)
                n_pass_filter += len(h_chunk_np)
                if len(h_chunk_np) == 0:
                    if verbose:
                        print(
                            f"[sample_bounded]   Batch {batch_idx+1}: "
                            f"0/{n_batch} pass tadpole filter  [{_elapsed()}]",
                            flush=True,
                        )
                    continue
                h_chunk = jnp.array(h_chunk_np, dtype=float)

                new_batch_fluxes: List[np.ndarray] = []
                new_batch_moduli: List[np.ndarray] = []
                new_batch_taus: List[complex] = []

                # Core ISD processing — shared with enumerate_fluxes
                new_f, new_m, new_t = self._process_h_chunk(
                    h_chunk, P,
                    use_linearised_shifts=use_linearised_shifts,
                    n_isd_iters=n_isd_iters,
                    n_moduli_batches=n_moduli_batches,
                    constraints=constraints,
                    rns_key=rns_key,
                    n_sample=n_mod,
                )

                # Dedup and accumulate (optionally map to FD first)
                for fv, mv, tv in zip(new_f, new_m, new_t):
                    fv, mv, tv = self._map_result_to_fd(fv, mv, tv)
                    flux_key = np.round(np.asarray(fv).real).astype(np.int32).tobytes()
                    if flux_key not in seen_fluxes:
                        seen_fluxes.add(flux_key)
                        valid_fluxes.append(fv)
                        valid_moduli.append(mv)
                        valid_taus.append(complex(tv))
                        if refine:
                            new_batch_fluxes.append(fv)
                            new_batch_moduli.append(mv)
                            new_batch_taus.append(complex(tv))

                # When refine=True, Newton-refine the new candidates from this
                # batch immediately so the stopping criterion reflects actual vacua.
                n_batch_conv = 0
                n_batch_patch = 0
                if refine and new_batch_fluxes:
                    flux_arr = jnp.array(new_batch_fluxes)
                    moduli_arr = jnp.array(new_batch_moduli, dtype=complex)
                    tau_arr = jnp.array(new_batch_taus, dtype=complex)

                    moduli_out, tau_out, residuals = self.newton_refine_batch(
                        moduli_arr, tau_arr, flux_arr,
                        step_size=newton_step_size,
                        tol=newton_tol,
                        max_iters=newton_max_iters,
                    )

                    converged_np = np.asarray(residuals < newton_tol)
                    in_patch_np = np.asarray(self.in_patch_batch(moduli_out, tau_out))
                    residuals_np = np.asarray(residuals)
                    n_batch_conv = int(np.sum(converged_np))
                    n_batch_patch = int(np.sum(converged_np & in_patch_np))
                    for idx in np.where(converged_np & in_patch_np)[0]:
                        refined_results.append({
                            "flux": new_batch_fluxes[idx],
                            "moduli": np.asarray(moduli_out[idx]),
                            "tau": complex(tau_out[idx]),
                            "residual": float(residuals_np[idx]),
                        })

                if verbose:
                    n_found = len(refined_results) if refine else len(valid_fluxes)
                    yield_pct = 100.0 * n_found / n_tried if n_tried > 0 else 0.0
                    if refine and new_batch_fluxes:
                        n_cands = len(new_batch_fluxes)
                        print(
                            f"[sample_bounded]   Batch {batch_idx + 1}/{max_batches} "
                            f"| found {n_found}/{n_target} "
                            f"| candidates: {n_cands}, converged: {n_batch_conv}/{n_cands}, "
                            f"in-patch: {n_batch_patch}/{n_batch_conv if n_batch_conv else 1}"
                            f"  [{_elapsed()}]",
                            flush=True,
                        )
                    else:
                        print(
                            f"[sample_bounded]   Batch {batch_idx + 1}/{max_batches} "
                            f"| found {n_found}/{n_target} "
                            f"| {n_tried} tried, yield {yield_pct:.2f}%"
                            f"  [{_elapsed()}]",
                            flush=True,
                        )
        except KeyboardInterrupt:
            _interrupted = True
            n_found = len(refined_results) if refine else len(valid_fluxes)
            print(
                f"\n[sample_bounded] Interrupted — returning "
                f"{n_found} results found so far.  [{_elapsed()}]",
                flush=True,
            )

        if verbose and not _interrupted:
            n_final = len(refined_results) if refine else len(valid_fluxes)
            yield_pct = 100.0 * n_final / n_tried if n_tried > 0 else 0.0
            if refine:
                print(
                    f"[sample_bounded] Done: {n_final} refined vacua from "
                    f"{len(valid_fluxes)} candidates "
                    f"({n_tried} samples, yield {yield_pct:.2f}%)  [{_elapsed()}]",
                    flush=True,
                )
            else:
                print(
                    f"[sample_bounded] Done: {n_final} unique flux "
                    f"candidates from {n_tried} samples "
                    f"(yield {yield_pct:.2f}%)  [{_elapsed()}]",
                    flush=True,
                )

        if not refine or _interrupted:
            if return_moduli or _interrupted:
                return [
                    {"flux": fv, "moduli": zm, "tau": tv}
                    for fv, zm, tv in zip(valid_fluxes, valid_moduli, valid_taus)
                ]
            return valid_fluxes

        return refined_results

    # =========================================================================
    #  Individual bound checks
    # =========================================================================

    def bound_h1_local(self) -> Tuple[bool, str]:
        r"""
        **Description:**
        Checks the local bound on :math:`\|h_1\|^2`:

        .. math:: s\,\tilde\mu_{\min}\,\|h_1\|^2 \leq N_{\rm flux}\,.

        Returns:
            Tuple[bool, str]: ``(satisfied, "h1 local")``.
        """
        return bool(self.s * self.tilde_mu_min * self.h1norm <= self.Nflux), "h1 local"

    def bound_h1_global(self) -> Tuple[bool, str]:
        r"""
        **Description:**
        Checks the global bound on :math:`\|h_1\|^2`:

        .. math:: s_{\min}\,\tilde\mu_{\min}^{\rm gl}\,\|h_1\|^2 \leq N_{\max}\,.

        Returns:
            Tuple[bool, str]: ``(satisfied, "h1 global")``.
        """
        return (
            bool(self.dil_min * self.tilde_mu_min_gl * self.h1norm <= self.Nmax),
            "h1 global",
        )

    def bound_h2_local(self) -> Tuple[bool, str]:
        r"""
        **Description:**
        Checks the local bound on :math:`\|h_2\|^2`:

        .. math:: s\,\mu_{\min}\,\|h_2\|^2 \leq N_{\rm flux}\,.

        Returns:
            Tuple[bool, str]: ``(satisfied, "h2 local")``.
        """
        return bool(self.s * self.mu_min * self.h2norm <= self.Nflux), "h2 local"

    def bound_h2_global(self) -> Tuple[bool, str]:
        r"""
        **Description:**
        Checks the global bound on :math:`\|h_2\|^2`:

        .. math:: s_{\min}\,\mu_{\min}^{\rm gl}\,\|h_2\|^2 \leq N_{\max}\,.

        Returns:
            Tuple[bool, str]: ``(satisfied, "h2 global")``.
        """
        return (
            bool(self.dil_min * self.mu_min_gl * self.h2norm <= self.Nmax),
            "h2 global",
        )

    def bound_f1_local(self) -> Tuple[Tuple[bool, ...], str]:
        r"""

        **Description:**
        Checks the four local bounds on the :math:`f_1` sector derived from
        the ISD condition (arXiv:2501.03984).  Uses
        :math:`\tilde f_1 = f_1 - c_0 h_1` and
        :math:`\tilde\mu_{\min/\max}` — the extreme eigenvalues of
        :math:`\operatorname{Im}(\mathcal{N}^{-1})`.

        .. admonition:: Details
            :class: dropdown

            The four local inequalities are:

            .. math::
                \tilde\mu_{\min}\,\|\tilde f_1\|^2 &\leq s\,N_{\rm flux}
                    \quad\text{(b1)}\,,\\
                \tilde\mu_{\min}({\|f_1\|^2 + c_0^2\|h_1\|^2})
                    - 2|c_0|\tilde\mu_{\max}\sqrt{\|f_1\|^2\|h_1\|^2}
                    &\leq s\,N_{\rm flux}
                    \quad\text{(b2)}\,,\\
                \tilde\mu_{\min}({\|f_1\|^2 + (c_0^2+\tfrac{3}{4})\|h_1\|^2})
                    - 2|c_0|\tilde\mu_{\max}\sqrt{\|f_1\|^2\|h_1\|^2}
                    &\leq s\,N_{\rm flux}
                    \quad\text{(b5)}\,,\\
                \tilde\mu_{\min}({\|\tilde f_1\|^2
                    + (c_0^2+\tfrac{3}{4})\|h_1\|^2})
                    &\leq s\,N_{\rm flux}
                    \quad\text{(b6)}\,.

        Returns:
            Tuple[Tuple[bool, bool, bool, bool], str]:
            Four bounds ``(b1, b2, b5, b6)`` and label ``"f1 local"``.
        """
        sqrt_f1h1 = float(jnp.sqrt(self.f1norm * self.h1norm))
        b1 = bool(self.tilde_mu_min * self.f1tilde_norm <= self.s * self.Nflux)
        b2 = bool(
            self.tilde_mu_min * (self.f1norm + self.c0 ** 2 * self.h1norm)
            - 2.0 * abs(self.c0) * self.tilde_mu_max * sqrt_f1h1
            <= self.s * self.Nflux
        )
        b5 = bool(
            self.tilde_mu_min * (self.f1norm + (self.c0 ** 2 + 0.75) * self.h1norm)
            - 2.0 * abs(self.c0) * self.tilde_mu_max * sqrt_f1h1
            <= self.s * self.Nflux
        )
        b6 = bool(
            self.tilde_mu_min * (self.f1tilde_norm + (self.c0 ** 2 + 0.75) * self.h1norm)
            <= self.s * self.Nflux
        )
        return (b1, b2, b5, b6), "f1 local"

    def bound_f1_global(self) -> Tuple[Tuple[bool, bool], str]:
        r"""

        **Description:**
        Checks the two global bounds on the :math:`f_1` sector using the
        global eigenvalue extrema :math:`\tilde\mu_{\min/\max}^{\rm gl}`.

        .. admonition:: Details
            :class: dropdown

            The two global inequalities are:

            .. math::
                \tilde\mu_{\min}^{\rm gl}
                    (\|f_1\|^2 + \tfrac{3}{4}\|h_1\|^2)
                    - \tilde\mu_{\max}^{\rm gl}\sqrt{\|f_1\|^2\|h_1\|^2}
                    &\leq s_{\max}\,N_{\max}
                    \quad\text{(b3)}\,,\\
                \tilde\mu_{\min}^{\rm gl}
                    (\|\tilde f_1\|^2 + \tfrac{3}{4}\|h_1\|^2)
                    &\leq s_{\max}\,N_{\max}
                    \quad\text{(b4)}\,.

        Returns:
            Tuple[Tuple[bool, bool], str]: Two bounds ``(b3, b4)`` and
            label ``"f1 global"``.
        """
        sqrt_f1h1 = float(jnp.sqrt(self.f1norm * self.h1norm))
        b3 = bool(
            self.tilde_mu_min_gl * (self.f1norm + 0.75 * self.h1norm)
            - self.tilde_mu_max_gl * sqrt_f1h1
            <= self.dil_max * self.Nmax
        )
        b4 = bool(
            self.tilde_mu_min_gl * (self.f1tilde_norm + 0.75 * self.h1norm)
            <= self.dil_max * self.Nmax
        )
        return (b3, b4), "f1 global"

    def bound_f2_local(self) -> Tuple[Tuple[bool, ...], str]:
        r"""

        **Description:**
        Checks the four local bounds on the :math:`f_2` sector derived from
        the ISD condition (arXiv:2501.03984).  Uses
        :math:`\tilde f_2 = f_2 - c_0 h_2` and
        :math:`\mu_{\min/\max}` — the extreme eigenvalues of
        :math:`-\operatorname{Im}(\mathcal{N})`.

        .. admonition:: Details
            :class: dropdown

            The four local inequalities are:

            .. math::
                \mu_{\min}\,\|\tilde f_2\|^2 &\leq s\,N_{\rm flux}
                    \quad\text{(b1)}\,,\\
                \mu_{\min}(\|f_2\|^2 + c_0^2\|h_2\|^2)
                    - 2|c_0|\mu_{\max}\sqrt{\|f_2\|^2\|h_2\|^2}
                    &\leq s\,N_{\rm flux}
                    \quad\text{(b2)}\,,\\
                \mu_{\min}(\|f_2\|^2 + (c_0^2+\tfrac{3}{4})\|h_2\|^2)
                    - \mu_{\max}|c_0|\sqrt{\|f_2\|^2\|h_2\|^2}
                    &\leq s\,N_{\rm flux}
                    \quad\text{(b5)}\,,\\
                \mu_{\min}(\|\tilde f_2\|^2
                    + (c_0^2+\tfrac{3}{4})\|h_2\|^2)
                    &\leq s\,N_{\rm flux}
                    \quad\text{(b6)}\,.

        Returns:
            Tuple[Tuple[bool, bool, bool, bool], str]:
            Four bounds ``(b1, b2, b5, b6)`` and label ``"f2 local"``.
        """
        sqrt_f2h2 = float(jnp.sqrt(self.f2norm * self.h2norm))
        b1 = bool(self.mu_min * self.f2tilde_norm <= self.s * self.Nflux)
        b2 = bool(
            self.mu_min * (self.f2norm + self.c0 ** 2 * self.h2norm)
            - 2.0 * abs(self.c0) * self.mu_max * sqrt_f2h2
            <= self.s * self.Nflux
        )
        b5 = bool(
            self.mu_min * (self.f2norm + (self.c0 ** 2 + 0.75) * self.h2norm)
            - self.mu_max * abs(self.c0) * sqrt_f2h2
            <= self.s * self.Nflux
        )
        b6 = bool(
            self.mu_min * (self.f2tilde_norm + (self.c0 ** 2 + 0.75) * self.h2norm)
            <= self.s * self.Nflux
        )
        return (b1, b2, b5, b6), "f2 local"

    def bound_f2_global(self) -> Tuple[Tuple[bool, bool], str]:
        r"""

        **Description:**
        Checks the two global bounds on the :math:`f_2` sector using the
        global eigenvalue extrema :math:`\mu_{\min/\max}^{\rm gl}`.

        .. admonition:: Details
            :class: dropdown

            The two global inequalities are:

            .. math::
                \mu_{\min}^{\rm gl}(\|f_2\|^2 + \tfrac{3}{4}\|h_2\|^2)
                    - \mu_{\max}^{\rm gl}\sqrt{\|f_2\|^2\|h_2\|^2}
                    &\leq s_{\max}\,N_{\max}
                    \quad\text{(b3)}\,,\\
                \mu_{\min}^{\rm gl}(\|\tilde f_2\|^2
                    + \tfrac{3}{4}\|h_2\|^2)
                    &\leq s_{\max}\,N_{\max}
                    \quad\text{(b4)}\,.

        Returns:
            Tuple[Tuple[bool, bool], str]: Two bounds ``(b3, b4)`` and
            label ``"f2 global"``.
        """
        sqrt_f2h2 = float(jnp.sqrt(self.f2norm * self.h2norm))
        b3 = bool(
            self.mu_min_gl * (self.f2norm + 0.75 * self.h2norm)
            - self.mu_max_gl * sqrt_f2h2
            <= self.dil_max * self.Nmax
        )
        b4 = bool(
            self.mu_min_gl * (self.f2tilde_norm + 0.75 * self.h2norm)
            <= self.dil_max * self.Nmax
        )
        return (b3, b4), "f2 global"

    def bound_s_local(self) -> Tuple[Tuple[bool, bool, bool], str]:
        r"""
        **Description:**
        Checks the local bounds on :math:`s = \operatorname{Im}(\tau)`:

        .. math::
            s_{\min} \leq s \leq \lambda_{\max} N_{\rm flux}\,,\quad
            s \leq \frac{\lambda_{\max} N_{\rm flux}}{\|h\|^2}
                  + \frac{\|h\|^2}{4\,\lambda_{\max}}\,.

        Returns:
            Tuple[Tuple[bool, bool, bool], str]:
            Three bounds and label ``"s local"``.
        """
        b1 = bool(self.s >= self.dil_min)
        b2 = bool(self.s <= self.lambda_max * self.Nflux)
        b3 = (
            bool(
                self.s
                <= self.lambda_max * self.Nflux / self.hnorm
                + self.hnorm / (4.0 * self.lambda_max)
            )
            if self.hnorm > 0
            else True
        )
        return (b1, b2, b3), "s local"

    def bound_s_global(self) -> Tuple[Tuple[bool, bool], str]:
        r"""
        **Description:**
        Checks the global bounds on :math:`s`.

        Returns:
            Tuple[Tuple[bool, bool], str]: Two bounds and label ``"s global"``.
        """
        b1 = bool(self.s >= self.dil_min)
        b2 = bool(self.s <= self.lambda_max_gl * self.Nmax)
        return (b1, b2), "s global"

    def bound_h_local(self) -> Tuple[bool, str]:
        r"""
        **Description:**
        Checks the local bound on :math:`\|h\|^2`:

        .. math:: \|h\|^2 \leq \frac{N_{\rm flux}\,\lambda_{\max}}{s}\,.

        Returns:
            Tuple[bool, str]: ``(satisfied, "h local")``.
        """
        return bool(self.hnorm <= self.Nflux * self.lambda_max / self.s), "h local"

    def bound_h_global(self) -> Tuple[bool, str]:
        r"""
        **Description:**
        Checks the global bound on :math:`\|h\|^2`:

        .. math:: \|h\|^2 \leq \frac{N_{\max}\,\lambda_{\max}^{\rm gl}}{s_{\min}}\,.

        Returns:
            Tuple[bool, str]: ``(satisfied, "h global")``.
        """
        return (
            bool(self.hnorm <= self.Nmax * self.lambda_max_gl / self.dil_min),
            "h global",
        )

    def bound_f_local(self) -> Tuple[Tuple[bool, bool], str]:
        r"""
        **Description:**
        Checks the local bounds on :math:`\|f\|^2`:

        .. math::
            \frac{s\,N_{\rm flux}}{\lambda_{\max}} \leq \|f\|^2
            \leq \frac{\lambda_{\max}^2 N_{\rm flux}^2}{\|h\|^2}
                 \Bigl(1 + \frac{c_0^2}{s^2}\Bigr)\,.

        Returns:
            Tuple[Tuple[bool, bool], str]: Two bounds and label ``"f local"``.
        """
        b2 = bool(self.fnorm >= self.s * self.Nflux / self.lambda_max)
        b3 = (
            bool(
                self.fnorm
                <= self.lambda_max ** 2 * self.Nflux ** 2 / self.hnorm
                * (1.0 + self.c0 ** 2 / self.s ** 2)
            )
            if self.hnorm > 0
            else True
        )
        return (b2, b3), "f local"

    def bound_f_global(self) -> Tuple[Tuple[bool, bool, bool], str]:
        r"""
        **Description:**
        Checks the global bounds on :math:`\|f\|^2`.

        Returns:
            Tuple[Tuple[bool, bool, bool], str]:
            Three bounds and label ``"f global"``.
        """
        b1 = bool(self.fnorm >= self.dil_min * self.Nmax / self.lambda_max_gl)
        b3 = bool(self.fnorm <= self.lambda_max_gl ** 2 * self.Nmax ** 2 * 4.0 / 3.0)
        b4 = (
            bool(
                self.fnorm
                <= self.lambda_max_gl ** 2 * self.Nflux ** 2 / self.hnorm
                + self.hnorm / 4.0
            )
            if self.hnorm > 0
            else True
        )
        return (b1, b3, b4), "f global"

    # =========================================================================
    #  Compound bound checks
    # =========================================================================

    # Cached list of bound_* method names (populated once on first use).
    _bound_method_names: List[str] | None = None

    @classmethod
    def _get_bound_method_names(cls) -> List[str]:
        """
        **Description:**
        Return cached sorted list of bound_* method names on this class.
        """
        if cls._bound_method_names is None:
            cls._bound_method_names = sorted(
                name for name in dir(cls)
                if name.startswith("bound_") and callable(getattr(cls, name))
            )
        return cls._bound_method_names

    def check_bounds(
        self,
        moduli: Array,
        tau: complex,
        flux: Array,
    ) -> List[Tuple]:
        r"""
        **Description:**
        Checks all eigenvalue bounds for a given flux configuration.

        Calls :func:`update_local` to refresh the local state and then
        evaluates every ``bound_*`` method.

        Args:
            moduli (Array): Complex structure moduli, shape ``(h^{1,2},)``.
            tau (complex): Axio-dilaton :math:`\tau = c_0 + \mathrm{i}\,s`.
            flux (Array): Full flux vector ``[f \mid h]`` of length
                ``2 * n_fluxes``.

        Returns:
            List[Tuple]: One entry per ``bound_*`` method of the form
            ``(result, label)`` where *result* is a ``bool`` or a tuple of
            bools.
        """
        self.update_local(moduli, tau, flux)
        return [getattr(self, name)() for name in self._get_bound_method_names()]

    def check_bounds_flat(self) -> Tuple[bool, List[Tuple]]:
        r"""
        **Description:**
        Evaluates all ``bound_*`` methods using the *current* local state
        (set by a prior :func:`update_local` call) and returns an aggregate
        pass/fail flag alongside the detailed results.

        Returns:
            Tuple[bool, List[Tuple]]: ``(all_pass, results)`` where
            *all_pass* is ``True`` iff every individual bound is satisfied.
        """
        results = [getattr(self, name)() for name in self._get_bound_method_names()]
        all_pass = True
        for result, _ in results:
            if isinstance(result, tuple):
                if not all(result):
                    all_pass = False
                    break
            elif not result:
                all_pass = False
                break
        return all_pass, results

    # =========================================================================
    #  Flux utility helpers
    # =========================================================================

    def get_nflux(self, flux: Array) -> float:
        r"""
        **Description:**
        Computes the D3-tadpole charge :math:`N_{\rm flux}` for a flux vector.

        Args:
            flux (Array): Full flux vector ``[f \mid h]``.

        Returns:
            float: D3-tadpole charge :math:`N_{\rm flux}`.
        """
        return float(self.model.tadpole(flux).real)

    def get_fh(self, flux: Array) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        **Description:**
        Splits the full flux vector into its RR- and NSNS-flux components.

        Args:
            flux (Array): Full flux vector ``[f \mid h]`` of length
                ``2 * n_fluxes``.

        Returns:
            Tuple[ndarray, ndarray]: ``(f, h)`` each of length ``n_fluxes``.
        """
        arr = np.asarray(flux).real
        return arr[:self.n_fluxes], arr[self.n_fluxes:]

    def get_flux_split(self, flux_half: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        **Description:**
        Splits a half-flux vector ``f`` or ``h`` (length ``n_fluxes``) into
        its two sub-sector components.

        Args:
            flux_half (ndarray): Half-flux vector of length ``n_fluxes``.

        Returns:
            Tuple[ndarray, ndarray]: ``(flux_1, flux_2)`` each of length
            ``dimension_H3``.
        """
        arr = np.asarray(flux_half).real
        return arr[:self.dimension_H3], arr[self.dimension_H3:]

    def get_subvector(self, flux: Array) -> Tuple:
        r"""
        **Description:**
        Splits the full flux vector into all four sub-components
        :math:`(h_1, h_2, f_1, f_2)`.

        Args:
            flux (Array): Full flux vector ``[f \mid h]``.

        Returns:
            Tuple[ndarray, ndarray, ndarray, ndarray]: ``(h1, h2, f1, f2)``.
        """
        f, h   = self.get_fh(flux)
        h1, h2 = self.get_flux_split(h)
        f1, f2 = self.get_flux_split(f)
        return h1, h2, f1, f2

    def compute_norm(self, v: np.ndarray) -> float:
        r"""
        **Description:**
        Computes the squared Euclidean norm :math:`\|v\|^2 = v \cdot v`.

        Args:
            v (ndarray): Real (sub-)vector.

        Returns:
            float: Squared norm :math:`\|v\|^2`.
        """
        arr = np.asarray(v).real
        return float(arr @ arr)

    # =========================================================================
    #  Cluster parallelisation: save / load / reconstruct pipeline
    # =========================================================================

    @staticmethod
    def _save_pipeline(pipeline: dict, config: dict, output_dir: str):
        r"""
        **Description:**
        Save the precomputed pipeline data and scalar config to disk.

        Writes ``pipeline.npz`` (numerical arrays) and ``config.json``
        (scalar parameters) into ``output_dir``.
        """
        import os, json

        # --- Save numerical arrays ---
        evs = pipeline['evs_batch']
        np.savez_compressed(
            os.path.join(output_dir, "pipeline.npz"),
            moduli_slice=np.asarray(pipeline['moduli_slice']),
            tau_slice=np.asarray(pipeline['tau_slice']),
            M0_sigma_real=np.asarray(pipeline['M0_sigma'].real),
            M0_sigma_imag=np.asarray(pipeline['M0_sigma'].imag),
            M_inv_all=pipeline['M_inv_all_np'],
            s_vec=np.asarray(pipeline['s_vec']),
            c0_vec=np.asarray(pipeline['c0_vec']),
            sigma=np.asarray(pipeline['sigma']),
            evs_0=np.asarray(evs[0]),
            evs_1=np.asarray(evs[1]),
            evs_2=np.asarray(evs[2]),
            evs_3=np.asarray(evs[3]),
            evs_4=np.asarray(evs[4]),
        )

        # --- Save scalar config ---
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)

    @staticmethod
    def _load_pipeline(output_dir: str) -> Tuple:
        r"""
        **Description:**
        Load pipeline data and config from disk.

        Returns:
            Tuple of (pipeline_arrays dict, config dict).
        """
        import os, json

        data = dict(np.load(os.path.join(output_dir, "pipeline.npz"),
                            allow_pickle=False))
        with open(os.path.join(output_dir, "config.json")) as f:
            config = json.load(f)
        return data, config

    @staticmethod
    def _reconstruct_pipeline(data: dict, config: dict) -> dict:
        r"""
        **Description:**
        Rebuild the full pipeline dict (including filter closures) from
        serialised numerical data loaded by :meth:`_load_pipeline`.

        The filter closures (``_smax_filter``, ``_continuous_tadpole_filter``)
        are reconstructed from the saved scalar parameters and M_inv matrices.

        Args:
            data (dict): Arrays loaded from ``pipeline.npz``.
            config (dict): Scalars loaded from ``config.json``.

        Returns:
            dict: Pipeline dict compatible with :meth:`_process_h_chunk`.
        """
        import time as _time

        gl = config['gl_params']
        gl_params = tuple(gl)

        moduli_slice = jnp.array(data['moduli_slice'])
        tau_slice = jnp.array(data['tau_slice'])
        M0_sigma = jnp.array(
            data['M0_sigma_real'] + 1j * data['M0_sigma_imag'])
        M_inv_all_np = np.array(data['M_inv_all'])
        s_vec = jnp.array(data['s_vec'])
        c0_vec = jnp.array(data['c0_vec'])
        sigma = jnp.array(data['sigma'])
        evs_batch = tuple(
            jnp.array(data[f'evs_{i}']) for i in range(5))

        moduli_slice_np = np.asarray(moduli_slice)
        tau_slice_np = np.asarray(tau_slice)

        # Reconstruct scalar params from gl_params
        _lm = gl[0]          # lambda_max_gl
        _dil_min = gl[5]     # dil_min
        _Nmax = gl[7]        # Nmax
        _s_min_tad = _dil_min

        # --- Rebuild s_max filter closure ---
        def _smax_filter(h_chunk_np, h_norms_sq=None):
            if h_norms_sq is None:
                h_norms_sq = np.einsum(
                    'ij,ij->i', h_chunk_np, h_chunk_np).astype(np.float64)
            else:
                h_norms_sq = h_norms_sq.astype(np.float64)
            h_norms = np.sqrt(h_norms_sq)
            nonzero = h_norms > 0
            s_max = np.where(
                nonzero,
                _lm * _Nmax / np.where(nonzero, h_norms, 1.0)
                + np.where(nonzero, h_norms, 0.0) / (4.0 * _lm),
                np.inf)
            keep = s_max >= _dil_min
            return h_chunk_np[keep], h_norms_sq[keep]

        # --- Rebuild continuous tadpole filter closure ---
        M_inv_jax = jnp.array(M_inv_all_np)

        @jax.jit
        def _tadpole_scores(h_arr):
            Minv_h = jnp.einsum('mij,hj->mhi', M_inv_jax, h_arr)
            hMinvh = jnp.einsum('hi,mhi->mh', h_arr, Minv_h)
            return jnp.min(jnp.abs(_s_min_tad * hMinvh), axis=0)

        def _continuous_tadpole_filter(h_chunk_np):
            if len(h_chunk_np) == 0:
                return h_chunk_np
            h_jax = jnp.array(h_chunk_np, dtype=jnp.float_)
            scores = np.asarray(_tadpole_scores(h_jax))
            return h_chunk_np[scores <= _Nmax]

        t0 = _time.perf_counter()

        def _elapsed():
            s = _time.perf_counter() - t0
            if s < 120:
                return f"{s:.1f}s"
            elif s < 3600:
                return f"{s / 60:.1f}m"
            else:
                h = int(s // 3600)
                m = int((s % 3600) // 60)
                return f"{h}h {m}m"

        return {
            'moduli_slice': moduli_slice,
            'tau_slice': tau_slice,
            'moduli_slice_np': moduli_slice_np,
            'tau_slice_np': tau_slice_np,
            'evs_batch': evs_batch,
            'M0_sigma': M0_sigma,
            's_vec': s_vec,
            'c0_vec': c0_vec,
            'gl_params': gl_params,
            'sigma': sigma,
            'M_inv_all_np': M_inv_all_np,
            'h1_box': config['h1_box'],
            'h2_box': config['h2_box'],
            'h_box': config['h_box'],
            '_smax_filter': _smax_filter,
            '_continuous_tadpole_filter': _continuous_tadpole_filter,
            't0': t0,
            '_elapsed': _elapsed,
        }

    # =========================================================================
    #  Cluster parallelisation: export / process / merge
    # =========================================================================

    def export_cluster_job(
        self,
        output_dir: str,
        mode: str = "enumerate",
        chunk_size: int = 100_000,
        n_total_samples: int = 5_000_000,
        seed: int = 42,
        n_sample: int = 500,
        n_isd_per_h: int = 20,
        moduli_regions=None,
        use_linearised_shifts: bool = False,
        n_isd_iters: int = 5,
        generate_slurm: bool = False,
        slurm_opts: dict = None,
        verbose: bool = True,
    ) -> dict:
        r"""
        **Description:**
        Export the flux search pipeline to disk for cluster-parallel execution.

        Precomputes the ISD pipeline, generates pre-filtered h-chunks,
        and saves everything so that each chunk can be processed
        independently by a cluster worker via :meth:`process_chunk_from_disk`.

        Args:
            output_dir (str): Directory to write pipeline, chunks, and scripts.
            mode (str): ``"enumerate"`` for exhaustive search,
                ``"sample"`` for stochastic importance sampling.
            chunk_size (int): Target h-vectors per chunk file.
            n_total_samples (int): Total h-vectors to generate in sample mode.
            seed (int): Base random seed for sample mode.
            n_sample (int): Moduli points for eigenvalue bounds.
            n_isd_per_h (int): Moduli points per h-vector for ISD completion.
            moduli_regions: Optional list of (lo, hi) moduli bands.
            use_linearised_shifts (bool): Use linearised_shifts pipeline.
            n_isd_iters (int): Iterations for linearised_shifts.
            generate_slurm (bool): Generate SLURM array job script.
            slurm_opts (dict): Override SLURM directives.
            verbose (bool): Print progress.

        Returns:
            dict: Summary with keys ``n_chunks``, ``output_dir``,
            ``estimated_disk_mb``, ``n_h_total``.
        """
        import os

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "chunks"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)

        # --- Step 1: Prepare pipeline ---
        pipeline = self._prepare_isd_pipeline(
            n_sample=n_sample,
            n_isd_per_h=n_isd_per_h,
            moduli_regions=moduli_regions,
            use_linearised_shifts=use_linearised_shifts,
            n_isd_iters=n_isd_iters,
            verbose=verbose,
            label="export",
        )

        smax_filter = pipeline['_smax_filter']
        tad_filter = pipeline['_continuous_tadpole_filter']

        # --- Step 2: Build config ---
        gl = pipeline['gl_params']
        config = {
            'gl_params': [float(x) for x in gl],
            'n_fluxes': int(self.n_fluxes),
            'dimension_H3': int(self.model.dimension_H3),
            'h1_box': float(pipeline['h1_box']),
            'h2_box': float(pipeline['h2_box']),
            'h_box': float(pipeline['h_box']),
            'Nmax': float(self.Nmax),
            'dil_min': float(self.dil_min),
            'dil_max': float(self.dil_max),
            'mode': mode,
            'chunk_size': chunk_size,
            'use_linearised_shifts': use_linearised_shifts,
            'n_isd_iters': n_isd_iters,
            'version': 2,        # bumped: sampler-bounds serialisation added
            'h12': int(self.model.h12),
            'model_type': getattr(self.model, 'model_type', 'KS'),
        }
        # --- Sampler bounds (direction-aware box + cone specs) ---
        # Serialised so a cluster worker can reconstruct an identical
        # data_sampler without having to guess the parameters.  All
        # entries are written as JSON-friendly values (lists + scalars).
        if self.sampler is not None:
            _lo = np.asarray(getattr(self.sampler, 'moduli_lower', []),
                             dtype=float).ravel().tolist()
            _hi = np.asarray(getattr(self.sampler, 'moduli_upper', []),
                             dtype=float).ravel().tolist()
            _excl = np.asarray(getattr(self.sampler, 'exclude_walls', []),
                               dtype=bool).ravel().tolist()
            config['sampler_moduli_lower']  = _lo
            config['sampler_moduli_upper']  = _hi
            config['sampler_stretching']    = float(
                getattr(self.sampler, 'stretching', 0.0))
            config['sampler_exclude_walls'] = _excl
            config['sampler_cone_cutoff']   = float(
                getattr(self.sampler, 'cone_cutoff', 0.0))
            config['sampler_axion_lower']   = float(
                getattr(self.sampler, 'axion_lower', -0.5))
            config['sampler_axion_upper']   = float(
                getattr(self.sampler, 'axion_upper',  0.5))
            config['sampler_s_lower']       = float(
                getattr(self.sampler, 's_lower', 2.0))
            config['sampler_s_upper']       = float(
                getattr(self.sampler, 's_upper', 10.0))

        # --- Step 3: Generate and save h-chunks ---
        n_chunks = 0
        n_h_total = 0
        total_bytes = 0

        if mode == "enumerate":
            h_box_sq = float(pipeline['h_box']) ** 2
            raw_gen = self._iter_h_chunks_streaming(h_box_sq)
            fixed_gen = self._stream_fixed_chunks(raw_gen, chunk_size)

            if verbose:
                print("[export] Generating and filtering h-chunks ...",
                      flush=True)

            for h_chunk, h_norms_sq in fixed_gen:
                h_chunk, h_norms_sq = smax_filter(h_chunk, h_norms_sq)
                if len(h_chunk) == 0:
                    continue
                h_chunk = tad_filter(h_chunk)
                if len(h_chunk) == 0:
                    continue

                path = os.path.join(
                    output_dir, "chunks",
                    f"chunk_{n_chunks:04d}.npy")
                np.save(path, h_chunk.astype(np.int32))
                total_bytes += h_chunk.nbytes
                n_h_total += len(h_chunk)
                n_chunks += 1

                if verbose and n_chunks % 10 == 0:
                    print(f"  {n_chunks} chunks, {n_h_total:,} h-vectors ...",
                          flush=True)

        elif mode == "sample":
            rng = np.random.default_rng(seed)
            M_inv_data = pipeline['M_inv_all_np']
            n_generated = 0

            if verbose:
                print(f"[export] Sampling {n_total_samples:,} h-vectors ...",
                      flush=True)

            while n_generated < n_total_samples:
                n_batch = min(chunk_size, n_total_samples - n_generated)
                h_batch = self._sample_h_vectors_importance(
                    n_batch, M_inv_data, rng=rng)
                h_batch = tad_filter(h_batch.astype(float))
                if len(h_batch) > 0:
                    path = os.path.join(
                        output_dir, "chunks",
                        f"chunk_{n_chunks:04d}.npy")
                    np.save(path, h_batch.astype(np.int32))
                    total_bytes += h_batch.nbytes
                    n_h_total += len(h_batch)
                    n_chunks += 1
                n_generated += n_batch
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Must be 'enumerate' or 'sample'.")

        config['n_chunks'] = n_chunks

        # --- Step 4: Save pipeline and config ---
        self._save_pipeline(pipeline, config, output_dir)

        # --- Step 5: Generate worker template ---
        worker_path = os.path.join(output_dir, "worker.py")
        with open(worker_path, 'w') as f:
            f.write(f'''#!/usr/bin/env python
"""Auto-generated cluster worker for jaxvacua flux search.
Edit the MODEL SETUP section below for your model.
"""
import sys, os, json
import jax; jax.config.update("jax_enable_x64", True)

# ---- MODEL SETUP (edit this!) ----
import jaxvacua as jvc
model = jvc.FluxVacuaFinder(
    h12={config['h12']},
    model_ID=1,  # <-- set your model_ID
    model_type="{config['model_type']}",
)

# ---- SAMPLER RECONSTRUCTION (auto-restored from config.json) ----
# The exporter's data_sampler bounds were serialised alongside the
# pipeline so cluster workers apply the same in-patch / stretched-cone
# / cutoff filtering during ISD refinement.  Edit if you want to widen
# or narrow the patch on the worker side.
with open(os.path.join(os.path.dirname(__file__), "config.json")) as _f:
    _cfg = json.load(_f)

def _bounds(lo_key, hi_key, h12):
    lo = _cfg.get(lo_key, [1.0] * h12)
    hi = _cfg.get(hi_key, [5.0] * h12)
    return (lo, hi)

sampler = jvc.data_sampler(
    model,
    moduli_bounds=_bounds("sampler_moduli_lower",
                          "sampler_moduli_upper",
                          int(_cfg.get("h12", {config['h12']}))),
    axion_bounds=(_cfg.get("sampler_axion_lower", -0.5),
                  _cfg.get("sampler_axion_upper",  0.5)),
    dilaton_bounds=(_cfg.get("sampler_s_lower", 2.0),
                    _cfg.get("sampler_s_upper", 10.0)),
    stretching=_cfg.get("sampler_stretching", 0.0),
    exclude_walls=[i for i, b in enumerate(_cfg.get("sampler_exclude_walls", []))
                   if b] or None,
    cone_cutoff=_cfg.get("sampler_cone_cutoff", None),
)
# ----------------------------------

from jaxvacua.flux_bounding import bounded_fluxes

output_dir = sys.argv[1]
chunk_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", sys.argv[2]))

bounded_fluxes.process_chunk_from_disk(
    output_dir=output_dir,
    chunk_id=chunk_id,
    model=model,
    sampler=sampler,
)
''')

        # --- Step 6: Optional SLURM script ---
        if generate_slurm:
            self._generate_slurm_script(output_dir, n_chunks, slurm_opts)

        disk_mb = total_bytes / 1024 / 1024
        if verbose:
            print(f"\n[export] Done: {n_chunks} chunks, "
                  f"{n_h_total:,} h-vectors, {disk_mb:.1f} MB on disk")
            print(f"[export] Output: {output_dir}")

        return {
            'n_chunks': n_chunks,
            'output_dir': output_dir,
            'estimated_disk_mb': disk_mb,
            'n_h_total': n_h_total,
        }

    @staticmethod
    def process_chunk_from_disk(
        output_dir: str,
        chunk_id: int,
        model,
        sampler=None,
        verbose: bool = True,
    ) -> None:
        r"""
        **Description:**
        Process a single h-chunk from disk using a reconstructed pipeline.
        Designed to be called from a cluster worker script.

        Args:
            output_dir (str): Directory with pipeline.npz, config.json, chunks/.
            chunk_id (int): Index of the chunk to process.
            model: A reconstructed FluxEFT or FluxVacuaFinder instance.
            sampler: Optional data_sampler (only for linearised_shifts mode).
            verbose (bool): Print progress.
        """
        import os, time as _time
        t0 = _time.perf_counter()

        data, config = bounded_fluxes._load_pipeline(output_dir)
        pipeline = bounded_fluxes._reconstruct_pipeline(data, config)

        chunk_path = os.path.join(
            output_dir, "chunks", f"chunk_{chunk_id:04d}.npy")
        h_chunk_np = np.load(chunk_path).astype(float)

        if verbose:
            print(f"[worker {chunk_id}] Loaded {len(h_chunk_np)} h-vectors",
                  flush=True)

        bf = bounded_fluxes(model, sampler, Nmax=config.get('Nmax', 10))
        gl = config['gl_params']
        bf.lambda_max_gl = gl[0]
        bf.mu_min_gl = gl[1]
        bf.mu_max_gl = gl[2]
        bf.tilde_mu_min_gl = gl[3]
        bf.tilde_mu_max_gl = gl[4]
        bf.dil_min = gl[5]
        bf.dil_max = gl[6]
        # Set box radii so bounds_initialized property returns True
        bf._h1_box = config.get('h1_box', 1.0)
        bf._h2_box = config.get('h2_box', 1.0)
        bf._h_box = config.get('h_box', 1.0)

        h_chunk_jax = jnp.array(h_chunk_np)
        new_fluxes, new_moduli, new_taus = bf._process_h_chunk(
            h_chunk_jax, pipeline,
            use_linearised_shifts=config.get('use_linearised_shifts', False),
            n_isd_iters=config.get('n_isd_iters', 5),
        )

        results_dir = os.path.join(output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        if new_fluxes:
            fluxes_arr = np.array(new_fluxes)
            moduli_arr = np.array(new_moduli)
            taus_arr = np.array(new_taus)
        else:
            n_fl = config['n_fluxes']
            h12 = config['h12']
            fluxes_arr = np.zeros((0, 2 * n_fl))
            moduli_arr = np.zeros((0, h12), dtype=complex)
            taus_arr = np.zeros((0,), dtype=complex)

        result_path = os.path.join(
            results_dir, f"result_{chunk_id:04d}.npz")
        np.savez_compressed(
            result_path,
            fluxes=fluxes_arr, moduli=moduli_arr,
            taus=taus_arr, n_processed=len(h_chunk_np),
        )

        dt = _time.perf_counter() - t0
        if verbose:
            print(
                f"[worker {chunk_id}] Done: {len(new_fluxes)} valid fluxes "
                f"from {len(h_chunk_np)} h-vectors ({dt:.1f}s)", flush=True)

    @classmethod
    def merge_cluster_results(
        cls,
        output_dir: str,
        model=None,
        sampler=None,
        refine: bool = False,
        newton_tol: float = 1e-10,
        newton_max_iters: int = 100,
        newton_step_size: float = 1.0,
        return_moduli: bool = True,
        database=None,
        method: str = "enumerate_cluster",
        tags: list = None,
        verbose: bool = True,
        map_to_fd: bool = False,
        designate: bool = False,
        label: str = None,
        committed_by: str = None,
        validate_before_designate: bool = True,
    ) -> list:
        r"""
        **Description:**
        Merge results from all cluster workers, deduplicate, optionally
        Newton-refine, and optionally write to the vacua database.

        Two levels of database integration:

        1. ``database=<CYDatabase>``: write the merged results as a
           **session-tier** batch via
           :meth:`stringforge.lcs_database.LCSDatabase.vacua_writer`.  The
           results are queryable via :meth:`query_vacua` but not yet
           permanent.
        2. ``designate=True`` (requires ``database``, ``label``,
           ``committed_by``): additionally promote the merged results to
           the **permanent vault** via
           :meth:`stringforge.lcs_database.LCSDatabase.designate_vacua`.  The
           results are retrievable via :meth:`load_local_vacua`.

        Args:
            output_dir (str): Directory containing results/ subdirectory.
            model: FluxEFT instance (needed for refine or database write).
            sampler: data_sampler instance (needed for in-patch checking).
            refine (bool): Newton-refine the merged results.
            newton_tol (float): Newton convergence tolerance.
            newton_max_iters (int): Max Newton iterations.
            newton_step_size (float): Newton step size.
            return_moduli (bool): Include moduli/tau in output dicts.
            database: Optional CYDatabase instance for writing results.
            method (str): Method label for database catalog.
            tags (list): Searchable tags for database catalog.
            verbose (bool): Print progress.
            map_to_fd (bool): If ``True`` and *model* has ``map_to_fd``,
                map each vacuum to the fundamental domain before
                deduplicating.  Defaults to ``False``.
            designate (bool): If True, also promote the merged results
                to the permanent vault via
                :meth:`designate_vacua`.  Requires *database*, *label*,
                and *committed_by*.
            label (str | None): Designation label (required when
                ``designate=True``).
            committed_by (str | None): Contributor identifier (required
                when ``designate=True``).
            validate_before_designate (bool): Run
                :meth:`validate_vacua` before designation.  Defaults to
                True.

        Returns:
            list: Merged results as list of dicts.
        """
        import os, json

        with open(os.path.join(output_dir, "config.json")) as f:
            config = json.load(f)

        n_expected = config.get('n_chunks', 0)
        results_dir = os.path.join(output_dir, "results")

        present = set()
        for fname in sorted(os.listdir(results_dir)):
            if fname.startswith("result_") and fname.endswith(".npz"):
                cid = int(fname.replace("result_", "").replace(".npz", ""))
                present.add(cid)

        missing = set(range(n_expected)) - present
        if verbose:
            print(f"[merge] Found {len(present)}/{n_expected} result files")
            if missing:
                ids = sorted(missing)
                show = ids[:20]
                print(f"[merge] Missing chunks: {show}"
                      f"{'...' if len(ids) > 20 else ''}")

        all_fluxes, all_moduli, all_taus = [], [], []
        n_processed_total = 0

        for cid in sorted(present):
            path = os.path.join(results_dir, f"result_{cid:04d}.npz")
            d = np.load(path, allow_pickle=False)
            if len(d['fluxes']) > 0:
                all_fluxes.append(d['fluxes'])
                all_moduli.append(d['moduli'])
                all_taus.append(d['taus'])
            n_processed_total += int(d['n_processed'])

        if not all_fluxes:
            if verbose:
                print("[merge] No valid results found.")
            return []

        fluxes = np.concatenate(all_fluxes, axis=0)
        moduli = np.concatenate(all_moduli, axis=0)
        taus = np.concatenate(all_taus, axis=0)

        if verbose:
            print(f"[merge] Raw: {len(fluxes)} candidates "
                  f"from {n_processed_total:,} h-vectors")

        # Deduplicate (optionally map to FD first)
        _do_fd = map_to_fd and model is not None and hasattr(model, 'map_to_fd')
        seen = set()
        keep = []
        for i in range(len(fluxes)):
            if _do_fd:
                moduli_fd, tau_fd, fluxes_fd = model.map_to_fd(
                    jnp.asarray(moduli[i]), complex(taus[i]), jnp.asarray(fluxes[i]),
                )
                fluxes[i] = np.asarray(fluxes_fd, dtype=np.int32)
                moduli[i] = np.asarray(moduli_fd)
                taus[i] = complex(tau_fd)
            # Strip any zero-magnitude imaginary part (JAX → numpy
            # roundoff on the upstream solver) before casting to int32.
            fl_i = np.asarray(fluxes[i])
            if np.iscomplexobj(fl_i):
                max_im = float(np.max(np.abs(fl_i.imag)))
                assert max_im < 1e-9, (
                    f"flux[{i}] has non-negligible imaginary part "
                    f"(max |imag|={max_im:.3e}) — should be a real "
                    f"integer vector."
                )
                fl_i = fl_i.real
            key = np.round(fl_i).astype(np.int32).tobytes()
            if key not in seen:
                seen.add(key)
                keep.append(i)

        fluxes = fluxes[keep]
        moduli = moduli[keep]
        taus = taus[keep]

        if verbose:
            print(f"[merge] After dedup: {len(fluxes)} unique flux vectors")

        # Optional Newton refinement
        res_arr = None
        if refine and model is not None:
            bf = cls(model, sampler, Nmax=config.get('Nmax', 10))
            mod_jax = jnp.array(moduli, dtype=complex)
            tau_jax = jnp.array(taus, dtype=complex)
            fl_jax = jnp.array(fluxes)

            mod_out, tau_out, residuals = bf.newton_refine_batch(
                mod_jax, tau_jax, fl_jax,
                step_size=newton_step_size,
                tol=newton_tol,
                max_iters=newton_max_iters,
            )

            converged = np.asarray(residuals) < newton_tol
            if sampler is not None:
                in_patch = np.asarray(bf.in_patch_batch(mod_out, tau_out))
                keep_mask = converged & in_patch
            else:
                keep_mask = converged

            moduli = np.asarray(mod_out)[keep_mask]
            taus = np.asarray(tau_out)[keep_mask]
            fluxes = np.asarray(fl_jax)[keep_mask]
            res_arr = np.asarray(residuals)[keep_mask]

            if verbose:
                print(f"[merge] After Newton: {int(keep_mask.sum())} "
                      f"converged (of {len(converged)})")

        # Build result list
        results = []
        for i in range(len(fluxes)):
            entry = {'flux': fluxes[i]}
            if return_moduli:
                entry['moduli'] = moduli[i]
                entry['tau'] = complex(taus[i])
            if res_arr is not None:
                entry['residual'] = float(res_arr[i])
            results.append(entry)

        # Optional database write (session tier)
        if database is not None and model is not None and len(results) > 0:
            try:
                with database.vacua_writer(
                        model, method=method, tags=tags) as writer:
                    writer.append_batch(
                        moduli=moduli, tau=taus, fluxes=fluxes,
                        residual=res_arr)
                if verbose:
                    print(f"[merge] Wrote {len(results)} results to database")
            except Exception as e:
                if verbose:
                    print(f"[merge] Database write failed: {e}")

        # Optional promotion to permanent vault via designate_vacua
        if designate and len(results) > 0:
            if database is None:
                raise ValueError(
                    "designate=True requires a CYDatabase instance passed "
                    "via `database=...`."
                )
            if model is None:
                raise ValueError(
                    "designate=True requires `model=...` for identity "
                    "extraction and validation."
                )
            if not label or not label.strip():
                raise ValueError(
                    "designate=True requires a non-empty `label` kwarg."
                )
            if not committed_by or not committed_by.strip():
                raise ValueError(
                    "designate=True requires a non-empty `committed_by` "
                    "kwarg."
                )
            try:
                import pandas as pd
                # Same defensive cast as in the dedup loop above:
                # tolerate trace imaginary parts (JAX roundoff) but
                # assert they're below tolerance.
                def _as_real_int_list(fl):
                    fl = np.asarray(fl)
                    if np.iscomplexobj(fl):
                        max_im = float(np.max(np.abs(fl.imag)))
                        assert max_im < 1e-9, (
                            f"flux row has non-negligible imaginary part "
                            f"(max |imag|={max_im:.3e})"
                        )
                        fl = fl.real
                    return list(int(x) for x in fl)

                df_rows = [{
                    "flux":      _as_real_int_list(fluxes[i]),
                    "moduli_re": list(float(x) for x in np.asarray(moduli[i]).real),
                    "moduli_im": list(float(x) for x in np.asarray(moduli[i]).imag),
                    "tau_re":    float(np.asarray(taus[i]).real),
                    "tau_im":    float(np.asarray(taus[i]).imag),
                    "residual":  (float(res_arr[i]) if res_arr is not None else None),
                } for i in range(len(fluxes))]
                df = pd.DataFrame(df_rows)
                ids = database.designate_vacua(
                    df,
                    label=label,
                    committed_by=committed_by,
                    model=model,
                    tags=tags,
                    validate=validate_before_designate,
                    force=not validate_before_designate,
                )
                if verbose:
                    print(f"[merge] Designated {len(ids)} vacua to vault "
                          f"with label={label!r}")
            except Exception as e:
                if verbose:
                    print(f"[merge] Designation failed: {e}")

        if verbose:
            print(f"[merge] Final: {len(results)} flux vacua")

        return results

    @staticmethod
    def _generate_slurm_script(output_dir, n_chunks, slurm_opts=None):
        r"""
        **Description:**
        Generate a SLURM array job submission script.
        """
        import os

        defaults = {
            'job-name': 'jaxvacua_flux',
            'ntasks': '1',
            'cpus-per-task': '4',
            'mem': '8G',
            'time': '02:00:00',
        }
        if slurm_opts:
            defaults.update(slurm_opts)

        lines = ['#!/bin/bash']
        lines.append(f'#SBATCH --array=0-{n_chunks - 1}')
        for key, val in defaults.items():
            lines.append(f'#SBATCH --{key}={val}')

        logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        lines.append(f'#SBATCH --output={logs_dir}/chunk_%a.out')
        lines.append(f'#SBATCH --error={logs_dir}/chunk_%a.err')
        lines.append('')
        lines.append(
            f'JAX_ENABLE_X64=1 python {output_dir}/worker.py '
            f'{output_dir} $SLURM_ARRAY_TASK_ID')
        lines.append('')

        script_path = os.path.join(output_dir, "submit_array.sh")
        with open(script_path, 'w') as f:
            f.write('\n'.join(lines))
        os.chmod(script_path, 0o755)
        return script_path
