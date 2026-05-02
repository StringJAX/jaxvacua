# ==============================================================================
# This code is written by Andreas Schachner.
#
# If any questions arise, please feel free to reach out to me (Andreas) either at
# andreas.schachner@gmx.net or at as3475@cornell.edu .
# ==============================================================================
"""``jaxvacua.conifold.zcf_solver`` — z_cf physics (bulk EFT after integrating out the conifold modulus).

Mathematical structure (notes/main.tex eq:Wtilde1Explicit, eq:zcf_corrected)::

    ∂_zcf W_coni = log_prefactor · ln(-2πi·z_cf) + W_log_coeff + O(z_cf)
    log_prefactor = (M¹ - τH¹) · n_cf / (2πi)
    log_prefactor · ln(-2πi·z_cf) + W_log_coeff [+ log_coeff_K_corr] = 0
    ⟹ z_cf = -(1/2πi) · exp(-(W_log_coeff [+ log_coeff_K_corr]) / log_prefactor)

Layout (top to bottom: building blocks → log-coeff routines → dispatcher):
    - ``W_bulk``, ``dK_cf_bulk``, ``log_prefactor`` — F-term ingredients.
    - ``_split_conifold_bulk`` — bulk/conifold projection of topological data.
    - ``_W_log_coeff_pfv`` / ``_W_log_coeff_manual`` / ``_W_log_coeff_autodiff``
      — three independent W̃₁ routes (manual & autodiff agree numerically; pfv
      is a deliberate racetrack approximation).
    - ``W_log_coeff`` — public dispatcher.
    - ``log_coeff_K_corr``, ``_zcf_from_log_coeff`` — Kähler correction +
      final exponentiation.
    - ``compute_zcf`` / ``compute_zcf_x`` / ``zcf_handling`` — top-level
      entries (complex-coord, real-coord, real-coord with no-op mode).

The deprecated ``DWbulk_x`` / ``dDWbulk_x`` were hard-removed on 2026-05-01
once ``private/promotion/vacuum_promotion.py`` was migrated to the
:class:`jaxvacua.freezer.ConifoldFreezer` interface (``DW_x_light`` /
``dDW_x_light``).

All public functions take ``flux`` and are written as plain ``def``s of
``self``; method attachment to ``FluxEFT`` lives in
:mod:`jaxvacua.conifold` (the package ``__init__.py``).
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

from jaxpolylog import jax_polylog_vmap


# ---- Building blocks -----------------------------------------------------

@partial(jit, static_argnums=(4,5,))
def W_bulk(self, z, tau, flux, conj=False, normalise=False):

    if not self.lcs_tree.conifold_basis:
        raise NotImplementedError("compute_zcf_compact is only implemented for conifold_basis=True.")

    coeff = 2 * jnp.pi * 1j
    if conj:
        coeff = -coeff

    ncf = self.lcs_tree.conifold.ncf

    zbulk = z

    M0, H0, M1, H1, Malpha, Halpha, P1, K1, Palpha, Kalpha, P0, K0 = self.conifold_fluxes(flux)

    F0 = self.F_coniLCS_exp(zbulk, conj=conj, n=0)
    dF0 = self.dF_coniLCS_exp(zbulk, conj=conj, n=0)
    F1 = self.F_coniLCS_exp(zbulk, conj=conj, n=1)

    tmp  = (M0 - tau * H0) * (2*F0 - zbulk@dF0- 2*ncf*jax.scipy.special.zeta(3, q=1)/coeff**3)
    tmp += (M1 - tau * H1) * F1
    tmp += (Malpha - tau * Halpha) @ dF0
    tmp += (-1) * (P0 - tau * K0)
    tmp += (-1) * (Palpha - tau * Kalpha) @ zbulk

    if normalise:
        tmp = jnp.sqrt(2./jnp.pi)*tmp

    return tmp


@jit
def dK_cf_bulk(self, z, cz):

    coeff = 2 * jnp.pi * 1j

    zbulk = z
    czbulk = cz
    ncf = self.lcs_tree.conifold.ncf

    F0 = self.F_coniLCS_exp(zbulk, conj=False, n=0)
    cF0 = self.F_coniLCS_exp(czbulk, conj=True, n=0)
    dF0 = self.dF_coniLCS_exp(zbulk, conj=False, n=0)
    dcF0 = self.dF_coniLCS_exp(czbulk, conj=True, n=0)

    F1 = self.F_coniLCS_exp(zbulk, conj=False, n=1)
    dF1 = self.dF_coniLCS_exp(zbulk, conj=False, n=1)
    cF1 = self.F_coniLCS_exp(czbulk, conj=True, n=1)

    numerator = F1-cF1 - zbulk@dF1 + czbulk@dF1

    FF0 = 2*F0 - zbulk@dF0- 2*ncf*jax.scipy.special.zeta(3, q=1)/coeff**3
    cFF0 = 2*cF0 - czbulk@dcF0- 2*ncf*jax.scipy.special.zeta(3, q=1)/(-coeff)**3

    denom = czbulk@dF0 - zbulk@dcF0 + FF0 - cFF0

    dK = -numerator / denom

    return dK


# Mathematical structure of the F-term equation
# (notes/main.tex eq:Wtilde1Explicit, eq:zcf_corrected):
#
#     ∂_zcf W_coni = log_prefactor · ln(-2πi·z_cf) + W_log_coeff + O(z_cf)
#     log_prefactor = (M¹ - τH¹) · n_cf / (2πi)
#
# Including the Kähler-covariant correction δW̃₁ = (∂_zcf K)|₀ · W_bulk:
#
#     log_prefactor · ln(-2πi·z_cf) + W_log_coeff + log_coeff_K_corr = 0
#
# ⟹  z_cf = -(1/2πi) · exp(-(W_log_coeff [+ log_coeff_K_corr]) / log_prefactor).

@partial(jit, static_argnums=(3,))
def log_prefactor(self, tau, flux, conj=False):
    r"""Coefficient of :math:`\ln(-2\pi i z_{\rm cf})` in :math:`\partial_{z_{\rm cf}} W_{\rm coni}`:

        ``log_prefactor = (M¹ - τH¹) · n_cf / (2πi)``.

    The leading-order F-term equation reads
    ``log_prefactor · ln(-2πi·z_cf) + W_log_coeff = 0``.
    """
    coeff = 2 * jnp.pi * 1j
    if conj:
        coeff = -coeff
    _, _, M1, H1, *_ = self.conifold_fluxes(flux)
    return (M1 - tau * H1) * self.lcs_tree.conifold.ncf / coeff


def _split_conifold_bulk(self):
    r"""Project the topological data (``intnums``, ``a_matrix``, ``b_vector``)
    onto the conifold-vs-bulk split.

    Returns a 5-tuple ``(kappa_011, kappa_001, a00, a01, b_coni)`` where:

    - ``kappa_011``: bulk-bulk slice along the conifold direction, shape
      ``(h12-1, h12-1)``.
    - ``kappa_001``: bulk slice along two conifold directions, shape ``(h12-1,)``.
    - ``a00`` / ``a01``: pieces of the a-matrix projected on the conifold and
      conifold-bulk directions respectively (scalar / shape ``(h12-1,)``).
    - ``b_coni``: scalar piece of the b-vector along the conifold direction.

    Under ``conifold_basis=True`` these reduce to plain array slicing
    (``kappa[0, 1:, 1:]``, ``kappa[0, 0, 1:]``, ``a_matrix[0, 0]``,
    ``a_matrix[0, 1:]``, ``b_vector[0]``). Under ``conifold_basis=False`` the
    same data is recovered via ``einsum`` projections through the conifold
    direction ``q = conifold.conifold_curve`` and the orthogonal-complement matrix
    ``w_proj = conifold.projection``.
    """
    kappa = self.lcs_tree.intnums
    if self.lcs_tree.conifold_basis:
        kappa_011 = kappa[0, 1:, 1:]
        kappa_001 = kappa[0, 0, 1:]
        a_coni = self.lcs_tree.a_matrix[:, 0]
        a00 = a_coni[0]
        a01 = a_coni[1:]
        b_coni = self.lcs_tree.b_vector[0]
    else:
        q = self.lcs_tree.conifold.conifold_curve
        w_proj = self.lcs_tree.conifold.projection
        kappa_011 = jnp.einsum("ijk,i,jl,km", kappa, q, w_proj, w_proj)
        kappa_001 = jnp.einsum("ijk,i,j,km", kappa, q, q, w_proj)
        a_coni = self.lcs_tree.a_matrix @ q
        a00 = a_coni @ q
        a01 = a_coni @ w_proj
        b_coni = self.lcs_tree.b_vector @ q
    return kappa_011, kappa_001, a00, a01, b_coni


@partial(jit, static_argnums=(4,))
def _W_log_coeff_pfv(self, z, tau, flux, conj=False):
    r"""``W_log_coeff`` (≡ W̃₁) in the **PFV / linear-racetrack approximation**.

    The PFV approximation replaces the bulk VEVs by the racetrack solution
    parametrised through the integer fluxes :math:`(M^a, K_a)` and the
    axio-dilaton :math:`\tau = c_0 + i/g_s`. The resulting ``W̃₁`` (and hence
    ``z_cf`` via :func:`_zcf_from_log_coeff`) differs from the closed-form
    ``manual`` / ``autodiff`` routes, which use the actual bulk moduli ``z``;
    PFV is an approximation that becomes exact at the racetrack-stationary
    point.

    Returned in the same ``W̃₁`` convention as :func:`_W_log_coeff_manual` and
    :func:`_W_log_coeff_autodiff`, so that the unified
    ``compute_zcf(mode="pfv")`` dispatch composes via
    :func:`_zcf_from_log_coeff` without any special-casing.

    Note: ``conifold_basis=True`` only — the racetrack split assumes
    axis-aligned ``M`` / ``K`` components.
    """
    if not self.lcs_tree.conifold_basis:
        raise NotImplementedError(
            "_W_log_coeff_pfv is only implemented for conifold_basis=True."
        )

    f1, f2, h1, h2 = self._split_fluxes(flux)

    Mvec = f2[1:]
    Kvec = h1[1:]
    P1   = f1[1]

    # Racetrack split of the bulk fluxes.
    N      = self.lcs_tree.intnums @ Mvec
    pvec   = jnp.linalg.inv(N[1:, 1:]) @ Kvec[1:]
    Kprime = Kvec[0] - N[0, 1:] @ pvec

    # PFV: c0 = Re(τ), gs = 1/Im(τ).
    c0 = jnp.real(tau)
    s  = jnp.imag(tau)
    gs = 1.0 / s

    phase_comb = -1j * (self.lcs_tree.a_matrix[0] @ Mvec - P1 + c0 * Kprime)
    if conj:
        phase_comb = -phase_comb

    radial = Kprime / gs
    return 1j * (phase_comb + radial)


@partial(jit, static_argnums=(4,))
def _W_log_coeff_manual(self, z, tau, flux, conj=False):
    r"""``W_log_coeff`` (≡ W̃₁) via closed-form assembly from
    ``kappa`` / ``a_matrix`` / ``b_vector`` + worldsheet-instanton ``Li`` sums.

    Body extracted from :func:`compute_zcf_explicit` minus the final
    exponentiation. Must agree with :func:`_W_log_coeff_autodiff` numerically.
    """
    coeff = 2 * jnp.pi * 1j
    if conj:
        coeff = -coeff

    ncf = self.lcs_tree.conifold.ncf
    zbulk = z

    M0, H0, M1, H1, Malpha, Halpha, P1, K1, Palpha, Kalpha, P0, K0 = self.conifold_fluxes(flux)

    kappa_011, kappa_001, a00, a01, b_coni = self._split_conifold_bulk()

    F1b  = ((kappa_011 @ zbulk) @ zbulk) / 2 + b_coni - jnp.pi**2 / 6 * ncf / coeff**2
    F2b  = -((kappa_001 @ zbulk)) + a00
    dF1b = -((kappa_011 @ zbulk)) + a01

    if self.periods.include_mirror_wsi:
        if self.lcs_tree.limit in ["coniLCS_series", "coniLCS_bulk"]:
            bulk_invs = self.lcs_tree.gv_invariants
            bulk_charges = self.lcs_tree.gv_charges
        else:
            bulk_invs = self.periods.delete_coni_index(self.lcs_tree.gv_invariants, self.lcs_tree.coni_index)
            bulk_charges = self.periods.delete_coni_index(self.lcs_tree.gv_charges, self.lcs_tree.coni_index)

        if self.lcs_tree.conifold_basis:
            beta1 = jnp.asarray(bulk_charges[:, 0])
            bulk_charges_proj = jnp.asarray(bulk_charges[:, 1:])
            etpz = jnp.exp(coeff * jnp.einsum("ki,i", bulk_charges_proj, zbulk))
        else:
            q = self.lcs_tree.conifold.conifold_curve
            beta1 = jnp.asarray(bulk_charges @ q)
            w_proj = self.lcs_tree.conifold.projection
            raise NotImplementedError("ISSUE HERE!!!")
            #bulk_charges_proj = jnp.asarray(bulk_charges @ w_proj)
            bulk_charges_proj = jnp.asarray(bulk_charges)
            etpz = jnp.exp(coeff * jnp.einsum("ki,i", bulk_charges_proj, w_proj @ zbulk))

        Li1 = jax_polylog_vmap(etpz, 1, self.periods.prange)
        F2b  += -1 / coeff * jnp.sum(bulk_invs * beta1**2 * Li1)
        dF1b += -1 / coeff * jnp.sum((bulk_invs * Li1 * beta1)[:, None] * bulk_charges_proj, axis=0)

        Li2 = jax_polylog_vmap(etpz, 2, self.periods.prange)
        F1b += -1 / coeff**2 * jnp.sum(bulk_invs * beta1 * (Li2 - (coeff * bulk_charges_proj @ zbulk) * Li1))

    tmp  = (M0 - tau * H0) * F1b
    tmp += (M1 - tau * H1) * F2b
    tmp += (Malpha - tau * Halpha) @ dF1b
    tmp += (-1) * (P1 - tau * K1)
    return tmp


@partial(jit, static_argnums=(4,))
def _W_log_coeff_autodiff(self, z, tau, flux, conj=False):
    r"""``W_log_coeff`` (≡ W̃₁) via the css-side ``F_coniLCS_exp(zbulk, n=1, 2)`` +
    ``dF_coniLCS_exp(zbulk, n=1)`` series helpers.

    Body extracted from :func:`compute_zcf_compact` minus the final
    exponentiation. Must agree with :func:`_W_log_coeff_manual` numerically.
    """
    if not self.lcs_tree.conifold_basis:
        raise NotImplementedError(
            "_W_log_coeff_autodiff is only implemented for conifold_basis=True."
        )

    coeff = 2 * jnp.pi * 1j
    if conj:
        coeff = -coeff

    ncf = self.lcs_tree.conifold.ncf
    zbulk = z

    M0, H0, M1, H1, Malpha, Halpha, P1, K1, Palpha, Kalpha, P0, K0 = self.conifold_fluxes(flux)

    F2 = self.F_coniLCS_exp(zbulk, conj=conj, n=2)
    F1 = self.F_coniLCS_exp(zbulk, conj=conj, n=1)
    dF1 = self.dF_coniLCS_exp(zbulk, conj=conj, n=1)

    F1b = F1 - zbulk @ dF1
    F2b = F2 + ncf * 3 / 2 / coeff

    tmp  = (M0 - tau * H0) * F1b
    tmp += (M1 - tau * H1) * F2b
    tmp += (Malpha - tau * Halpha) @ dF1
    tmp += (-1) * (P1 - tau * K1)
    return tmp


@partial(jit, static_argnums=(4, 5))
def W_log_coeff(self, z, tau, flux, mode="manual", conj=False):
    r"""Public dispatcher for the regular :math:`O(z_{\rm cf}^0)` coefficient
    :math:`\widetilde W_1` of :math:`\partial_{z_{\rm cf}} W_{\rm coni}` near the
    conifold.

    The F-term equation reads
    ``log_prefactor · ln(-2πi·z_cf) + W_log_coeff + O(z_cf) = 0``,
    so the leading-order solution is
    ``z_cf = -(1/2πi) · exp(-W_log_coeff / log_prefactor)``.

    Args:
        z: bulk moduli (h12-1 complex components).
        tau: axio-dilaton.
        flux: full flux vector.
        mode: ``"manual"`` for the closed-form ``kappa`` / ``a_matrix`` /
            ``b_vector`` + ``Li`` assembly; ``"autodiff"`` for the assembly via
            the css-side ``F_coniLCS_exp`` + ``dF_coniLCS_exp`` helpers
            (``manual`` and ``autodiff`` agree numerically — cross-checked in
            the smoke test); ``"pfv"`` for the PFV / linear-racetrack
            approximation, which uses only the integer fluxes and does NOT
            agree with manual/autodiff (it's a deliberate approximation that
            becomes exact at the racetrack-stationary point).
        conj: take complex-conjugate branch.
    """
    if mode == "manual":
        return self._W_log_coeff_manual(z, tau, flux, conj=conj)
    if mode == "autodiff":
        return self._W_log_coeff_autodiff(z, tau, flux, conj=conj)
    if mode == "pfv":
        return self._W_log_coeff_pfv(z, tau, flux, conj=conj)
    raise ValueError(f"mode must be 'manual', 'autodiff', or 'pfv'; got {mode!r}")


@partial(jit, static_argnums=(6,))
def log_coeff_K_corr(self, z, cz, tau, ctau, flux, conj=False):
    r"""Kähler-covariant correction to ``W_log_coeff``:

        :math:`\delta\widetilde W_1 = \partial_{z_{\rm cf}} K|_0 \cdot W_{\rm bulk}`.

    The corrected leading-order F-term equation reads
    ``log_prefactor · ln(-2πi·z_cf) + W_log_coeff + log_coeff_K_corr = 0``.
    """
    if conj:
        Wbulk = self.W_bulk(cz, ctau, flux, conj=conj)
    else:
        Wbulk = self.W_bulk(z, tau, flux, conj=conj)
    return self.dK_cf_bulk(z, cz) * Wbulk


@partial(jit, static_argnums=(4,))
def _zcf_from_log_coeff(self, W, tau, flux, conj=False):
    r"""Exponentiate the F-term equation:

        ``z_cf = -(1/2πi) · exp(-W / log_prefactor)``,

    where ``W`` is either ``W_log_coeff`` (leading-order) or
    ``W_log_coeff + log_coeff_K_corr`` (with Kähler correction applied).
    """
    coeff = 2 * jnp.pi * 1j
    if conj:
        coeff = -coeff
    return (-1 / coeff) * jnp.exp(-W / self.log_prefactor(tau, flux, conj=conj))




# ---- Dispatcher + bulk F-terms -------------------------------------------

@partial(jit, static_argnums=(6, 7, 8))   # mode, apply_correction, conj
def compute_zcf(self,
                z_bulk,
                cz_bulk,
                tau,
                ctau,
                flux,
                mode="manual",
                apply_correction=False,
                conj=False):
    r"""
    **Description:**
    Unified solver for the conifold modulus :math:`z_{\rm cf}` after integrating
    out the heavy direction.

    The leading-order F-term equation is
    ``log_prefactor · ln(-2πi·z_cf) + W_log_coeff = 0``, optionally augmented
    by the Kähler-covariant correction ``log_coeff_K_corr``. The exponential
    inversion is delegated to :func:`_zcf_from_log_coeff`.

    Args:
        z_bulk (Array): Bulk moduli vector.
        cz_bulk (Array): Complex-conjugate bulk moduli vector.
        tau (Array): Axio-dilaton.
        ctau (Array): Complex-conjugate axio-dilaton.
        flux (Array): Flux vector.
        mode ({"manual", "autodiff"}): Which :func:`W_log_coeff` route to use.
            ``"manual"`` is the closed-form ``kappa`` / ``a_matrix`` / ``b_vector``
            + ``Li`` assembly (replaces the old ``compute_zcf_explicit``);
            ``"autodiff"`` is the css-side ``F_coniLCS_exp`` route (replaces the
            old ``compute_zcf_compact``). Both must agree numerically.
        apply_correction (bool): If True, add :func:`log_coeff_K_corr` to
            ``W_log_coeff`` before exponentiating (replaces the old
            ``compute_zcf_correction`` multiplicative factor).
        conj (bool): Take the complex-conjugate branch.

    Returns:
        complex: Value of the conifold modulus :math:`z_{\rm cf}`.
    """

    if conj:
        W = self.W_log_coeff(cz_bulk, ctau, flux, mode=mode, conj=conj)
    else:
        W = self.W_log_coeff(z_bulk, tau, flux, mode=mode, conj=conj)

    if apply_correction:
        W = W + self.log_coeff_K_corr(z_bulk, cz_bulk, tau, ctau, flux, conj=conj)

    if conj:
        return self._zcf_from_log_coeff(W, ctau, flux, conj=conj)
    else:
        return self._zcf_from_log_coeff(W, tau, flux, conj=conj)


@partial(jit, static_argnums=(3, 4, 5))   # mode, apply_correction, conj
def compute_zcf_x(self, x_bulk, flux, mode="manual", apply_correction=False, conj=False):
    r"""
    **Description:**
    Unified solver for the conifold modulus :math:`z_{\rm cf}` after integrating
    out the heavy direction.

    The leading-order F-term equation is
    ``log_prefactor · ln(-2πi·z_cf) + W_log_coeff = 0``, optionally augmented
    by the Kähler-covariant correction ``log_coeff_K_corr``. The exponential
    inversion is delegated to :func:`_zcf_from_log_coeff`.

    Args:
        x_bulk (Array): Full real-coord vector containing the bulk moduli and axio-dilaton (without the conifold modulus).
        flux (Array): Flux vector.
        mode ({"manual", "autodiff"}): Which :func:`W_log_coeff` route to use.
            ``"manual"`` is the closed-form ``kappa`` / ``a_matrix`` / ``b_vector``
            + ``Li`` assembly (replaces the old ``compute_zcf_explicit``);
            ``"autodiff"`` is the css-side ``F_coniLCS_exp`` route (replaces the
            old ``compute_zcf_compact``). Both must agree numerically.
        apply_correction (bool): If True, add :func:`log_coeff_K_corr` to
            ``W_log_coeff`` before exponentiating (replaces the old
            ``compute_zcf_correction`` multiplicative factor).
        conj (bool): Take the complex-conjugate branch.

    Returns:
        complex: Value of the conifold modulus :math:`z_{\rm cf}`.
    """

    z_bulk, cz_bulk, tau, ctau = self._convert_real_to_complex(x_bulk)

    return self.compute_zcf(z_bulk, cz_bulk, tau, ctau , flux, mode=mode, apply_correction=apply_correction, conj=conj)




@partial(jax.jit, static_argnums=(3, 4, 5))   # mode, apply_correction, conj
def zcf_handling(self, x_bulk, flux, mode=None, apply_correction=False, conj=False):
    r"""
    **Description:**
    Handle the conifold modulus depending on the mode chosen.

    Args:
        x_bulk (Array): Full real-coord vector containing the bulk moduli
            and axio-dilaton (without the conifold modulus).
        flux (Array): Flux vector.
        mode (str | None, optional): If ``None``, return ``[0, 0, *x_bulk]``
            (no z_cf solve). Otherwise one of ``{"manual", "autodiff", "pfv"}``,
            forwarded to :func:`compute_zcf_x`. Defaults to ``None``.
        apply_correction (bool, optional): Forwarded to :func:`compute_zcf_x`.
            Adds the Kähler-covariant ``log_coeff_K_corr`` to the log
            coefficient before exponentiating. Ignored when ``mode is None``.
        conj (bool, optional): Forwarded to :func:`compute_zcf_x`. Take the
            complex-conjugate branch. Ignored when ``mode is None``.

    Returns:
        Array: Real variables including the conifold modulus.
    """
    if not self.lcs_tree.conifold_basis:
        raise NotImplementedError("zcf_handling is only implemented for conifold_basis=True.")

    if mode is None:
        xcz = jnp.zeros(2)
    else:
        zcf = self.compute_zcf_x(
            x_bulk, flux,
            mode=mode, apply_correction=apply_correction, conj=conj,
        )
        xcz = jnp.array([zcf.real, zcf.imag])

    return jnp.append(xcz, x_bulk)


# NOTE: ``DWbulk_x`` and ``dDWbulk_x`` were hard-removed on 2026-05-01 after
# ``private/promotion/vacuum_promotion.py`` migrated to the
# :class:`jaxvacua.freezer.ConifoldFreezer` interface (``DW_x_light`` /
# ``dDW_x_light``).
