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

"""Stateless flux-vector and solution-classification helpers.

Purpose
-------
Provide small, reusable helpers for converting flux representations,
normalising candidate solutions and applying post-processing checks outside
the main ``FluxEFT`` and ``FluxVacuaFinder`` classes.

Main public API
---------------
- PFV conversions: ``flux_to_pfv``, ``pfv_to_flux`` and ``pfv_to_moduli``.
- Solution identity and classification: ``dedup_key``,
  ``classify_solution`` and ``is_physical``.
- Fundamental-domain mapping: ``map_to_fd`` and class methods that delegate
  to this module.

Design notes
------------
Functions are written to be attached to or called from EFT/search objects and
therefore take ``self`` as their first argument.  Keeping this code stateless
reduces coupling between solver workflows and post-hoc analysis.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from typing import Tuple, Any
from jax import jit
from functools import partial

from .util import auto_vmap

# ---------------------------------------------------------------------------
# PFV algebra
# ---------------------------------------------------------------------------

@partial(jit, static_argnums = ())
def flux_to_pfv(
                self, 
                flux: Array, 
                ) -> Tuple[Array,Array]:
    r"""
    **Description:**
    Returns M- and K-vector specifying PFV from full flux vector.
    
    Args:
        flux (Array): Flux-vector.           
    
    Returns:
        M (Array): M-vector.
        K (Array): K-vector.  
    
    """
    _,f2,h1,_ = self._split_fluxes(flux)

    Mvec = f2[1:]
    Kvec = h1[1:]

    return Mvec,Kvec

@partial(jit, static_argnums = ())
def pfv_to_flux(
                self, 
                M: Array, 
                K: Array
                ) -> Array:
    r"""
    **Description:**
    Returns full flux vector from M- and K-vector specifying PFV.
    
    Args:
        M (Array): M-vector.
        K (Array): K-vector.            
    
    Returns:
        Array: Flux vector.
    
    """

    a = self.lcs_tree.a_matrix
    b = self.lcs_tree.b_vector

    if "coniLCS" in self.periods.limit:
        #tmp = jnp.zeros(b.shape[0])
        #tmp = tmp.at[0].set(1.)
        if self.lcs_tree.conifold_basis:
            tmp = self.lcs_tree.conifold.conifold_curve0
        else:
            tmp = self.lcs_tree.conifold.conifold_curve
            
        b = b+tmp*self.lcs_tree.conifold.ncf/24

    f0 = M@b
    f2 = (a.T@M)
    f2 = jnp.append(jnp.ones(1)*f0,f2)
    f2 = jnp.append(f2,jnp.zeros(1))
    f = jnp.append(f2,M)

    h = jnp.append(jnp.zeros(1),K)
    h = jnp.append(h,jnp.zeros(K.shape[0]+1))

    return jnp.append(f,h)


@partial(jit, static_argnums = ())
def pfv_to_moduli(
                self, 
                M: Array, 
                K: Array,
                tau: complex
                ) -> Array:
    r"""
    **Description:**
    Returns values of the complex structure moduli at the level of the PFV.
    
    Args:
        M (Array): M-vector.
        K (Array): K-vector.
        tau (complex): Axio-dilaton value.
    
    Returns:
        Array: Values of complex structure moduli.
    
    """

    if "coniLCS" in self.periods.limit:
        gs = 1/tau.imag
        c0 = tau.real
        ncf = self.lcs_tree.conifold.ncf
        # Conifold/bulk split (see ``_pfv_coni_split``).
        Nhat, Ncb, Kbulk, Kcf, Mcf = _pfv_coni_split(self, M, K)

        phat   = jnp.linalg.inv(Nhat) @ Kbulk
        zbulk  = phat * tau
        Kprime = Kcf - Ncb @ phat

        phase_comb = -1j*(+c0*Kprime)
        radial = Kprime/gs
        zcf = -1/(2*jnp.pi*1j)*jnp.exp(2*jnp.pi/ncf/Mcf*(phase_comb+radial))

        if self.lcs_tree.conifold_basis:
            z0 = jnp.append(jnp.ones(1)*zcf, zbulk)
        else:
            # reconstruct the full modulus: z = z_cf·e_q + bulk_embedding·z_bulk
            e_q = self.lcs_tree.conifold.embedding
            be  = self.lcs_tree.conifold.bulk_embedding
            z0 = zcf*jnp.asarray(e_q, dtype=zbulk.dtype) + be @ zbulk
    else:
        p = jnp.linalg.inv(_N_matrix(self, M))@K
        z0 = p*tau

    return z0


# ---------------------------------------------------------------------------
# PFV algebra: N-matrix, p-vector, condition checker and analytic racetrack.
#
# The PFV and conifold-PFV conditions follow arXiv:2512.17095 §6.2 eq (6.27) and
# §6.4 eq (7.28); the 2-term racetrack follows §6.2 eqs (6.28), (6.38), (6.40).
# The public functions ``N_matrix``, ``pfv_p_vector``, ``pfv_conditions`` and
# ``pfv_racetrack`` are attached to ``FluxEFT`` and accept batches of ``(M, K)``.
# ---------------------------------------------------------------------------


def _pfv_guard(self) -> None:
    r"""
    **Description:**
    Reject models for which the PFV algebra is undefined (one-modulus /
    hypergeometric).

    Raises:
        NotImplementedError: If ``h12 < 2`` or the model is of hypergeometric
            type.
    """
    lt = self.lcs_tree
    model_type = getattr(lt, "model_type", None)
    if int(lt.h12) < 2:
        raise NotImplementedError(
            "PFV algebra requires h12 >= 2; one-modulus / hypergeometric "
            f"models are unsupported (got h12={lt.h12}, model_type={model_type})."
        )
    if model_type == "hypergeometric":
        raise NotImplementedError(
            "PFV algebra is not defined for hypergeometric (one-modulus) models."
        )


def _N_matrix(self, M: Array) -> Array:
    r"""
    **Description:**
    N-matrix :math:`N_{ab} = \kappa_{abc}\, M^c`.

    Args:
        M (Array): M-vector.

    Returns:
        Array: The :math:`(h^{12}, h^{12})` N-matrix as a float array.
    """
    intnums = jnp.asarray(self.lcs_tree.intnums, dtype=float)
    return intnums @ jnp.asarray(M, dtype=float)


def _pfv_coni_split(self, M: Array, K: Array) -> Tuple[Array, Array, Array, Array, Array]:
    r"""
    **Description:**
    Basis-aware conifold/bulk split of the flux data.

    .. admonition:: Details
        :class: dropdown

        In the aligned basis (``conifold_basis=True``) the conifold sits at
        index 0 and the split is plain slicing; otherwise it is projected via
        the conifold embedding ``e_q`` and ``bulk_embedding``, with ``K``
        covariant and ``M`` contravariant through the conifold curve ``q``.

    Args:
        M (Array): M-vector.
        K (Array): K-vector.

    Returns:
        tuple: ``(Nhat, Ncb, Kbulk, Kcf, Mcf)`` — the bulk-bulk N-block, the
        conifold-bulk row, the bulk K-flux, and the conifold K- and M-fluxes.
    """
    lt = self.lcs_tree
    N = _N_matrix(self, M)
    Mf = jnp.asarray(M, dtype=float)
    Kf = jnp.asarray(K, dtype=float)
    if lt.conifold_basis:
        return N[1:, 1:], N[0, 1:], Kf[1:], Kf[0], Mf[0]
    e_q = jnp.asarray(lt.conifold.embedding, dtype=float)
    q = jnp.asarray(lt.conifold.conifold_curve, dtype=float)
    be = jnp.asarray(lt.conifold.bulk_embedding, dtype=float)
    return be.T @ N @ be, e_q @ N @ be, Kf @ be, Kf @ e_q, Mf @ q


def _pfv_flat_direction(self, M: Array, K: Array) -> Array:
    r"""
    **Description:**
    Full-length linear flat direction :math:`p^a` such that :math:`z^a = p^a\tau`.
    For LCS this is :math:`p = N^{-1}K`; for coniLCS it is :math:`p^a=(0,\hat{p})`
    (the conifold slot is zero, as ``z_cf`` is the separate exponential throat).
    Returns ``NaN`` when ``N`` is singular.

    Args:
        M (Array): M-vector.
        K (Array): K-vector.

    Returns:
        Array: The length-``h12`` flat-direction vector :math:`p^a`.
    """
    lt = self.lcs_tree
    if "coniLCS" in self.periods.limit:
        Nhat, Ncb, Kbulk, Kcf, Mcf = _pfv_coni_split(self, M, K)
        det_ok = jnp.abs(jnp.linalg.det(Nhat)) > 0.5
        phat = jnp.where(det_ok, jnp.linalg.inv(Nhat) @ Kbulk, jnp.nan)
        if lt.conifold_basis:
            return jnp.concatenate([jnp.zeros(1, dtype=float), phat])
        be = jnp.asarray(lt.conifold.bulk_embedding, dtype=float)
        return be @ phat
    N = _N_matrix(self, M)
    det_ok = jnp.abs(jnp.linalg.det(N)) > 0.5
    return jnp.where(det_ok, jnp.linalg.inv(N) @ jnp.asarray(K, dtype=float), jnp.nan)


def _N_matrix_public(self, M: Array) -> Array:
    r"""
    **Description:**
    The N-matrix :math:`N_{ab} = \kappa_{abc}\, M^c`.

    Args:
        M (Array): M-vector.

    Returns:
        Array: The :math:`(h^{12}, h^{12})` N-matrix.
    """
    _pfv_guard(self)
    return _N_matrix(self, M)


def _pfv_p_vector(self, M: Array, K: Array) -> Array:
    r"""
    **Description:**
    The PFV p-vector.  For LCS this is :math:`p = N^{-1}K` (length ``h12``);
    for coniLCS it is the bulk-only :math:`\hat{p} = \hat{N}^{-1} K_{\rm bulk}`
    (length ``h12-1``), for which :math:`z = \hat{p}\,\tau` holds only in the
    bulk (``z_cf`` is the separate exponential throat).

    Args:
        M (Array): M-vector.
        K (Array): K-vector.

    Returns:
        Array: The p-vector (length ``h12`` for LCS, ``h12-1`` for coniLCS).
    """
    _pfv_guard(self)
    if "coniLCS" in self.periods.limit:
        Nhat, Ncb, Kbulk, Kcf, Mcf = _pfv_coni_split(self, M, K)
        return jnp.linalg.inv(Nhat) @ Kbulk
    return jnp.linalg.inv(_N_matrix(self, M)) @ jnp.asarray(K, dtype=float)


def _pfv_conditions(self, M: Array, K: Array) -> dict:
    r"""
    **Description:**
    Evaluate the perturbatively-flat-vacuum conditions of arXiv:2512.17095 for
    the flux quanta ``(M, K)``; accepts batches.

    .. admonition:: Details
        :class: dropdown

        - **LCS**, Eq. (6.27) with :math:`N_{ab}=\kappa_{abc}M^c` and
          :math:`p^a=(N^{-1}K)^a`: :math:`\det N\neq 0`; :math:`p\in\mathcal{K}_X`
          (Kähler-cone interior); :math:`K_a p^a=0`;
          :math:`\tilde{a}_{ab}M^b\in\mathbb{Z}`;
          :math:`\tilde{c}_a M^a\in 24\mathbb{Z}`.
        - **coniLCS**, Eq. (7.28) with :math:`p^a=(0,\hat{p})`,
          :math:`\hat{p}=\hat{N}^{-1}K_{\rm bulk}`: :math:`\det\hat{N}\neq 0`;
          :math:`\hat{p}\in\mathcal{K}_{\rm cf}` (the facet where the conifold
          curve collapses); :math:`K\cdot p=K_{\rm bulk}\cdot\hat{p}=0`;
          :math:`M_{\rm cf}\neq 0`; the same integrality conditions (with the
          conifold ``b'`` shift).
        - **Both** — tadpole (``pfvs`` condition #5):
          :math:`0\le -\tfrac12 M\cdot K\le Q_{D3}`.

        A flux that does not define a PFV (singular ``N``) is reported with
        ``det N != 0`` false and ``p`` set to ``NaN``.

    Args:
        M (Array): M-vector.
        K (Array): K-vector.

    Returns:
        dict: Mapping ``name -> (ok, value)`` where ``ok`` is a boolean array and
        ``value`` the underlying quantity; ``'p'`` carries the p-vector.
    """
    _pfv_guard(self)
    lt = self.lcs_tree
    Mf = jnp.asarray(M, dtype=float)
    Kf = jnp.asarray(K, dtype=float)
    coni = "coniLCS" in self.periods.limit
    H = getattr(lt, "hyperplanes", None)
    out = {}

    if coni:
        Nhat, Ncb, Kbulk, Kcf, Mcf = _pfv_coni_split(self, M, K)
        det = jnp.linalg.det(Nhat)
        det_ok = jnp.abs(det) > 0.5
        phat = jnp.where(det_ok, jnp.linalg.inv(Nhat) @ Kbulk, jnp.nan)
        out["det N!=0"] = (det_ok, det)
        out["p"] = (det_ok, phat)
        Kp = Kbulk @ phat                                    # K_a p^a with p^a=(0,p_hat)
        out["K.p==0"] = (det_ok & (jnp.abs(Kp) < 1e-9), Kp)
        out["Mcf!=0"] = (jnp.abs(Mcf) > 0.5, Mcf)
        if H is not None:
            H = jnp.asarray(H, dtype=float)
            if lt.conifold_basis:
                p_full = jnp.concatenate([jnp.zeros(1, dtype=float), phat])
            else:
                be = jnp.asarray(lt.conifold.bulk_embedding, dtype=float)
                p_full = be @ phat
            hp = H @ p_full
            out["p in K_cf"] = (det_ok & jnp.all(hp >= -1e-9), hp)
    else:
        N = _N_matrix(self, M)
        det = jnp.linalg.det(N)
        det_ok = jnp.abs(det) > 0.5
        p = jnp.where(det_ok, jnp.linalg.inv(N) @ Kf, jnp.nan)
        out["det N!=0"] = (det_ok, det)
        out["p"] = (det_ok, p)
        Kp = Kf @ p
        out["K.p==0"] = (det_ok & (jnp.abs(Kp) < 1e-9), Kp)
        if H is not None:
            H = jnp.asarray(H, dtype=float)
            hp = H @ p
            out["p in K_X"] = (det_ok & jnp.all(hp > 1e-9), hp)

    # Integrality (both limits), mirroring pfv_to_flux (a.T @ M; conifold b' shift).
    a = jnp.asarray(lt.a_matrix, dtype=float)
    b = jnp.asarray(lt.b_vector, dtype=float)
    if coni:
        tmp = jnp.asarray(
            lt.conifold.conifold_curve0 if lt.conifold_basis else lt.conifold.conifold_curve,
            dtype=float,
        )
        b = b + tmp * float(lt.conifold.ncf) / 24.0
    aTM = a.T @ Mf
    bM = b @ Mf
    out["a.T@M in Z"] = (jnp.all(jnp.abs(aTM - jnp.round(aTM)) < 1e-9), aTM)
    out["24*b@M in 24Z"] = (jnp.abs(bM - jnp.round(bM)) < 1e-9, 24.0 * bM)

    # Tadpole (pfvs README condition #5): 0 <= Qflux <= Q_D3, jaxvacua Qflux = -M.K/2.
    Qflux = -0.5 * (Mf @ Kf)
    Q_D3 = float(self.Q())
    out["tadpole 0<=Qflux<=Q"] = ((Qflux >= -1e-9) & (Qflux <= Q_D3 + 1e-9), Qflux)
    return out


def _pfv_racetrack(self, M: Array, K: Array) -> dict:
    r"""
    **Description:**
    Leading-order analytic estimate of the PFV effective-theory vacuum from the
    2-term racetrack (arXiv:2512.17095 §6.2 eqs 6.28, 6.38, 6.40).

    .. admonition:: Details
        :class: dropdown

        Series from the GV data:
        :math:`W_{\rm exp}=\sum_i c_i\,e^{2\pi i\tau e_i}` with exponents
        :math:`e_i=q_i\cdot p` (grouped over equal exponents) and coefficients
        :math:`c_i=\sum_q n_q\,(M\cdot q)`.  ``tau0`` and ``W0`` are evaluated in
        logarithmic form to avoid underflow of the exponentially small
        superpotential; the prefactor :math:`1/(2^{3/2}\pi^{5/2})` matches
        ``self.W(normalise=True)``.  This is a leading-order estimate (the
        2-term-racetrack extremum, slightly displaced from the full F-term
        solution in ``tau``); use it for fast scanning and seeding.

    Args:
        M (Array): M-vector.
        K (Array): K-vector.

    Returns:
        dict: ``{tau0, W0, log10_W0, gs, valid}`` — the racetrack axio-dilaton
        VEV, the flux superpotential and its base-10 logarithm, the string
        coupling ``gs = 1/Im(tau0)``, and a validity flag (``|c1| > |c0|`` with
        at least two nonzero terms).
    """
    _pfv_guard(self)
    lt = self.lcs_tree
    p_full = _pfv_flat_direction(self, M, K)
    q = jnp.asarray(lt.gv_charges, dtype=float)
    n = jnp.asarray(lt.gv_invariants, dtype=float)
    Mf = jnp.asarray(M, dtype=float)
    Nq = q.shape[0]

    e = q @ p_full
    c = n * (q @ Mf)
    pos = e > 1e-9
    e_m = jnp.where(pos, e, jnp.inf)
    c_m = jnp.where(pos, c, 0.0)

    order = jnp.argsort(e_m)
    es, cs = e_m[order], c_m[order]
    same = jnp.concatenate([jnp.array([False]), jnp.isclose(es[1:], es[:-1], atol=1e-9)])
    seg = jnp.cumsum(jnp.where(same, 0, 1)) - 1
    c_grp = jax.ops.segment_sum(cs, seg, num_segments=Nq)
    e_grp = jax.ops.segment_min(es, seg, num_segments=Nq)

    good = (jnp.abs(c_grp) > 1e-30) & jnp.isfinite(e_grp)
    e_pick = jnp.where(good, e_grp, jnp.inf)
    o2 = jnp.argsort(e_pick)
    i0, i1 = o2[0], o2[1]
    e0, c0 = e_grp[i0], c_grp[i0]
    e1, c1 = e_grp[i1], c_grp[i1]
    valid = (jnp.sum(good) >= 2) & (jnp.abs(c1) > jnp.abs(c0))

    mag = jnp.log(jnp.abs((c0 * e0) / (c1 * e1)))
    mag = jnp.where(c0 * c1 > 0, mag + 1j * jnp.pi, mag.astype(complex))
    tau0 = mag * 1j / (2 * jnp.pi * (e0 - e1))
    tau0 = jnp.where(valid, tau0, jnp.nan + 1j * jnp.nan)

    pref = 1.0 / ((2.0 ** 1.5) * (jnp.pi ** 2.5))
    base = jnp.abs(-(c0 * e0) / (c1 * e1))
    power = e1 / (e1 - e0)
    log10_W0 = (power * jnp.log10(base)
                + jnp.log10(jnp.abs(c1 * (e0 - e1) / e0))
                + jnp.log10(pref))
    log10_W0 = jnp.where(valid, log10_W0, jnp.nan)
    W0 = jnp.where(valid, 10.0 ** log10_W0, jnp.nan)
    gs = jnp.where(valid, 1.0 / jnp.imag(tau0), jnp.nan)
    return {"tau0": tau0, "W0": W0, "log10_W0": log10_W0, "gs": gs, "valid": valid}


# Public PFV algebra, vectorised over ``(M, K)``.
N_matrix = auto_vmap(M=1, validate_shapes=False)(_N_matrix_public)
pfv_p_vector = auto_vmap(M=1, K=1, validate_shapes=False)(_pfv_p_vector)
pfv_conditions = auto_vmap(M=1, K=1, validate_shapes=False)(_pfv_conditions)
pfv_racetrack = auto_vmap(M=1, K=1, validate_shapes=False)(_pfv_racetrack)


# ---------------------------------------------------------------------------
# Streaming deduplication key (single-vacuum hashable key)
# ---------------------------------------------------------------------------
# Distinct from ``FluxVacuaFinder.deduplicate_vacua``, which is a *batched*
# jnp.unique pass over an entire solution set.  ``dedup_key`` builds a
# hashable key for *one* ``(flux, moduli, tau)`` triple, suitable for
# incremental Python ``set`` membership tests inside a sampling loop.


def dedup_key(
    moduli: Any,
    tau: Any,
    fluxes: Any,
    n_digits: int = 6,
) -> Tuple[bytes, bytes, bytes, Tuple[float, float]]:
    r"""
    **Description:**
    Build a hashable dedup key from a single ``(moduli, tau, fluxes)`` triple.

    .. admonition:: Details
        :class: dropdown

        The ``fluxes`` vector is rounded to integers and packed as raw bytes
        (it represents an integer flux quantum in practice). The continuous
        ``moduli`` and ``tau`` are rounded to ``n_digits`` decimal places to
        absorb numerical noise from solver convergence. The result is
        hashable, so it can be used as a Python ``set`` element or ``dict``
        key.

        Note that a single flux vector can admit multiple distinct critical
        points (because the moduli equations are non-linear), so the key
        deliberately includes the moduli + ``tau``, not just the flux.

        Distinct from :func:`FluxVacuaFinder.deduplicate_vacua`, which is a
        batched ``jnp.unique`` pass: this helper is for streaming /
        incremental dedup of one solution at a time.

    Args:
        moduli (Array): Complex structure moduli. Real and imaginary parts
            are each rounded to ``n_digits`` decimal places.
        tau (complex): Axio-dilaton. Real and imaginary parts are each
            rounded to ``n_digits`` decimal places.
        fluxes (Array): Flux vector. Real part is rounded to ``int32``;
            imaginary part is ignored.
        n_digits (int, optional): Decimal precision for rounding continuous
            components. Defaults to ``6``.

    Returns:
        tuple: 4-tuple ``(flux_bytes, moduli_re_bytes, moduli_im_bytes, tau_tuple)``,
        hashable for use in a ``set`` / ``dict``.

    **Example:**

    .. code-block:: python

        seen = set()
        for vac in candidate_vacua:
            k = dedup_key(vac.moduli, vac.tau, vac.fluxes)
            if k in seen:
                continue
            seen.add(k)
            ...

    """
    f_part = np.round(np.asarray(fluxes).real).astype(np.int32).tobytes()
    z_re   = np.around(np.asarray(moduli).real, n_digits)
    z_im   = np.around(np.asarray(moduli).imag, n_digits)
    t_part = (round(complex(tau).real, n_digits), round(complex(tau).imag, n_digits))
    return (f_part, z_re.tobytes(), z_im.tobytes(), t_part)


# ---------------------------------------------------------------------------
# Single-point critical-point classification
# ---------------------------------------------------------------------------


def classify_solution(
    model: Any,
    x: Any,
    flux: Any,
    noscale: bool = True,
    min_tol: float = 1e-6,
) -> dict:
    r"""
    **Description:**
    Classify a converged critical-point candidate ``(x, flux)``.

    .. admonition:: Details
        :class: dropdown

        Computes the scalar potential :math:`V`, the F-term norm
        :math:`|DW|=\sum_i|D_iW|`, and the Hessian eigenvalues
        :math:`\mathrm{eig}(\partial^2 V)`, then labels the point as

        - SUSY iff :math:`|DW|<10^{-6}` (fixed threshold);
        - minimum iff every Hessian eigenvalue exceeds ``min_tol``,
          otherwise saddle.

        Eager-evaluation Python routine suitable for one-shot post-Newton
        classification; a JIT-vmapped batched variant lives on the finder
        for sample-time use.

    Args:
        model (FluxEFT): Finder instance providing
            :func:`_convert_real_to_complex`, :func:`scalar_potential`,
            :func:`DW`, :func:`ddV_x` and :func:`tadpole`. Passed explicitly
            (rather than via implicit ``self``) so the helper works with any
            ``FluxEFT`` subclass without inheritance assumptions.
        x (Array): Converged real-coord solution vector.
        flux (Array): Flux vector at the candidate point.
        noscale (bool, optional): Pass-through to :func:`scalar_potential`
            and :func:`ddV_x`. Defaults to ``True``.
        min_tol (float, optional): Minimum-classification threshold. A
            point is labelled ``is_minimum=True`` iff every Hessian
            eigenvalue exceeds ``min_tol``. Defaults to ``1e-6``.

    Returns:
        dict: Mapping with keys

            - ``'V'`` (float): Scalar potential at the point.
            - ``'|DW|'`` (float): :math:`\sum_i |D_i W|`.
            - ``'eigenvalues'`` (np.ndarray): Sorted real Hessian eigenvalues.
            - ``'is_susy'`` (bool): ``|DW| < 1e-6``.
            - ``'is_minimum'`` (bool): All eigenvalues ``> min_tol``.
            - ``'Nflux'`` (float): ``|tadpole(flux)|``.

    """
    x_jax  = jnp.array(x)
    fl_jax = jnp.array(flux)

    z, z_c, tau, tau_c = model._convert_real_to_complex(x_jax)
    V       = float(model.scalar_potential(z, z_c, tau, tau_c, fl_jax, noscale=noscale).real)
    DW      = model.DW(z, z_c, tau, tau_c, fl_jax)
    dw_norm = float(jnp.sum(jnp.abs(DW)))
    ddV     = model.ddV_x(x_jax, fl_jax, noscale=noscale)
    eigs    = np.sort(np.asarray(jnp.linalg.eigvalsh(ddV).real))
    tad     = abs(float(jnp.real(model.tadpole(fl_jax))))

    return {
        'V':           V,
        '|DW|':        dw_norm,
        'eigenvalues': eigs,
        'is_susy':     dw_norm < 1e-6,
        'is_minimum':  bool(np.all(eigs > min_tol)),
        'Nflux':       tad,
    }


# ---------------------------------------------------------------------------
# Multi-tier physicality cascade
# ---------------------------------------------------------------------------


def is_physical(
    model: Any,
    sampler: Any,
    x: Any,
    moduli_max: float = None,
    verbose: bool = False,
) -> bool:
    r"""
    **Description:**
    Decide whether the converged candidate point ``x`` lies in the physical
    region of moduli space.

    .. admonition:: Details
        :class: dropdown

        Five checks are applied in increasing cost order.  Hyperplane failure
        rejects immediately, but hyperplane success is not treated as
        conclusive: Kähler-metric positivity is still checked.

        +---+-----------------------------+--------+-----------------------------+
        | # | Check                       | Cost   | Rejected iff                |
        +===+=============================+========+=============================+
        | 0 | Runaway bound               | O(1)   | ``max(|moduli|, |tau|) >    |
        |   |                             |        | moduli_max`` (if provided)  |
        +---+-----------------------------+--------+-----------------------------+
        | 1 | Dilaton floor               | O(1)   | ``Im(tau) <= s_lower``      |
        +---+-----------------------------+--------+-----------------------------+
        | 2 | Kähler-cone hyperplanes     | O(h12²)| ``hp @ Im(z)`` not all > 0  |
        |   | (if available; early-return)|        |                             |
        +---+-----------------------------+--------+-----------------------------+
        | 3 | Kähler metric positivity    | O(h12³)| any ``eig(K) <= 0``         |
        |   | (fallback; early-return)    |        |                             |
        +---+-----------------------------+--------+-----------------------------+
        | 4 | Basic ``Im(z_i) > 0``       | O(h12) | any ``Im(z_i) <= 0``        |
        +---+-----------------------------+--------+-----------------------------+

        If ``model.lcs_tree.hyperplanes`` is set, check 2 provides the exact
        linear cone test.  Check 3 is still tried afterwards; if it raises,
        the function falls through to check 4 as a last-resort sanity check.

    Args:
        model (FluxEFT): Finder instance providing
            :func:`_convert_real_to_complex`, :attr:`lcs_tree` and (for
            check 3) :func:`kahler_metric`.
        sampler (data_sampler): Sampler instance providing the
            ``s_lower`` attribute (dilaton floor). Passed explicitly so the
            helper is decoupled from any particular finder configuration.
        x (Array): Converged real-coord solution vector.
        moduli_max (float, optional): If not ``None``, reject points with
            ``max(|moduli|, |tau|) > moduli_max`` (runaway check). Defaults
            to ``None`` (no runaway check).
        verbose (bool, optional): If ``True``, print which check rejected
            the point (useful for debugging "why was my candidate filtered?").
            Defaults to ``False``.

    Returns:
        bool: ``True`` if ``x`` is in the physical region by all applicable
        checks, ``False`` otherwise.

    """
    x_jax = jnp.array(x)
    moduli, _, tau, _ = model._convert_real_to_complex(x_jax)
    im_z = jnp.imag(moduli)
    s    = float(jnp.imag(tau))

    # Check 0: runaway bound (cheapest)
    if moduli_max is not None:
        max_abs = float(jnp.max(jnp.abs(jnp.append(moduli, tau))))
        if max_abs > moduli_max:
            if verbose:
                print(f"[is_physical] REJECT check 0 (runaway): "
                      f"max(|moduli|, |tau|) = {max_abs:.3e} > {moduli_max}")
            return False

    # Check 1: dilaton floor
    s_lo = getattr(sampler, 's_lower', 0.0)
    if s <= s_lo:
        if verbose:
            print(f"[is_physical] REJECT check 1 (dilaton floor): "
                  f"Im(tau) = {s:.4f} <= s_lower = {s_lo:.4f}")
        return False

    # Check 2: Kähler-cone hyperplanes.  Passing the linear cone test is
    # necessary but not sufficient in numerical/high-dimensional scans:
    # near walls or with instanton corrections the Kähler metric can still be
    # indefinite.  Therefore this check rejects on failure but does not accept
    # early on success.
    hp = getattr(model.lcs_tree, 'hyperplanes', None)
    if hp is not None:
        dots = jnp.array(hp) @ im_z
        ok   = bool(jnp.all(dots > 0))
        if verbose and not ok:
            print(f"[is_physical] REJECT check 2 (Kähler-cone hyperplanes): "
                  f"hp @ Im(z) = {np.asarray(dots)} (need all > 0)")
        if not ok:
            return False

    # Check 3: Kähler-metric positivity.
    try:
        moduli_c = jnp.conj(moduli)
        tau_c    = jnp.conj(tau)
        K        = model.kahler_metric(moduli, moduli_c, tau, tau_c)
        eigs     = jnp.linalg.eigvalsh(K)
        ok       = bool(jnp.all(eigs > 0))
        if verbose and not ok:
            print(f"[is_physical] REJECT check 3 (Kähler metric): "
                  f"eigvals(K) = {np.asarray(eigs)} (need all > 0)")
        elif verbose:
            print(f"[is_physical] ACCEPT (checks 0,1,2,3 passed; Kähler-metric check)")
        return ok
    except Exception as e:
        if verbose:
            print(f"[is_physical] check 3 (Kähler metric) raised — falling back to check 4: {e}")

    # Check 4: basic Im(z_i) > 0 (last-resort sanity)
    ok = bool(jnp.all(im_z > 0))
    if verbose:
        if ok:
            print(f"[is_physical] ACCEPT (checks 0,1,4 passed; basic Im(z)>0 check)")
        else:
            print(f"[is_physical] REJECT check 4 (basic Im(z)>0): "
                  f"Im(z) = {np.asarray(im_z)}")
    return ok


# ---------------------------------------------------------------------------
# Fundamental-domain mapping at the numpy boundary
# ---------------------------------------------------------------------------


def map_to_fd(
    model: Any,
    moduli: Any,
    tau: Any,
    fluxes: Any,
    enabled: bool = False,
) -> Tuple[Any, Any, Any]:
    r"""
    **Description:**
    Optionally map ``(moduli, tau, fluxes)`` to the
    :math:`\text{SL}(2,\mathbb{Z}) \times` monodromy fundamental domain via
    :func:`FluxEFT.map_to_fd`, handling the numpy/JAX boundary so call sites
    in eager Python loops stay clean.

    .. admonition:: Details
        :class: dropdown

        The real mathematical work
        (:math:`\text{SL}(2,\mathbb{Z})` action on :math:`\tau` + monodromy
        shifts on the moduli axions + boundary snap) is delegated to
        :func:`model.map_to_fd`. This helper adds three pieces of
        bookkeeping on top:

        1. **Opt-out short-circuit**: if ``enabled=False`` the inputs are
           returned unchanged. Lets call sites be written without an outer
           ``if`` block.
        2. **numpy ↔ JAX marshalling**: inputs are cast to JAX
           (``jnp.asarray``, ``complex``); outputs come back as numpy /
           Python scalars (``np.asarray`` with ``int32`` fluxes,
           ``complex`` tau). Intended for eager-Python sampling loops
           that consume the result on the numpy side.
        3. **Output dtype contract**: ``fluxes`` is returned as ``int32``
           (they represent integer flux quanta).

    Args:
        model (FluxEFT): Finder instance providing :func:`map_to_fd`. Passed
            explicitly so the helper works with any ``FluxEFT`` subclass.
        moduli (Array): Complex structure moduli.
        tau (complex): Axio-dilaton.
        fluxes (Array): Flux vector.
        enabled (bool, optional): If ``False``, the helper is a no-op and
            returns the inputs verbatim. If ``True``, the FD mapping is
            applied. Defaults to ``False``.

    Returns:
        Tuple[Array, complex, Array]: ``(moduli_fd, tau_fd, fluxes_fd)``.
        When ``enabled=True`` these are numpy arrays / Python ``complex``;
        when ``enabled=False`` they are the original input objects unchanged.

    """
    if not enabled:
        return moduli, tau, fluxes
    moduli_fd, tau_fd, fluxes_fd = model.map_to_fd(
        jnp.asarray(moduli),
        complex(tau),
        jnp.asarray(fluxes),
    )
    return (
        np.asarray(moduli_fd),
        complex(tau_fd),
        np.asarray(fluxes_fd, dtype=np.int32),
    )


# ===========================================================================
# Optional `pfvs` availability
# ===========================================================================
# The jaxvacua <-> pfvs conversion lives on the topology container as
# ``lcs_tree.to_cydata()`` / ``lcs_tree.to_cydata_kwargs()`` (external PFV
# enumerator, github.com/natemacfadden/pfvs).  pfvs is an OPTIONAL dependency;
# ``has_pfvs()`` reports whether it is importable and ``_import_pfvs`` is the
# cached lazy importer those methods use.

_PFVS = None
_PFVS_CHECKED = False


def _import_pfvs():
    r"""Cached lazy import of the optional ``pfvs`` package (a ``try: import``,
    NOT ``find_spec`` — pfvs eagerly imports latticepts/ortools/flint/matplotlib,
    so a spec can exist yet the import crash).  Returns the module or ``None``."""
    global _PFVS, _PFVS_CHECKED
    if not _PFVS_CHECKED:
        _PFVS_CHECKED = True
        try:
            import pfvs as _p
            _PFVS = _p
        except Exception:                                    # noqa: BLE001
            _PFVS = None
    return _PFVS


def has_pfvs() -> bool:
    r"""
    **Description:**
    Whether the optional :mod:`pfvs` package is importable.

    Returns:
        bool: ``True`` if ``import pfvs`` succeeds.
    """
    return _import_pfvs() is not None
