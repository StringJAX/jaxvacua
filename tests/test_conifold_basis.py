# Copyright 2024 Andreas Schachner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A/B invariance tests for conifold_basis=True vs conifold_basis=False.

Purpose
-------
Verify that the coniLCS machinery gives basis-INVARIANT physics in a general
(non-conifold-aligned) basis.  The same NB13 geometry is built two ways:

- ``M_aligned``  : ``basis_change=L``, ``conifold_basis=True``  (conifold at z^1)
- ``M_general``  : ``basis_change=None``, ``conifold_basis=False`` (conifold = q)

A physical point has moduli ``z_aligned`` in the rotated frame and
``z_general = Lᵀ z_aligned`` in the original frame (``L q = (1,0,0)``).

What is / isn't invariant
-------------------------
- The period vectors of the two bases are related by an INTEGER SYMPLECTIC
  monodromy ``M``: ``Π_aligned = M·Π_general`` with ``Mᵀ Σ M = Σ``.  This is the
  precise statement of basis-consistency (verified to ~1e-13).
- The prepotential ``F`` is therefore NOT a naive invariant: it is the
  symplectic generating function ``½ Xᴵ Fᴵ``, and ``M`` mixes the electric
  (``Xᴵ``) and magnetic (``Fᴵ``) period blocks, so ``F`` shifts.  ``dF`` is NOT
  a quadratic in the moduli ``z`` (checked) — it is the generating-function
  shift.  We therefore do NOT compare ``F``.
- The genuine (gauge-/symplectic-)invariants do match:
  - Kähler potential ``e^{-K} = i·Π̄ᵀ Σ Π`` (``MᵀΣM=Σ``), to ~1e-12.
  - Superpotential ``W = (f−τh)·Σ·Π`` once the fluxes are transported by the
    SAME ``M`` (``f_al = M f_gen``), to ~1e-12.
- ``F_coniLCS_series`` must be finite in both bases.

Design notes
------------
Heavy cytools build at import (gated/skipped if cytools is unavailable).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import jaxvacua as jvc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util import TestCase  # noqa: E402

warnings.filterwarnings("ignore")
jax.config.update("jax_enable_x64", True)

try:
    from cytools import Polytope as _Polytope
    _CYTOOLS_ERR = None
except Exception as _exc:                                                  # noqa: BLE001
    _Polytope = None
    _CYTOOLS_ERR = f"{type(_exc).__name__}: {_exc}"

# NB13 fixture (mirrors test_coniLCS_limit.py).
_POLYTOPE_PTS = np.array([
    [-1, 3, -2, -1], [1, -1, 0, 0], [-1, 0, 0, 1],
    [-1, 0, 0, 0],   [-1, 0, 1, 1], [-1, 0, 2, 0],
    [-1, 0, 1, 0],
])
_L = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 1]])   # L q = (1,0,0)
_Q = np.array([-1, 1, 0])
_NCF, _MAXDEG, _PRANGE = 2, 6, 100
_ALIGNED_POINTS = np.array([
    [0.10 + 2.00j, -0.05 + 2.20j, 0.00 + 2.50j],
    [-0.18 + 1.90j, 0.12 + 2.35j, 0.07 + 2.10j],
    [0.24 + 2.45j, -0.16 + 1.85j, 0.19 + 2.70j],
    [-0.08 + 2.65j, -0.22 + 2.05j, 0.14 + 1.95j],
    [0.03 + 1.75j, 0.18 + 2.55j, -0.20 + 2.30j],
    [0.21 + 2.15j, 0.04 + 1.82j, -0.13 + 2.62j],
    [-0.27 + 2.38j, 0.26 + 2.18j, 0.11 + 1.88j],
    [0.15 + 1.92j, -0.24 + 2.68j, -0.06 + 2.42j],
    [-0.12 + 2.28j, 0.09 + 1.98j, 0.23 + 2.58j],
    [0.28 + 2.72j, -0.11 + 2.08j, -0.17 + 1.78j],
    [-0.04 + 1.86j, -0.19 + 2.48j, 0.25 + 2.20j],
    [0.06 + 2.32j, 0.22 + 1.90j, -0.28 + 2.66j],
], dtype=complex)
_BULK_POINTS = np.array([
    [0.05 + 1.95j, -0.03 + 2.05j],
    [-0.12 + 2.20j, 0.08 + 1.88j],
    [0.17 + 2.35j, -0.09 + 2.10j],
    [-0.04 + 1.82j, -0.16 + 2.28j],
], dtype=complex)
_FLUX_SAMPLES = np.array([
    [1, 0, -2, 3, -1, 2, 0, -3, 2, -1, 1, 0, -2, 3, -1, 1],
    [-3, 2, 1, 0, 4, -1, -2, 1, 0, 3, -4, 2, 1, -1, 2, -3],
    [2, -2, 0, 1, -3, 3, 1, -1, -1, 4, 2, -2, 0, 1, -4, 3],
    [0, 1, -3, 2, 2, -4, 3, 0, 3, 0, -1, 1, -2, 2, 4, -1],
], dtype=float)

_MA = _MG = None
_BUILD_ERR = None


def _build(basis_change, conifold_basis):
    cy = _Polytope(_POLYTOPE_PTS).triangulate().get_cy()
    return jvc.FluxVacuaFinder(
        h12=int(cy.h11()), Q=int(cy.h11()) + int(cy.h12()) + 2,
        use_cytools=True, mirror_cy=cy, ncf=_NCF, use_gvs=True,
        maximum_degree=_MAXDEG, basis_change=basis_change,
        conifold_curve=_Q, limit="coniLCS", prange=_PRANGE,
        conifold_basis=conifold_basis,
    )


if _Polytope is not None:
    try:
        _MA = _build(_L, True)
        _MG = _build(None, False)
    except Exception as _exc:                                              # noqa: BLE001
        _BUILD_ERR = f"{type(_exc).__name__}: {_exc}"

_NEEDS = pytest.mark.skipif(
    _MA is None or _MG is None,
    reason=f"cytools/model unavailable ({_CYTOOLS_ERR or _BUILD_ERR})",
)


def _emK(model, z):
    """Kähler potential e^{-K} = i·Π̄ᵀ Σ Π (symplectic invariant; real)."""
    Pi = model.period_vector(z)
    return complex(1j * (jnp.conj(Pi) @ model.periods.sigma @ Pi))


def _recover_monodromy(MA, MG, Lt, points=_ALIGNED_POINTS):
    """Recover the integer monodromy M with Π_aligned(z') = M·Π_general(Lᵀz').

    The GL(h,ℤ) moduli basis change embeds into Sp(2(h+1),ℤ); M is read off
    numerically from matched period vectors and rounded to the nearest integer.
    """
    PA, PG = [], []
    for z in points:
        z_al = jnp.asarray(z)
        PA.append(np.asarray(MA.period_vector(z_al)))
        PG.append(np.asarray(MG.period_vector(Lt @ z_al)))
    M = (np.array(PA).T @ np.linalg.pinv(np.array(PG).T))
    return np.rint(M.real)


@_NEEDS
class TestConifoldBasisInvariance(TestCase):
    r"""A/B invariance of the coniLCS prepotential layer across bases."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.MA, cls.MG = _MA, _MG
        cls.Lt = jnp.asarray(_L.T, dtype=complex)

    def _points(self, n=8):
        for z in _ALIGNED_POINTS[:n]:
            z_al = jnp.asarray(z)
            yield z_al, self.Lt @ z_al

    def test_setup_flags(self):
        self.assertTrue(bool(self.MA.lcs_tree.conifold_basis))
        self.assertFalse(bool(self.MG.lcs_tree.conifold_basis))
        coni = self.MG.lcs_tree.conifold
        # bulk embedding / projection stored as (h12, h12-1) matrices
        be = np.asarray(coni.bulk_embedding); bp = np.asarray(coni.bulk_projection)
        self.assertEqual(be.shape, (3, 2))
        self.assertEqual(bp.shape, (3, 2))
        # left-inverse relation: bulk_embedding^T @ bulk_projection = I_{h12-1}
        self.assertTrue(np.allclose(be.T @ bp, np.eye(2)))
        # q·e_q = 1 and q·bulk_embedding = 0 (conifold curve orthogonal to bulk)
        self.assertEqual(int(np.rint(_Q @ np.asarray(coni.embedding))), 1)
        self.assertTrue(np.allclose(_Q @ be, 0))

    def test_kahler_potential_invariant(self):
        """e^{-K} = i·Π̄ᵀΣΠ matches across bases (the real symplectic invariant)."""
        for z_al, z_gen in self._points():
            ea, eg = _emK(self.MA, z_al), _emK(self.MG, z_gen)
            self.assertTrue(np.isfinite(ea) and np.isfinite(eg))
            rel = abs(ea - eg) / max(abs(ea), 1e-12)
            self.assertLess(rel, 1e-10, f"e^-K mismatch rel={rel:.2e}")

    def test_F_coniLCS_series_finite(self):
        """F_coniLCS_series must be finite in both bases (general NaN cleared)."""
        for z_al, z_gen in self._points():
            self.assertTrue(np.isfinite(complex(self.MA.F_coniLCS_series(z_al))))
            self.assertTrue(np.isfinite(complex(self.MG.F_coniLCS_series(z_gen))))

    def test_zcf_manual_equals_autodiff_general(self):
        """The two independent z_cf-solver routes (closed-form ``manual`` and
        autodiff via ``F_coniLCS_exp``) must agree in BOTH bases.

        This is the gate that caught the bulk projection/embedding bug: in a
        general basis the bulk EMBEDDING (Λ[1:]ᵀ, build a vector / project a
        charge) and the bulk PROJECTION (Λ⁻¹[:,1:], extract bulk coords of a
        period) are different matrices; conflating them made the two routes
        disagree (rel ~5).  With the distinction they agree to ~1e-10.
        """
        for M in (self.MA, self.MG):
            nf = M.n_fluxes
            for flux_arr, z_arr in zip(_FLUX_SAMPLES, _BULK_POINTS):
                flux = jnp.asarray(flux_arr[: 2 * nf])
                z_bulk = jnp.asarray(z_arr[: M.h12 - 1])
                tau = complex(0.1, 8.0)
                wm = complex(M.W_log_coeff(z_bulk, tau, flux, mode="manual"))
                wa = complex(M.W_log_coeff(z_bulk, tau, flux, mode="autodiff"))
                self.assertTrue(np.isfinite(wm) and np.isfinite(wa))
                self.assertLess(abs(wm - wa) / max(abs(wm), 1e-12), 1e-7,
                                f"manual vs autodiff W_log_coeff mismatch ({wm} vs {wa})")

    def test_conifold_flux_projection_convention(self):
        r"""``conifold_fluxes`` (general basis) must split the flux blocks by
        their covariant/contravariant character — the bug that an `e^{-K}` or
        manual==autodiff gate cannot catch (it is common-mode in the solver).

        From ``W = f1·X − f2·𝓕``:
        - ``f1, h1`` pair with electric periods ``X`` ⇒ COVARIANT ⇒ conifold
          component via ``e_q``, bulk via ``bulk_embedding``;
        - ``f2, h2`` pair with magnetic periods ``𝓕`` ⇒ CONTRAVARIANT ⇒ conifold
          component via ``q`` (= ``conifold_curve``), bulk via ``bulk_projection``.

        Getting ``f2,h2`` wrong (using ``e_q`` instead of ``q``) inflates the
        conifold flux by ``q·q`` and breaks the physical ``z_cf`` (verified at
        the arXiv:2009.03312 vacua).
        """
        coni = self.MG.lcs_tree.conifold
        q  = np.asarray(coni.conifold_curve, dtype=float)
        eq = np.asarray(coni.embedding, dtype=float)
        be = np.asarray(coni.bulk_embedding, dtype=float)
        bp = np.asarray(coni.bulk_projection, dtype=float)
        dH = self.MG.dimension_H3
        nf = self.MG.n_fluxes

        flux = _FLUX_SAMPLES[0, : 2 * nf]
        f1, f2 = flux[:dH], flux[dH:nf]
        h1, h2 = flux[nf:nf + dH], flux[nf + dH:]
        (M0, H0, M1, H1, Malpha, Halpha,
         P1, K1, Palpha, Kalpha, P0, K0) = self.MG.conifold_fluxes(jnp.asarray(flux))

        # fundamental (X^0) components are the plain index-0 slices in any basis
        self.assertAlmostEqual(float(M0), float(f2[0]))
        self.assertAlmostEqual(float(P0), float(f1[0]))
        # f2, h2 contravariant: conifold via q, bulk via bulk_projection
        self.assertAlmostEqual(float(M1), float(f2[1:] @ q), places=9)
        self.assertAlmostEqual(float(H1), float(h2[1:] @ q), places=9)
        self.assertTrue(np.allclose(np.asarray(Malpha), f2[1:] @ bp))
        self.assertTrue(np.allclose(np.asarray(Halpha), h2[1:] @ bp))
        # f1, h1 covariant: conifold via e_q, bulk via bulk_embedding
        self.assertAlmostEqual(float(P1), float(f1[1:] @ eq), places=9)
        self.assertAlmostEqual(float(K1), float(h1[1:] @ eq), places=9)
        self.assertTrue(np.allclose(np.asarray(Palpha), f1[1:] @ be))
        self.assertTrue(np.allclose(np.asarray(Kalpha), h1[1:] @ be))

    def test_pfv_route_general_basis(self):
        r"""The PFV pipeline runs and is self-consistent in the general basis:
        ``pfv_to_flux`` / ``pfv_to_moduli`` produce finite output, the conifold
        modulus ``q·z0`` is the conifold (bulk-orthogonal) component (small ⇒
        near-conifold), and the ``pfv`` z_cf route is finite.  The PFV integers
        are transported into the original frame as ``M_orig = M@Λ`` (M-flux
        contravariant) and ``K_orig = K@Λ⁻ᵀ`` (K-flux covariant)."""
        import jaxvacua as jvc
        Lam = np.asarray(jvc.get_basis_change(_Q))
        M = np.array([4, -8, 8]); K = np.array([-8, 3, -6])
        Morig = jnp.asarray((M @ Lam).astype(float))
        Korig = jnp.asarray((K @ np.linalg.inv(Lam).T).astype(float))
        tau = complex(0.0, 5.0)

        z0 = np.asarray(self.MG.pfv_to_moduli(Morig, Korig, tau))
        self.assertTrue(np.all(np.isfinite(z0)))
        q = np.asarray(self.MG.lcs_tree.conifold.conifold_curve)
        self.assertLess(abs(complex(q @ z0)), 0.5)          # near the conifold

        flux = self.MG.pfv_to_flux(Morig, Korig)
        self.assertTrue(np.all(np.isfinite(np.asarray(flux))))
        bp = np.asarray(self.MG.lcs_tree.conifold.bulk_projection)
        z_bulk = jnp.asarray(z0 @ bp)
        w = complex(self.MG.W_log_coeff(z_bulk, tau, flux, mode="pfv"))
        self.assertTrue(np.isfinite(w))

    def test_conifold_freezer_general_basis(self):
        r"""``ConifoldFreezer`` integrate-out works in the general basis.  The
        light↔full map is ``z_full = z_cf·e_q + bulk_embedding·z_light`` (not an
        index scatter), so the reconstructed full modulus has conifold component
        ``q·z_full == solved z_cf`` and round-trips
        (``z_full @ bulk_projection == z_light``); the reduced ``V``/``DW`` are
        finite."""
        import jaxvacua as jvc
        fr = jvc.ConifoldFreezer(self.MG)
        Lam = np.asarray(jvc.get_basis_change(_Q))
        M = np.array([4, -8, 8]); K = np.array([-8, 3, -6])
        flux = self.MG.pfv_to_flux(jnp.asarray((M @ Lam).astype(float)),
                                   jnp.asarray((K @ np.linalg.inv(Lam).T).astype(float)))
        z_light = jnp.asarray(np.array([0.05 + 1.95j, -0.03 + 2.05j]))
        tau = complex(0.1, 5.0)

        z_cf = complex(fr.solve_heavy(z_light, tau, flux)[0])
        z_full = np.asarray(fr.reconstruct_full_moduli(z_light, tau, flux))
        q = np.asarray(self.MG.lcs_tree.conifold.conifold_curve)
        bp = np.asarray(self.MG.lcs_tree.conifold.bulk_projection)
        # conifold component of the reconstructed full modulus IS the solved z_cf
        self.assertLess(abs(complex(q @ z_full) - z_cf), 1e-9)
        # bulk components round-trip
        self.assertTrue(np.allclose(z_full @ bp, np.asarray(z_light)))
        # reduced potential + light F-terms finite
        x_light = np.array([0.05, 1.95, -0.03, 2.05, 0.1, 5.0])
        self.assertTrue(np.isfinite(float(fr.V_x_light(jnp.asarray(x_light), flux))))
        self.assertTrue(np.all(np.isfinite(np.asarray(fr.DW_x_light(jnp.asarray(x_light), flux)))))

    def test_map_to_fd_general_basis(self):
        r"""``map_to_fd`` in the general basis applies bulk-only monodromy
        (``n ∈ ker(q)``): the conifold modulus ``z_cf = q·z`` is preserved, the
        bulk ``Re(z_al)`` is mapped into the FD ``(lo, hi]``, and
        monodromy-equivalent vacua (a ``ker(q)`` shift + the matching flux
        monodromy) deduplicate to the same representative.  Uses
        ``limit="coniLCS_series"`` (the conifold-skip only fires for
        ``coniLCS_series``/``coniLCS_bulk``)."""
        import jaxvacua as jvc
        cy = _Polytope(_POLYTOPE_PTS).triangulate().get_cy()
        MGs = jvc.FluxVacuaFinder(
            h12=int(cy.h11()), Q=int(cy.h11()) + int(cy.h12()) + 2,
            use_cytools=True, mirror_cy=cy, ncf=_NCF, use_gvs=True,
            maximum_degree=_MAXDEG, basis_change=None, conifold_curve=_Q,
            limit="coniLCS_series", prange=10, conifold_basis=False, nmax=2,
        )
        Lam = np.asarray(jvc.get_basis_change(_Q))
        Lam_inv_T = np.linalg.inv(Lam).T
        flux = MGs.pfv_to_flux(
            jnp.asarray((np.array([4, -8, 8]) @ Lam).astype(float)),
            jnp.asarray((np.array([-8, 3, -6]) @ Lam_inv_T).astype(float)),
        )
        z = jnp.asarray(np.array([0.2 + 2.0j, 2.6 + 2.1j, -1.7 + 1.9j]))
        tau = complex(0.1, 2.5)

        zfd, _, ffd = MGs.map_to_fd(z, tau, jnp.asarray(flux))
        zfd = np.asarray(zfd); ffd = np.asarray(ffd)
        # z_cf = q·z preserved
        self.assertLess(abs(complex(_Q @ np.asarray(z)) - complex(_Q @ zfd)), 1e-9)
        # bulk Re(z_al) in (-0.5, 0.5]
        zal = Lam_inv_T @ zfd
        self.assertTrue(np.all(zal[1:].real > -0.5 - 1e-8))
        self.assertTrue(np.all(zal[1:].real <= 0.5 + 1e-8))
        # dedup: monodromy-equivalent vacuum maps to the SAME representative
        n_bulk = np.rint(Lam.T @ np.array([0, 1, -1])).astype(int)
        self.assertEqual(int(_Q @ n_bulk), 0)
        z2, flux2 = MGs.apply_monodromy(z, jnp.asarray(flux), n_bulk)
        zfd2, _, ffd2 = MGs.map_to_fd(z2, tau, flux2)
        self.assertLess(np.max(np.abs(zfd - np.asarray(zfd2))), 1e-9)
        self.assertEqual(int(np.max(np.abs(ffd - np.asarray(ffd2)))), 0)

    def test_periods_related_by_integer_symplectic_monodromy(self):
        """Π_aligned(z') = M·Π_general(Lᵀz') with M integer and symplectic.

        This is the precise statement of basis-consistency: the GL(h,ℤ) moduli
        basis change embeds into Sp(2(h+1),ℤ).  e^{-K} invariance is a corollary
        of MᵀΣM=Σ.
        """
        M = _recover_monodromy(self.MA, self.MG, self.Lt)
        Sig = np.asarray(self.MA.periods.sigma)
        # integer + exactly symplectic
        self.assertLess(np.max(np.abs(M - np.rint(M))), 1e-6)
        self.assertEqual(np.max(np.abs(M.T @ Sig @ M - Sig)), 0.0)
        # reconstruction at fresh points
        for z in _ALIGNED_POINTS[2:6]:
            z_al = jnp.asarray(z)
            pa = np.asarray(self.MA.period_vector(z_al))
            pg = np.asarray(self.MG.period_vector(self.Lt @ z_al))
            self.assertLess(np.max(np.abs(M @ pg - pa)), 1e-9)

    def test_W_invariant_under_monodromy_flux_transport(self):
        """W = (f−τh)·Σ·Π is invariant once fluxes are transported by the same M
        (f_al = M f_gen).  This pins the symplectic flux transform."""
        M = _recover_monodromy(self.MA, self.MG, self.Lt)
        nf = self.MG.n_fluxes
        Mfull = np.block([[M, np.zeros_like(M)], [np.zeros_like(M), M]])
        tau = complex(0.3, 7.0)
        for z, flux_arr in zip(_ALIGNED_POINTS[6:10], _FLUX_SAMPLES):
            z_al = jnp.asarray(z)
            flux_gen = flux_arr[: 2 * nf]
            flux_al = Mfull @ flux_gen
            Wg = complex(self.MG.superpotential(self.Lt @ z_al, tau, jnp.asarray(flux_gen)))
            Wa = complex(self.MA.superpotential(z_al, tau, jnp.asarray(flux_al)))
            self.assertLess(abs(Wg - Wa) / max(abs(Wg), 1e-12), 1e-10)

    def test_F_differs_by_monodromy(self):
        """Sanity: F itself is NOT invariant.  F is the symplectic generating
        function ½XᴵFᴵ; M mixes the electric/magnetic period blocks, so F shifts
        (the shift is NOT a quadratic in the moduli z — it is a generating-function
        shift).  The physics (e^{-K}, W) is invariant; F is not."""
        z_al = jnp.asarray(np.array([0.1 + 2.0j, -0.05 + 2.2j, 0.0 + 2.5j]))
        z_gen = self.Lt @ z_al
        dF = abs(complex(self.MA.F_coniLCS_series(z_al))
                 - complex(self.MG.F_coniLCS_series(z_gen)))
        self.assertGreater(dF, 1.0)   # the documented monodromy difference


if __name__ == "__main__":
    import unittest
    unittest.main()
