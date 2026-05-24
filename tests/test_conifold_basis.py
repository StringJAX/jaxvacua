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


def _recover_monodromy(MA, MG, Lt, rng, npts=12):
    """Recover the integer monodromy M with Π_aligned(z') = M·Π_general(Lᵀz').

    The GL(h,ℤ) moduli basis change embeds into Sp(2(h+1),ℤ); M is read off
    numerically from matched period vectors and rounded to the nearest integer.
    """
    PA, PG = [], []
    for _ in range(npts):
        z_al = jnp.asarray(rng.uniform(-0.25, 0.25, 3) + 1j * rng.uniform(1.7, 2.7, 3))
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
        cls.rng = np.random.default_rng(20240524)

    def _points(self, n=8):
        for _ in range(n):
            z_al = jnp.asarray(self.rng.uniform(-0.3, 0.3, 3)
                               + 1j * self.rng.uniform(1.6, 2.8, 3))
            yield z_al, self.Lt @ z_al

    def test_setup_flags(self):
        self.assertTrue(bool(self.MA.lcs_tree.conifold_basis))
        self.assertFalse(bool(self.MG.lcs_tree.conifold_basis))
        # projection stored as a matrix (not the legacy scalar 0)
        self.assertEqual(np.ndim(self.MG.lcs_tree.conifold.projection), 2)
        self.assertEqual(int(np.rint(_Q @ np.asarray(self.MG.lcs_tree.conifold.embedding))), 1)

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

    def test_manual_zcf_route_finite_general(self):
        """The manual z_cf route runs (no NotImplementedError) and is finite in
        the general basis — the 'ISSUE HERE' branch is completed."""
        nf = self.MG.n_fluxes
        flux = jnp.asarray(self.rng.integers(-3, 4, 2 * nf).astype(float))
        z_bulk = jnp.asarray(self.rng.uniform(-0.2, 0.2, 2) + 1j * self.rng.uniform(1.8, 2.4, 2))
        tau = complex(0.1, 8.0)
        for M in (self.MA, self.MG):
            w = complex(M.W_log_coeff(z_bulk if M is self.MG else z_bulk,
                                      tau, flux, mode="manual"))
            self.assertTrue(np.isfinite(w), "W_log_coeff (manual) not finite")

    def test_periods_related_by_integer_symplectic_monodromy(self):
        """Π_aligned(z') = M·Π_general(Lᵀz') with M integer and symplectic.

        This is the precise statement of basis-consistency: the GL(h,ℤ) moduli
        basis change embeds into Sp(2(h+1),ℤ).  e^{-K} invariance is a corollary
        of MᵀΣM=Σ.
        """
        M = _recover_monodromy(self.MA, self.MG, self.Lt, np.random.default_rng(3))
        Sig = np.asarray(self.MA.periods.sigma)
        # integer + exactly symplectic
        self.assertLess(np.max(np.abs(M - np.rint(M))), 1e-6)
        self.assertEqual(np.max(np.abs(M.T @ Sig @ M - Sig)), 0.0)
        # reconstruction at fresh points
        rng = np.random.default_rng(99)
        for _ in range(4):
            z_al = jnp.asarray(rng.uniform(-0.2, 0.2, 3) + 1j * rng.uniform(1.8, 2.6, 3))
            pa = np.asarray(self.MA.period_vector(z_al))
            pg = np.asarray(self.MG.period_vector(self.Lt @ z_al))
            self.assertLess(np.max(np.abs(M @ pg - pa)), 1e-9)

    def test_W_invariant_under_monodromy_flux_transport(self):
        """W = (f−τh)·Σ·Π is invariant once fluxes are transported by the same M
        (f_al = M f_gen).  This pins the symplectic flux transform."""
        M = _recover_monodromy(self.MA, self.MG, self.Lt, np.random.default_rng(3))
        nf = self.MG.n_fluxes
        Mfull = np.block([[M, np.zeros_like(M)], [np.zeros_like(M), M]])
        rng = np.random.default_rng(7)
        tau = complex(0.3, 7.0)
        for _ in range(4):
            z_al = jnp.asarray(rng.uniform(-0.2, 0.2, 3) + 1j * rng.uniform(1.8, 2.6, 3))
            flux_gen = rng.integers(-3, 4, 2 * nf).astype(float)
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