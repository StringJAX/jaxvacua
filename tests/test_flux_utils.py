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

"""Tests for flux-vector utility functions.

Purpose
-------
Validate stateless helpers for PFV conversions, solution classification,
physicality checks, deduplication keys and fundamental-domain mapping.

Main public API
---------------
- ``TestFluxUtils``: exercises ``jaxvacua.flux_utils`` through small EFT
  fixtures and class-delegated helper methods.

Design notes
------------
These tests protect post-processing behaviour independently from the full
vacuum-search workflow.
"""

import sys, os, warnings
import jax
import jax.numpy as jnp
import numpy as np
import chex
from functools import partial
from util import *

jax.config.update("jax_enable_x64", True)

sys.path.append("./../")
import jaxvacua

# Suppress warnings
warnings.filterwarnings("ignore")


# ==============================================================================
#  TestFluxUtils
# ==============================================================================

class TestFluxUtils(TestCase):
    r"""
    **Description:**
    Test suite for flux utility functions that convert between the full flux
    vector :math:`[f \mid h]` and the Primitive Flux Vector (PFV) representation
    :math:`(M, K)`.

    .. admonition:: Background
        :class: dropdown

        In Type IIB flux compactifications, the 3-form flux :math:`G_3 = F_3 - \tau H_3`
        is specified by integer RR-flux :math:`f` and NSNS-flux :math:`h`, each of
        length :math:`n = h^{1,2}+1`.  The PFV decomposition extracts the moduli-space
        content :math:`M` (related to :math:`f`) and :math:`K` (related to :math:`h`),
        each of length :math:`h^{1,2}`, which parameterise the primitive part of the flux.

        Given the PFV :math:`(M, K)` and the axio-dilaton :math:`\tau`, the moduli values
        at the level of the PFV are determined by solving :math:`z^i = N^{-1}_{ij} K_j \cdot \tau`
        where :math:`N_{ij} = \kappa_{ijk} M^k` are the intersection numbers contracted with :math:`M`.

    Attributes:
        model (FluxEFT): Physics model with :math:`h^{1,2}=2`, KS type.
        fl (Array): Test flux vector of length :math:`4(h^{1,2}+1) = 12`.
        tau (complex): Test axio-dilaton value.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        h12 = 2
        cls.model = jaxvacua.FluxEFT(
            h12=h12, model_ID=1, model_type="KS", maximum_degree=0
        )
        cls.n_fluxes = cls.model.n_fluxes  # h12+1 = 3
        cls.h12 = h12

        # A generic test flux vector [f | h] of length 2*n_fluxes = 12
        # f = [1, 0, -2, 0, 3, -1],  h = [2, 1, 0, -1, 1, 0]
        cls.fl = jnp.array([1., 0., -2., 0., 3., -1., 2., 1., 0., -1., 1., 0.], dtype=float)
        cls.tau = -0.3 + 5.0j

    # ------------------------------------------------------------------
    # flux_to_pfv  /  pfv_to_flux  round-trip
    # ------------------------------------------------------------------

    @chex.variants(with_jit=True, without_jit=True)
    def test_flux_to_pfv_shapes(self):
        r"""
        **Description:**
        Verify that :func:`flux_to_pfv` returns two arrays :math:`(M, K)` each
        of length :math:`h^{1,2}`.

        The M-vector corresponds to a subset of the RR-flux components and the
        K-vector to a subset of the NSNS-flux components.  Both have length
        :math:`h^{1,2}` (not :math:`h^{1,2}+1`) because the zeroth component is
        fixed by the Freed-Witten quantisation condition.
        """
        fn = self.variant(self.model.flux_to_pfv)
        M, K = fn(self.fl)
        # M-vector must have length h12 (RR-flux primitive components)
        chex.assert_shape(M, (self.h12,))
        # K-vector must have length h12 (NSNS-flux primitive components)
        chex.assert_shape(K, (self.h12,))

    @chex.variants(with_jit=True, without_jit=True)
    def test_pfv_roundtrip(self):
        r"""
        **Description:**
        Verify that the round-trip ``pfv_to_flux(flux_to_pfv(f))`` reconstructs
        a flux vector of the correct shape.

        .. note::
            The reconstructed flux may differ from the original because the PFV
            embedding imposes additional structure (e.g. :math:`f_0 = M \cdot b`
            where :math:`b` is the b-vector from the prepotential).
        """
        fn_to = self.variant(self.model.flux_to_pfv)
        fn_from = self.variant(self.model.pfv_to_flux)
        M, K = fn_to(self.fl)
        fl_recon = fn_from(M, K)
        # The reconstruction should match the original flux vector shape
        chex.assert_shape(fl_recon, self.fl.shape)

    # ------------------------------------------------------------------
    # pfv_to_moduli
    # ------------------------------------------------------------------

    @chex.variants(with_jit=True, without_jit=True)
    def test_pfv_to_moduli_shape(self):
        r"""
        **Description:**
        Verify that :func:`pfv_to_moduli` returns an array of length :math:`h^{1,2}`.

        Given the PFV :math:`(M, K)` and the axio-dilaton :math:`\tau`, the function
        solves for the complex structure moduli :math:`z^i` at the PFV level.
        """
        fn_to = self.variant(self.model.flux_to_pfv)
        fn_mod = self.variant(self.model.pfv_to_moduli)
        M, K = fn_to(self.fl)
        z0 = fn_mod(M, K, self.tau)
        chex.assert_shape(z0, (self.h12,))

    @chex.variants(with_jit=True, without_jit=True)
    def test_pfv_to_moduli_complex(self):
        r"""
        **Description:**
        Verify that the moduli returned by :func:`pfv_to_moduli` are complex-valued.

        The complex structure moduli :math:`z^i = a^i + \mathrm{i}\,v^i` are
        inherently complex, with :math:`v^i > 0` in the physical (Kähler cone) region.
        """
        fn_to = self.variant(self.model.flux_to_pfv)
        fn_mod = self.variant(self.model.pfv_to_moduli)
        M, K = fn_to(self.fl)
        z0 = fn_mod(M, K, self.tau)
        chex.assert_type(z0, complex)

    @chex.variants(with_jit=True, without_jit=True)
    def test_pfv_to_moduli_finite(self):
        r"""
        **Description:**
        Verify that the moduli values are finite (no NaN or Inf) for a
        reasonable test flux vector.

        Non-finite values would indicate a singular intersection number matrix
        :math:`N_{ij} = \kappa_{ijk} M^k`, which would mean the PFV does not
        determine a valid vacuum.
        """
        fn_to = self.variant(self.model.flux_to_pfv)
        fn_mod = self.variant(self.model.pfv_to_moduli)
        M, K = fn_to(self.fl)
        z0 = fn_mod(M, K, self.tau)
        self.assertTrue(jnp.all(jnp.isfinite(z0)))


# ==============================================================================
#  TestPFVAlgebra — N-matrix, p-vector, condition checker and analytic racetrack
# ==============================================================================

class TestPFVAlgebra(TestCase):
    r"""
    **Description:**
    Test suite for the PFV algebra added to :mod:`jaxvacua.flux_utils`:
    ``N_matrix``, ``pfv_p_vector``, ``pfv_conditions`` and ``pfv_racetrack``.

    The reference geometry is the degree-18 hypersurface in
    :math:`\mathbb{CP}^4_{[1,1,1,6,9]}` (``h12=2``, KS, ``model_ID=1``); the
    perturbatively flat vacuum ``M=[-16, 50]``, ``K=[3, -4]`` is the worked
    example of arXiv:2512.17095 §6.2, with ``p = [0.4, 0.3]`` and
    :math:`|W_0|\sim 10^{-8}`.

    Attributes:
        model (FluxEFT): ``h12=2`` KS model with degree-2 GV data (needed for
            the racetrack series).
        M, K (Array): The reference PFV flux quanta.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.h12 = 2
        cls.model = jaxvacua.FluxEFT(
            h12=2, model_ID=1, model_type="KS", maximum_degree=2
        )
        cls.M = jnp.array([-16.0, 50.0], dtype=float)
        cls.K = jnp.array([3.0, -4.0], dtype=float)

    def test_N_matrix(self):
        r"""**Description:** :math:`N_{ab}=\kappa_{abc}M^c` matches the direct
        contraction and is a float array."""
        N = self.model.N_matrix(self.M)
        ref = jnp.asarray(self.model.lcs_tree.intnums, dtype=float) @ self.M
        self.assertAllClose(N, ref, rtol=0, atol=0)
        self.assertEqual(jnp.asarray(N).dtype, jnp.asarray(ref).dtype)

    def test_pfv_p_vector(self):
        r"""**Description:** the LCS p-vector equals :math:`N^{-1}K = [0.4, 0.3]`
        and batches over ``(M, K)``."""
        p = self.model.pfv_p_vector(self.M, self.K)
        self.assertAllClose(p, jnp.array([0.4, 0.3]), rtol=0, atol=1e-12)
        # batch: a stacked input must equal the per-item loop
        Mb = jnp.stack([self.M, self.M])
        Kb = jnp.stack([self.K, self.K])
        pb = self.model.pfv_p_vector(Mb, Kb)
        chex.assert_shape(pb, (2, self.h12))
        self.assertAllClose(pb, jnp.stack([p, p]), rtol=0, atol=1e-12)

    def test_pfv_conditions_hold_for_reference_pfv(self):
        r"""**Description:** every §6.2 condition (incl. the tadpole, pfvs #5)
        holds for the reference PFV."""
        c = self.model.pfv_conditions(self.M, self.K)
        for key in ("det N!=0", "p in K_X", "K.p==0",
                    "a.T@M in Z", "24*b@M in 24Z", "tadpole 0<=Qflux<=Q"):
            self.assertIn(key, c)
            self.assertTrue(bool(c[key][0]), msg=f"condition {key!r} must hold")

    def test_pfv_conditions_singular_N_is_safe(self):
        r"""**Description:** a non-PFV flux with singular ``N`` reports
        ``det N != 0`` false and ``p = NaN`` without raising."""
        M_sing = jnp.array([-4.0, 0.0], dtype=float)  # kappa[:,:,1] is rank-deficient
        c = self.model.pfv_conditions(M_sing, self.K)
        self.assertFalse(bool(c["det N!=0"][0]))
        self.assertFalse(bool(np.all(np.isfinite(np.asarray(c["p"][1])))))

    def test_pfv_guard_rejects_one_modulus(self):
        r"""**Description:** the PFV algebra raises for ``h12 < 2`` /
        hypergeometric models."""
        from types import SimpleNamespace
        from jaxvacua.flux_utils import _pfv_guard
        stub = SimpleNamespace(
            lcs_tree=SimpleNamespace(h12=1, model_type="hypergeometric")
        )
        with self.assertRaises(NotImplementedError):
            _pfv_guard(stub)

    def test_pfv_racetrack(self):
        r"""**Description:** the analytic 2-term racetrack reproduces the §6.2
        estimate (``tau0 ~ 6.91i``, ``gs ~ 0.145``) and its ``W0`` matches
        ``W(normalise=True)`` at ``tau0``."""
        rt = self.model.pfv_racetrack(self.M, self.K)
        self.assertTrue(bool(rt["valid"]))
        tau0 = complex(rt["tau0"])
        self.assertAllClose(tau0.imag, 6.909, rtol=0, atol=2e-2)
        self.assertAllClose(abs(tau0.real), 0.0, rtol=0, atol=1e-6)
        self.assertAllClose(float(rt["gs"]), 0.1447, rtol=0, atol=1e-3)
        # W0 convention: racetrack log10_W0 matches log10|W(normalise=True)| at tau0
        flux = self.model.pfv_to_flux(self.M, self.K)
        p = np.asarray(self.model.pfv_p_vector(self.M, self.K))
        W = abs(complex(self.model.W(jnp.asarray(p * tau0), tau0,
                                     jnp.asarray(flux), normalise=True)))
        self.assertAllClose(float(rt["log10_W0"]), float(np.log10(W)), rtol=0, atol=1e-3)


try:
    import pfvs as _pfvs
    _HAS_PFVS = True
except Exception:                                             # noqa: BLE001
    _HAS_PFVS = False

import pytest


@pytest.mark.skipif(not _HAS_PFVS, reason="optional pfvs package not installed")
class TestPFVSBridge(TestCase):
    r"""
    **Description:**
    Cross-check the ``jaxvacua.flux_utils`` <-> ``pfvs`` bridge on the
    CP[1,1,1,6,9] LCS reference PFV: the bridge builds a ``pfvs.CYData`` whose
    PFV p-vector and shared conditions match jaxvacua's.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = jaxvacua.FluxEFT(h12=2, model_ID=1, model_type="KS",
                                     maximum_degree=2)
        cls.M = jnp.array([-16.0, 50.0], dtype=float)
        cls.K = jnp.array([3.0, -4.0], dtype=float)

    @staticmethod
    def _cond(conditions, *subs):
        for k, (ok, _v) in conditions.items():
            if all(s.lower() in k.lower() for s in subs):
                return bool(ok)
        return None

    def test_kwargs_h21_is_h11_not_h12(self):
        kw = self.model.lcs_tree.to_cydata_kwargs()
        self.assertEqual(kw["kappa"].shape[0], self.model.h12)
        self.assertEqual(kw["h21"], int(self.model.lcs_tree.h11))
        self.assertNotEqual(kw["h21"], self.model.h12)
        self.assertIsNone(kw["coni_curve"])                   # LCS: no conifold

    def test_p_vector_matches_pfvs(self):
        r"""``pfvs.PFV(data, K, M).p == pfv_p_vector(M, K)`` (pfvs arg order K, M;
        integer flux for flint)."""
        data = self.model.lcs_tree.to_cydata()
        self.assertEqual(int(data.h11), self.model.h12)
        self.assertEqual(int(data.h21), int(self.model.lcs_tree.h11))
        pf = _pfvs.PFV(data, np.asarray(self.K).astype(int),
                       np.asarray(self.M).astype(int))
        self.assertAllClose(np.asarray(pf.p, dtype=float),
                            np.asarray(self.model.pfv_p_vector(self.M, self.K), dtype=float),
                            rtol=0, atol=1e-9)

    def test_shared_conditions_agree(self):
        data = self.model.lcs_tree.to_cydata()
        pf = _pfvs.PFV(data, np.asarray(self.K).astype(int),
                       np.asarray(self.M).astype(int))
        cond = self.model.pfv_conditions(self.M, self.K)
        self.assertEqual(bool(pf.check_Ninvertible()), self._cond(cond, "det"))
        self.assertTrue(bool(pf.check_NpK()))
        self.assertEqual(bool(pf.check_orthogonality()), self._cond(cond, "k", "p"))
        self.assertEqual(bool(pf.check_b()), self._cond(cond, "24"))

    def test_absent_pfvs_friendly_error(self):
        r"""With pfvs unavailable, ``lcs_tree.to_cydata`` raises a friendly
        ImportError (not a raw dependency ImportError)."""
        import jaxvacua.flux_utils as fu
        saved = (fu._PFVS, fu._PFVS_CHECKED)
        fu._PFVS, fu._PFVS_CHECKED = None, True
        try:
            with self.assertRaises(ImportError) as ctx:
                self.model.lcs_tree.to_cydata()
            self.assertIn("pfvs", str(ctx.exception).lower())
        finally:
            fu._PFVS, fu._PFVS_CHECKED = saved


try:
    from cytools import Polytope as _Polytope
    _HAS_CYTOOLS = True
except Exception:                                             # noqa: BLE001
    _HAS_CYTOOLS = False


@pytest.mark.skipif(not (_HAS_PFVS and _HAS_CYTOOLS),
                    reason="needs optional pfvs + cytools")
class TestPFVSFromCy(TestCase):
    r"""
    **Description:**
    Field-level cross-check of :meth:`jaxvacua.lcs.lcs_tree.to_cydata_kwargs`
    against ``pfvs.CYData.from_cy(cy)`` for a geometry that is both a jaxvacua
    LCS model and a ``cytools.CalabiYau``.  This pins the topology-field
    convention independently of the p-vector agreement.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        pts = np.array([[-1, 3, -2, -1], [1, -1, 0, 0], [-1, 0, 0, 1],
                        [-1, 0, 0, 0], [-1, 0, 1, 1], [-1, 0, 2, 0], [-1, 0, 1, 0]])
        cls.cy = _Polytope(pts).triangulate().get_cy()
        cls.model = jaxvacua.FluxVacuaFinder(
            h12=int(cls.cy.h11()), use_cytools=True, mirror_cy=cls.cy,
            limit="LCS", maximum_degree=2, use_gvs=True, prange=20)

    def test_kappa_c2_and_hodge_match_from_cy(self):
        r"""kappa, c2, h11 (kappa dim) and h21 agree exactly with
        ``pfvs.CYData.from_cy``.  In particular ``c2`` is passed directly (NOT
        ``c2/24``)."""
        import pfvs
        kw = self.model.lcs_tree.to_cydata_kwargs()
        ref = pfvs.CYData.from_cy(self.cy)
        self.assertEqual(kw["kappa"].shape[0], int(ref.h11))
        self.assertEqual(kw["h21"], int(ref.h21))
        self.assertTrue(np.array_equal(kw["kappa"], np.asarray(ref.kappa)))
        self.assertTrue(np.array_equal(kw["c2"], np.asarray(ref.c2)))  # NOT /24

    def test_hyperplanes_encode_the_cone_differently(self):
        r"""Documented convention difference: jaxvacua stores the simplicial
        Kähler-cone facets while ``pfvs.CYData.from_cy`` stores the full cone, so
        the ``H`` arrays differ in row count.  This affects only
        ``check_pcontainment`` (excluded from the condition cross-check); kappa /
        c2 / p are unaffected."""
        import pfvs
        kw = self.model.lcs_tree.to_cydata_kwargs()
        ref = pfvs.CYData.from_cy(self.cy)
        self.assertEqual(kw["H"].shape[1], np.asarray(ref.H).shape[1])   # same dim
        # row counts legitimately differ (simplicial facets vs full cone)
        self.assertNotEqual(kw["H"].shape[0], np.asarray(ref.H).shape[0])


if __name__ == "__main__":
    import unittest
    unittest.main()
