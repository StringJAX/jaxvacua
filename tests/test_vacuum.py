# Copyright 2024-2026 Andreas Schachner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

r"""Tests for the vacuum data containers (:mod:`jaxvacua.vacuum`).

Purpose
-------
Validate the two layers of :mod:`jaxvacua.vacuum`:

* :class:`~jaxvacua.vacuum.PFVData` — construction from flux quanta, delegation
  to the model, array-aware equality and the ``_model``-free serialization
  round-trip;
* :class:`~jaxvacua.vacuum.Vacuum` / :class:`~jaxvacua.vacuum.PFV` — the core
  content invariant (a flux **and** a location), the ``limit``-gated conifold
  fields, array-aware exact equality vs. tolerant equivalence, the registry-routed
  serialization round-trip, the permissive minimal ``{moduli, tau, flux}`` loader,
  the tagged-JSON storage tier, the structured ``diagnostics`` / ``is_consistent``
  report on good *and* deliberately broken vacua, the PFV racetrack-estimate
  properties, and the :math:`SL(2, \mathbb{Z}) \times` monodromy equivalence/dedup
  helpers.  (The promotion-layer ``afvs.AFV`` / ``afvs.PromotedPFV`` types, which
  carry ``genealogy`` / ``success`` / ``trajectory``, are tested in ``afvs``.)
"""

import os
import sys
import tempfile
import warnings

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from util import TestCase

jax.config.update("jax_enable_x64", True)

sys.path.append("./../")
import jaxvacua
from jaxvacua.flux_utils import map_to_fd
from jaxvacua.vacuum import (
    PFV, PFVData, Vacuum, VacuumAnalysis, complex_to_real, dedup_vacua,
    load_vacua, real_to_complex, register_vacuum_kind, save_vacua, unique_vacua,
    vacuum_from_json, vacuum_to_json,
)

warnings.filterwarnings("ignore")


def _vacuum_from_pfv(model, M, K, tau):
    r"""Build a bare :class:`Vacuum` at the PFV point ``(M, K, tau)`` and return
    it together with the complex moduli ``z``."""
    z = model.pfv_to_moduli(M, K, tau)
    x = model._convert_complex_to_real(z, jnp.conj(z), complex(tau), np.conj(complex(tau)))
    flux = model.pfv_to_flux(M, K)
    return Vacuum(x=x, flux=jnp.asarray(flux, dtype=float)), z


class TestPFVData(TestCase):
    r"""
    **Description:**
    Test suite for :class:`jaxvacua.vacuum.PFVData`, the light PFV algebra
    container, using the CP[1,1,1,6,9] reference PFV (``M=[-16,50]``,
    ``K=[3,-4]``).
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = jaxvacua.FluxEFT(
            h12=2, model_ID=1, model_type="KS", maximum_degree=2
        )
        cls.M = jnp.array([-16.0, 50.0], dtype=float)
        cls.K = jnp.array([3.0, -4.0], dtype=float)
        cls.tau = 6.85j

    def test_from_fluxes_delegates_to_model(self):
        r"""**Description:** ``to_flux``/``moduli`` match the model helpers and
        the conditions are populated."""
        pd = PFVData.from_fluxes(self.model, self.M, self.K)
        self.assertAllClose(pd.to_flux(), self.model.pfv_to_flux(self.M, self.K),
                            rtol=0, atol=1e-12)
        self.assertAllClose(pd.moduli(self.tau),
                            self.model.pfv_to_moduli(self.M, self.K, self.tau),
                            rtol=0, atol=1e-12)
        self.assertIsNotNone(pd.p)
        self.assertIn("det N!=0", pd.check())

    def test_model_via_convenience_method(self):
        r"""**Description:** ``model.pfv_data(M, K)`` returns an equivalent
        :class:`PFVData`."""
        pd = self.model.pfv_data(self.M, self.K)
        self.assertTrue(pd.equals(PFVData.from_fluxes(self.model, self.M, self.K)))

    def test_to_dict_drops_model_and_downcasts(self):
        r"""**Description:** ``to_dict`` omits the ``_model`` handle and returns
        NumPy arrays; ``from_dict`` yields a model-free, data-equal record."""
        pd = PFVData.from_fluxes(self.model, self.M, self.K)
        d = pd.to_dict()
        self.assertNotIn("_model", d)
        self.assertIsInstance(d["M"], np.ndarray)
        self.assertIsInstance(d["p"], np.ndarray)
        pd2 = PFVData.from_dict(d)
        self.assertIsNone(pd2._model)
        self.assertTrue(pd.equals(pd2))

    def test_equals_is_scalar_bool(self):
        r"""**Description:** array-aware equality returns a scalar ``bool``
        (the auto-generated ``__eq__`` would raise on array fields)."""
        pd = PFVData.from_fluxes(self.model, self.M, self.K)
        result = pd.equals(PFVData.from_fluxes(self.model, self.M, self.K))
        self.assertIsInstance(result, bool)
        self.assertTrue(result)
        other = PFVData.from_fluxes(self.model, jnp.array([8.0, -25.0]), self.K)
        self.assertFalse(pd.equals(other))

    def test_singular_N_sets_p_none_without_raising(self):
        r"""**Description:** a non-PFV flux with singular ``N`` builds without
        raising and leaves ``p`` unset."""
        pd = PFVData.from_fluxes(self.model, jnp.array([-4.0, 0.0]), self.K)
        self.assertIsNone(pd.p)
        self.assertFalse(bool(pd.check()["det N!=0"][0]))

    def test_repr(self):
        r"""**Description:** ``__repr__`` is a plain string carrying ``M``/``K``."""
        pd = PFVData.from_fluxes(self.model, self.M, self.K)
        self.assertIn("PFVData(", repr(pd))


class TestVacuumBase(TestCase):
    r"""
    **Description:**
    Test suite for the :class:`jaxvacua.vacuum.Vacuum` base class — array-aware
    equality and the ``"_kind"``-routed serialization round-trip (no model
    required).
    """

    def _sample_vacuum(self):
        return Vacuum(
            x=jnp.array([1.0, 2.0, 3.0, 6.85]),
            flux=jnp.array([1.0, 2.0, 3.0, 4.0]),
            W0=complex(1.0, -2.0), DW=jnp.array([0.0, 1e-10]),
            residual=1e-10, gs=0.3, metadata={"note": "hi"},
        )

    def test_roundtrip_recasts_arrays(self):
        r"""**Description:** ``to_dict`` downcasts arrays to NumPy and embeds the
        ``"_kind"`` discriminator; ``from_dict`` recasts them to ``jax.Array``."""
        v = self._sample_vacuum()
        d = v.to_dict()
        self.assertEqual(d["_kind"], "Vacuum")
        self.assertIsInstance(d["x"], np.ndarray)
        v2 = Vacuum.from_dict(d)
        self.assertIsInstance(v2, Vacuum)
        self.assertIsInstance(v2.x, jnp.ndarray)
        self.assertTrue(v.equals(v2))

    def test_equals_is_scalar_bool_and_nan_aware(self):
        r"""**Description:** ``equals`` returns a scalar ``bool`` and treats the
        ``NaN`` defaults as equal (the auto ``__eq__`` would raise on arrays)."""
        result = self._sample_vacuum().equals(self._sample_vacuum())
        self.assertIsInstance(result, bool)
        self.assertTrue(result)
        # two undiagnosed vacua (all-NaN solved fields) compare equal
        kw = dict(x=jnp.array([1.0, 2.0, 3.0, 6.85]), flux=jnp.array([1.0, 2.0]))
        self.assertTrue(Vacuum(**kw).equals(Vacuum(**kw)))

    def test_exception_metadata_stringified(self):
        r"""**Description:** an exception stored in ``metadata`` is serialised as
        its string message so the payload stays picklable."""
        d = Vacuum(x=jnp.array([1.0, 2.0, 3.0, 6.85]), flux=jnp.array([1.0, 2.0]),
                   metadata={"exception": RuntimeError("boom")}).to_dict()
        self.assertEqual(d["metadata"]["exception"], "boom")

    def test_cross_subclass_not_equal(self):
        r"""**Description:** ``equals`` is ``False`` across different subclasses."""
        pfv = PFV(x=jnp.array([1.0, 2.0, 3.0, 6.85]), flux=jnp.array([1.0, 2.0]))
        self.assertFalse(self._sample_vacuum().equals(pfv))

    def test_summaries_omit_conifold_rows_for_an_lcs_vacuum(self):
        r"""**Description:** the display helpers build strings without raising, and
        an LCS vacuum is **not** padded with ``nan``s for a bulk/conifold split
        that does not exist for it (the defect that motivated the refactor).  The
        promotion trajectory table lives in ``afvs`` now, not here."""
        v = self._sample_vacuum()
        short, long = v._short_summary_str(), v._long_summary_str()
        self.assertIn("Vacuum[", short)
        self.assertIn("limit=LCS", short)
        for text in (short, long):
            self.assertNotIn("zcf", text)
            self.assertNotIn("nan", text.lower())
        self.assertNotIn("trajectory", long)      # provenance is an afvs concept

    def test_summaries_show_conifold_rows_in_a_conifold_limit(self):
        r"""**Description:** the contrast with the LCS case: in a conifold limit
        the conifold trio is present."""
        v = Vacuum(x=jnp.array([1.0, 2.0, 3.0, 6.85]), flux=jnp.array([1.0, 2.0]),
                   limit="coniLCS", zcf=complex(0.1, 0.2),
                   residual_bulk=1e-9, residual_conifold=1e-10)
        self.assertIn("|zcf|", v._short_summary_str())
        self.assertIn("residual_conifold", v._long_summary_str())

    def test_save_load_roundtrip(self):
        r"""**Description:** ``save_vacua``/``load_vacua`` round-trips a mixed list
        preserving each subclass."""
        vacua = [self._sample_vacuum(),
                 PFV(flux=jnp.array([1.0, 2.0]), z=np.array([1.0 + 2.0j]),
                     tau=complex(0.0, 6.85), tau_input=complex(0.0, 6.85))]
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "vacua.p")
        save_vacua(vacua, path)
        reloaded = load_vacua(path)
        self.assertEqual([type(v).__name__ for v in reloaded], ["Vacuum", "PFV"])
        self.assertTrue(vacua[0].equals(reloaded[0]))
        self.assertTrue(vacua[1].equals(reloaded[1]))

    def test_minimal_moduli_tau_flux_loads_as_vacuum(self):
        r"""**Description:** ``from_dict`` accepts a minimal ``{moduli, tau, flux}``
        payload (no full ``x``), building ``x`` from the fixed real/imag
        interleaving; the result is a base :class:`Vacuum`."""
        v = Vacuum.from_dict({"moduli": [2.5j, 3.0j], "tau": complex(0.0, 4.0),
                              "flux": [1, 0, -2, 3, 0, 1]})
        self.assertIsInstance(v, Vacuum)
        self.assertAllClose(np.asarray(v.x), [0.0, 2.5, 0.0, 3.0, 0.0, 4.0],
                            rtol=0, atol=0)


class TestPFV(TestCase):
    r"""
    **Description:**
    Test suite for :class:`jaxvacua.vacuum.PFV` — construction from quantum
    numbers, the analytic racetrack-estimate properties (distinct from the
    solved fields) and the ``.data``-preserving round-trip, on the CP[1,1,1,6,9]
    reference PFV (``M=[-16,50]``, ``K=[3,-4]``).
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = jaxvacua.FluxEFT(
            h12=2, model_ID=1, model_type="KS", maximum_degree=2
        )
        cls.M = jnp.array([-16.0, 50.0], dtype=float)
        cls.K = jnp.array([3.0, -4.0], dtype=float)
        cls.tau = 6.85j

    def test_from_quantum_numbers_populates_seed(self):
        r"""**Description:** ``from_quantum_numbers`` builds ``data``/``flux`` and
        seeds ``x`` from the flat-direction moduli ``z = p*tau``."""
        pfv = PFV.from_quantum_numbers(self.model, self.M, self.K, self.tau)
        self.assertIsNotNone(pfv.data)
        self.assertIsNotNone(pfv.x)
        self.assertAllClose(pfv.flux, self.model.pfv_to_flux(self.M, self.K),
                            rtol=0, atol=1e-12)
        self.assertIsNone(pfv.analysis)          # nothing derived until asked

    def test_singular_N_raises_naming_the_reason(self):
        r"""**Description:** a non-PFV flux (singular ``N``) has no flat direction,
        hence no location to seed from, so construction **raises** and names the
        cause.  (It used to return an object with ``x=None``, which violated the
        invariant that a vacuum has a location and could not be used for
        anything.)  ``PFVData`` itself still tolerates it, reporting ``p=None``."""
        M_singular = jnp.array([-4.0, 0.0])
        with self.assertRaises(ValueError) as ctx:
            PFV.from_quantum_numbers(self.model, M_singular, self.K, self.tau)
        self.assertIn("do not define a PFV", str(ctx.exception))
        self.assertIn("det N", str(ctx.exception))
        # the algebra layer is deliberately more permissive
        self.assertIsNone(PFVData.from_fluxes(self.model, M_singular, self.K).p)

    def test_racetrack_estimate_matches_core_and_is_distinct(self):
        r"""**Description:** the ``tau0``/``W0_estimate``/``gs_estimate``/
        ``log10_W0_estimate`` properties match ``model.pfv_racetrack`` and are
        distinct from the (unset) solved ``W0``/``gs`` fields."""
        pfv = PFV.from_quantum_numbers(self.model, self.M, self.K, self.tau)
        rt = self.model.pfv_racetrack(self.M, self.K)
        self.assertAllClose(pfv.tau0, rt["tau0"], rtol=0, atol=1e-12)
        self.assertAllClose(pfv.W0_estimate, rt["W0"], rtol=0, atol=1e-12)
        self.assertAllClose(pfv.gs_estimate, rt["gs"], rtol=0, atol=1e-12)
        self.assertAllClose(pfv.log10_W0_estimate, rt["log10_W0"], rtol=0, atol=1e-12)
        # the solved fields are unset (NaN) on a freshly-seeded, un-promoted PFV
        self.assertTrue(np.isnan(pfv.W0))
        self.assertTrue(np.isfinite(complex(pfv.W0_estimate)))

    def test_roundtrip_preserves_data(self):
        r"""**Description:** a :class:`PFV` round-trips (routed by ``"_kind"``),
        rebuilding ``.data`` (without a ``_model``) and preserving ``M``/``K``."""
        pfv = PFV.from_quantum_numbers(self.model, self.M, self.K, self.tau)
        d = pfv.to_dict()
        self.assertEqual(d["_kind"], "PFV")
        self.assertIsInstance(d["data"], dict)
        self.assertNotIn("_model", d["data"])
        pfv2 = Vacuum.from_dict(d)
        self.assertIsInstance(pfv2, PFV)
        self.assertIsNone(pfv2.data._model)
        self.assertAllClose(pfv2.data.M, self.M, rtol=0, atol=0)
        self.assertAllClose(pfv2.data.K, self.K, rtol=0, atol=0)
        self.assertTrue(pfv.equals(pfv2))


class TestVacuumEquivalence(TestCase):
    r"""
    **Description:**
    Test suite for the :math:`SL(2, \mathbb{Z}) \times` monodromy equivalence /
    dedup helpers (:meth:`Vacuum.canonical_key`, :meth:`Vacuum.equivalent_to`,
    :func:`unique_vacua`) on the CP[1,1,1,6,9] reference PFV.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = jaxvacua.FluxEFT(
            h12=2, model_ID=1, model_type="KS", maximum_degree=2
        )
        cls.M = jnp.array([-16.0, 50.0], dtype=float)
        cls.K = jnp.array([3.0, -4.0], dtype=float)
        cls.tau = 6.85j

    def test_canonical_key_is_deterministic(self):
        r"""**Description:** ``canonical_key`` is stable across repeated calls."""
        v, _ = _vacuum_from_pfv(self.model, self.M, self.K, self.tau)
        self.assertEqual(v.canonical_key(self.model), v.canonical_key(self.model))

    def test_fd_image_is_equivalent(self):
        r"""**Description:** a vacuum and its fundamental-domain image collapse to
        the same canonical key."""
        v, z = _vacuum_from_pfv(self.model, self.M, self.K, self.tau)
        flux = self.model.pfv_to_flux(self.M, self.K)
        z_fd, tau_fd, flux_fd = map_to_fd(self.model, z, complex(self.tau), flux, enabled=True)
        x_fd = self.model._convert_complex_to_real(
            jnp.asarray(z_fd), jnp.conj(jnp.asarray(z_fd)),
            complex(tau_fd), np.conj(complex(tau_fd)))
        v_fd = Vacuum(x=x_fd, flux=jnp.asarray(flux_fd, dtype=float))
        self.assertTrue(v.equivalent_to(v_fd, self.model))

    def test_distinct_pfv_not_equivalent(self):
        r"""**Description:** two genuinely distinct PFVs are not identified."""
        v, _ = _vacuum_from_pfv(self.model, self.M, self.K, self.tau)
        v_other, _ = _vacuum_from_pfv(
            self.model, jnp.array([-8.0, 25.0]), jnp.array([1.0, -2.0]), 5.0j)
        self.assertFalse(v.equivalent_to(v_other, self.model))

    def test_unique_vacua_collapses_duplicates(self):
        r"""**Description:** ``unique_vacua`` collapses exact + FD-equivalent
        duplicates while keeping distinct vacua (``dedup_vacua`` is an alias)."""
        v, z = _vacuum_from_pfv(self.model, self.M, self.K, self.tau)
        v_copy, _ = _vacuum_from_pfv(self.model, self.M, self.K, self.tau)
        flux = self.model.pfv_to_flux(self.M, self.K)
        z_fd, tau_fd, flux_fd = map_to_fd(self.model, z, complex(self.tau), flux, enabled=True)
        x_fd = self.model._convert_complex_to_real(
            jnp.asarray(z_fd), jnp.conj(jnp.asarray(z_fd)),
            complex(tau_fd), np.conj(complex(tau_fd)))
        v_fd = Vacuum(x=x_fd, flux=jnp.asarray(flux_fd, dtype=float))
        v_other, _ = _vacuum_from_pfv(
            self.model, jnp.array([-8.0, 25.0]), jnp.array([1.0, -2.0]), 5.0j)
        reps = unique_vacua([v, v_copy, v_fd, v_other], self.model)
        self.assertEqual(len(reps), 2)
        self.assertIs(dedup_vacua, unique_vacua)


class TestCoreContentInvariant(TestCase):
    r"""
    **Description:**
    A vacuum is defined by its **flux** and its **location**; the location may be
    given as the interleaved real vector ``x`` or as ``(z, tau)``, and the missing
    one is derived.  These are the constructor-level guarantees everything else
    relies on.
    """

    def test_flux_is_required(self):
        r"""**Description:** omitting ``flux`` is a ``TypeError``; passing
        ``flux=None`` explicitly is a ``ValueError`` naming the requirement."""
        with self.assertRaises(TypeError):
            Vacuum(x=jnp.array([0.0, 1.0, 0.0, 6.85]))
        with self.assertRaises(ValueError) as ctx:
            Vacuum(flux=None, x=jnp.array([0.0, 1.0, 0.0, 6.85]))
        self.assertIn("requires `flux`", str(ctx.exception))

    def test_location_is_required(self):
        r"""**Description:** a flux alone is not a vacuum."""
        with self.assertRaises(ValueError) as ctx:
            Vacuum(flux=jnp.array([1.0, 2.0]))
        self.assertIn("requires a location", str(ctx.exception))
        # z without tau is incomplete
        with self.assertRaises(ValueError):
            Vacuum(flux=jnp.array([1.0, 2.0]), z=np.array([1.0 + 1.0j]))

    def test_z_tau_and_x_are_two_views_of_one_location(self):
        r"""**Description:** either input yields the same, fully populated pair."""
        z, tau = np.array([1.0 + 2.0j, 3.0 + 4.0j]), complex(0.5, 6.85)
        from_ztau = Vacuum(flux=jnp.array([1.0, 2.0]), z=z, tau=tau)
        from_x = Vacuum(flux=jnp.array([1.0, 2.0]),
                        x=jnp.asarray(complex_to_real(z, tau)))
        self.assertAllClose(np.asarray(from_ztau.x), np.asarray(from_x.x),
                            rtol=0, atol=0)
        for v in (from_ztau, from_x):
            self.assertAllClose(v.z, z, rtol=0, atol=0)
            self.assertEqual(v.tau, tau)

    def test_conflicting_x_and_z_tau_raises(self):
        r"""**Description:** a mismatch raises rather than silently preferring
        one -- the caller would otherwise never learn which was used."""
        with self.assertRaises(ValueError) as ctx:
            Vacuum(flux=jnp.array([1.0, 2.0]),
                   x=jnp.array([1.0, 2.0, 0.0, 6.85]),
                   z=np.array([9.0 + 9.0j]), tau=complex(0.0, 6.85))
        self.assertIn("different points", str(ctx.exception))

    def test_real_to_complex_is_the_inverse_of_complex_to_real(self):
        r"""**Description:** the model-free layout helpers round-trip exactly,
        including the ``n_light = 0`` empty-moduli case."""
        z, tau = np.array([1.0 + 2.0j, 3.0 + 4.0j]), complex(0.5, 6.85)
        z2, tau2 = real_to_complex(complex_to_real(z, tau))
        self.assertAllClose(z2, z, rtol=0, atol=0)
        self.assertEqual(tau2, tau)
        z3, tau3 = real_to_complex(complex_to_real(np.array([]), tau))
        self.assertEqual(z3.size, 0)
        self.assertEqual(tau3, tau)
        with self.assertRaises(ValueError):
            real_to_complex(np.array([1.0, 2.0, 3.0]))       # odd length

    def test_limit_gates_the_conifold_fields(self):
        r"""**Description:** ``has_conifold`` follows ``limit``; for a plain LCS
        vacuum the bulk/conifold split is undefined, not merely unknown, and the
        fields default to ``None`` rather than ``NaN`` (``NaN != NaN`` broke
        ``equals`` once)."""
        kw = dict(flux=jnp.array([1.0, 2.0]), x=jnp.array([1.0, 2.0, 0.0, 6.85]))
        lcs = Vacuum(**kw)
        self.assertFalse(lcs.has_conifold)
        for name in ("zcf", "residual_bulk", "residual_conifold"):
            self.assertIsNone(getattr(lcs, name))
        for limit in ("coniLCS", "coniLCS_series", "coniLCS_bulk"):
            self.assertTrue(Vacuum(limit=limit, **kw).has_conifold)


class TestVacuumRegistryAndStorage(TestCase):
    r"""
    **Description:**
    The ``_kind`` subclass registry and the two storage tiers (gzip+pickle for
    local use, tagged JSON for anything others download).
    """

    def _vac(self):
        return Vacuum(x=jnp.array([1.0, 2.0, 3.0, 6.85]),
                      flux=jnp.array([1.0, 2.0, 3.0, 4.0]),
                      W0=complex(1.0, -2.0), DW=jnp.array([0.0, 1e-10]),
                      residual=1e-10, gs=0.3, metadata={"note": "hi"})

    def test_registered_subclass_rebuilds_as_itself(self):
        r"""**Description:** a downstream subclass round-trips as itself once
        registered -- the mechanism that lets ``afvs`` types survive a load."""

        @register_vacuum_kind
        @dataclass(eq=False)
        class _Downstream(Vacuum):
            extra: str = "tag"

        v = _Downstream(x=jnp.array([1.0, 2.0, 3.0, 6.85]),
                        flux=jnp.array([1.0, 2.0]), extra="kept")
        back = Vacuum.from_dict(v.to_dict())
        self.assertIsInstance(back, _Downstream)
        self.assertEqual(back.extra, "kept")

    def test_unknown_kind_warns_and_degrades(self):
        r"""**Description:** an unregistered kind degrades to a base ``Vacuum``
        with a warning -- a load must not fail on an unknown producer."""
        d = self._vac().to_dict()
        d["_kind"] = "NotRegistered"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            v = Vacuum.from_dict(d)
        self.assertIs(type(v), Vacuum)
        self.assertTrue(any("Unknown vacuum kind" in str(w.message) for w in caught))

    def test_to_dict_omits_derived_and_stamps_the_schema(self):
        r"""**Description:** ``z``/``tau`` are re-derived on load, so storing them
        would duplicate the location and could go stale."""
        d = self._vac().to_dict()
        self.assertNotIn("z", d)
        self.assertNotIn("tau", d)
        self.assertIn("_schema_version", d)

    def test_tagged_json_tier_round_trips_and_drops_analysis(self):
        r"""**Description:** the pickle-free tier round-trips arrays and complex
        scalars exactly, and ``analysis=False`` drops the derived block."""
        v = self._vac()
        v.analysis = VacuumAnalysis(hessian_eigenvalues=np.array([1.0, 2.0]))
        back = vacuum_from_json(vacuum_to_json(v))
        self.assertTrue(v.equals(back))
        self.assertIsInstance(back.x, jnp.ndarray)
        self.assertAllClose(back.analysis.hessian_eigenvalues, [1.0, 2.0],
                            rtol=0, atol=0)
        self.assertEqual(complex(back.W0), complex(1.0, -2.0))
        minimal = vacuum_from_json(vacuum_to_json(v, analysis=False))
        self.assertIsNone(minimal.analysis)
        self.assertTrue(v.equals(minimal))       # analysis is payload, not identity


class TestVacuumDiagnostics(TestCase):
    r"""
    **Description:**
    The structured ``{name: (ok, value, reason)}`` report and its reduction
    ``is_consistent`` -- on a physically sound point *and* on deliberately broken
    ones, asserting the failing **key and reason** rather than just the bool.

    Uses the CP[1,1,1,6,9] reference PFV point (``M=[-16,50]``, ``K=[3,-4]``).
    ``residual`` is left unrecorded (``NaN``) on the sound vacuum: a PFV seed is
    not a solved point, and a *skipped* check must not read as a passed one.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = jaxvacua.FluxEFT(h12=2, model_ID=1, model_type="KS",
                                     maximum_degree=2)
        cls.M = jnp.array([-16.0, 50.0], dtype=float)
        cls.K = jnp.array([3.0, -4.0], dtype=float)
        cls.tau = 6.85j
        z = cls.model.pfv_to_moduli(cls.M, cls.K, cls.tau)
        cls.x = cls.model._convert_complex_to_real(
            z, jnp.conj(z), complex(cls.tau), np.conj(complex(cls.tau)))
        cls.flux = jnp.asarray(cls.model.pfv_to_flux(cls.M, cls.K), dtype=float)

    def _vac(self, **kw):
        kw.setdefault("x", self.x)
        kw.setdefault("flux", self.flux)
        return Vacuum(**kw)

    @staticmethod
    def _graded(report):
        return {k: v for k, v in report.items()
                if not str(v[2]).startswith("skipped")}

    def test_sound_point_passes_every_graded_check(self):
        r"""**Description:** every check that can run at the reference point
        passes, and ``is_consistent`` agrees with the report."""
        v = self._vac()
        report = v.diagnostics(self.model)
        graded = self._graded(report)
        failing = {k: report[k][2] for k in graded if not graded[k][0]}
        self.assertEqual(failing, {}, f"unexpected failures: {failing}")
        self.assertIn("kahler_metric_pd", graded)     # the 2026-06-02 check ran
        self.assertIn("tadpole", graded)
        self.assertTrue(v.is_consistent(self.model))

    def test_skipped_checks_are_excluded_from_the_verdict(self):
        r"""**Description:** a check whose inputs are unavailable reports
        ``"skipped: ..."`` and is excluded -- a missing input can never silently
        *pass*.  A bare ``FluxEFT`` has no sampler, and no ``moduli_max`` is given."""
        report = self._vac().diagnostics(self.model)
        self.assertTrue(report["dilaton_floor"][2].startswith("skipped"))
        self.assertIn("no sampler", report["dilaton_floor"][2])
        self.assertTrue(report["runaway_bound"][2].startswith("skipped"))
        self.assertTrue(report["hessian_min_eig"][2].startswith("skipped"))
        self.assertIn("stability=True", report["hessian_min_eig"][2])
        # residual is unrecorded here -> skipped, NOT passed
        self.assertTrue(report["residual"][2].startswith("skipped"))
        # every skipped entry still carries ok=True so a .values() scan is clean
        for name, (ok, _val, why) in report.items():
            if str(why).startswith("skipped"):
                self.assertTrue(ok, f"{name} skipped but ok=False")

    def test_supplying_the_inputs_un_skips_the_checks(self):
        r"""**Description:** the same checks run once their inputs are given."""
        report = self._vac().diagnostics(self.model, s_min=1.0, moduli_max=1e3)
        self.assertFalse(report["dilaton_floor"][2].startswith("skipped"))
        self.assertFalse(report["runaway_bound"][2].startswith("skipped"))
        self.assertTrue(report["dilaton_floor"][0])   # Im(tau)=6.85 > 1.0

    def test_model_free_checks_still_run_without_a_model(self):
        r"""**Description:** ``im_tau_positive`` and ``flux_integrality`` need no
        geometry, so they are graded even when every model-dependent check is
        skipped.  (Worth pinning: it means ``diagnostics`` is never entirely
        empty, and a caller cannot infer "no model" from an empty report.)"""
        report = self._vac().diagnostics(None)
        graded = self._graded(report)
        self.assertEqual(set(graded), {"im_tau_positive", "flux_integrality"})
        for name in ("kahler_cone", "kahler_metric_pd", "tadpole"):
            self.assertTrue(report[name][2].startswith("skipped"))

    def test_is_consistent_is_false_when_nothing_can_be_graded(self):
        r"""**Description:** ``all([])`` is vacuously ``True``, so the reduction
        guards against it: an unverifiable vacuum must not read as a verified one.
        Exercised directly, since the two model-free checks above mean a real
        report is never entirely skipped."""

        class _AllSkipped(Vacuum):
            def diagnostics(self, model, **kwargs):
                return {"a": (True, None, "skipped: no inputs"),
                        "b": (True, None, "skipped: no inputs")}

        v = _AllSkipped(x=self.x, flux=self.flux)
        self.assertFalse(v.is_consistent(self.model))
        # ... and True once a single check can actually be graded
        class _OneGraded(_AllSkipped):
            def diagnostics(self, model, **kwargs):
                out = super().diagnostics(model, **kwargs)
                out["c"] = (True, 1.0, "")
                return out
        self.assertTrue(_OneGraded(x=self.x, flux=self.flux).is_consistent(self.model))

    def test_broken_vacua_name_the_failing_key_and_reason(self):
        r"""**Description:** the point of the whole exercise -- each deliberate
        defect is reported by name, with a human-readable reason, and *only* that
        check fails.  ``flux_utils.is_physical`` returns a bare bool and names the
        failure only via ``print(verbose=True)``."""
        cases = []

        # (1) Im(tau) < 0 -- the only positivity requirement on the location
        x_bad = np.asarray(self.x, dtype=float).copy()
        x_bad[-1] = -6.85
        cases.append(("im_tau_positive", self._vac(x=jnp.asarray(x_bad)),
                      "not positive"))

        # (2) non-integral flux
        f_bad = np.asarray(self.flux, dtype=float).copy()
        f_bad[0] += 0.25
        cases.append(("flux_integrality", self._vac(flux=jnp.asarray(f_bad)),
                      "deviates from integers"))

        # (3) NEGATIVE D3 charge.  Flipping the WHOLE flux vector would not do:
        # N_flux = f.Sigma.h is bilinear, so the value would not move at all.
        # Flipping only the RR block does flip the sign.
        n = int(self.flux.shape[0] // 2)
        f_neg = np.asarray(self.flux, dtype=float).copy()
        f_neg[:n] *= -1.0
        cases.append(("tadpole", self._vac(flux=jnp.asarray(f_neg)),
                      "negative D3 charge"))

        # (4) residual above tolerance
        cases.append(("residual", self._vac(residual=1e-3), "exceeds residual_tol"))

        # (5) runaway
        cases.append(("runaway_bound", self._vac(), "exceeds moduli_max"))

        for expected, vac, fragment in cases:
            kw = {"moduli_max": 1e-6} if expected == "runaway_bound" else {}
            report = vac.diagnostics(self.model, **kw)
            failing = [k for k, val in self._graded(report).items() if not val[0]]
            self.assertIn(expected, failing,
                          f"{expected} not reported; failing = {failing}")
            self.assertIn(fragment, report[expected][2])
            self.assertFalse(vac.is_consistent(self.model, **kw))

    def test_negative_tadpole_check_actually_moves_the_value(self):
        r"""**Description:** guard on the test above -- confirm the RR-block flip
        really changes ``N_flux`` (a whole-vector flip leaves it unchanged, which
        would make case (3) verify nothing)."""
        n = int(self.flux.shape[0] // 2)
        f = np.asarray(self.flux, dtype=float)
        base = float(np.real(self.model.tadpole(jnp.asarray(f))))
        whole = float(np.real(self.model.tadpole(jnp.asarray(-f))))
        rr = f.copy()
        rr[:n] *= -1.0
        flipped = float(np.real(self.model.tadpole(jnp.asarray(rr))))
        self.assertAlmostEqual(base, whole, places=9)      # bilinear: unchanged
        self.assertAlmostEqual(flipped, -base, places=9)   # RR flip: sign flips
        self.assertLess(flipped, 0.0)

    def test_report_is_cached_and_invalidated_on_changed_tolerances(self):
        r"""**Description:** results are attached to the vacuum and reused, so the
        expensive pieces run once and stay inspectable; changing a tolerance
        recomputes."""
        v = self._vac(residual=1e-6)
        first = v.diagnostics(self.model, residual_tol=1e-4)
        self.assertIs(v.diagnostics(self.model, residual_tol=1e-4), first)
        self.assertTrue(first["residual"][0])
        second = v.diagnostics(self.model, residual_tol=1e-12)
        self.assertIsNot(second, first)
        self.assertFalse(second["residual"][0])
        self.assertIs(v.analysis.checks, second)

    def test_analysis_carries_eigenvalues_and_survives_the_round_trip(self):
        r"""**Description:** what was computed is kept on the object -- and can be
        dropped wholesale for bulk storage."""
        v = self._vac()
        v.diagnostics(self.model)
        eigs = v.analysis.kahler_metric_eigenvalues
        self.assertIsNotNone(eigs)
        self.assertTrue(np.all(np.asarray(eigs) > 0))
        back = Vacuum.from_dict(v.to_dict())
        self.assertAllClose(back.analysis.kahler_metric_eigenvalues, eigs,
                            rtol=0, atol=0)
        self.assertIn("ok", back.analysis.summary())
        self.assertIsNone(Vacuum.from_dict(v.to_dict(analysis=False)).analysis)

    def test_alignment_only_in_a_conifold_limit(self):
        r"""**Description:** the alignment scalar needs a conifold modulus, so it
        stays ``None`` for an LCS vacuum instead of reporting a meaningless
        number."""
        lcs = self._vac()
        lcs.diagnostics(self.model)
        self.assertIsNone(lcs.analysis.alignment)
        cf = self._vac(limit="coniLCS", zcf=complex(1e-3, 1e-3), gs=1 / 6.85,
                       W0=complex(1e-3, 0.0))
        cf.diagnostics(self.model)
        self.assertIsNotNone(cf.analysis.alignment)
        self.assertTrue(np.isfinite(cf.analysis.alignment))

    def test_pfv_diagnostics_enforce_the_pfv_algebra(self):
        r"""**Description:** a record labelled a PFV is tested against *being* one:
        the PFV conditions join the report under ``pfv:`` keys (prefixed to avoid
        colliding with the base ``tadpole`` check, which tests a different
        quantity)."""
        pfv = PFV.from_quantum_numbers(self.model, self.M, self.K, self.tau)
        report = pfv.diagnostics(self.model)
        pfv_keys = [k for k in report if k.startswith("pfv:")]
        self.assertIn("pfv:det N!=0", pfv_keys)
        self.assertIn("pfv:K.p==0", pfv_keys)
        self.assertNotIn("pfv:p", report)          # a value, not a verdict
        self.assertTrue(all(report[k][0] for k in pfv_keys))
        # a base Vacuum at the same point has no such entries
        self.assertEqual([k for k in self._vac().diagnostics(self.model)
                          if k.startswith("pfv:")], [])

    def test_pfv_violating_its_conditions_is_not_consistent(self):
        r"""**Description:** ``is_consistent`` therefore covers the PFV algebra: a
        "PFV" whose conditions fail reports ``False`` with the condition named."""
        M_bad = jnp.array([-6.0, -6.0], dtype=float)
        cond = self.model.pfv_conditions(M_bad, self.K)
        failing = [n for n, e in cond.items()
                   if n != "p" and not bool(np.asarray(e[0]).all())]
        self.assertTrue(failing, "fixture no longer violates any PFV condition")
        pfv = PFV.from_quantum_numbers(self.model, M_bad, self.K, self.tau)
        report = pfv.diagnostics(self.model)
        for name in failing:
            self.assertFalse(report[f"pfv:{name}"][0])
            self.assertIn("violated", report[f"pfv:{name}"][2])
        self.assertFalse(pfv.is_consistent(self.model))


class TestVacuumTolerantEquivalence(TestCase):
    r"""
    **Description:**
    The three notions of sameness: exact record equality (:meth:`Vacuum.equals`),
    the same point up to rounding (``equivalent_to`` without a model), and up to
    duality (``equivalent_to`` with one).
    """

    def _pair(self, delta):
        x = np.array([1.0, 2.0, 3.0, 6.85])
        flux = jnp.array([1.0, 2.0, 3.0, 4.0])
        x2 = x.copy()
        x2[0] += delta
        return (Vacuum(x=jnp.asarray(x), flux=flux),
                Vacuum(x=jnp.asarray(x2), flux=flux))

    def test_equals_is_exact_and_equivalent_to_is_tolerant(self):
        r"""**Description:** ``equals`` is bit-exact on purpose (it proves a round
        trip lost nothing); solver noise therefore breaks it, while
        ``equivalent_to`` absorbs it via ``dedup_key``'s rounding."""
        a, b = self._pair(1e-12)
        self.assertFalse(a.equals(b))
        self.assertTrue(a.equivalent_to(b))

    def test_genuinely_different_points_are_not_equivalent(self):
        a, b = self._pair(1e-3)
        self.assertFalse(a.equivalent_to(b))

    def test_n_digits_controls_the_tolerance(self):
        a, b = self._pair(1e-4)
        self.assertFalse(a.equivalent_to(b))               # 6 dp: distinct
        self.assertTrue(a.equivalent_to(b, n_digits=2))    # 2 dp: same bin

    def test_equivalence_ignores_the_subclass(self):
        r"""**Description:** a base ``Vacuum`` re-solved from scratch and a
        ``PFV`` at the same point are the same vacuum, though not the same
        *record*."""
        a, _ = self._pair(0.0)
        p = PFV(x=a.x, flux=a.flux, tau_input=complex(0.0, 6.85))
        self.assertTrue(a.equivalent_to(p))
        self.assertFalse(a.equals(p))

    def test_differing_flux_is_a_different_vacuum(self):
        a, _ = self._pair(0.0)
        b = Vacuum(x=a.x, flux=jnp.array([1.0, 2.0, 3.0, 5.0]))
        self.assertFalse(a.equivalent_to(b))


if __name__ == "__main__":
    import unittest
    unittest.main()