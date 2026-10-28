# Copyright 2026 Andreas Schachner
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

"""Time limits of the CYTools steps in :mod:`jaxvacua.cytools_interface`.

Purpose
-------
``time_out=None`` (the default) runs each CYTools step in-process without a limit; a number runs
it in a child process killed at the limit, with the documented fallbacks. These tests check that
the default never starts a child, that the two modes give the same model data, that an expired
limit falls back (cones, conifolds) or raises (invariants, ``eft_to_cone``), that a real error
falls back with its reason, and that ``KeyboardInterrupt`` raised inside CYTools is not
swallowed.

Design notes
------------
Two cheap geometries: the CY of a 7-point polytope with h11 = 3 (three-dimensional cones, two
conifolds, so ordering and pickling of the conifold data are exercised) and the quintic (h11 = 1)
for the invariants. Every CYTools step takes well under a second; the limit 1e-3 s expires before
a child can even start, which makes the fallbacks deterministic. Each child imports CYTools and
JAXVacua (about 2 s).
"""

import inspect
import types
import warnings

import numpy as np
import pytest

cytools = pytest.importorskip("cytools")

from jaxvacua import cytools_interface as ci  # noqa: E402

H11_3 = [[-1, 3, -2, -1], [1, -1, 0, 0], [-1, 0, 0, 1], [-1, 0, 0, 0], [-1, 0, 1, 1], [-1, 0, 2, 0],
         [-1, 0, 1, 0]]
QUINTIC = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
TINY = 1e-3
CONE_KEYS = ("hyperplanes", "tip_skc", "kahler_generators", "kahler_rays", "mori_rays", "intnums", "c2")


@pytest.fixture(scope="module")
def cy():
    return cytools.Polytope(H11_3).triangulate().get_cy()


@pytest.fixture(scope="module")
def quintic():
    return cytools.Polytope(QUINTIC).triangulate().get_cy()


def _model(cy, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = ci.cytools_model_data_init(cy, **kwargs)
    return data, [str(w.message) for w in caught]


def test_default_is_unlimited_and_starts_no_child(cy, monkeypatch):
    for function, name in ((ci.cytools_model_data_init, "time_out"),
                           (ci.cytools_instanton_data_init, "time_out"), (ci.eft_to_cone, "timeout")):
        assert inspect.signature(function).parameters[name].default is None

    def no_child(*args, **kwargs):
        raise AssertionError("a child process was started without a time limit")

    monkeypatch.setattr(ci, "run_in_subprocess", no_child)
    data, messages = _model(cy)
    assert not [m for m in messages if "not computed" in m]
    assert data["mori_rays"].shape == (3, 3) and len(data["conifolds"]) == 2


def test_child_process_gives_the_same_model_data(cy):
    in_process, _ = _model(cy, time_out=None)
    in_child, messages = _model(cy, time_out=120)
    assert not [m for m in messages if "not computed" in m]
    for key in CONE_KEYS:
        np.testing.assert_array_equal(np.asarray(in_child[key]), np.asarray(in_process[key]), err_msg=key)
    assert len(in_child["conifolds"]) == len(in_process["conifolds"]) == 2
    for a, b in zip(in_child["conifolds"], in_process["conifolds"]):
        np.testing.assert_array_equal(np.asarray(a.conifold_curve), np.asarray(b.conifold_curve))


def test_expired_limit_falls_back_to_the_toric_cone(cy):
    data, messages = _model(cy, time_out=TINY)
    assert any("Mori cone cap" in m and f"time_out={TINY}" in m for m in messages)
    assert any("Extremal hyperplanes" in m for m in messages)
    toric = cy.toric_kahler_cone()
    np.testing.assert_array_equal(np.asarray(data["hyperplanes"]), np.asarray(toric.hyperplanes()))
    assert len(data["kahler_rays"]) == 0 and len(data["kahler_generators"]) == 0
    assert data["conifolds"] is None


def test_invalid_limits_are_rejected(cy, quintic):
    for bad in (0, -1, "10"):
        with pytest.raises(ValueError, match="time_out"):
            ci.cytools_model_data_init(cy, time_out=bad)
        with pytest.raises(ValueError, match="time_out"):
            ci.cytools_instanton_data_init(quintic, 1, time_out=bad)


def test_error_falls_back_with_its_reason(cy, monkeypatch):
    def broken(self, *args, **kwargs):
        raise RuntimeError("cap failed on purpose")

    monkeypatch.setattr(type(cy), "mori_cone_cap", broken)
    data, messages = _model(cy)
    for what in ("Kähler cone from the Mori cone cap", "Mori cone cap generators"):
        assert any(what in m and "RuntimeError: cap failed on purpose" in m for m in messages), what
    # the cone properties still succeed, on the toric Kähler cone
    toric = cy.toric_kahler_cone()
    np.testing.assert_array_equal(np.asarray(data["hyperplanes"]), np.asarray(toric.extremal_hyperplanes()))
    np.testing.assert_array_equal(np.asarray(data["kahler_generators"]), np.asarray(toric.extremal_rays()))


def test_keyboard_interrupt_inside_cytools_is_not_swallowed(cy, monkeypatch):
    def interrupted(self, *args, **kwargs):
        raise KeyboardInterrupt

    def reached(*args, **kwargs):  # the conifold search also calls mori_cone_cap: fail before it
        raise AssertionError("the interrupt was swallowed by the cone step")

    monkeypatch.setattr(type(cy), "mori_cone_cap", interrupted)
    monkeypatch.setattr(ci, "find_conifolds", reached)
    with pytest.raises(KeyboardInterrupt):
        ci.cytools_model_data_init(cy)


def test_invariants_in_child_match_and_an_expired_limit_raises(quintic):
    in_process = ci.cytools_instanton_data_init(quintic, 2, time_out=None)
    in_child = ci.cytools_instanton_data_init(quintic, 2, time_out=120)
    assert in_child["gvs"] == in_process["gvs"] and len(in_process["gvs"]) > 0
    np.testing.assert_array_equal(np.asarray(in_child["grading_vector"]),
                                  np.asarray(in_process["grading_vector"]))
    with pytest.raises(TimeoutError, match=f"time_out={TINY}"):
        ci.cytools_instanton_data_init(quintic, 2, time_out=TINY)


def test_eft_to_cone_limits(cy):
    tree = types.SimpleNamespace(
        polytope_points=cy.polytope().points(),
        heights=cy.triangulation().heights(),
        generators_kahler_cone=None,
    )
    eft = types.SimpleNamespace(lcs_tree=tree)
    in_process = ci.eft_to_cone(eft)
    in_child = ci.eft_to_cone(eft, timeout=120)
    np.testing.assert_array_equal(np.asarray(in_child.rays()), np.asarray(in_process.rays()))
    assert np.asarray(in_process.rays()).shape[1] == 3
    with pytest.raises(TimeoutError, match="Kähler cone"):
        ci.eft_to_cone(eft, timeout=TINY)


def test_large_h12_branch_has_no_mori_generators(cy, monkeypatch):
    # h12 >= 100 skips the cone computations; it used to leave M_cap_generators unassigned
    monkeypatch.setattr(type(cy), "h11", lambda self: 100)
    data, _ = _model(cy)
    assert data["mori_rays"].item() is None
    np.testing.assert_array_equal(np.asarray(data["hyperplanes"]),
                                  np.asarray(cy.toric_kahler_cone().hyperplanes()))


def test_child_runs_on_the_cpu():
    import os

    assert ci._run_timed(os.getenv, ("JAX_PLATFORMS",), 60) == "cpu"
