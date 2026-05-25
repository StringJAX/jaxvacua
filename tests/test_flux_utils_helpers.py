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

"""Unit tests for stateless flux utility helpers.

Purpose
-------
Lock the eager Python helper contracts in :mod:`jaxvacua.flux_utils` without
requiring a full flux-vacuum search.

Main public API
---------------
- ``dedup_key`` for streaming deduplication.
- ``map_to_fd`` for numpy/JAX boundary marshalling.
- ``is_physical`` and ``classify_solution`` for post-processing candidates.

Design notes
------------
Small dummy models isolate helper behaviour from the conifold and period
subsystems while still exercising the same public interfaces.
"""

from types import SimpleNamespace

import numpy as np

import jax.numpy as jnp

from jaxvacua.flux_utils import classify_solution, dedup_key, is_physical, map_to_fd


class _PhysicalityModel:
    def __init__(self, moduli, tau, hyperplanes=None, metric=None, metric_raises=False):
        self.moduli = jnp.asarray(moduli)
        self.tau = complex(tau)
        self.lcs_tree = SimpleNamespace(hyperplanes=hyperplanes)
        self.metric = metric
        self.metric_raises = metric_raises

    def _convert_real_to_complex(self, x):
        return self.moduli, jnp.conj(self.moduli), self.tau, complex(self.tau).conjugate()

    def kahler_metric(self, moduli, moduli_c, tau, tau_c):
        if self.metric_raises:
            raise RuntimeError("metric unavailable in dummy model")
        return jnp.asarray(self.metric)


class _ClassifierModel:
    def _convert_real_to_complex(self, x):
        z = jnp.array([1.0 + 2.0j])
        tau = 0.1 + 3.0j
        return z, jnp.conj(z), tau, complex(tau).conjugate()

    def scalar_potential(self, z, z_c, tau, tau_c, flux, noscale=True):
        return 1.25 + 0.0j

    def DW(self, z, z_c, tau, tau_c, flux):
        return jnp.array([2.0e-7 + 0.0j, 3.0e-7 + 0.0j])

    def ddV_x(self, x, flux, noscale=True):
        return jnp.diag(jnp.array([2.0, 3.0, 4.0, 5.0]))

    def tadpole(self, flux):
        return -7.0


class _FundamentalDomainModel:
    def __init__(self):
        self.seen_types = None

    def map_to_fd(self, moduli, tau, fluxes):
        self.seen_types = (type(moduli), type(tau), type(fluxes))
        return moduli + (1.0 + 0.0j), tau + 1.0j, fluxes + 0.25


def test_dedup_key_is_hashable_and_rounds_numerical_noise():
    r"""Tiny continuous perturbations collapse to the same streaming key."""
    moduli = np.array([1.234561 + 2.345671j, -0.456781 + 3.456781j])
    tau = 0.123451 + 4.567891j
    fluxes = np.array([1.0, 2.01, -3.49])

    key_a = dedup_key(moduli, tau, fluxes, n_digits=5)
    key_b = dedup_key(moduli + (1.0e-7 - 1.0e-7j), tau + 1.0e-7j, fluxes, n_digits=5)
    key_c = dedup_key(moduli, tau + 1.0e-3j, fluxes, n_digits=5)

    assert key_a == key_b
    assert key_a != key_c
    assert key_a in {key_b}


def test_map_to_fd_noop_returns_original_objects_when_disabled():
    r"""The disabled path is a true no-op for eager sampling call sites."""
    model = _FundamentalDomainModel()
    moduli = np.array([0.2 + 2.0j])
    tau = 0.3 + 4.0j
    fluxes = np.array([1.0, 2.0])

    out = map_to_fd(model, moduli, tau, fluxes, enabled=False)

    assert out[0] is moduli
    assert out[1] is tau
    assert out[2] is fluxes
    assert model.seen_types is None


def test_map_to_fd_enabled_marshals_outputs_to_numpy_boundary():
    r"""The enabled path delegates to the model and returns eager Python types."""
    model = _FundamentalDomainModel()
    moduli = np.array([0.2 + 2.0j])
    tau = 0.3 + 4.0j
    fluxes = np.array([1.0, 2.0])

    moduli_fd, tau_fd, fluxes_fd = map_to_fd(model, moduli, tau, fluxes, enabled=True)

    assert isinstance(moduli_fd, np.ndarray)
    assert isinstance(tau_fd, complex)
    assert isinstance(fluxes_fd, np.ndarray)
    assert fluxes_fd.dtype == np.int32
    np.testing.assert_allclose(moduli_fd, np.array([1.2 + 2.0j]))
    assert tau_fd == tau + 1.0j
    np.testing.assert_array_equal(fluxes_fd, np.array([1, 2], dtype=np.int32))


def test_is_physical_accepts_point_passing_all_checks():
    r"""A point inside the cone with positive metric is accepted."""
    model = _PhysicalityModel(
        moduli=jnp.array([0.1 + 2.0j, -0.2 + 3.0j]),
        tau=0.0 + 4.0j,
        hyperplanes=jnp.eye(2),
        metric=jnp.eye(3),
    )
    sampler = SimpleNamespace(s_lower=1.0)

    assert is_physical(model, sampler, x=jnp.zeros(6), moduli_max=10.0)


def test_is_physical_rejects_runaway_dilaton_and_cone_failures():
    r"""The cheap physicality checks reject before accepting a candidate."""
    sampler = SimpleNamespace(s_lower=1.0)
    base_kwargs = dict(
        moduli=jnp.array([0.1 + 2.0j, -0.2 + 3.0j]),
        hyperplanes=jnp.eye(2),
        metric=jnp.eye(3),
    )

    assert not is_physical(_PhysicalityModel(tau=0.0 + 4.0j, **base_kwargs), sampler, x=0, moduli_max=2.5)
    assert not is_physical(_PhysicalityModel(tau=0.0 + 0.5j, **base_kwargs), sampler, x=0, moduli_max=10.0)

    cone_fail = _PhysicalityModel(
        moduli=jnp.array([0.1 - 2.0j, -0.2 + 3.0j]),
        tau=0.0 + 4.0j,
        hyperplanes=jnp.eye(2),
        metric=jnp.eye(3),
    )
    assert not is_physical(cone_fail, sampler, x=0, moduli_max=10.0)


def test_is_physical_falls_back_when_metric_is_unavailable():
    r"""If the metric check raises, the helper falls back to Im(z)>0."""
    sampler = SimpleNamespace(s_lower=1.0)
    positive = _PhysicalityModel(
        moduli=jnp.array([0.1 + 2.0j, -0.2 + 3.0j]),
        tau=0.0 + 4.0j,
        hyperplanes=None,
        metric_raises=True,
    )
    negative = _PhysicalityModel(
        moduli=jnp.array([0.1 - 2.0j, -0.2 + 3.0j]),
        tau=0.0 + 4.0j,
        hyperplanes=None,
        metric_raises=True,
    )

    assert is_physical(positive, sampler, x=0)
    assert not is_physical(negative, sampler, x=0)


def test_classify_solution_reports_scalar_outputs_and_flags():
    r"""Classification exposes the expected scalar diagnostics and labels."""
    result = classify_solution(_ClassifierModel(), x=jnp.zeros(4), flux=jnp.ones(4))

    assert result["V"] == 1.25
    assert np.isclose(result["|DW|"], 5.0e-7)
    np.testing.assert_allclose(result["eigenvalues"], np.array([2.0, 3.0, 4.0, 5.0]))
    assert result["is_susy"] is True
    assert result["is_minimum"] is True
    assert result["Nflux"] == 7.0
