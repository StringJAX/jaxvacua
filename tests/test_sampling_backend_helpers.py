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

"""Regression tests for backend-neutral sampling helper methods.

Purpose
-------
Exercise small :class:`jaxvacua.sampling.data_sampler` helper methods without
building a geometry model.

Main public API
---------------
- JAX-key dispatch.
- Batch collection.
- Backend integer-bound consistency.

Design notes
------------
Instances are created via ``object.__new__`` and populated only with the helper
attributes required by the methods under test.  This avoids conifold, CYTools
and flux-EFT dependencies in this non-conifold pass.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jaxvacua.sampling import data_sampler
from jaxvacua.util import PRNGSequence


def _bare_sampler(use_jax):
    sampler = object.__new__(data_sampler)
    sampler.use_jax = use_jax
    sampler._rng = PRNGSequence(123)
    return sampler


def test_get_jax_key_uses_raw_key_without_splitting():
    r"""Passing a raw PRNG key returns that exact key."""
    sampler = _bare_sampler(use_jax=True)
    key = jax.random.PRNGKey(5)

    out = sampler._get_jax_key(key)

    np.testing.assert_array_equal(np.asarray(out), np.asarray(key))


def test_get_jax_key_advances_sequence_inputs():
    r"""Sequence-backed key dispatch is deterministic and advances."""
    sampler = _bare_sampler(use_jax=True)
    sequence_a = PRNGSequence(7)
    sequence_b = PRNGSequence(7)

    key_a0 = sampler._get_jax_key(sequence_a)
    key_b0 = sampler._get_jax_key(sequence_b)
    key_a1 = sampler._get_jax_key(sequence_a)

    np.testing.assert_array_equal(np.asarray(key_a0), np.asarray(key_b0))
    assert not np.array_equal(np.asarray(key_a0), np.asarray(key_a1))


def test_collect_batches_trims_numpy_and_jax_backends():
    r"""Batch accumulation preserves the backend array family and trims rows."""
    sampler_np = _bare_sampler(use_jax=False)
    out_np = sampler_np._collect_batches(
        [np.ones((2, 3)), 2.0 * np.ones((2, 3))],
        n=3,
    )
    assert isinstance(out_np, np.ndarray)
    np.testing.assert_allclose(out_np, [[1, 1, 1], [1, 1, 1], [2, 2, 2]])

    sampler_jax = _bare_sampler(use_jax=True)
    out_jax = sampler_jax._collect_batches(
        [jnp.ones((2, 3)), 2.0 * jnp.ones((2, 3))],
        n=3,
    )
    assert isinstance(out_jax, jax.Array)
    np.testing.assert_allclose(np.asarray(out_jax), [[1, 1, 1], [1, 1, 1], [2, 2, 2]])


@pytest.mark.xfail(
    reason=(
        "Known bug: data_sampler._rand_integer documents NumPy-style exclusive "
        "upper bounds, but the JAX path delegates to random_integer, whose public "
        "contract is inclusive."
    ),
    strict=True,
)
def test_rand_integer_jax_backend_uses_exclusive_upper_bound_like_numpy():
    r"""The private sampler helper should use NumPy randint bound semantics."""
    sampler = _bare_sampler(use_jax=True)
    out = np.asarray(
        sampler._rand_integer(
            -1,
            2,
            shape=(10000,),
            rns_key=jax.random.PRNGKey(0),
        )
    )

    assert out.min() >= -1
    assert out.max() < 2
