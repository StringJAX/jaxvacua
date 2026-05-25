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

"""Regression tests for package-wide utility helpers.

Purpose
-------
Cover lightweight, non-conifold helpers in :mod:`jaxvacua.util` whose behaviour
is used by several higher-level workflows.

Main public API
---------------
- PRNG and random wrappers.
- Flattening, JIT/vmap adapter, matrix diagnostics and compressed pickle IO.

Design notes
------------
The tests avoid constructing physics models, so they remain fast and isolate
generic contracts from the flux and conifold sectors.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jaxvacua.util import (
    PRNGSequence,
    check_nan,
    compute_evs_hermitian,
    flatten,
    flatten_top,
    is_outlier,
    jit_with_static_args,
    load_zipped_pickle,
    mergeDictionary,
    random_integer,
    random_integer_jit,
    random_uniform,
    random_uniform_jit,
    rank_matrix,
    save_zipped_pickle,
    subsets,
    vmapping_func_cached,
)


def _scaled_square(x, scale=1):
    return scale * x * x


def test_prng_sequence_reproducible_and_advances():
    r"""Identical seeds give identical streams, and successive keys differ."""
    seq_a = PRNGSequence(123)
    seq_b = PRNGSequence(123)

    a0 = next(seq_a)
    b0 = next(seq_b)
    a1 = next(seq_a)
    b1 = next(seq_b)

    np.testing.assert_array_equal(np.asarray(a0), np.asarray(b0))
    np.testing.assert_array_equal(np.asarray(a1), np.asarray(b1))
    assert not np.array_equal(np.asarray(a0), np.asarray(a1))


def test_random_helpers_are_bounded_and_reproducible():
    r"""The public random helpers honour shape, bounds and deterministic seeds."""
    u1 = random_uniform(-2.0, 3.0, seed=7, shape=(32, 2))
    u2 = random_uniform(-2.0, 3.0, seed=7, shape=(32, 2))
    chex_u = np.asarray(u1)

    np.testing.assert_allclose(chex_u, np.asarray(u2))
    assert u1.shape == (32, 2)
    assert np.all(chex_u >= -2.0)
    assert np.all(chex_u < 3.0)

    ints = np.asarray(random_integer(-1, 1, seed=11, shape=(256,)))
    assert ints.shape == (256,)
    assert set(np.unique(ints)).issubset({-1, 0, 1})


def test_random_jit_helpers_match_bounds_contract():
    r"""JIT random helpers preserve the non-JIT bound conventions."""
    key_u, key_i = jax.random.split(jax.random.PRNGKey(0))

    uniform = np.asarray(random_uniform_jit(key_u, 0.0, 1.0, shape=(16,)))
    assert uniform.shape == (16,)
    assert np.all(uniform >= 0.0)
    assert np.all(uniform < 1.0)

    integers = np.asarray(random_integer_jit(key_i, 2, 4, shape=(256,)))
    assert integers.shape == (256,)
    assert set(np.unique(integers)).issubset({2, 3, 4})


def test_vmapping_func_cached_reuses_kernel_object():
    r"""Repeated cached vmap construction returns the same wrapper object."""
    vmapped_a = vmapping_func_cached(_scaled_square, in_axes=0, scale=3)
    vmapped_b = vmapping_func_cached(_scaled_square, in_axes=0, scale=3)

    assert vmapped_a is vmapped_b
    np.testing.assert_allclose(np.asarray(vmapped_a(jnp.arange(4.0))), [0, 3, 12, 27])


def test_jit_with_static_args_handles_static_python_values():
    r"""The static-argument wrapper works for small Python configuration values."""
    def affine(x, scale, offset):
        return scale * x + offset

    wrapped = jit_with_static_args(affine, static_argnums=(1, 2))
    np.testing.assert_allclose(np.asarray(wrapped(jnp.array([1.0, 2.0]), 2.0, 5.0)), [7, 9])


def test_flatten_helpers_cover_nested_and_top_level_forms():
    r"""Recursive and top-level flattening keep their distinct semantics."""
    nested = [[1, [2, 3]], np.array([4, 5])]

    assert flatten(nested) == [1, 2, 3, 4, 5]
    np.testing.assert_array_equal(flatten(nested, as_np_arr=True), np.array([1, 2, 3, 4, 5]))
    assert list(flatten(nested, as_gen=True)) == [1, 2, 3, 4, 5]

    with pytest.raises(ValueError):
        flatten(nested, as_gen=True, as_np_arr=True)

    top = [[[1], [2]], [[3], [4]]]
    assert flatten_top(top, N=1) == [[1], [2], [3], [4]]
    assert flatten_top(top, N=2) == [1, 2, 3, 4]


def test_matrix_and_nan_helpers():
    r"""Linear-algebra helpers return stable scalar and eigenvalue contracts."""
    matrix = jnp.array([[2.0, 1.0], [1.0, 2.0]])
    eigs = np.asarray(compute_evs_hermitian(matrix))
    np.testing.assert_allclose(eigs, [1.0, 3.0])

    rank_input = jnp.array([[1.0, 2.0], [2.0, 4.0 + 1e-12]])
    assert int(rank_matrix(rank_input, tolerance=1e-8)) == 1
    assert bool(check_nan(jnp.array([0.0, jnp.nan])))
    assert not bool(check_nan(jnp.array([0.0, 1.0])))


def test_zipped_pickle_roundtrip(tmp_path):
    r"""Compressed pickle helpers preserve nested Python and NumPy data."""
    path = tmp_path / "payload.pkl.gz"
    payload = {"name": "jaxvacua", "values": np.arange(5), "nested": {"a": 1}}

    save_zipped_pickle(payload, str(path))
    loaded = load_zipped_pickle(str(path))

    assert loaded["name"] == payload["name"]
    assert loaded["nested"] == payload["nested"]
    np.testing.assert_array_equal(loaded["values"], payload["values"])


def test_dictionary_outlier_and_subset_helpers():
    r"""Small collection helpers keep their documented edge contracts."""
    merged = mergeDictionary(
        {"x": np.array([[1], [2]]), "left": np.array([10])},
        {"x": np.array([[3]]), "right": np.array([20])},
    )
    np.testing.assert_array_equal(merged["x"], np.array([[3], [1], [2]]))
    np.testing.assert_array_equal(merged["left"], np.array([10]))
    np.testing.assert_array_equal(merged["right"], np.array([20]))

    mask = is_outlier(np.array([0.0, 1.0, 2.0, 100.0]), percentile_cut=25)
    assert mask.tolist() == [True, False, False, True]

    assert subsets(range(4), 2) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
