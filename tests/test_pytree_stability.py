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

"""Regression tests for pytree-flatten stability of registered classes.

Purpose
-------
Lock the JAX pytree contract for ``FluxVacuaFinder`` (and, by extension, the
other ``register_pytree_node`` classes that share ``util.flatten_func``):

- the treedef must be hashable (else jit/vmap caching fails);
- the treedef must be **stable** across lazy-cache mutation — accessing the
  ``sampler`` property or populating calibration state must not change it
  (otherwise jit silently recompiles, and finders cannot share kernels);
- jit'd kernels must give identical results before and after those mutations;
- the cache/scratch attributes are excluded from the flattened pytree.

These guard the fragility described in review item #9 (see
``private/jaxvacua_public_review_risks_2026-05-17.md`` §9b): the generic
``__dict__`` flattener would otherwise let a lazily-set attribute perturb the
treedef.

Design notes
------------
Uses the ``h12=2, model_ID=1`` KS fixture, matching the other finder tests.
"""

import sys, os, warnings
import jax
import jax.numpy as jnp
import numpy as np
from util import *

jax.config.update("jax_enable_x64", True)

sys.path.append("./../")
import jaxvacua
from jaxvacua.util import _PYTREE_IGNORE, flatten_func

warnings.filterwarnings("ignore")


class TestPytreeStability(TestCase):
    r"""
    **Description:**
    Verifies that a :class:`FluxVacuaFinder` flattens to a hashable, stable
    pytree treedef across lazy-cache mutation (sampler access, calibration
    state), and that the JIT'd kernels are unaffected.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.finder = jaxvacua.FluxVacuaFinder(
            h12=2, model_ID=1, model_type="KS", maximum_degree=0,
            flux_bounds=[-5, 5], moduli_bounds=(2, 5),
            dilaton_bounds=(1.0, 10.0), axion_bounds=(-0.5, 0.5),
        )
        cls.z = jnp.array([1.0 + 2.0j, 1.0 + 2.0j])
        cls.tau = complex(0.0, 2.0)
        cls.fl = jnp.zeros(2 * cls.finder.n_fluxes)

    def _treedef(self, obj):
        return jax.tree_util.tree_structure(obj)

    def test_treedef_hashable(self):
        """The treedef must be hashable (jit/vmap cache key requirement)."""
        td = self._treedef(self.finder)
        # Should not raise.
        self.assertIsInstance(hash(td), int)

    def test_cache_attrs_excluded_from_pytree(self):
        """`_sampler` and calibration scratch are dropped from the flatten."""
        # Access the sampler + set calibration state so the attrs exist.
        _ = self.finder.sampler
        self.finder._calibrated_sigmas = {"H": 1.0}
        self.finder._M_eigvecs = np.eye(2)
        children, aux = flatten_func(self.finder)
        aux_keys = {a[0] if isinstance(a, tuple) else a for a in aux}
        for k in _PYTREE_IGNORE:
            self.assertNotIn(k, aux_keys, f"{k} leaked into aux_data")

    def test_treedef_stable_across_sampler_access(self):
        """Accessing the lazy `sampler` must not change the treedef."""
        f = jaxvacua.FluxVacuaFinder(
            h12=2, model_ID=1, model_type="KS", maximum_degree=0,
            flux_bounds=[-5, 5], moduli_bounds=(2, 5),
            dilaton_bounds=(1.0, 10.0), axion_bounds=(-0.5, 0.5),
        )
        td_before = self._treedef(f)
        _ = f.sampler  # mutates _sampler: None -> data_sampler
        td_after = self._treedef(f)
        self.assertEqual(td_before, td_after)
        self.assertEqual(hash(td_before), hash(td_after))

    def test_treedef_stable_across_calibration(self):
        """Setting calibration state (dict / numpy) must not change the treedef."""
        f = jaxvacua.FluxVacuaFinder(
            h12=2, model_ID=1, model_type="KS", maximum_degree=0,
            flux_bounds=[-5, 5], moduli_bounds=(2, 5),
            dilaton_bounds=(1.0, 10.0), axion_bounds=(-0.5, 0.5),
        )
        td_before = self._treedef(f)
        f._calibrated_sigmas = {"H": 1.0, "F": 2.0}
        f._M_eigvecs = np.eye(2)
        f._M_scales = np.ones(2)
        td_after = self._treedef(f)
        self.assertEqual(td_before, td_after)

    def test_jit_kernel_identical_before_after_sampler(self):
        """A jit'd kernel gives identical output before and after sampler access."""
        f = jaxvacua.FluxVacuaFinder(
            h12=2, model_ID=1, model_type="KS", maximum_degree=0,
            flux_bounds=[-5, 5], moduli_bounds=(2, 5),
            dilaton_bounds=(1.0, 10.0), axion_bounds=(-0.5, 0.5),
        )
        kernel = jax.jit(lambda mm, ff: mm.tadpole(ff))
        before = complex(kernel(f, self.fl))
        _ = f.sampler
        after = complex(kernel(f, self.fl))
        self.assertEqual(before, after)

    def test_vmap_over_finder_method(self):
        """vmap of a finder kernel works (treedef round-trips through vmap)."""
        f = self.finder
        zb = jnp.stack([self.z, self.z])
        out = jax.vmap(f.linearised_shifts_H, in_axes=(0, None, None))(
            zb, self.tau, self.fl
        )
        self.assertEqual(len(out), 3)


if __name__ == "__main__":
    import unittest
    unittest.main()