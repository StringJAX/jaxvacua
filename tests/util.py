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

"""Shared test utilities for JAXVacua.

Purpose
-------
Provide a common ``TestCase`` base class with deterministic JAX key handling
and array assertions used throughout the test suite.

Main public API
---------------
- ``TestCase``: extends ``chex.TestCase`` with PRNG-key helpers and
  ``assertAllEqual``, ``assertAllClose`` and ``assertAllTrue``.

Design notes
------------
The helpers keep individual tests concise while preserving JAX-friendly array
comparisons.
"""

import chex
import jax.random
import numpy as np
import jax.numpy as jnp
from time import time_ns


class TestCase(chex.TestCase):

    def setUp(self):
        seed = time_ns()
        self.init_random(seed)

    def init_random(self, seed: int):
        self._key = jax.random.PRNGKey(seed)

    def next_key(self, num=1):
        self._key, *key = jax.random.split(self._key, num=num+1)
        return key[0] if num == 1 else key

    def assertAllEqual(self, x, y, msg=None):
        x, y = map(np.asarray, (x, y))
        self.assertTrue(np.all(x == y), msg=msg)

    def assertAllClose(self, x, y, rtol=1e-5, atol=1e-8, msg=None):
        x, y = map(np.asarray, (x, y))
        self.assertTrue(jnp.allclose(x, y, rtol=rtol, atol=atol), msg=msg)

    def assertAllTrue(self, t, msg=None):
        self.assertTrue(jnp.all(t), msg=msg)
    
