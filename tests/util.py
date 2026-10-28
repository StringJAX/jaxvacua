# Copyright 2024 Andreas Schachner
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
    
