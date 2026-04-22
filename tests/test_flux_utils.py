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


if __name__ == "__main__":
    import unittest
    unittest.main()
