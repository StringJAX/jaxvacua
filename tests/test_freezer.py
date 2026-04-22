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

import sys, os, warnings, unittest
import jax
import jax.numpy as jnp
import numpy as np
from util import *

jax.config.update("jax_enable_x64", True)

sys.path.append("./../")
from jaxvacua.freezer import Freezer, ConifoldFreezer

# Suppress warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
#  Attempt to load a conifold model for integration tests.
# ---------------------------------------------------------------------------
_MODEL = None
_MODEL_LOAD_ERROR = None

try:
    import jaxvacua
    _MODEL = jaxvacua.FluxEFT(
        h12=2, model_ID=1, model_type="KS",
        maximum_degree=0, limit="coniLCS",
    )
except Exception as exc:
    _MODEL_LOAD_ERROR = str(exc)


# ==============================================================================
#  TestFreezerAbstract
# ==============================================================================

class TestFreezerAbstract(TestCase):
    r"""
    **Description:**
    Test suite for the :class:`Freezer` abstract base class.

    The ``Freezer`` defines the interface for a moduli-freezing procedure in
    which a subset of complex-structure moduli (the "heavy" moduli) are solved
    algebraically from their leading-order equations of motion as functions of
    the remaining "light" moduli, the axio-dilaton, and the flux quanta.
    Substituting back yields a reduced effective field theory with fewer
    degrees of freedom.

    Because ``Freezer`` is abstract (it inherits from ``ABC`` and declares
    ``heavy_indices``, ``solve_heavy``, and ``_real_light_to_full`` as abstract
    methods), it cannot be instantiated directly.  These tests verify that
    the abstract contract is enforced.
    """

    def test_freezer_cannot_be_instantiated(self):
        r"""
        **Description:**
        Freezer is abstract and raises ``TypeError`` on direct instantiation.
        The ``Freezer`` class uses Python's ``abc.ABC`` mechanism, so attempting
        to create an instance without implementing the abstract methods
        (``heavy_indices``, ``solve_heavy``, ``_real_light_to_full``) must
        raise ``TypeError``.
        """
        # Verify that direct instantiation of the ABC raises TypeError
        with self.assertRaises(TypeError):
            Freezer(model=None)

    def test_freezer_subclass_must_implement_abstract_methods(self):
        r"""
        **Description:**
        A partial subclass that omits abstract methods still raises ``TypeError``.
        This ensures the ABC contract is enforced: all three abstract methods
        (``heavy_indices``, ``solve_heavy``, ``_real_light_to_full``) must be
        overridden before a subclass can be instantiated.
        """

        class IncompleteFreezer(Freezer):
            # Only override heavy_indices, leave solve_heavy and
            # _real_light_to_full abstract.
            @property
            def heavy_indices(self):
                return (0,)

        # Verify that a subclass missing some abstract methods cannot be instantiated
        with self.assertRaises(TypeError):
            IncompleteFreezer(model=None)


# ==============================================================================
#  TestConifoldFreezer
# ==============================================================================

@unittest.skipIf(
    _MODEL is None,
    f"Conifold model could not be loaded: {_MODEL_LOAD_ERROR}",
)
class TestConifoldFreezer(TestCase):
    r"""
    **Description:**
    Test suite for the :class:`ConifoldFreezer` concrete implementation.

    The ``ConifoldFreezer`` integrates out the conifold modulus
    :math:`z_{\text{cf}}` (by default at index 0) in coniLCS models.  Near
    the conifold locus this modulus acquires a parametrically large mass from
    the flux superpotential and can be expressed as a function of the
    remaining bulk (light) moduli, the axio-dilaton :math:`\tau`, and the
    flux quanta.

    These tests verify the basic constructor behaviour and the consistency
    of the index-partitioning properties (``heavy_indices``,
    ``light_indices``, ``n_heavy``, ``n_light``).
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = _MODEL
        cls.freezer = ConifoldFreezer(cls.model)

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def test_conifold_freezer_is_freezer(self):
        r"""
        **Description:**
        Verifies that ``ConifoldFreezer`` is a proper subclass of ``Freezer``,
        ensuring the inheritance hierarchy is correct so that any code accepting
        a generic ``Freezer`` will also accept a ``ConifoldFreezer``.
        """
        # Check that the concrete freezer is an instance of the abstract base class
        self.assertIsInstance(self.freezer, Freezer)

    def test_conifold_freezer_stores_model(self):
        r"""
        **Description:**
        Verifies that the constructor stores a reference to the underlying
        flux EFT model, which is needed for superpotential evaluations and
        period computations during the moduli-freezing procedure.
        """
        # Check that the freezer's model attribute points to the same object
        self.assertIs(self.freezer.model, self.model)

    # ------------------------------------------------------------------
    # Index properties
    # ------------------------------------------------------------------

    def test_heavy_indices_default(self):
        r"""
        **Description:**
        Verifies that the default ``heavy_indices`` is ``(0,)``, corresponding
        to the conifold modulus z_cf at index 0, which acquires a parametrically
        large mass near the conifold locus and is the natural candidate for freezing.
        """
        # The conifold modulus at index 0 should be the sole heavy modulus
        self.assertEqual(self.freezer.heavy_indices, (0,))

    def test_light_indices_complement(self):
        r"""
        **Description:**
        Verifies that ``light_indices`` equals the sorted complement of
        ``heavy_indices`` in ``range(h12)``, i.e. all bulk moduli that remain
        as free variables in the reduced effective field theory after freezing.
        """
        h12 = self.model.h12
        expected_light = tuple(range(1, h12))
        # Light indices should be {1, ..., h12-1} when the conifold at index 0 is frozen
        self.assertEqual(self.freezer.light_indices, expected_light)

    def test_n_heavy_plus_n_light_equals_h12(self):
        r"""
        **Description:**
        Checks the key consistency condition ``n_heavy + n_light == h12``:
        every complex-structure modulus must be classified as either heavy
        (frozen) or light (free), with no overlaps or gaps.
        """
        h12 = self.model.h12
        # The total number of heavy + light moduli must equal h12
        self.assertEqual(
            self.freezer.n_heavy + self.freezer.n_light, h12,
            msg=f"n_heavy ({self.freezer.n_heavy}) + n_light ({self.freezer.n_light}) "
                f"!= h12 ({h12})",
        )

    def test_n_heavy_is_one(self):
        r"""
        **Description:**
        Verifies that the default ``ConifoldFreezer`` freezes exactly one
        modulus (the conifold modulus z_cf), which is the physically motivated
        choice near a single conifold singularity.
        """
        # Exactly one modulus should be classified as heavy
        self.assertEqual(self.freezer.n_heavy, 1)

    def test_n_light_is_h12_minus_one(self):
        r"""
        **Description:**
        Verifies that ``n_light == h12 - 1`` for the single-conifold freezer,
        confirming that all remaining bulk moduli are light and survive as free
        parameters in the reduced EFT.
        """
        # With one frozen modulus, the number of light moduli is h12 - 1
        self.assertEqual(self.freezer.n_light, self.model.h12 - 1)

    # ------------------------------------------------------------------
    # Custom conifold index
    # ------------------------------------------------------------------

    def test_custom_conifold_index(self):
        r"""
        **Description:**
        Verifies that passing a custom ``conifold_index`` correctly changes
        which modulus is treated as heavy, supporting models where the
        conifold modulus sits at a non-default position in the moduli array.
        """
        custom = ConifoldFreezer(self.model, conifold_index=1)
        # The heavy index should be the custom conifold index
        self.assertEqual(custom.heavy_indices, (1,))
        # Still exactly one heavy modulus
        self.assertEqual(custom.n_heavy, 1)
        # Index 1 should be excluded from light indices since it is now heavy
        self.assertNotIn(1, custom.light_indices)
        # Index 0 should now be a light modulus instead of heavy
        self.assertIn(0, custom.light_indices)

    def test_custom_ncf(self):
        r"""
        **Description:**
        Verifies that a custom conifold degree ``ncf`` is stored correctly by
        the constructor, since ncf controls the order of the conifold
        singularity and enters the exponential formula for z_cf.
        """
        custom = ConifoldFreezer(self.model, ncf=3)
        # The internal conifold degree attribute should match the input value
        self.assertEqual(custom._ncf, 3)

    # ------------------------------------------------------------------
    # Partition consistency (union / disjointness)
    # ------------------------------------------------------------------

    def test_indices_are_disjoint(self):
        r"""
        **Description:**
        Verifies that the heavy and light index sets are disjoint, which is
        required for a well-defined moduli-freezing procedure where each
        modulus is either solved algebraically or kept as a free variable.
        """
        heavy = set(self.freezer.heavy_indices)
        light = set(self.freezer.light_indices)
        # The intersection of heavy and light indices must be empty
        self.assertEqual(heavy & light, set())

    def test_indices_cover_all_moduli(self):
        r"""
        **Description:**
        Verifies that the union of heavy and light indices covers the full set
        ``{0, ..., h12-1}``, ensuring every complex-structure modulus is
        accounted for in the freezing partition.
        """
        h12 = self.model.h12
        all_indices = set(self.freezer.heavy_indices) | set(self.freezer.light_indices)
        # The union must equal the complete set of moduli indices
        self.assertEqual(all_indices, set(range(h12)))


if __name__ == "__main__":
    unittest.main()
