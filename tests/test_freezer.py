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
        h12=5, model_ID="aule", maximum_degree=5, limit="coniLCS",
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

    def test_ncf_property_reads_from_lcs_tree(self):
        r"""
        **Description:**
        Verifies that ``ncf`` is now exposed as a property that reads directly
        from ``model.lcs_tree.conifold.ncf`` (single source of truth).  The
        previous behaviour stored a copy via the ``ncf=`` constructor kwarg;
        that kwarg has been removed so the freezer never mirrors geometric
        data already carried by the model.
        """
        freezer = ConifoldFreezer(self.model)
        self.assertEqual(freezer.ncf, int(self.model.lcs_tree.conifold.ncf))

    def test_ncf_kwarg_no_longer_accepted(self):
        r"""
        **Description:**
        Constructing ``ConifoldFreezer(model, ncf=...)`` must raise
        ``TypeError`` after the kwarg removal.
        """
        with self.assertRaises(TypeError):
            ConifoldFreezer(self.model, ncf=3)

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


# ==============================================================================
#  TestConifoldFreezerIntegration — exercises solve_heavy / _real_light_to_full
#  end-to-end against the new conifold_utils API.  Uses the same "aule"
#  PromotionModels fixture as tests/test_conifold_bulk_eft.py so we get a
#  loadable coniLCS model regardless of the standalone FluxEFT(h12=2,
#  model_ID=1) fixture above.
# ==============================================================================

import pytest as _pytest

_INT_NAME  = "aule"
_INT_MVEC0 = np.array([20, 4, 8, -18, -20])
_INT_KVEC0 = np.array([-5, -1, 0, 1, -1])
_INT_PVEC0 = np.array([0.0, 0.020833333333333332, 0.041666666666666664,
                       0.020833333333333332, 0.0])
_INT_TAU0  = 1j / 0.04317129968232153
_INT_ATOL  = 1e-10

_INT_MODELS = None
_INT_PFV = None
_INT_LOAD_ERROR = None


def _try_load_int_models():
    """Build the (bulk, series, conilcs) PromotionModels triple from the
    'aule' lcs_tree, plus a PFV seed point. Returns (models, pfv) on success,
    or raises with a descriptive error."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    promo_dir = os.path.join(repo_root, "private", "promotion")
    if promo_dir not in sys.path:
        sys.path.insert(0, promo_dir)
    import vacuum_promotion as vp  # noqa: E402
    import jaxvacua as jvc  # noqa: E402

    lcs_tree = jvc.periods(h12=len(_INT_MVEC0), model_ID=_INT_NAME, limit="coniLCS").lcs_tree
    models = vp.PromotionModels.from_lcs_tree(
        lcs_tree, conifold_basis=True, ncf=2, prange=20, maximum_degree=2,
    )
    pfv = vp.PFV.from_quantum_numbers(
        models, M_vec=_INT_MVEC0, K_vec=_INT_KVEC0, p_vec=_INT_PVEC0, tau=_INT_TAU0,
        metadata={"model_name": _INT_NAME},
    )
    return models, pfv


try:
    _INT_MODELS, _INT_PFV = _try_load_int_models()
except Exception as _exc:
    _INT_LOAD_ERROR = f"{type(_exc).__name__}: {_exc}"


_NEEDS_INT_MODEL = _pytest.mark.skipif(
    _INT_MODELS is None,
    reason=f"PromotionModels unavailable ({_INT_LOAD_ERROR})",
)


@_NEEDS_INT_MODEL
class TestConifoldFreezerIntegration(TestCase):
    r"""
    Integration tests that pin :class:`ConifoldFreezer` to the new
    :func:`jaxvacua.conifold.zcf_solver.compute_zcf` and
    :func:`jaxvacua.conifold.zcf_solver.zcf_handling` dispatchers exactly.

    Each test runs against the ``models.bulk`` model from the "aule"
    ``PromotionModels`` fixture; this model has ``conifold_basis=True`` and
    ``maximum_degree=2`` (instanton corrections enabled).
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model   = _INT_MODELS.bulk
        cls.freezer = ConifoldFreezer(cls.model)
        cls.x_full  = jnp.asarray(_INT_PFV.x)
        cls.flux    = jnp.asarray(_INT_PFV.flux)
        h12 = len(_INT_MVEC0)
        # Bulk-only real vector: drop the first 2 real components (Re/Im of z_cf).
        cls.x_bulk  = jnp.concatenate([cls.x_full[2:2 * h12], cls.x_full[2 * h12:]])
        # Complex-coord pieces.
        z, _, tau, _ = cls.model._convert_real_to_complex(cls.x_full)
        cls.z_bulk = z[1:]
        cls.tau    = tau

    def test_solve_heavy_matches_compute_zcf(self):
        r"""
        ``freezer.solve_heavy(z_bulk, tau, flux, mode=m)[0]`` must match
        ``model.compute_zcf(z_bulk, jnp.conj(z_bulk), tau, jnp.conj(tau),
        flux, mode=m)`` for every dispatcher mode.
        """
        for m in ("manual", "autodiff", "pfv"):
            via_freezer = complex(self.freezer.solve_heavy(
                self.z_bulk, self.tau, self.flux, mode=m)[0])
            via_model = complex(self.model.compute_zcf(
                self.z_bulk, jnp.conj(self.z_bulk),
                self.tau, jnp.conj(self.tau),
                self.flux, mode=m,
            ))
            self.assertAllClose(via_freezer, via_model, atol=_INT_ATOL,
                                msg=f"mode={m}")

    def test_solve_heavy_apply_correction_toggle(self):
        r"""
        ``apply_correction=True`` must (a) differ from the default and
        (b) match ``model.compute_zcf(..., apply_correction=True)``.
        """
        z_off = complex(self.freezer.solve_heavy(
            self.z_bulk, self.tau, self.flux, mode="manual",
            apply_correction=False)[0])
        z_on  = complex(self.freezer.solve_heavy(
            self.z_bulk, self.tau, self.flux, mode="manual",
            apply_correction=True)[0])
        z_on_via_model = complex(self.model.compute_zcf(
            self.z_bulk, jnp.conj(self.z_bulk),
            self.tau, jnp.conj(self.tau),
            self.flux, mode="manual", apply_correction=True,
        ))
        # Toggle is non-trivial.
        self.assertGreater(abs(z_on - z_off), 1e-12 * abs(z_off))
        # And matches the model dispatcher exactly.
        self.assertAllClose(z_on, z_on_via_model, atol=_INT_ATOL)

    def test_real_light_to_full_matches_zcf_handling(self):
        r"""
        ``freezer._real_light_to_full(x_bulk, flux, mode=m)`` must equal
        ``model.zcf_handling(x_bulk, flux, mode=m)`` for every dispatcher mode.
        """
        for m in ("manual", "autodiff", "pfv"):
            via_freezer = self.freezer._real_light_to_full(
                self.x_bulk, self.flux, mode=m)
            via_model = self.model.zcf_handling(self.x_bulk, self.flux, mode=m)
            self.assertAllClose(via_freezer, via_model, atol=_INT_ATOL,
                                msg=f"mode={m}")

    # NOTE: previously this section had ``test_DW_x_light_matches_DWbulk_x``,
    # a deprecation-era bridge test pinning ``freezer.DW_x_light`` to
    # ``model.DWbulk_x``.  After 2026-05-01 ``DWbulk_x`` / ``dDWbulk_x`` were
    # hard-removed (vacuum_promotion.py migrated to the freezer interface).
    # The slice-equivalence semantics are now covered by
    # ``test_conifold_bulk_eft.py::TestDWxLight::test_slice_equivalence``,
    # which compares ``freezer.DW_x_light`` against
    # ``DW_x(zcf_handling(x_bulk, ...), flux)[2:]`` directly.

    def test_ncf_property_matches_lcs_tree(self):
        r"""
        ``freezer.ncf`` must read directly from
        ``model.lcs_tree.conifold.ncf`` after the kwarg removal.
        """
        self.assertEqual(self.freezer.ncf,
                         int(self.model.lcs_tree.conifold.ncf))


if __name__ == "__main__":
    unittest.main()
