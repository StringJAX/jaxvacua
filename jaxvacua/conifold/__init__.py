"""``jaxvacua.conifold`` — conifold subsystem.

Phase 2 split (2026-05-01): the contents of the old monolithic
``jaxvacua/conifold_utils.py`` are now distributed across:

    - :mod:`jaxvacua.conifold.conifold_utils`        — structural helpers
      (basis-change algebra, flux splitting, index manipulation).
    - :mod:`jaxvacua.conifold.coni`                  — :class:`Conifold` class
      + ``find_conifolds`` + pytree registration.
    - :mod:`jaxvacua.conifold.coniLCS_prepotential`  — coniLCS prepotential
      family (per-period + per-modulus, no flux dependence).
    - :mod:`jaxvacua.conifold.zcf_solver`            — z_cf physics
      (W_log_coeff, compute_zcf, zcf_handling, …).

This ``__init__.py`` defines the :class:`_ConifoldGated` descriptor and
re-exports the public symbols.  **Method attachment to the consumer classes
(``periods``, ``css``, ``FluxEFT``) lives in the consumer modules themselves**
— each consumer imports the symbols it needs from this package and runs its
own ``setattr`` block.  The locality matches the pre-split convention and
avoids load-order coupling between the conifold subpackage and the consumers.
"""

from types import MethodType


class _ConifoldGated:
    r"""
    **Description:**
    Class-level descriptor: surfaces a method only when the instance's limit
    is in the coniLCS family. Otherwise raises AttributeError so hasattr()
    returns False.

    Works for `periods` (reads `instance.limit` directly) and for `css` /
    `FluxEFT` (which delegate to `instance.periods.limit`).
    """
    def __init__(self, fn):
        self.fn = fn
        # Pre-populate ``__name__`` so the AttributeError message in ``__get__``
        # is informative when the descriptor is attached at *runtime* (via
        # ``setattr(cls, name, _ConifoldGated(fn))``) rather than as a class-
        # body assignment.  ``__set_name__`` only fires for class-body
        # assignments, so without this, ``self.__name__`` would itself raise
        # AttributeError inside ``__get__`` and mask the real "limit ∉ coniLCS
        # family" check.
        self.__name__ = getattr(fn, "__name__", "<conifold-method>")
    def __set_name__(self, owner, name): self.__name__ = name
    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        # Try direct (periods sets self.limit); fall back to nested periods.
        limit = instance.__dict__.get("limit")
        if limit is None:
            periods = instance.__dict__.get("periods")
            if periods is not None:
                limit = getattr(periods, "limit", None)
        if limit is None or "coniLCS" not in str(limit):
            raise AttributeError(
                f"{self.__name__!r} requires limit ∈ coniLCS family "
                f"(got {limit!r})"
            )
        return MethodType(self.fn, instance)


# ---- Submodule re-exports -------------------------------------------------
# Consumer modules (periods.py, css.py, flux_eft.py) can import from
# ``jaxvacua.conifold`` directly instead of reaching into the submodules.

# Structural helpers (basis algebra + flux/index manipulation).
from jaxvacua.conifold.conifold_utils import (        # noqa: E402,F401
    extended_euclidean,
    orthogonal_lattice,
    get_basis_change,
    getAMatrix,
    get_projection,
    conifold_fluxes,
    delete_coni_index,
)

# Conifold class + discovery.
from jaxvacua.conifold.coni import (                   # noqa: E402,F401
    Conifold, find_conifolds,
)

# Prepotential family (per-period + per-modulus).
from jaxvacua.conifold.coniLCS_prepotential import (   # noqa: E402,F401
    F_coniLCS_bulk_per,
    F_coniLCS_poly_split_per,
    dF_coniLCS_poly_per,
    ddF_coniLCS_poly_per,
    dddF_coniLCS_poly_per,
    ddddF_coniLCS_poly_per,
    F_inst_per_coni,
    F_coni_per,
    F_coniLCS_exp_per,
    F_coniLCS_series_per,
    F_coniLCS_bulk,
    F_coniLCS_exp,
    dF_coniLCS_exp,
    F_coniLCS_series,
)

# z_cf physics.
from jaxvacua.conifold.zcf_solver import (             # noqa: E402,F401
    W_bulk,
    dK_cf_bulk,
    log_prefactor,
    _split_conifold_bulk,
    _W_log_coeff_pfv,
    _W_log_coeff_manual,
    _W_log_coeff_autodiff,
    W_log_coeff,
    log_coeff_K_corr,
    _zcf_from_log_coeff,
    compute_zcf,
    compute_zcf_x,
    zcf_handling,
)


# ---- Method-name lists: single source of truth for attachments ----------
#
# Each consumer module (``periods.py``, ``css.py``, ``flux_eft.py``) imports
# the relevant list from here and runs its own ``setattr`` loop at the bottom
# of the module.  Adding a new attached method only requires:
#   1. defining it in the appropriate submodule, and
#   2. appending its name to the list below.
# No edit to the consumer module is needed.

_PERIODS_METHODS = (
    # From coniLCS_prepotential.py
    "F_coniLCS_series_per",
    "F_coniLCS_exp_per",
    "F_coni_per",
    "F_inst_per_coni",
    "F_coniLCS_poly_split_per",
    "dF_coniLCS_poly_per",
    "ddF_coniLCS_poly_per",
    "dddF_coniLCS_poly_per",
    "ddddF_coniLCS_poly_per",
    "F_coniLCS_bulk_per",
    # From conifold_utils.py
    "delete_coni_index",
)

_CSS_METHODS = (
    # From coniLCS_prepotential.py
    "F_coniLCS_bulk",
    "F_coniLCS_series",
    "F_coniLCS_exp",
    "dF_coniLCS_exp",
    # From zcf_solver.py
    "dK_cf_bulk",
)

_FLUXEFT_METHODS = (
    # From zcf_solver.py
    "compute_zcf",
    "compute_zcf_x",
    "zcf_handling",
    "W_bulk",
    "log_prefactor",
    "_W_log_coeff_manual",
    "_W_log_coeff_autodiff",
    "_W_log_coeff_pfv",
    "W_log_coeff",
    "log_coeff_K_corr",
    "_zcf_from_log_coeff",
    "_split_conifold_bulk",
    # From conifold_utils.py
    "conifold_fluxes",
)


__all__ = [
    # Geometric layer
    "Conifold", "find_conifolds",
    # Structural helpers
    "extended_euclidean", "orthogonal_lattice",
    "get_basis_change", "getAMatrix", "get_projection",
    "conifold_fluxes", "delete_coni_index",
    # z_cf physics (attached as methods to FluxEFT)
    "compute_zcf", "compute_zcf_x", "zcf_handling",
    "W_log_coeff", "_W_log_coeff_manual", "_W_log_coeff_autodiff", "_W_log_coeff_pfv",
    "log_prefactor", "log_coeff_K_corr", "_zcf_from_log_coeff",
    "W_bulk", "dK_cf_bulk", "_split_conifold_bulk",
    # Prepotential family (attached as methods to periods / css)
    "F_coniLCS_bulk", "F_coniLCS_exp", "dF_coniLCS_exp", "F_coniLCS_series",
    "F_coniLCS_bulk_per", "F_coniLCS_series_per", "F_coniLCS_exp_per",
    "F_coniLCS_poly_split_per",
    "dF_coniLCS_poly_per", "ddF_coniLCS_poly_per",
    "dddF_coniLCS_poly_per", "ddddF_coniLCS_poly_per",
    "F_inst_per_coni", "F_coni_per",
    # Descriptor + method-name lists used by the consumer modules' setattr blocks
    "_ConifoldGated",
    "_PERIODS_METHODS", "_CSS_METHODS", "_FLUXEFT_METHODS",
]
