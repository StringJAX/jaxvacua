# Copyright 2022-2026 Andreas Schachner
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

"""Conifold subsystem public interface.

Purpose
-------
Expose the conifold-specific data structures, structural helpers,
prepotential functions and conifold-modulus solvers used by the main
``periods``, ``css`` and ``FluxEFT`` classes.

Main public API
---------------
- ``Conifold`` and ``find_conifolds`` for representing and discovering
  conifold limits.
- Basis and flux helpers from ``conifold_utils`` such as ``get_basis_change``,
  ``compute_a_matrix``, ``get_projection``, ``conifold_fluxes`` and
  ``delete_coni_index``.
- coniLCS prepotential functions from ``coniLCS_prepotential``.
- ``z_cf`` physics helpers from ``zcf_solver`` such as ``W_log_coeff``,
  ``compute_zcf`` and ``zcf_handling``.

Design notes
------------
The ``_ConifoldGated`` descriptor exposes attached methods only for coniLCS
limits.  Consumer modules attach the methods they need locally, which avoids
load-order coupling between this subpackage and the main physics classes.
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
    compute_a_matrix,
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
    "get_basis_change", "compute_a_matrix", "get_projection",
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
