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

"""
**Description:**
JAXVacua: A library for analysing string compactifications and constructing string vacua.

Sub-modules:
	periods: Contains `periods` class implementing standard formulas in terms of the periods.
	css: Contains `css` class containing functions to handle
		the Kähler geometry of the complex structure sector.
	flux_eft: Contains `FluxEFT` class for computations of the flux scalar potential induced by
		3-form flux backgrounds and EFT conditions.
	flux_vacua_finder: Contains `FluxVacuaFinder` class for searching and constructing flux vacua.
	util: Contains utility functions.

"""

# ── Precision detection ───────────────────────────────────────────────────
# Must run before any other jaxvacua import to configure JAX correctly.
# Metal (Apple GPU) only supports float32; CPU and CUDA support float64.
import jax as _jax
import jax.numpy as _jnp

def set_precision(mode="float64"):
    """Set the floating-point precision for all JAXVacua computations.

    Args:
        mode (str): ``"float64"`` (default, full precision) or ``"float32"``
            (reduced precision, required for some GPU backends).

    Example::

        import jaxvacua as jvc
        jvc.set_precision("float32")   # switch to single precision
        print(jvc.precision)           # "float32"
        print(jvc.FLOAT)              # jnp.float32
    """
    global precision, FLOAT, COMPLEX
    if mode == "float64":
        precision = "float64"
        _jax.config.update("jax_enable_x64", True)
        FLOAT = _jnp.float64
        COMPLEX = _jnp.complex128
    elif mode == "float32":
        precision = "float32"
        _jax.config.update("jax_enable_x64", False)
        FLOAT = _jnp.float32
        COMPLEX = _jnp.complex64
    else:
        raise ValueError(f"Unknown precision mode '{mode}'. Use 'float64' or 'float32'.")

# Auto-detect precision based on backend
_backend = _jax.default_backend()
if _backend == "METAL":
    import warnings as _warnings
    _warnings.warn(
        "JAXVacua detected the Apple Metal backend. Metal does not currently "
        "support complex numbers, which are required for period and superpotential "
        "calculations. Falling back to CPU. For GPU acceleration, use CUDA.",
        stacklevel=2,
    )
    _jax.config.update("jax_default_device", _jax.devices("cpu")[0])

# Default: float64, overridable via JAXVACUA_PRECISION env var
import os as _os
_default_prec = _os.environ.get("JAXVACUA_PRECISION", "float64")
set_precision(_default_prec)
# ──────────────────────────────────────────────────────────────────────────

# ── Data directory ────────────────────────────────────────────────────────
# Default location for all database cache and vacua storage.
# Override with JAXVACUA_DATA_DIR env var or jvc.set_data_dir().
_DEFAULT_DATA_DIR = _os.path.join(_os.getcwd(), ".jaxvacua_cache")
data_dir = _os.environ.get("JAXVACUA_DATA_DIR", _DEFAULT_DATA_DIR)

def set_data_dir(path):
    r"""
    **Description:**
    Set the global data directory for all jaxvacua database operations
    (HuggingFace cache, vacua storage, designated solutions).

    New :class:`~jaxvacua.lcs_database.LCSDatabase` instances created after
    this call will use the specified directory unless overridden by an
    explicit ``cache_dir`` argument.

    Args:
        path (str | Path): Absolute or relative path to the data
            directory.  The directory is created on first use.

    Returns:
        None
    """
    global data_dir
    data_dir = str(path)
# ──────────────────────────────────────────────────────────────────────────

# ── Vacua vault directory ────────────────────────────────────────────────
# Permanent storage for designated vacuum solutions.  Resolves in priority:
#   1. JAXVACUA_VAULT env var (explicit override), or value set via
#      jvc.set_vault_dir(path)
#   2. <repo_root>/vacua_vault/ when inside a jaxvacua source checkout
#   3. <cwd>/vacua_vault/ otherwise
# The vault is **not** under the cache dir — it persists across
# clear_cache() calls.

def set_vault_dir(path):
    r"""
    **Description:**
    Set the vault directory for designated vacuum solutions by
    exporting ``JAXVACUA_VAULT`` into the environment.  Takes effect
    for all subsequent :class:`~jaxvacua.lcs_database.LCSDatabase` calls.

    Args:
        path (str | Path): Absolute or relative path to the vault
            directory.  Pass ``None`` to clear the override and fall
            back to repo-root / cwd auto-detection.

    Returns:
        None
    """
    if path is None:
        _os.environ.pop("JAXVACUA_VAULT", None)
    else:
        _os.environ["JAXVACUA_VAULT"] = str(path)


def set_vault_repo(repo_id):
    r"""
    **Description:**
    Set the HuggingFace dataset repo ID used for uploading / fetching
    community vacuum solutions.  Sets the ``JAXVACUA_VAULT_REPO`` env
    var.

    Args:
        repo_id (str | None): ``"user/repo"`` on HuggingFace Hub, or
            ``None`` to clear the override and fall back to the
            package default ``aschachner/vacua_vault``.

    Returns:
        None
    """
    if repo_id is None:
        _os.environ.pop("JAXVACUA_VAULT_REPO", None)
    else:
        _os.environ["JAXVACUA_VAULT_REPO"] = str(repo_id)
# ──────────────────────────────────────────────────────────────────────────

from .util import *
from .utils_jaxvacua import *
from .cytools_interface import *
from .conifold_utils import *
from .lcs import *
from .periods import *
from .css import *
from .flux_eft import *
from .flux_vacua_finder import *
from .sampling import *

# ── Re-apply precision setting ────────────────────────────────────────────
# Some dependencies (e.g. jaxpolylog) re-enable x64 at import time.
# We enforce the correct setting after all imports are done.
if precision == "float32":
    _jax.config.update("jax_enable_x64", False)
# ──────────────────────────────────────────────────────────────────────────


__version__ = '0.1.0'
