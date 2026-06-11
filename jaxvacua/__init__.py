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

"""Top-level package interface for JAXVacua.

Purpose
-------
Configure JAX precision and expose the main public classes for analysing
Calabi-Yau compactifications, flux effective field theories, and flux vacua.

Main public API
---------------
- ``set_precision`` plus the module-level ``precision``, ``FLOAT`` and
  ``COMPLEX`` aliases used throughout the package.
- Geometry and periods: ``lcs_tree``, ``periods``, ``css``.
- Flux physics and searches: ``FluxEFT``, ``FluxVacuaFinder``,
  ``bounded_fluxes`` and ``data_sampler``.
- Reduced EFT and special-model helpers: ``Freezer``, ``ConifoldFreezer`` and
  ``HypergeometricModels``.
- Conifold, CYTools, utility and flux helper functions re-exported for common
  interactive workflows.

Import notes
------------
Precision is configured before importing the rest of JAXVacua.  The package
defaults to ``float64`` unless ``JAXVACUA_PRECISION`` is set, while Apple Metal
is redirected to CPU because the current calculations require complex numbers.
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

# ── Data directory + vault setters ───────────────────────────────────────
# These were moved to ``stringforge/__init__.py`` on 2026-04-30 alongside
# the cy_io / vacuavault extraction.  Use ``stringforge.set_data_dir``,
# ``stringforge.set_vault_dir``, and ``stringforge.set_vault_repo`` directly.
# Env vars ``STRINGFORGE_DATA_DIR`` / ``STRINGFORGE_VAULT`` /
# ``STRINGFORGE_VAULT_REPO`` (formerly ``JAXVACUA_*``) override the
# defaults; ``stringforge.data_dir`` is the global the I/O layer reads.
# ──────────────────────────────────────────────────────────────────────────

from .util import *
from .cytools_interface import *
# Conifold subsystem: 2026-05-01 Phase 2 split moved the contents of the old
# ``conifold_utils.py`` into the ``conifold/`` subpackage.  Importing it here
# (BEFORE ``periods``/``css``/``flux_eft``) makes the symbols visible to those
# modules' bottom-of-file ``setattr`` blocks via ``from jaxvacua import conifold``.
from .conifold import *
from .lcs import *
from .periods import *
from .css import *
from .flux_eft import *
from .flux_vacua_finder import *
from .sampling import *
from .hypergeometric_models import *
from .freezer import *

# ── Re-apply precision setting ────────────────────────────────────────────
# Some dependencies (e.g. jaxpolylog) re-enable x64 at import time.
# We enforce the correct setting after all imports are done.
if precision == "float32":
    _jax.config.update("jax_enable_x64", False)
# ──────────────────────────────────────────────────────────────────────────


__version__ = '0.1.1'
