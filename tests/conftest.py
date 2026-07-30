"""Pytest session configuration for the JAXVacua test suite.

Purpose
-------
Set test-wide JAX precision, configure local import paths and mark expensive
JIT-compilation tests as slow.

Main public API
---------------
- ``pytest_configure``: emits debugging information about the active
  ``jaxpolylog`` installation.
- ``pytest_collection_modifyitems``: applies the ``slow`` marker to tests
  following the ``__with_jit`` naming convention.

Design notes
------------
The float64 environment setting must happen before other test imports can
initialise JAX.
"""

import os
import sys
from pathlib import Path

# Force float64 BEFORE any JAX import elsewhere in the test session.
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax  # noqa: E402  (import after env-var set on purpose)
jax.config.update("jax_enable_x64", True)

import pytest  # noqa: E402

_tests_dir = Path(__file__).resolve().parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

# Debug: print jaxpolylog location
def pytest_configure(config):
    import jaxpolylog
    print(f"\n*** jaxpolylog: {jaxpolylog.__file__}")
    print(f"*** has jax_polylog_vmap: {hasattr(jaxpolylog, 'jax_polylog_vmap')}")
    print(f"*** sys.path[0:5]: {sys.path[:5]}")


def pytest_collection_modifyitems(config, items):
    # Tests following the "__with_jit" naming convention pay the full
    # JAX tracing + XLA compile cost and are auto-routed to the slow
    # suite. Covers both plain "test_foo__with_jit" and parametrized
    # "test_foo__with_jit[case-1]". See workflows/stringforge/jaxvacua_ci.md.
    for item in items:
        if item.name.endswith("__with_jit") or "__with_jit[" in item.name:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def precision_guard():
    """Snapshot and restore the global JAX/JAXVacua precision around each test.

    ``jaxvacua.set_precision`` mutates the process-global ``jax_enable_x64`` flag
    and the module-level ``FLOAT``/``COMPLEX`` dtypes; without a restore a stray
    precision change would leak into subsequent tests and make results
    order-dependent.  This does not pin a particular precision, so the
    ``JAXVACUA_PRECISION=float32`` whole-run mode is unaffected.
    """
    import jaxvacua as _jvc

    x64 = jax.config.jax_enable_x64
    flt = getattr(_jvc, "FLOAT", None)
    cplx = getattr(_jvc, "COMPLEX", None)
    prec = getattr(_jvc, "precision", None)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", x64)
        if flt is not None:
            _jvc.FLOAT = flt
        if cplx is not None:
            _jvc.COMPLEX = cplx
        if prec is not None:
            _jvc.precision = prec
