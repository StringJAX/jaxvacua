import os
import sys
from pathlib import Path

# Force float64 BEFORE any JAX import elsewhere in the test session.
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax  # noqa: E402  (import after env-var set on purpose)
jax.config.update("jax_enable_x64", True)

_tests_dir = Path(__file__).resolve().parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

# Debug: print jaxpolylog location
def pytest_configure(config):
    import jaxpolylog
    print(f"\n*** jaxpolylog: {jaxpolylog.__file__}")
    print(f"*** has jax_polylog_vmap: {hasattr(jaxpolylog, 'jax_polylog_vmap')}")
    print(f"*** sys.path[0:5]: {sys.path[:5]}")
