import sys
from pathlib import Path

_tests_dir = Path(__file__).resolve().parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

# Debug: print jaxpolylog location
def pytest_configure(config):
    import jaxpolylog
    print(f"\n*** jaxpolylog: {jaxpolylog.__file__}")
    print(f"*** has jax_polylog_vmap: {hasattr(jaxpolylog, 'jax_polylog_vmap')}")
    print(f"*** sys.path[0:5]: {sys.path[:5]}")
