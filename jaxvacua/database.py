# ==============================================================================
# This code is written by Andreas Schachner. Without the author's permission, this
# code must not be shared with anyone else or used for any other projects than
# those involving the author directly.
#
# If any questions arise, please feel free to reach out to me (Andreas) either at
# andreas.schachner@gmx.net or at as3475@cornell.edu or at a.schachner@lmu.de.
# ==============================================================================
#
# ------------------------------------------------------------------------------
# This file implements the CYDatabase interface for loading Calabi-Yau geometry
# data from a HuggingFace dataset repository.
#
# The database is organised into sub-datasets identified by a short tag:
#   - "tdf": toric divisor flux models from the Kreuzer-Skarke list, identified
#            by (ks_id, triang_id).
#   - "cicy": Complete Intersection Calabi-Yau models, identified by cicy_id.
#
# Layout of the HuggingFace repository (one config per sub-dataset):
#
#   <HF_REPO_ID>/<dataset>/
#       catalog.parquet           — lightweight index; downloaded once and cached
#       lcs_data/h11_{N}/
#           data-{shard_id}.parquet   — geometry: intnums, c2, cone data, ...
#       gv/h11_{N}/
#           data-{shard_id}.parquet   — Gopakumar-Vafa / Gromov-Witten invariants
#       conifolds/h11_{N}/
#           data-{shard_id}.parquet   — one row per (ks_id, triang_id, conifold_id)
#       extra/
#           data-{shard_id}.parquet   — additional precomputed model properties
#       polytope/
#           data-{shard_id}.parquet   — polytope vertex data (tdf only)
#
# The catalog stores shard pointers (shard_id + row_index) for every split, so a
# single in-memory lookup suffices to locate any piece of data without scanning.
# ------------------------------------------------------------------------------


# Important standard libraries
import datetime
import hashlib
import os
import json
import threading
import uuid
import warnings
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# Important numerical libraries
import numpy as np


# ---------------------------------------------------------------------------
# Module-level constants — update HF_REPO_ID once the repository is live.
# ---------------------------------------------------------------------------

#: HuggingFace repository identifier.  Override with the ``JAXVACUA_HF_REPO``
#: environment variable, e.g. for private forks.
HF_REPO_ID: str = os.environ.get("JAXVACUA_HF_REPO", "aschachner/cy-database")

#: Threshold in MB for the one-time cache size warning.
_CACHE_SIZE_WARNING_MB: int = 500


def _get_default_data_dir() -> str:
    r"""
    **Description:**
    Return the current global data directory from ``jaxvacua.data_dir``.

    Falls back to ``~/.cache/jaxvacua`` if jaxvacua is not importable
    (e.g. when ``database.py`` is used standalone).

    Returns:
        str: Path to the data directory.
    """
    try:
        from . import data_dir
        return data_dir
    except ImportError:
        return os.path.join(Path.home(), ".jaxvacua_cache")


#: Default local cache directory (kept for backwards compatibility).
DEFAULT_CACHE_DIR: str = _get_default_data_dir()

#: Name of the permanent vacuum-solutions directory.
VAULT_DIRNAME: str = "vacua_vault"

#: Default HuggingFace dataset repository for community vacuum uploads.
DEFAULT_VAULT_REPO: str = "aschachner/vacua_vault"


def _resolve_vault_repo() -> str:
    r"""
    **Description:**
    Return the HuggingFace dataset repo ID for vacua uploads.  Honours
    ``JAXVACUA_VAULT_REPO`` env var, falling back to
    :data:`DEFAULT_VAULT_REPO`.
    """
    return os.environ.get("JAXVACUA_VAULT_REPO", DEFAULT_VAULT_REPO)


def _find_jaxvacua_repo_root(start: Optional[Path] = None) -> Optional[Path]:
    r"""
    **Description:**
    Walk up from *start* (default: cwd) looking for a jaxvacua source
    checkout.  A directory counts as the repo root when it contains both
    a ``setup.py`` with ``name="jaxvacua"`` and a ``jaxvacua/`` package
    directory.

    Args:
        start (Path | None): Starting directory.  Defaults to cwd.

    Returns:
        Path | None: Absolute path to the repo root, or None if not
        found.
    """
    current = Path(start or os.getcwd()).resolve()
    for candidate in [current, *current.parents]:
        setup_py = candidate / "setup.py"
        pkg_dir  = candidate / "jaxvacua"
        if setup_py.is_file() and pkg_dir.is_dir():
            try:
                text = setup_py.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if 'name="jaxvacua"' in text or "name='jaxvacua'" in text:
                return candidate
    return None


def _resolve_vault_dir() -> Path:
    r"""
    **Description:**
    Resolve the location of the permanent vacuum-solutions directory
    (``vacua_vault/``).  Resolution order:

    1. ``JAXVACUA_VAULT`` env var (explicit override).
    2. ``<repo_root>/vacua_vault/`` when cwd is inside a jaxvacua source
       checkout.
    3. ``<cwd>/vacua_vault/`` otherwise.

    The directory is **not** created by this function; callers create it
    on demand.

    Returns:
        Path: Absolute path to the vault directory (may not exist yet).
    """
    override = os.environ.get("JAXVACUA_VAULT")
    if override:
        return Path(override).expanduser().resolve()

    repo_root = _find_jaxvacua_repo_root()
    if repo_root is not None:
        return repo_root / VAULT_DIRNAME

    return Path(os.getcwd()).resolve() / VAULT_DIRNAME

#: Recognised sub-dataset identifiers and their model_type strings.
_DATASET_CONFIGS: Dict[str, str] = {
    "tdf":  "KS",
    "cicy": "CICY",
}

# ---------------------------------------------------------------------------
# Schema versioning
# ---------------------------------------------------------------------------

#: Schema version that this version of the code was written against.
#: Increment whenever a breaking change is made to any parquet file in the
#: database (column rename, removal, type change, required new column).
SCHEMA_VERSION: int = 1

#: Human-readable description of what changed in each schema version.
SCHEMA_CHANGELOG: Dict[int, str] = {
    1: "Initial versioned schema. Conifold basis-change matrix stored as "
       "'basis_change' column; 'n_conifolds' and 'D3_tadpole' added to "
       "catalog.parquet; conifold_catalog.parquet introduced.",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SchemaVersionError(RuntimeError):
    r"""
    Raised when the schema version of the local cache does not match the
    version expected by this release of the code.
    """


class ValidationError(RuntimeError):
    r"""
    Raised when vacuum solutions fail validation checks before designation.
    The ``report`` attribute contains per-solution diagnostics.
    """

    def __init__(self, message: str, report: Optional[List[dict]] = None):
        """Initialise with message and optional per-solution diagnostic report."""
        super().__init__(message)
        self.report = report or []


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _require_pandas() -> Any:
    r"""
    **Description:**
    Import and return the ``pandas`` module, raising a clear error if absent.
    """
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise ImportError(
            "The 'pandas' package is required to use CYDatabase.  "
            "Install it with:  pip install pandas"
        )


def _require_pyarrow() -> Any:
    r"""
    **Description:**
    Import and return the ``pyarrow`` module, raising a clear error if absent.
    """
    try:
        import pyarrow.parquet as pq
        return pq
    except ImportError:
        raise ImportError(
            "The 'pyarrow' package is required to use CYDatabase.  "
            "Install it with:  pip install pyarrow"
        )


def _require_hf_hub() -> Any:
    r"""
    **Description:**
    Import and return the ``hf_hub_download`` function, raising a clear
    error if ``huggingface_hub`` is absent.
    """
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download
    except ImportError:
        raise ImportError(
            "The 'huggingface_hub' package is required to use CYDatabase.  "
            "Install it with:  pip install huggingface-hub"
        )


def _require_hf_api() -> Any:
    r"""
    **Description:**
    Import and return a fresh ``HfApi`` instance, raising a clear error
    if ``huggingface_hub`` is absent.  Used for write operations
    (uploading files, opening pull requests, identifying the
    authenticated user).
    """
    try:
        from huggingface_hub import HfApi
        return HfApi()
    except ImportError:
        raise ImportError(
            "The 'huggingface_hub' package is required for upload "
            "operations. Install it with:  pip install huggingface-hub"
        )


def _safe_concat(frames: List[Any], **kwargs: Any) -> Any:
    r"""
    Concatenate DataFrames, filtering out empty frames to avoid the pandas
    ``FutureWarning`` about all-NA columns during concatenation.
    """
    pd = _require_pandas()
    non_empty = [f for f in frames if len(f) > 0]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, **kwargs)


def _decode_array(value: Any) -> Optional[np.ndarray]:
    r"""
    **Description:**
    Convert a parquet column entry to a numpy array.

    Parquet stores multi-dimensional arrays either as nested lists or as
    bytes-serialised numpy arrays (depending on the serialisation scheme used
    when the dataset was written).  This helper normalises both cases.

    Args:
        value: Raw column value from a parquet row.

    Returns:
        ``np.ndarray`` if the value is array-like, ``None`` if the value is
        ``None`` or a pandas ``NA``.
    """
    if value is None:
        return None
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.ndarray):
        return value
    return np.array(value)


def _parse_lcs_row(row: Any, dataset: str) -> Dict[str, Any]:
    r"""
    **Description:**
    Convert a single row from the ``lcs_data`` parquet split into a keyword-
    argument dictionary suitable for constructing an :class:`~jaxvacua.lcs.lcs_tree`.

    The expected column names in the parquet row are:

    .. code-block:: text

        h11, h12, chi,
        intnums_coo_i, intnums_coo_j, intnums_coo_k, intnums_coo_v,
        c2, a_matrix,
        hyperplanes, kahler_generators, kahler_rays, mori_rays,
        polytope_points,  # tdf only
        heights,          # tdf only
        extra_data        # JSON string

    Args:
        row: A ``pandas.Series`` corresponding to one row of the parquet file.
        dataset (str): Sub-dataset identifier (``"tdf"`` or ``"cicy"``).

    Returns:
        Dict[str, Any]: Keyword arguments for ``lcs_tree.__init__``.
    """
    # ---- Reconstruct sparse intersection-number array (COO format) ----------
    coo_i = _decode_array(row.get("intnums_coo_i"))
    coo_j = _decode_array(row.get("intnums_coo_j"))
    coo_k = _decode_array(row.get("intnums_coo_k"))
    coo_v = _decode_array(row.get("intnums_coo_v"))

    if all(x is not None for x in (coo_i, coo_j, coo_k, coo_v)):
        intnums_coo = np.stack([coo_i, coo_j, coo_k, coo_v], axis=1)
    else:
        intnums_coo = None

    # ---- Parse extra_data JSON string ---------------------------------------
    raw_extra = row.get("extra_data")
    if isinstance(raw_extra, str):
        try:
            extra_data = json.loads(raw_extra)
        except json.JSONDecodeError:
            extra_data = {}
    elif isinstance(raw_extra, dict):
        extra_data = raw_extra
    else:
        extra_data = {}

    # ---- Inject primary-key fields into extra_data -------------------------
    if dataset == "tdf":
        extra_data.setdefault("ks_id",     int(row.get("ks_id",     -1)))
        extra_data.setdefault("triang_id", int(row.get("triang_id",  0)))
        model_ID = int(row.get("ks_id", -1))
    else:
        extra_data.setdefault("cicy_id", int(row.get("cicy_id", -1)))
        model_ID = int(row.get("cicy_id", -1))

    kwargs = dict(
        h11          = int(row["h11"]),
        h12          = int(row["h12"]),
        chi          = int(row["chi"])  if "chi"  in row.index else None,
        intnums_coo  = intnums_coo,
        c2           = _decode_array(row.get("c2")),
        a_matrix     = _decode_array(row.get("a_matrix")),
        hyperplanes  = _decode_array(row.get("hyperplanes")),
        kahler_generators = _decode_array(row.get("kahler_generators")),
        kahler_rays  = _decode_array(row.get("kahler_rays")),
        mori_rays    = _decode_array(row.get("mori_rays")),
        tip_skc    = _decode_array(row.get("tip_skc")),
        model_type   = _DATASET_CONFIGS[dataset],
        model_ID     = model_ID,
        extra_data   = extra_data or None,
    )

    # KS-specific fields
    if dataset == "tdf":
        kwargs["polytope_points"] = _decode_array(row.get("polytope_points"))
        kwargs["heights"]         = _decode_array(row.get("heights"))

    return kwargs


def _parse_gv_row(row: Any) -> Dict[str, Any]:
    r"""
    **Description:**
    Convert a single row from the ``gv`` parquet split into a dictionary of
    Gopakumar-Vafa / Gromov-Witten data.

    Expected columns: ``gv_charges``, ``gv_invariants``, ``gw_charges``,
    ``gw_invariants``.

    Args:
        row: A ``pandas.Series`` from the GV parquet file.

    Returns:
        Dict[str, Any]: Dictionary with keys ``"GVs"`` and ``"GWs"``, each a
        dict with ``"charges"`` and ``"invariants"`` arrays.
    """
    return {
        "GVs": {
            "charges":    _decode_array(row.get("gv_charges")),
            "invariants": _decode_array(row.get("gv_invariants")),
        },
        "GWs": {
            "charges":    _decode_array(row.get("gw_charges")),
            "invariants": _decode_array(row.get("gw_invariants")),
        },
        "grading_vector": _decode_array(row.get("grading_vector")),
    }


def _parse_conifold_rows(rows: Any) -> List[Dict[str, Any]]:
    r"""
    **Description:**
    Convert all rows belonging to one model from the ``conifolds`` parquet
    split into a list of conifold-data dictionaries.

    Expected columns per row: ``conifold_id``, ``conifold_curve``, ``ncf``,
    ``basis_change``, ``flop_edge``, ``one_face_divisors``.

    Args:
        rows: A ``pandas.DataFrame`` containing all conifold rows for one model
            (i.e. all rows sharing the same ``ks_id`` and ``triang_id``).

    Returns:
        List[Dict[str, Any]]: One dict per conifold, sorted by ``conifold_id``.
    """
    result = []
    for _, row in rows.sort_values("conifold_id").iterrows():
        result.append({
            "conifold_id":       int(row["conifold_id"]),
            "conifold_curve":    _decode_array(row.get("conifold_curve")),
            "ncf":               int(row["ncf"]) if ("ncf" in row.index and row["ncf"] is not None) else 2,
            "basis_change":      _decode_array(row.get("basis_change")),
            "flop_edge":         _decode_array(row.get("flop_edge")),
            "one_face_divisors": _decode_array(row.get("one_face_divisors")),
        })
    return result


def _parse_polytope_row(row: Any) -> Dict[str, Any]:
    r"""
    **Description:**
    Convert a single row from the ``polytope`` parquet split into a dict.

    The mandatory column is ``polytope_points``.  Any additional columns
    produced by ``process_polytope`` (e.g. ``is_favorable``, ``volume``,
    ``glsm_charge_matrix``) are collected into a ``"polytope_data"`` sub-dict
    so they do not collide with ``lcs_tree`` constructor arguments.

    Args:
        row: A ``pandas.Series`` from the polytope parquet file.

    Returns:
        Dict[str, Any]: Contains ``"polytope_points"`` and optionally
        ``"polytope_data"`` with any extra fields.
    """
    _RESERVED = {"ks_id", "h11", "h12", "polytope_points"}
    result: Dict[str, Any] = {
        "polytope_points": _decode_array(row.get("polytope_points")),
    }
    extra = {
        k: (v.tolist() if hasattr(v, "tolist") else v)
        for k, v in row.items()
        if k not in _RESERVED and v is not None
    }
    if extra:
        result["polytope_data"] = extra
    return result


# ---------------------------------------------------------------------------
# In-memory shard cache
# ---------------------------------------------------------------------------

class _LRUShardCache:
    r"""
    Thread-safe LRU cache for parquet shard DataFrames.

    Keeps the ``maxsize`` most-recently-used ``(split, shard_id)`` shards in
    memory so that repeated loads within the same h11 bucket avoid redundant
    disk reads.

    Args:
        maxsize (int): Maximum number of shards to keep in memory.
    """

    def __init__(self, maxsize: int = 32) -> None:
        """Initialise the LRU shard cache with a given maximum size."""
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Any:
        """Return the cached DataFrame for *key*, or ``None`` if not cached."""
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: tuple, value: Any) -> None:
        """Insert *value* under *key*, evicting the LRU entry if full."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    def clear(self) -> None:
        """Evict all cached shards."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CYDatabase:
    r"""
    **Description:**
    Interface for loading Calabi-Yau geometry data from a HuggingFace dataset
    repository.

    The database is organised into sub-datasets (``dataset`` parameter).
    Currently supported:

    - ``"tdf"``: toric models from the Kreuzer-Skarke list, identified by
      ``(ks_id, triang_id)``.
    - ``"cicy"``: Complete Intersection Calabi-Yau models, identified by
      ``cicy_id``.

    A lightweight catalog is downloaded once and kept in memory.  Geometry
    data (``lcs_data`` split), GV invariants, conifold data, and additional
    model properties (``extra`` split) are fetched on demand and cached
    locally.

    Example usage::

        db = CYDatabase(dataset="tdf")
        db.info()

        # Browse available models
        df = db.query(h11=2, h12=128)

        # Load a single model as an lcs_tree
        tree = db.load(ks_id=12345, triang_id=0)
        tree = db.load(ks_id=12345, triang_id=0, include_gv=True, maximum_degree=5)
        tree = db.load(ks_id=12345, triang_id=0, include_conifolds=True)

        # Load directly as a flux_sector model
        model = db.load_model(ks_id=12345, triang_id=0, Q=24)

        # Batch and random sampling
        trees = db.load_batch(df.head(20))
        trees = db.sample(h11=2, n=10, seed=42)

    """

    def __init__(
        self,
        dataset: str = "tdf",
        hf_repo: str = HF_REPO_ID,
        cache_dir: Optional[str] = None,
        offline: bool = False,
        shard_cache_size: int = 32,
        cache_mode: str = "persistent",
    ) -> None:
        r"""
        **Description:**
        Initialise a :class:`CYDatabase` instance.

        The catalog parquet file is downloaded on first use and cached to
        ``cache_dir``.  Subsequent calls are served from disk.  Set
        ``offline=True`` to suppress any network access.

        Args:
            dataset (str): Sub-dataset identifier.  Must be one of
                ``"tdf"`` (Kreuzer-Skarke toric models) or ``"cicy"``
                (Complete Intersection CY models).  Defaults to ``"tdf"``.
            hf_repo (str): HuggingFace repository identifier of the form
                ``"org/repo-name"``.  Defaults to :data:`HF_REPO_ID`.
            cache_dir (str | None): Local directory for cached parquet
                files and vacua storage.  If ``None`` (default), uses the
                global ``jaxvacua.data_dir`` (which defaults to
                ``{cwd}/.jaxvacua`` and can be overridden via
                :func:`jaxvacua.set_data_dir` or the
                ``JAXVACUA_DATA_DIR`` environment variable).
            offline (bool): If ``True``, only serve data from the local
                cache and raise an error on any cache miss.  Defaults to
                ``False``.
            shard_cache_size (int): Number of parquet shards to keep in
                memory after loading.  Repeated accesses to the same shard
                (common when iterating over models with the same h11) are
                served from memory without a disk read.  Set to ``0`` to
                disable the in-memory cache.  Defaults to ``32``.
            cache_mode (str): Controls how downloaded shard files are
                managed.  ``"persistent"`` (default) keeps files on disk
                and in the LRU cache for repeated use.  ``"none"``
                downloads each shard, reads the needed row, then deletes
                the file — ideal for scanning millions of models without
                accumulating disk usage.

        Raises:
            ValueError: If ``dataset`` is not a recognised identifier.
        """
        if cache_dir is None:
            cache_dir = _get_default_data_dir()

        if dataset not in _DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset '{dataset}'.  "
                f"Choose one of: {list(_DATASET_CONFIGS)}"
            )

        self.dataset    = dataset
        self.hf_repo    = hf_repo
        self.cache_dir  = Path(cache_dir) / dataset
        self.offline    = offline
        self.cache_mode = cache_mode

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Catalog DataFrames — loaded lazily on first use.
        self._catalog              = None
        self._conifold_catalog     = None
        self._vacua_catalog        = None
        self._designated_catalog   = None

        # In-memory LRU cache for recently loaded parquet shards.
        self._shard_cache = _LRUShardCache(maxsize=shard_cache_size)

        # One-time flag for the cache-size warning.
        self._size_warned = False

        # Lock that serialises file downloads so concurrent threads never
        # race to download the same shard simultaneously.
        self._download_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_local(
        cls,
        path: Union[str, "Path"],
        dataset: str = "tdf",
        shard_cache_size: int = 32,
    ) -> "CYDatabase":
        r"""
        **Description:**
        Create a :class:`CYDatabase` backed by a local directory, with no
        network access.

        This is the recommended way to work with a database that was built
        locally (e.g. via ``build_tdf_database``).

        The ``path`` should point to the directory that **contains** the
        sub-dataset folder.  For example, if the catalog lives at
        ``/data/mydb/tdf/catalog.parquet``, pass ``path="/data/mydb"``.
        If ``path`` itself already ends with the dataset name (e.g.
        ``/data/mydb/tdf``), the parent is used automatically.

        Example::

            db = CYDatabase.from_local("/data/mydb")
            db = CYDatabase.from_local("/data/mydb/tdf")   # also works
            cat = db.query(h11=3)

        Args:
            path (str | Path): Root directory of the local database.
            dataset (str): Sub-dataset identifier (``"tdf"`` or ``"cicy"``).
            shard_cache_size (int): In-memory shard cache size.

        Returns:
            CYDatabase: A fully offline database instance.
        """
        path = Path(path)
        # If the user passed the sub-dataset directory itself, step up.
        if path.name == dataset and (path / "catalog.parquet").exists():
            path = path.parent
        return cls(
            dataset=dataset,
            cache_dir=str(path),
            offline=True,
            shard_cache_size=shard_cache_size,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        r"""
        **Description:**
        Returns a string representation of the database object.

        Returns:
            str: Description of the database.
        """
        status = "catalog loaded" if self._catalog is not None else "catalog not yet loaded"
        return (
            f"CYDatabase(dataset='{self.dataset}', "
            f"hf_repo='{self.hf_repo}', {status})"
        )

    # ------------------------------------------------------------------
    # Schema versioning
    # ------------------------------------------------------------------

    def _check_schema(self) -> None:
        r"""
        **Description:**
        Verify that the locally cached database was built against the same
        schema version that this release of the code expects.

        Downloads ``schema.json`` from the HuggingFace repository on first
        call (or reads it from the local cache).  Compares the stored
        ``schema_version`` against :data:`SCHEMA_VERSION` and raises
        :exc:`SchemaVersionError` if they differ, with a message that
        explains exactly what changed and how to fix the problem.

        When ``offline=True`` and no ``schema.json`` is cached, the check is
        skipped with a warning rather than blocking the user.

        Raises:
            SchemaVersionError: If the cached schema version does not match
                :data:`SCHEMA_VERSION`.
        """
        path = self.cache_dir / "schema.json"

        if not path.exists():
            if self.offline:
                warnings.warn(
                    "schema.json not found in local cache and offline=True; "
                    "schema version check skipped.",
                    stacklevel=3,
                )
                return
            try:
                self._download_file(f"{self.dataset}/schema.json", path)
            except RuntimeError:
                # schema.json may not exist yet for databases built before
                # versioning was introduced — skip the check gracefully.
                warnings.warn(
                    "schema.json not found in the remote database; "
                    "schema version check skipped.  Consider rebuilding "
                    "the database to enable versioning.",
                    stacklevel=3,
                )
                return

        with open(path) as f:
            stored = json.load(f).get("schema_version")

        if stored is None or stored == SCHEMA_VERSION:
            return

        if stored < SCHEMA_VERSION:
            changes = "\n".join(
                f"  v{v}: {msg}"
                for v, msg in SCHEMA_CHANGELOG.items()
                if v > stored
            )
            raise SchemaVersionError(
                f"Local cache has schema v{stored}, but jaxvacua expects "
                f"v{SCHEMA_VERSION}.\n"
                f"Changes since your cached version:\n{changes}\n"
                f"Run db.clear_cache() to delete the stale cache and "
                f"re-download the latest database."
            )

        raise SchemaVersionError(
            f"Local cache has schema v{stored}, but this version of jaxvacua "
            f"only supports up to v{SCHEMA_VERSION}.  "
            f"Please upgrade jaxvacua."
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(
        self,
        include_vacua: bool = False,
        include_designated: bool = False,
        keep_vault: bool = True,
    ) -> None:
        r"""
        **Description:**
        Delete all locally cached files for this dataset and reset the
        in-memory catalog.

        After calling this method, the next access will re-download the
        catalog, schema, and any requested shards from the HuggingFace
        repository.  Useful after a schema version mismatch or when the
        remote database has been updated.

        The **permanent vacua vault** (``vacua_vault/`` — see
        :func:`_resolve_vault_dir`) is **never** touched by this method:
        it lives outside the cache directory and is protected by design.
        The *include_designated* flag only controls the legacy
        ``<cache_dir>/designated_vacua/`` location (used as a fallback
        for pre-vault installations).

        By default, session vacua (``<cache_dir>/vacua/``) and the
        legacy designated dir are preserved.  Pass
        ``include_vacua=True`` / ``include_designated=True`` to delete
        them as well.

        Args:
            include_vacua (bool): If True, also delete the
                ``<cache_dir>/vacua/`` subdirectory containing session
                vacuum solutions.  Defaults to ``False``.
            include_designated (bool): If True, also delete the legacy
                ``<cache_dir>/designated_vacua/`` subdirectory.  The
                modern vault (``vacua_vault/``) is unaffected and must
                be removed manually if desired.  Defaults to ``False``.
            keep_vault (bool): Reserved for symmetry; ``vacua_vault/``
                is always protected and this flag has no effect in
                normal use.  Defaults to ``True``.

        Raises:
            RuntimeError: If ``offline=True``, since clearing the cache
                would leave the database inaccessible.
        """
        if self.offline:
            raise RuntimeError(
                "Cannot clear cache while offline=True — doing so would "
                "leave the database inaccessible.  Set offline=False first."
            )
        # keep_vault is documented for API symmetry; the vault lives
        # outside self.cache_dir so this method cannot touch it anyway.
        del keep_vault
        import shutil
        vacua_dir      = self.cache_dir / "vacua"
        designated_dir = self.cache_dir / "designated_vacua"
        protected: set = set()
        if not include_vacua and vacua_dir.exists():
            protected.add(vacua_dir)
        if not include_designated and designated_dir.exists():
            protected.add(designated_dir)

        if protected:
            for item in self.cache_dir.iterdir():
                if item in protected:
                    continue
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
        else:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._catalog              = None
        self._conifold_catalog     = None
        self._vacua_catalog        = None
        self._designated_catalog   = None
        self._shard_cache.clear()

    def _check_cache_size(self) -> None:
        r"""
        **Description:**
        Issue a one-time warning if the data directory exceeds
        :data:`_CACHE_SIZE_WARNING_MB` (default 500 MB).

        Called automatically on the first :meth:`load` or
        :meth:`vacua_writer` invocation.

        Returns:
            None
        """
        if self._size_warned:
            return
        try:
            total = sum(
                f.stat().st_size
                for f in self.cache_dir.rglob("*")
                if f.is_file()
            )
        except OSError:
            return
        if total > _CACHE_SIZE_WARNING_MB * 1024 * 1024:
            warnings.warn(
                f"jaxvacua data directory at '{self.cache_dir}' is "
                f"{total / 1024**2:.0f} MB. "
                f"Use db.clear_cache() to free space.",
                stacklevel=3,
            )
        self._size_warned = True

    # ------------------------------------------------------------------
    # Catalog management
    # ------------------------------------------------------------------

    def _ensure_catalog(self) -> None:
        r"""
        **Description:**
        Download the catalog parquet file if not already cached, then load it
        into ``self._catalog`` as a ``pandas.DataFrame``.

        The catalog is the central index: it maps every ``(ks_id, triang_id)``
        (or ``cicy_id``) to shard pointers for all data splits, enabling O(1)
        lookups without scanning the full dataset.

        Raises:
            FileNotFoundError: If ``offline=True`` and the catalog is not in
                the local cache.
            SchemaVersionError: If the cached database schema does not match
                :data:`SCHEMA_VERSION`.
        """
        if self._catalog is not None:
            return

        self._check_schema()

        pd = _require_pandas()
        catalog_path = self.cache_dir / "catalog.parquet"

        if not catalog_path.exists():
            if self.offline:
                raise FileNotFoundError(
                    f"Catalog not found in cache ({catalog_path}) and "
                    "offline=True.  Run with offline=False first to "
                    "download it."
                )
            self._download_file(f"{self.dataset}/catalog.parquet", catalog_path)

        self._catalog = pd.read_parquet(catalog_path)

    def _ensure_conifold_catalog(self) -> None:
        r"""
        **Description:**
        Download ``conifold_catalog.parquet`` if not already cached, then load
        it into ``self._conifold_catalog``.  One row per conifold limit with
        columns ``ks_id``, ``triang_id``, ``conifold_id``, ``h11``, ``h12``,
        ``ncf``, ``conifold_curve``, ``conifold_shard_id``,
        ``conifold_row_index``.

        Raises:
            FileNotFoundError: If ``offline=True`` and the file is not cached,
                or if the database was built without conifold data.
        """
        if self._conifold_catalog is not None:
            return

        pd = _require_pandas()
        path = self.cache_dir / "conifold_catalog.parquet"

        if not path.exists():
            if self.offline:
                raise FileNotFoundError(
                    f"Conifold catalog not found in cache ({path}) and "
                    "offline=True.  Run with offline=False first to download it, "
                    "or rebuild the database to generate conifold_catalog.parquet."
                )
            self._download_file(f"{self.dataset}/conifold_catalog.parquet", path)

        self._conifold_catalog = pd.read_parquet(path)

    def _download_file(self, hf_path: str, local_path: Path) -> None:
        r"""
        **Description:**
        Download a single file from the HuggingFace repository to a local path.

        Args:
            hf_path (str): Path within the HuggingFace repository.
            local_path (Path): Destination path on the local filesystem.

        Raises:
            RuntimeError: If the download fails.
        """
        hf_hub_download = _require_hf_hub()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Use a no-op tqdm class to avoid ipywidgets/threading crashes
            # when called from a thread pool inside Jupyter notebooks.
            _quiet = type("_QuietTqdm", (), {
                "__init__": lambda s, *a, **kw: None,
                "__enter__": lambda s: s,
                "__exit__": lambda s, *a: None,
                "update": lambda s, *a, **kw: None,
                "close": lambda s: None,
            })
            downloaded = hf_hub_download(
                repo_id=self.hf_repo,
                filename=hf_path,
                repo_type="dataset",
                local_dir=str(local_path.parent),
                tqdm_class=_quiet,
            )
            # huggingface_hub may place the file in a subdirectory; move if needed.
            if Path(downloaded) != local_path:
                import shutil
                shutil.move(downloaded, local_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download '{hf_path}' from '{self.hf_repo}': {exc}"
            ) from exc

    def _fetch_shard(self, split: str, shard_id: int) -> Any:
        r"""
        **Description:**
        Fetch a single parquet shard from the given split.

        Checks the in-memory LRU shard cache first.  On a cache miss the
        shard is read from disk (downloading it from HuggingFace if not yet
        cached locally), then stored in the LRU cache for future calls.

        Args:
            split (str): Name of the split directory, e.g. ``"lcs_data/h11_2"``
                or ``"gv"``.
            shard_id (int): Integer shard index.

        Returns:
            pandas.DataFrame: Contents of the shard.
        """
        key = (split, shard_id)

        cached = self._shard_cache.get(key)
        if cached is not None:
            return cached

        pd = _require_pandas()
        shard_filename = f"data-{shard_id:05d}.parquet"
        local_path = self.cache_dir / split / shard_filename

        if not local_path.exists():
            with self._download_lock:
                # Re-check inside the lock in case another thread downloaded
                # the file while this thread was waiting.
                if not local_path.exists():
                    if self.offline:
                        raise FileNotFoundError(
                            f"Shard '{split}/{shard_filename}' not found in cache "
                            f"({local_path}) and offline=True."
                        )
                    hf_path = f"{self.dataset}/{split}/{shard_filename}"
                    self._download_file(hf_path, local_path)

        df = pd.read_parquet(local_path)
        self._shard_cache.put(key, df)
        return df

    def _cleanup_shard(self, local_path: Path) -> None:
        r"""
        **Description:**
        Delete a shard file and remove directories left by
        ``hf_hub_download``.

        ``hf_hub_download`` creates ``.cache/huggingface/`` metadata
        trees and nested ``tdf/…`` subdirectories inside each split
        folder.  This helper removes the parquet file itself, any
        ``.cache`` tree in the same split directory, and prunes empty
        parent directories up to (but not including) ``self.cache_dir``.

        Args:
            local_path (Path): Path to the parquet shard file on disk.

        Returns:
            None
        """
        import shutil

        # Delete the parquet file.
        local_path.unlink(missing_ok=True)

        # Remove the .cache/huggingface tree that hf_hub_download creates
        # inside the split directory.
        split_dir = local_path.parent
        hf_cache = split_dir / ".cache"
        if hf_cache.exists():
            shutil.rmtree(hf_cache, ignore_errors=True)

        # Remove the nested tdf/… directories that hf_hub_download creates.
        nested = split_dir / self.dataset
        if nested.exists():
            shutil.rmtree(nested, ignore_errors=True)

        # Prune empty parent directories up to cache_dir.
        d = split_dir
        while d != self.cache_dir and d.exists():
            try:
                d.rmdir()          # only succeeds if empty
                d = d.parent
            except OSError:
                break              # not empty — stop climbing

    def _fetch_row(self, split: str, shard_id: int, row_index: int) -> Any:
        r"""
        **Description:**
        Fetch a single row from a parquet shard without caching the full
        DataFrame in memory.

        The shard file is downloaded to the local cache if not already
        present.  Only the requested row is read into memory using
        PyArrow.  When ``cache_mode="none"``, the shard file is deleted
        from disk after the row is extracted.

        Args:
            split (str): Name of the split directory, e.g.
                ``"lcs_data/h11_2"`` or ``"gv"``.
            shard_id (int): Integer shard index.
            row_index (int): Row offset within the shard.

        Returns:
            pandas.Series: The requested row.
        """
        import pyarrow.parquet as pq

        shard_filename = f"data-{shard_id:05d}.parquet"
        local_path = self.cache_dir / split / shard_filename

        if not local_path.exists():
            with self._download_lock:
                if not local_path.exists():
                    if self.offline:
                        raise FileNotFoundError(
                            f"Shard '{split}/{shard_filename}' not found "
                            f"in cache ({local_path}) and offline=True."
                        )
                    hf_path = f"{self.dataset}/{split}/{shard_filename}"
                    self._download_file(hf_path, local_path)

        table = pq.read_table(str(local_path))
        row = table.slice(row_index, 1).to_pandas().iloc[0]

        if self.cache_mode == "none":
            self._cleanup_shard(local_path)

        return row

    def _fetch_rows(self, split: str, shard_id: int, mask_fn: Any) -> Any:
        r"""
        **Description:**
        Fetch rows matching a filter from a parquet shard without caching
        the full DataFrame in memory.

        Like :meth:`_fetch_row` but applies a callable ``mask_fn`` to the
        loaded DataFrame and returns the matching subset.  Used for
        conifold data where multiple rows may belong to the same model.

        Args:
            split (str): Name of the split directory.
            shard_id (int): Integer shard index.
            mask_fn (callable): Function that takes a DataFrame and returns
                a boolean mask.

        Returns:
            pandas.DataFrame: Matching rows.
        """
        import pyarrow.parquet as pq

        shard_filename = f"data-{shard_id:05d}.parquet"
        local_path = self.cache_dir / split / shard_filename

        if not local_path.exists():
            with self._download_lock:
                if not local_path.exists():
                    if self.offline:
                        raise FileNotFoundError(
                            f"Shard '{split}/{shard_filename}' not found "
                            f"in cache ({local_path}) and offline=True."
                        )
                    hf_path = f"{self.dataset}/{split}/{shard_filename}"
                    self._download_file(hf_path, local_path)

        df = pq.read_table(str(local_path)).to_pandas()
        result = df[mask_fn(df)]

        if self.cache_mode == "none":
            self._cleanup_shard(local_path)

        return result

    def _lookup(
        self,
        ks_id: Optional[int],
        triang_id: Optional[int],
        cicy_id: Optional[int],
        h11: Optional[int] = None,
        h12: Optional[int] = None,
    ) -> Any:
        r"""
        **Description:**
        Look up the catalog row for a given model primary key.

        For ``tdf`` models the full primary key is
        ``(h11, h12, ks_id, triang_id)``.  When ``h11`` and ``h12`` are
        omitted, only ``(ks_id, triang_id)`` is used; if this yields multiple
        matches an error is raised asking the caller to disambiguate.

        Args:
            ks_id (int | None): Kreuzer-Skarke polytope index (tdf only).
            triang_id (int | None): Triangulation index (tdf only).
            cicy_id (int | None): CICY model index (cicy only).
            h11 (int | None): Hodge number :math:`h^{1,1}` (tdf only).
            h12 (int | None): Hodge number :math:`h^{1,2}` (tdf only).

        Returns:
            pandas.Series: The matching catalog row.

        Raises:
            KeyError: If no matching row is found.
            ValueError: If multiple rows match — supply ``h11`` and ``h12``
                to disambiguate.
        """
        self._ensure_catalog()
        cat = self._catalog

        if self.dataset == "tdf":
            mask = (cat["ks_id"] == ks_id) & (cat["triang_id"] == triang_id)
            if h11 is not None:
                mask = mask & (cat["h11"] == h11)
            if h12 is not None:
                mask = mask & (cat["h12"] == h12)
        else:
            mask = cat["cicy_id"] == cicy_id

        matches = cat[mask]
        if len(matches) == 0:
            key = f"ks_id={ks_id}, triang_id={triang_id}" if self.dataset == "tdf" \
                  else f"cicy_id={cicy_id}"
            if h11 is not None or h12 is not None:
                key += f", h11={h11}, h12={h12}"
            raise KeyError(
                f"No model found for {key} in the '{self.dataset}' dataset."
            )
        if len(matches) > 1:
            h_values = matches[["h11", "h12"]].drop_duplicates().values.tolist()
            raise ValueError(
                f"Multiple models match ks_id={ks_id}, triang_id={triang_id} "
                f"with (h11, h12) values {h_values}.  "
                f"Please pass h11 and h12 to disambiguate."
            )
        return matches.iloc[0]

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def info(self) -> None:
        r"""
        **Description:**
        Print a summary of the available models in this sub-dataset.

        The summary includes total model count, Hodge-number ranges, and the
        fraction of models with GV invariants and conifold data available.
        """
        self._ensure_catalog()
        cat = self._catalog

        n_total = len(cat)
        h11_min, h11_max = int(cat["h11"].min()), int(cat["h11"].max())
        h12_min, h12_max = int(cat["h12"].min()), int(cat["h12"].max())
        n_gv = int(cat["has_gv"].sum()) if "has_gv" in cat.columns else -1
        n_cf = int((cat["n_conifolds"] > 0).sum()) if "n_conifolds" in cat.columns else -1

        print(f"CYDatabase — sub-dataset: '{self.dataset}'")
        print(f"  Total models : {n_total:,}")
        print(f"  h11 range    : {h11_min} – {h11_max}")
        print(f"  h12 range    : {h12_min} – {h12_max}")
        if n_gv >= 0:
            print(f"  With GV data : {n_gv:,}  ({100*n_gv/n_total:.1f}%)")
        if n_cf >= 0:
            print(f"  With conifolds: {n_cf:,}  ({100*n_cf/n_total:.1f}%)")
        print(f"  Cache dir    : {self.cache_dir}")

    def query_conifolds(
        self,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ncf: Optional[int] = None,
        conifold_id: Optional[int] = None,
        **filters: Any,
    ) -> Any:
        r"""
        **Description:**
        Filter the conifold sub-catalog and return a ``pandas.DataFrame`` of
        matching conifold limits.  Each row represents one conifold limit and
        carries enough information to load the corresponding model directly via
        :meth:`load_from_conifold_row`.

        This method operates entirely on the in-memory conifold catalog and
        does not load any geometry data.

        For filters not covered by the keyword arguments, use pandas on the
        returned DataFrame::

            df = db.query_conifolds()
            df[df["conifold_curve"].apply(lambda c: c == [0, 1])]

        Args:
            h11 (int | None): Filter by :math:`h^{1,1}`.
            h12 (int | None): Filter by :math:`h^{1,2}`.
            ncf (int | None): Filter by the GV invariant of the conifold node.
            conifold_id (int | None): Filter by conifold index within a model
                (0-based).
            **filters: Additional exact-match filters on any catalog column.

        Returns:
            pandas.DataFrame: Matching rows from the conifold sub-catalog.
        """
        self._ensure_conifold_catalog()
        cat = self._conifold_catalog
        mask = np.ones(len(cat), dtype=bool)

        if h11 is not None:
            mask &= (cat["h11"] == h11).values
        if h12 is not None:
            mask &= (cat["h12"] == h12).values
        if ncf is not None:
            mask &= (cat["ncf"] == ncf).values
        if conifold_id is not None:
            mask &= (cat["conifold_id"] == conifold_id).values
        for col, val in filters.items():
            if col in cat.columns:
                mask &= (cat[col] == val).values
            else:
                warnings.warn(f"Column '{col}' not found in conifold catalog; filter ignored.")

        return cat[mask].reset_index(drop=True)

    def load_from_conifold_row(
        self,
        row: Any,
        include_gv: bool = False,
        conifold_basis: bool = True,
        include_extra_data: bool = True,
        include_polytope: bool = False,
        maximum_degree: Optional[int] = None,
    ) -> Any:
        r"""
        **Description:**
        Load the model corresponding to a single row from
        :meth:`query_conifolds` and return it as an
        :class:`~jaxvacua.lcs.lcs_tree` with that conifold limit active.

        Example::

            rows = db.query_conifolds(h11=2, ncf=2)
            tree = db.load_from_conifold_row(rows.iloc[0])

        Args:
            row: A single-row result from :meth:`query_conifolds` (a
                ``pandas.Series``).
            include_gv (bool): Attach GV invariants.  Defaults to ``False``.
            conifold_basis (bool): Rotate to the conifold basis.  Defaults to
                ``True``.
            include_extra_data (bool): Populate ``extra_data``.  Defaults to
                ``True``.
            include_polytope (bool): Attach polytope data.  Defaults to
                ``False``.
            maximum_degree (int | None): GV truncation degree.

        Returns:
            lcs_tree: The requested model with the selected conifold active.
        """
        return self.load(
            ks_id=int(row["ks_id"]) if self.dataset == "tdf" else None,
            triang_id=int(row["triang_id"]) if self.dataset == "tdf" else None,
            cicy_id=int(row["cicy_id"]) if self.dataset == "cicy" else None,
            include_conifolds=True,
            conifold_id=int(row["conifold_id"]),
            conifold_basis=conifold_basis,
            include_gv=include_gv,
            include_extra_data=include_extra_data,
            include_polytope=include_polytope,
            maximum_degree=maximum_degree,
        )

    def query(
        self,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        has_conifolds: Optional[bool] = None,
        n_conifolds_min: Optional[int] = None,
        n_conifolds_max: Optional[int] = None,
        has_gv: Optional[bool] = None,
        D3_tadpole_max: Optional[int] = None,
        **filters: Any,
    ) -> Any:
        r"""
        **Description:**
        Filter the catalog and return a ``pandas.DataFrame`` of matching models.

        This method operates entirely on the in-memory catalog and does not
        download any geometry data.

        For filters not covered by the keyword arguments, operate on the
        returned DataFrame directly, e.g.::

            df = db.query()
            df[df["chi"] < -200]

        Args:
            h11 (int | None): Filter by :math:`h^{1,1}`.
            h12 (int | None): Filter by :math:`h^{1,2}`.
            has_conifolds (bool | None): If ``True``, return only models with
                at least one conifold limit available.
            n_conifolds_min (int | None): Return only models with at least this
                many conifold limits, e.g. ``n_conifolds_min=2``.
            n_conifolds_max (int | None): Return only models with at most this
                many conifold limits.
            has_gv (bool | None): If ``True``, return only models for which
                Gopakumar-Vafa invariants are stored.
            D3_tadpole_max (int | None): Return only models whose D3-tadpole
                is at most this value.
            **filters: Additional exact-match filters on catalog columns,
                e.g. ``chi=-240``.

        Returns:
            pandas.DataFrame: Matching rows from the catalog.
        """
        self._ensure_catalog()
        mask = np.ones(len(self._catalog), dtype=bool)

        if h11 is not None:
            mask &= (self._catalog["h11"] == h11).values
        if h12 is not None:
            mask &= (self._catalog["h12"] == h12).values
        if "n_conifolds" in self._catalog.columns:
            if has_conifolds is not None:
                if has_conifolds:
                    mask &= (self._catalog["n_conifolds"] > 0).values
                else:
                    mask &= (self._catalog["n_conifolds"] == 0).values
            if n_conifolds_min is not None:
                mask &= (self._catalog["n_conifolds"] >= n_conifolds_min).values
            if n_conifolds_max is not None:
                mask &= (self._catalog["n_conifolds"] <= n_conifolds_max).values
        if has_gv is not None and "has_gv" in self._catalog.columns:
            mask &= (self._catalog["has_gv"] == has_gv).values
        if D3_tadpole_max is not None and "D3_tadpole" in self._catalog.columns:
            mask &= (self._catalog["D3_tadpole"] <= D3_tadpole_max).values
        for col, val in filters.items():
            if col in self._catalog.columns:
                mask &= (self._catalog[col] == val).values
            else:
                warnings.warn(f"Column '{col}' not found in catalog; filter ignored.")

        return self._catalog[mask].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        cicy_id: Optional[int] = None,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        include_gv: bool = False,
        include_conifolds: Union[bool, str] = False,
        conifold_id: Optional[int] = None,
        conifold_basis: bool = True,
        include_extra_data: bool = True,
        include_polytope: bool = False,
        extra_fields: Optional[List[str]] = None,
        maximum_degree: Optional[int] = None,
    ) -> Any:
        r"""
        **Description:**
        Load a single Calabi-Yau model from the database and return it as an
        :class:`~jaxvacua.lcs.lcs_tree` object.

        For ``tdf`` models the full primary key is
        ``(h11, h12, ks_id, triang_id)``.  When ``h11`` and ``h12`` are
        omitted the lookup falls back to ``(ks_id, triang_id)``; if that
        combination is ambiguous a ``ValueError`` is raised asking you to
        supply the Hodge numbers.

        For ``cicy`` models, only ``cicy_id`` is required.

        Args:
            ks_id (int | None): Kreuzer-Skarke polytope index (required for
                ``dataset="tdf"``).
            triang_id (int | None): Triangulation index (required for
                ``dataset="tdf"``).
            cicy_id (int | None): CICY model index (required for
                ``dataset="cicy"``).
            h11 (int | None): Hodge number :math:`h^{1,1}`.  Recommended for
                ``dataset="tdf"`` to ensure a unique match.
            h12 (int | None): Hodge number :math:`h^{1,2}`.  Recommended for
                ``dataset="tdf"`` to ensure a unique match.
            include_gv (bool): If ``True``, fetch and attach Gopakumar-Vafa
                and Gromov-Witten invariants.  Defaults to ``False``.
            include_conifolds (bool | str): If ``True``, fetch all available
                conifold limits and populate ``lcs_tree.ncf`` and
                ``lcs_tree.conifold_curve`` from the selected conifold
                (controlled by ``conifold_id``; defaults to 0).  Pass
                ``"all"`` to obtain a list of trees, one per conifold.
                Defaults to ``False``.
            conifold_id (int | None): Index of the conifold limit to use when
                ``include_conifolds=True``.  Ignored when
                ``include_conifolds="all"``.  Defaults to ``0``.
            conifold_basis (bool): If ``True``, rotate to the basis defined by
                the conifold curve when ``include_conifolds`` is active.
                Defaults to ``True``.
            include_extra_data (bool): If ``True``, populate
                ``lcs_tree.extra_data`` from the ``extra`` split (merged with
                the identifier fields always stored there).  Defaults to
                ``True``.
            include_polytope (bool): If ``True``, fetch the ``polytope`` split
                and attach ``polytope_points`` (and any extra polytope-level
                fields) to the returned tree.  Only meaningful for
                ``dataset="tdf"``.  Defaults to ``False``.
            extra_fields (list[str] | None): Subset of columns to load from
                the ``extra`` split when ``include_extra_data=True``.  If
                ``None``, all available columns are loaded.
            maximum_degree (int | None): If ``include_gv=True``, truncate GV
                invariants to at most this degree.  ``None`` keeps all stored
                invariants.

        Returns:
            lcs_tree | list[lcs_tree]: A single tree, or a list when
            ``include_conifolds="all"``.

        Raises:
            KeyError: If the requested model is not in the catalog.
            ValueError: If required primary-key arguments are missing, or if
                the ``(ks_id, triang_id)`` pair matches multiple models and
                ``h11``/``h12`` were not supplied.
        """
        self._check_cache_size()
        self._validate_key(ks_id, triang_id, cicy_id)
        entry = self._lookup(ks_id, triang_id, cicy_id, h11=h11, h12=h12)

        # ---- Fetch LCS geometry data ----------------------------------------
        h11 = int(entry["h11"])
        if self.cache_mode == "persistent":
            lcs_shard = self._fetch_shard(
                f"lcs_data/h11_{h11}", int(entry["lcs_shard_id"])
            )
            lcs_row = lcs_shard.iloc[int(entry["lcs_row_index"])]
        else:
            lcs_row = self._fetch_row(
                f"lcs_data/h11_{h11}",
                int(entry["lcs_shard_id"]),
                int(entry["lcs_row_index"]),
            )
        kwargs = _parse_lcs_row(lcs_row, self.dataset)

        # ---- Optionally fetch extra data ------------------------------------
        if include_extra_data and "extra_shard_id" in entry.index:
            if self.cache_mode == "persistent":
                extra_shard = self._fetch_shard("extra", int(entry["extra_shard_id"]))
                extra_row   = extra_shard.iloc[int(entry["extra_row_index"])]
            else:
                extra_row = self._fetch_row(
                    "extra", int(entry["extra_shard_id"]),
                    int(entry["extra_row_index"]),
                )
            extra = dict(extra_row)
            if extra_fields is not None:
                extra = {k: v for k, v in extra.items() if k in extra_fields}
            existing = kwargs.get("extra_data") or {}
            kwargs["extra_data"] = {**existing, **extra}

        # ---- Optionally fetch polytope data (tdf only) ----------------------
        if include_polytope and "polytope_shard_id" in entry.index:
            _psid = entry["polytope_shard_id"]
            _prid = entry["polytope_row_index"]
            _has_poly = False
            try:
                import pandas as pd
                _has_poly = _psid is not None and not pd.isna(_psid)
            except (TypeError, ValueError):
                _has_poly = _psid is not None
            if _has_poly:
                if self.cache_mode == "persistent":
                    poly_shard = self._fetch_shard("polytope", int(_psid))
                    poly_row   = poly_shard.iloc[int(_prid)]
                else:
                    poly_row = self._fetch_row("polytope", int(_psid), int(_prid))
                poly_data  = _parse_polytope_row(poly_row)
                kwargs["polytope_points"] = poly_data["polytope_points"]
                if "polytope_data" in poly_data:
                    existing = kwargs.get("extra_data") or {}
                    kwargs["extra_data"] = {**existing, "polytope_data": poly_data["polytope_data"]}

        # ---- Optionally fetch GV invariants --------------------------------
        gvs, gws, grading_vec = None, None, None
        if include_gv and entry.get("has_gv", False):
            _gv_split = f"gv/h11_{h11}"
            if self.cache_mode == "persistent":
                gv_shard = self._fetch_shard(_gv_split, int(entry["gv_shard_id"]))
                gv_row   = gv_shard.iloc[int(entry["gv_row_index"])]
            else:
                gv_row = self._fetch_row(
                    _gv_split, int(entry["gv_shard_id"]),
                    int(entry["gv_row_index"]),
                )
            gv_data  = _parse_gv_row(gv_row)
            gvs = gv_data["GVs"]
            gws = gv_data["GWs"]
            grading_vec = gv_data["grading_vector"]
            if maximum_degree is not None:
                gvs, gws = self._truncate_gv(gvs, gws, grading_vec,maximum_degree)
        kwargs["gvs"] = gvs
        kwargs["gws"] = gws
        kwargs["grading_vector"] = grading_vec
        
        
        # Flip Hodge numbers for input
        h11 = kwargs["h11"]
        h12 = kwargs["h12"]
        kwargs["h11"] = h12
        kwargs["h12"] = h11
        kwargs["chi"] = None
        
        # ---- Optionally fetch conifold data --------------------------------
        # NOTE: `h11` was swapped with h12 above (line 1524-1525) for API
        # conventions, so use the catalog's h11 for the shard path.
        _cat_h11 = int(entry["h11"])
        conifold_list = []
        if include_conifolds and int(entry.get("n_conifolds", 0)) > 0:
            _cf_shard_id = int(entry["conifold_shard_id"])
            _cf_split = f"conifolds/h11_{_cat_h11}"
            if self.cache_mode == "persistent":
                cf_shard = self._fetch_shard(_cf_split, _cf_shard_id)
            else:
                # For conifolds we need the full shard to filter by model
                # ID, so use _fetch_rows with a mask function.
                if self.dataset == "tdf":
                    def _cf_mask(df):
                        m = (df["ks_id"] == ks_id) & (df["triang_id"] == triang_id)
                        if "h11" in df.columns:
                            m = m & (df["h11"] == _cat_h11)
                        return m
                else:
                    _cf_mask = lambda df: df["cicy_id"] == cicy_id
                cf_rows = self._fetch_rows(_cf_split, _cf_shard_id, _cf_mask)
                conifold_list = _parse_conifold_rows(cf_rows)

            if self.cache_mode == "persistent":
                if self.dataset == "tdf":
                    mask = (
                        (cf_shard["ks_id"]     == ks_id) &
                        (cf_shard["triang_id"] == triang_id)
                    )
                    if "h11" in cf_shard.columns:
                        mask = mask & (cf_shard["h11"] == _cat_h11)
                    cf_rows = cf_shard[mask]
                else:
                    cf_rows = cf_shard[cf_shard["cicy_id"] == cicy_id]
                conifold_list = _parse_conifold_rows(cf_rows)

        # ---- Construct lcs_tree --------------------------------------------
        from .lcs import lcs_tree

        _CF_EXTRA_KEYS = ("flop_edge", "one_face_divisors")

        if include_conifolds == "all" and conifold_list:
            trees = []
            for cf in conifold_list:
                kw = {
                    **kwargs,
                    "ncf":            cf["ncf"],
                    "conifold_curve": cf["conifold_curve"],
                    "conifold_basis": conifold_basis,
                    "limit":          "coniLCS",
                }
                if conifold_basis and cf.get("basis_change") is not None:
                    kw["basis_change"] = cf["basis_change"]
                if include_extra_data:
                    cf_extra = {k: cf[k] for k in _CF_EXTRA_KEYS if cf.get(k) is not None}
                    if cf_extra:
                        kw["extra_data"] = {**(kw.get("extra_data") or {}), **cf_extra}
                trees.append(lcs_tree.from_dict(kw))
            return trees

        if conifold_list:
            # Select the requested conifold; fall back to id=0 if not found.
            _target_id = conifold_id if conifold_id is not None else 0
            _matches = [cf for cf in conifold_list if cf["conifold_id"] == _target_id]
            if not _matches:
                raise KeyError(
                    f"conifold_id={_target_id} not found for this model. "
                    f"Available ids: {[cf['conifold_id'] for cf in conifold_list]}"
                )
            cf = _matches[0]
            kwargs["ncf"]            = cf["ncf"]
            kwargs["conifold_curve"] = cf["conifold_curve"]
            kwargs["conifold_basis"] = conifold_basis
            kwargs["limit"]          = "coniLCS"
            if conifold_basis and cf.get("basis_change") is not None:
                kwargs["basis_change"] = cf["basis_change"]
            if include_extra_data:
                cf_extra = {k: cf[k] for k in _CF_EXTRA_KEYS if cf.get(k) is not None}
                if cf_extra:
                    kwargs["extra_data"] = {**(kwargs.get("extra_data") or {}), **cf_extra}

        return lcs_tree.from_dict(kwargs)

    def load_model(
        self,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        cicy_id: Optional[int] = None,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        include_gv: bool = False,
        include_conifolds: Union[bool, str] = False,
        maximum_degree: int = 0,
        **flux_sector_kwargs: Any,
    ) -> Any:
        r"""
        **Description:**
        Load a model from the database and return it as a fully initialised
        :class:`~jaxvacua.flux_eft.FluxEFT` object.

        This is a convenience wrapper around :meth:`load` followed by
        construction of a :class:`~jaxvacua.flux_eft.FluxEFT` with
        ``lcs_tree_input``.

        Args:
            ks_id (int | None): Kreuzer-Skarke polytope index (tdf only).
            triang_id (int | None): Triangulation index (tdf only).
            cicy_id (int | None): CICY index (cicy only).
            h11 (int | None): Hodge number :math:`h^{1,1}` (tdf only).
            h12 (int | None): Hodge number :math:`h^{1,2}` (tdf only).
            include_gv (bool): Attach GV invariants.  Defaults to ``False``.
            include_conifolds (bool): Attach conifold data.  Defaults to
                ``False``.
            maximum_degree (int): Maximum instanton degree; passed to
                :class:`~jaxvacua.flux_eft.FluxEFT`.  Defaults to ``0``
                (instanton corrections disabled).
            **flux_sector_kwargs: Additional keyword arguments forwarded to
                :class:`~jaxvacua.flux_eft.FluxEFT`, e.g. ``Q``,
                ``limit``, ``gauge_choice``.

        Returns:
            flux_sector: Initialised flux-sector model.

        Raises:
            KeyError: If the requested model is not in the catalog.
        """
        from .flux_eft import FluxEFT

        tree = self.load(
            ks_id=ks_id,
            triang_id=triang_id,
            cicy_id=cicy_id,
            h11=h11,
            h12=h12,
            include_gv=include_gv,
            include_conifolds=include_conifolds,
            maximum_degree=maximum_degree if include_gv else None,
        )
        return FluxEFT(
            lcs_tree_input=tree,
            maximum_degree=maximum_degree,
            **flux_sector_kwargs,
        )

    def load_batch(
        self,
        identifiers: Any = None,
        include_gv: bool = False,
        include_conifolds: Union[bool, str] = False,
        conifold_id: Optional[int] = None,
        conifold_basis: bool = True,
        include_extra_data: bool = True,
        include_polytope: bool = False,
        maximum_degree: Optional[int] = None,
        parallel: bool = True,
        n_workers: Optional[int] = None,
        **query_filters: Any,
    ) -> List[Any]:
        r"""
        **Description:**
        Load multiple models and return them as a list of
        :class:`~jaxvacua.lcs.lcs_tree` objects.

        By default shard I/O is parallelised across a thread pool so that
        models in different shards are fetched concurrently.  Results are
        always returned in the same order as ``identifiers``.

        Can be called in two ways:

        1. **With explicit identifiers:**
           ``db.load_batch(df)`` or ``db.load_batch([(ks_id, triang_id), ...])``
        2. **With query filters** (same kwargs as :meth:`query`):
           ``db.load_batch(h11=2)`` or ``db.load_batch(ks_id=12345)``

        When query filters are used, :meth:`query` is called first and
        the matching catalog entries are loaded.

        Args:
            identifiers: Either a ``pandas.DataFrame`` with columns
                ``ks_id`` and ``triang_id`` (or ``cicy_id``), or a list of
                ``(ks_id, triang_id)`` tuples for ``"tdf"``, or a list of
                ``cicy_id`` integers for ``"cicy"``.  If ``None``, the
                ``**query_filters`` are used instead.
            include_gv (bool): Attach GV invariants to each model.
            include_conifolds (bool | str): Attach conifold data.
            conifold_id (int | None): Index of the conifold limit to use.
                Defaults to ``0``. Ignored when ``include_conifolds="all"``.
            conifold_basis (bool): Rotate to the conifold basis when
                ``include_conifolds`` is active. Defaults to ``True``.
            include_extra_data (bool): Populate ``extra_data`` field.
            include_polytope (bool): Attach polytope data from the polytope split.
            maximum_degree (int | None): GV truncation degree.
            parallel (bool): If ``False``, load models sequentially (no thread
                pool).  Useful when the caller is already inside a parallel
                context or when debugging.  Defaults to ``True``.
            n_workers (int | None): Number of threads when ``parallel=True``.
                ``None`` uses ``min(len(identifiers), 8)``.
            **query_filters: Keyword arguments forwarded to :meth:`query`
                when ``identifiers`` is ``None``.  Common filters:
                ``h11``, ``h12``, ``ks_id``, ``has_conifolds``, ``has_gv``.

        Returns:
            list[lcs_tree]: Loaded trees in the same order as ``identifiers``
            (or in catalog order when using query filters).
        """
        _require_pandas()

        if identifiers is None:
            if not query_filters:
                raise ValueError(
                    "Either pass identifiers (DataFrame / list) or "
                    "query filters (e.g. h11=2, ks_id=12345)."
                )
            identifiers = self.query(**query_filters)
            if len(identifiers) == 0:
                warnings.warn("Query returned no models.")
                return []

        rows = self._identifiers_to_list(identifiers)

        if not rows:
            return []

        workers = 1 if not parallel else (min(len(rows), 8) if n_workers is None else n_workers)

        def _load_one(row):
            """Load a single model from an identifier row."""
            ks_id, triang_id, cicy_id, h11, h12 = row
            return self.load(
                ks_id=ks_id,
                triang_id=triang_id,
                cicy_id=cicy_id,
                h11=h11,
                h12=h12,
                include_gv=include_gv,
                include_conifolds=include_conifolds,
                conifold_id=conifold_id,
                conifold_basis=conifold_basis,
                include_extra_data=include_extra_data,
                include_polytope=include_polytope,
                maximum_degree=maximum_degree,
            )

        if workers == 1:
            return [_load_one(row) for row in rows]

        # Submit all jobs, preserving insertion order via index.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(_load_one, row): i
                for i, row in enumerate(rows)
            }
            results = [None] * len(rows)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()  # re-raises any exception

        return results

    def iter_batch(
        self,
        identifiers: Any,
        include_gv: bool = False,
        include_conifolds: Union[bool, str] = False,
        conifold_id: Optional[int] = None,
        conifold_basis: bool = True,
        include_extra_data: bool = True,
        include_polytope: bool = False,
        maximum_degree: Optional[int] = None,
        prefetch: int = 4,
    ) -> Iterator[Any]:
        r"""
        **Description:**
        Yield models one at a time from a (potentially large) set of
        identifiers, keeping memory usage constant regardless of the total
        number of models requested.

        Unlike :meth:`load_batch`, which loads all models into a list before
        returning, ``iter_batch`` is a generator: it yields each
        :class:`~jaxvacua.lcs.lcs_tree` as soon as it is ready, so the
        caller can begin processing immediately and only one model needs to be
        in memory at a time.

        A sliding prefetch window of size ``prefetch`` submits the next
        ``prefetch`` loads to a background thread pool while the caller
        processes the current model, keeping disk I/O overlapped with
        computation.  Set ``prefetch=1`` (or ``prefetch=0``) to disable
        parallelism and load strictly one model at a time.

        Example::

            df = db.query(h11=3)
            for tree in db.iter_batch(df, prefetch=8):
                result = analyse(tree)   # memory: ~1 model at a time

        Args:
            identifiers: Either a ``pandas.DataFrame`` with columns
                ``ks_id`` and ``triang_id`` (or ``cicy_id``), or a list of
                ``(ks_id, triang_id)`` tuples for ``"tdf"``, or a list of
                ``cicy_id`` integers for ``"cicy"``.
            include_gv (bool): Attach GV invariants to each model.
            include_conifolds (bool | str): Attach conifold data.
            conifold_id (int | None): Index of the conifold limit to use.
                Defaults to ``0``. Ignored when ``include_conifolds="all"``.
            conifold_basis (bool): Rotate to the conifold basis when
                ``include_conifolds`` is active. Defaults to ``True``.
            include_extra_data (bool): Populate ``extra_data`` field.
            include_polytope (bool): Attach polytope data.
            maximum_degree (int | None): GV truncation degree.
            prefetch (int): Number of models to load ahead in background
                threads.  Defaults to ``4``.  Set to ``0`` or ``1`` for
                strictly sequential loading.

        Yields:
            lcs_tree: One loaded model per iteration, in the same order as
            ``identifiers``.
        """
        _require_pandas()
        rows = self._identifiers_to_list(identifiers)

        if not rows:
            return

        def _load_one(row):
            """Load a single model from an identifier row for iteration."""
            ks_id, triang_id, cicy_id, h11, h12 = row
            return self.load(
                ks_id=ks_id,
                triang_id=triang_id,
                cicy_id=cicy_id,
                h11=h11,
                h12=h12,
                include_gv=include_gv,
                include_conifolds=include_conifolds,
                conifold_id=conifold_id,
                conifold_basis=conifold_basis,
                include_extra_data=include_extra_data,
                include_polytope=include_polytope,
                maximum_degree=maximum_degree,
            )

        # Sequential path — no thread pool overhead.
        if prefetch <= 1:
            for row in rows:
                yield _load_one(row)
            return

        # Prefetch path — sliding window of futures keeps I/O and computation
        # overlapped.  The window size is capped to the number of rows.
        window = min(prefetch, len(rows))
        executor = ThreadPoolExecutor(max_workers=window)
        pending: deque = deque()

        try:
            # Prime the window with the first `window` rows.
            for i in range(window):
                pending.append(executor.submit(_load_one, rows[i]))

            # For each remaining row, pop the oldest future (blocking until
            # done), yield it, then immediately submit the next row so the
            # window stays full.
            for i in range(window, len(rows)):
                pending.append(executor.submit(_load_one, rows[i]))
                yield pending.popleft().result()

            # Drain the window.
            while pending:
                yield pending.popleft().result()

        finally:
            # Cancel any still-pending futures if the caller broke out of the
            # loop early, then shut down the executor cleanly.
            for f in pending:
                f.cancel()
            executor.shutdown(wait=False)

    def sample(
        self,
        n: int,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        has_conifolds: Optional[bool] = None,
        has_gv: Optional[bool] = None,
        seed: Optional[int] = None,
        include_gv: bool = False,
        include_conifolds: Union[bool, str] = False,
        conifold_id: Optional[int] = None,
        conifold_basis: bool = True,
        include_extra_data: bool = True,
        include_polytope: bool = False,
        maximum_degree: Optional[int] = None,
        parallel: bool = True,
        n_workers: Optional[int] = None,
    ) -> List[Any]:
        r"""
        **Description:**
        Draw a random sample of models from the database and return them as a
        list of :class:`~jaxvacua.lcs.lcs_tree` objects.

        Args:
            n (int): Number of models to draw.
            h11 (int | None): Restrict sample to models with this :math:`h^{1,1}`.
            h12 (int | None): Restrict sample to models with this :math:`h^{1,2}`.
            has_conifolds (bool | None): Restrict to models with conifold data.
            has_gv (bool | None): Restrict to models with GV invariants.
            seed (int | None): Random seed for reproducibility.
            include_gv (bool): Attach GV invariants to each sampled model.
            include_conifolds (bool | str): Attach conifold data.
            conifold_id (int | None): Index of the conifold limit to use.
                Defaults to ``0``. Ignored when ``include_conifolds="all"``.
            conifold_basis (bool): Rotate to the conifold basis when
                ``include_conifolds`` is active. Defaults to ``True``.
            include_extra_data (bool): Populate ``extra_data`` field.
            include_polytope (bool): Attach polytope data from the polytope split.
            maximum_degree (int | None): GV truncation degree.
            parallel (bool): If ``False``, load models sequentially. See
                :meth:`load_batch`.
            n_workers (int | None): Number of parallel threads. See
                :meth:`load_batch`.

        Returns:
            list[lcs_tree]: ``n`` randomly drawn trees.

        Raises:
            ValueError: If the filtered catalog contains fewer than ``n``
                models.
        """
        subset = self.query(
            h11=h11, h12=h12,
            has_conifolds=has_conifolds,
            has_gv=has_gv,
        )
        if len(subset) < n:
            raise ValueError(
                f"Requested {n} models but only {len(subset)} match the filter."
            )
        sampled = subset.sample(n=n, random_state=seed)
        return self.load_batch(
            sampled,
            include_gv=include_gv,
            include_conifolds=include_conifolds,
            conifold_id=conifold_id,
            conifold_basis=conifold_basis,
            include_extra_data=include_extra_data,
            include_polytope=include_polytope,
            maximum_degree=maximum_degree,
            parallel=parallel,
            n_workers=n_workers,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_key(
        self,
        ks_id: Optional[int],
        triang_id: Optional[int],
        cicy_id: Optional[int],
    ) -> None:
        r"""
        **Description:**
        Validate that the correct primary-key arguments are supplied for the
        active sub-dataset.

        Raises:
            ValueError: On missing or unexpected key arguments.
        """
        if self.dataset == "tdf":
            if ks_id is None or triang_id is None:
                raise ValueError(
                    "Both 'ks_id' and 'triang_id' are required for dataset='tdf'."
                )
        else:
            if cicy_id is None:
                raise ValueError("'cicy_id' is required for dataset='cicy'.")

    def _identifiers_to_list(
        self, identifiers: Any
    ) -> List[Tuple[Optional[int], Optional[int], Optional[int],
                     Optional[int], Optional[int]]]:
        r"""
        **Description:**
        Normalise ``identifiers`` to a list of
        ``(ks_id, triang_id, cicy_id, h11, h12)`` quintuples.

        Accepted input formats:

        - ``pandas.DataFrame`` with columns ``ks_id, triang_id`` (and
          optionally ``h11``, ``h12``) or ``cicy_id``.
        - List of ``(ks_id, triang_id)`` 2-tuples (for ``"tdf"``).
        - List of ``(h11, h12, ks_id, triang_id)`` 4-tuples (for ``"tdf"``,
          recommended to avoid ambiguity).
        - List of ``int`` values (for ``"cicy"``).

        Args:
            identifiers: See above.

        Returns:
            list of ``(ks_id, triang_id, cicy_id, h11, h12)`` quintuples.
        """
        pd = _require_pandas()
        result = []
        if isinstance(identifiers, pd.DataFrame):
            for _, row in identifiers.iterrows():
                if self.dataset == "tdf":
                    h11 = int(row["h11"]) if "h11" in row.index else None
                    h12 = int(row["h12"]) if "h12" in row.index else None
                    result.append((int(row["ks_id"]), int(row["triang_id"]),
                                   None, h11, h12))
                else:
                    result.append((None, None, int(row["cicy_id"]), None, None))
        elif isinstance(identifiers, (list, tuple)):
            for item in identifiers:
                if isinstance(item, (list, tuple)) and len(item) == 4:
                    # (h11, h12, ks_id, triang_id)
                    result.append((int(item[2]), int(item[3]), None,
                                   int(item[0]), int(item[1])))
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    # (ks_id, triang_id) — legacy format
                    result.append((int(item[0]), int(item[1]), None, None, None))
                elif isinstance(item, (int, np.integer)):
                    result.append((None, None, int(item), None, None))
                else:
                    raise ValueError(f"Cannot interpret identifier: {item!r}")
        else:
            raise TypeError(
                f"'identifiers' must be a DataFrame or a list of tuples/ints, "
                f"got {type(identifiers).__name__}."
            )
        return result

    @staticmethod
    def _truncate_gv(
        gvs: Dict[str, Any],
        gws: Dict[str, Any],
        grading_vec: Optional[np.ndarray],
        maximum_degree: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        r"""
        **Description:**
        Truncate Gopakumar-Vafa and Gromov-Witten invariants to a maximum
        total degree.

        The total degree of a curve with charge vector ``q`` is taken to be
        ``q.sum()``.  Entries with total degree greater than ``maximum_degree``
        are removed.

        Args:
            gvs (dict): GV data with keys ``"charges"`` and ``"invariants"``.
            gws (dict): GW data with keys ``"charges"`` and ``"invariants"``.
            maximum_degree (int): Maximum allowed total degree.

        Returns:
            Tuple[dict, dict]: Truncated ``(gvs, gws)``.
        """
        def _trunc(data: Dict[str, Any]) -> Dict[str, Any]:
            """Truncate data arrays to entries within the maximum degree."""
            charges = data.get("charges")
            invs    = data.get("invariants")
            if charges is None or invs is None or grading_vec is None:
                return data
            charges = np.asarray([np.array(c).astype(int) for c in charges])
            invs    = np.asarray(invs)
            deg = charges@grading_vec
            if np.any(deg < 0):
                raise ValueError("Found negative curve degrees. Grading vector may be incorrect.")
            mask    = deg <= maximum_degree
            return {"charges": charges[mask], "invariants": invs[mask]}
        
        return _trunc(gvs), _trunc(gws)

    # ------------------------------------------------------------------
    # Vacua storage
    # ------------------------------------------------------------------

    @property
    def _vacua_dir(self) -> Path:
        r"""Root directory for vacua storage."""
        return self.cache_dir / "vacua"

    def _ensure_vacua_catalog(self) -> None:
        r"""Load vacua catalog from disk if not yet in memory."""
        if self._vacua_catalog is not None:
            return
        pd = _require_pandas()
        catalog_path = self._vacua_dir / "vacua_catalog.parquet"
        if catalog_path.exists():
            self._vacua_catalog = pd.read_parquet(catalog_path)
        else:
            self._vacua_catalog = pd.DataFrame()

    @property
    def _designated_dir(self) -> Path:
        r"""
        Description:
        Root directory for designated (permanent) vacua storage.

        Resolves to the project-local ``vacua_vault/`` directory (see
        :func:`_resolve_vault_dir`).  Falls back to the legacy
        ``<cache_dir>/designated_vacua/`` location only when a user has
        pre-existing data there and the vault directory is empty — this
        preserves backwards compatibility without silent duplication.
        """
        vault = _resolve_vault_dir()
        legacy = self.cache_dir / "designated_vacua"
        # Prefer vault unless only legacy has data (one-time migration hint).
        if (not vault.exists()) and legacy.exists() and any(legacy.iterdir()):
            return legacy
        return vault

    def _resolve_vacua_dir(self, model: Any = None, **identity_kwargs: Any) -> Path:
        r"""
        Description:
        Return the per-model subdirectory inside the vault for designated
        vacua belonging to *model*.

        Path layout:

        - Local KS / CICY models (no ks_id / cicy_id in metadata) →
          ``<vault>/KS/h12_{h12}_model_{model_ID}/`` (analogous for
          CICY).
        - HF-downloaded TDF models → ``<vault>/tdf/ks_{ks_id}_tri_{triang_id}/``.
        - HF-downloaded CICY models → ``<vault>/cicy/cicy_{cicy_id}/``.
        - Fallback (custom model) → ``<vault>/custom/{model_hash}/``.

        Args:
            model: Optional ``flux_sector`` or ``lcs_tree`` used for
                identity extraction.
            **identity_kwargs: Explicit overrides forwarded to
                :func:`_extract_model_identity`.

        Returns:
            Path: Absolute directory path.  Not created on disk.
        """
        vault = _resolve_vault_dir()
        ident = _extract_model_identity(model=model, **identity_kwargs)

        ks = ident.get("ks_id", -1)
        tr = ident.get("triang_id", -1)
        cicy_id = ident.get("cicy_id", -1)
        h12    = ident.get("h12")
        name   = ident.get("model_name") or ""
        mhash  = ident.get("model_hash") or ""

        # HF-downloaded TDF models
        if ks >= 0 and tr >= 0:
            return vault / "tdf" / f"ks_{ks}_tri_{tr}"

        # HF-downloaded CICY models
        if cicy_id >= 0:
            return vault / "cicy" / f"cicy_{cicy_id}"

        # Local models: extract model_ID from auto-generated name
        # ("local_h12_{h12}_ID_{mid}")
        model_ID = None
        if name.startswith("local_h12_") and "_ID_" in name:
            try:
                model_ID = int(name.split("_ID_")[-1])
            except ValueError:
                model_ID = None

        if h12 is not None and model_ID is not None:
            # Distinguish KS from CICY for local models via stored tree
            # metadata if available; default to KS.
            tree = getattr(getattr(model, "periods", None), "lcs_tree", model)
            model_type = getattr(tree, "model_type", None) or "KS"
            return vault / model_type / f"h12_{h12}_model_{model_ID}"

        # Fallback: hash-keyed custom model
        return vault / "custom" / (mhash or "unknown")

    def _ensure_designated_catalog(self) -> None:
        r"""Load designated vacua catalog from disk if not yet in memory."""
        if self._designated_catalog is not None:
            return
        pd = _require_pandas()
        catalog_path = self._designated_dir / "designated_vacua_catalog.parquet"
        if catalog_path.exists():
            self._designated_catalog = pd.read_parquet(catalog_path)
        else:
            self._designated_catalog = pd.DataFrame()

    def vacua_writer(
        self,
        model: Any = None,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        conifold_id: int = -1,
        cicy_id: Optional[int] = None,
        model_name: Optional[str] = None,
        run_id: Optional[str] = None,
        method: str = "manual",
        compute_derived: bool = False,
        filter_fn: Any = None,
        deduplicate: bool = True,
        map_to_fd: bool = False,
        metadata: Optional[dict] = None,
    ) -> "_VacuaWriter":
        r"""
        **Description:**
        Return a context manager for writing vacuum solutions to local
        parquet storage.

        Identity fields (``h11``, ``h12``, ``ks_id``, ``triang_id``,
        ``conifold_id``) are extracted from *model* when available; explicit
        keyword arguments override.  For custom models (no database entry),
        set ``model_name`` for human-readable identification.

        Args:
            model: A ``flux_sector`` or ``lcs_tree`` instance.  Used both
                for identity extraction and (if *compute_derived* is True)
                for computing W, DW, N_flux.
            h11 (int | None): Override :math:`h^{1,1}`.
            h12 (int | None): Override :math:`h^{1,2}`.
            ks_id (int | None): Override Kreuzer-Skarke polytope index.
            triang_id (int | None): Override triangulation index.
            conifold_id (int): Conifold index (-1 for LCS).
            cicy_id (int | None): Override CICY index.
            model_name (str | None): Human-readable label for custom models.
            run_id (str | None): UUID for this run.  Auto-generated if None.
            method (str): Sampling method label.  Defaults to ``"manual"``.
            compute_derived (bool): Compute W, DW, N_flux at flush time.
            filter_fn: Optional callable ``result_dict → bool``.
            deduplicate (bool): Skip duplicate flux vectors.  Defaults to
                ``True``.
            map_to_fd (bool): If True and *model* has ``map_to_FD``, map each
                vacuum to the fundamental domain (SL(2,Z) for tau, monodromy
                for z) before storing and deduplicating.  Defaults to ``False``.
            metadata (dict | None): Arbitrary JSON-serialisable run metadata.

        Returns:
            _VacuaWriter: A context manager.  Use with ``with`` statement.

        Example::

            with db.vacua_writer(model=my_model, method="enumerate") as w:
                for result in results:
                    w.append(result, tags=["PFV"])
            print(f"Stored {w.count} vacua, skipped {w.n_duplicates_skipped}")
        """
        self._check_cache_size()
        identity = _extract_model_identity(
            model=model, h11=h11, h12=h12, ks_id=ks_id,
            triang_id=triang_id, conifold_id=conifold_id,
            cicy_id=cicy_id, model_name=model_name,
        )
        _run_id = run_id or str(uuid.uuid4())

        return _VacuaWriter(
            vacua_dir=self._vacua_dir,
            identity=identity,
            run_id=_run_id,
            method=method,
            metadata=metadata,
            model=model,
            compute_derived=compute_derived,
            filter_fn=filter_fn,
            deduplicate=deduplicate,
            map_to_fd=map_to_fd,
        )

    def query_vacua(
        self,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        conifold_id: Optional[int] = None,
        cicy_id: Optional[int] = None,
        model_hash: Optional[str] = None,
        run_id: Optional[str] = None,
        is_susy: Optional[bool] = None,
        N_flux_max: Optional[float] = None,
        tags: Optional[Any] = None,
        method: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Any:
        r"""
        **Description:**
        Query stored vacua and return matching rows as a ``pandas.DataFrame``.

        Filtering proceeds in two stages:

        1. **Catalog stage** — filter ``vacua_catalog.parquet`` by model
           identity and run-level fields (fast, in-memory).
        2. **Data stage** — for matching runs, load parquet files and apply
           row-level filters (``is_susy``, ``N_flux_max``, ``tags``).

        Args:
            h11, h12, ks_id, triang_id, conifold_id, cicy_id: Model identity
                filters (exact match).
            model_hash (str | None): Filter by geometric fingerprint hash.
            run_id (str | None): Filter by run UUID.
            is_susy (bool | None): Filter by SUSY flag.
            N_flux_max (float | None): Maximum tadpole value.
            tags: String or list of strings.  Rows must contain **all**
                specified tags.
            method (str | None): Filter by sampling method.
            limit (int | None): Maximum number of rows to return.

        Returns:
            pandas.DataFrame: Matching vacuum rows with ``run_id`` column
            injected.
        """
        pd = _require_pandas()
        self._ensure_vacua_catalog()
        cat = self._vacua_catalog
        if cat is None or len(cat) == 0:
            return pd.DataFrame()

        # Stage 1: catalog-level filters
        mask = np.ones(len(cat), dtype=bool)
        if h11 is not None:
            mask &= (cat["h11"] == h11).values
        if h12 is not None:
            mask &= (cat["h12"] == h12).values
        if ks_id is not None:
            mask &= (cat["ks_id"] == ks_id).values
        if triang_id is not None:
            mask &= (cat["triang_id"] == triang_id).values
        if conifold_id is not None:
            mask &= (cat["conifold_id"] == conifold_id).values
        if cicy_id is not None:
            mask &= (cat["cicy_id"] == cicy_id).values
        if model_hash is not None:
            mask &= (cat["model_hash"] == model_hash).values
        if run_id is not None:
            mask &= (cat["run_id"] == run_id).values
        if method is not None:
            mask &= (cat["method"] == method).values

        matching = cat[mask]
        if len(matching) == 0:
            return pd.DataFrame()

        # Stage 2: load and filter data files
        frames = []
        for _, run_row in matching.iterrows():
            path = self._vacua_dir / run_row["file_path"]
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            df["run_id"] = run_row["run_id"]

            if is_susy is not None and "is_susy" in df.columns:
                df = df[df["is_susy"] == is_susy]
            if N_flux_max is not None and "N_flux" in df.columns:
                df = df[df["N_flux"].notna() & (df["N_flux"] <= N_flux_max)]
            if tags is not None:
                if isinstance(tags, str):
                    tags = [tags]
                def _has_all_tags(tags_json, required=tags):
                    """Check whether tags_json contains all required tags."""
                    try:
                        stored = json.loads(tags_json) if isinstance(tags_json, str) else []
                    except (json.JSONDecodeError, TypeError):
                        stored = []
                    return all(t in stored for t in required)
                if "tags" in df.columns:
                    df = df[df["tags"].apply(_has_all_tags)]

            frames.append(df)

        if not frames:
            return pd.DataFrame()
        result = _safe_concat(frames, ignore_index=True)
        if limit is not None:
            result = result.head(limit)
        return result

    def load_vacua(self, run_id: str) -> Any:
        r"""
        **Description:**
        Load all vacuum solutions from a specific run.

        Args:
            run_id (str): UUID of the run.

        Returns:
            pandas.DataFrame: All rows from the run's parquet file.

        Raises:
            KeyError: If the run is not found in the catalog.
        """
        pd = _require_pandas()
        self._ensure_vacua_catalog()
        cat = self._vacua_catalog
        if cat is None or len(cat) == 0:
            raise KeyError(f"Run '{run_id}' not found — vacua catalog is empty.")
        matches = cat[cat["run_id"] == run_id]
        if len(matches) == 0:
            raise KeyError(f"Run '{run_id}' not found in vacua catalog.")
        row = matches.iloc[0]
        path = self._vacua_dir / row["file_path"]
        if not path.exists():
            raise FileNotFoundError(f"Run file not found: {path}")
        df = pd.read_parquet(path)
        df["run_id"] = run_id
        return df

    def solution_exists(
        self,
        flux: Any,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        model: Any = None,
    ) -> bool:
        r"""
        **Description:**
        Check whether a vacuum with the given flux vector is already stored
        for this model.  Uses exact integer comparison.

        Args:
            flux: Integer array of length ``2 * n_fluxes``.
            h11, h12, ks_id, triang_id: Model identity (used to filter runs).
            model: Alternative — extract identity from a ``flux_sector`` or
                ``lcs_tree``.

        Returns:
            bool: True if the flux vector is found in any stored run.
        """
        pd = _require_pandas()
        identity = _extract_model_identity(
            model=model, h11=h11, h12=h12, ks_id=ks_id, triang_id=triang_id,
        )
        flux_key = tuple(int(x) for x in np.asarray(flux))

        self._ensure_vacua_catalog()
        cat = self._vacua_catalog
        if cat is None or len(cat) == 0:
            return False

        # Filter catalog to this model
        mask = np.ones(len(cat), dtype=bool)
        if identity["ks_id"] >= 0:
            mask &= (cat["ks_id"] == identity["ks_id"]).values
            mask &= (cat["triang_id"] == identity["triang_id"]).values
            if "h11" in cat.columns:
                mask &= (cat["h11"] == identity["h11"]).values
        elif identity["model_hash"]:
            mask &= (cat["model_hash"] == identity["model_hash"]).values

        for _, run_row in cat[mask].iterrows():
            path = self._vacua_dir / run_row["file_path"]
            if not path.exists():
                continue
            df = pd.read_parquet(path, columns=["flux"])
            for stored_flux in df["flux"]:
                if tuple(int(x) for x in stored_flux) == flux_key:
                    return True
        return False

    def find_similar_vacua(
        self,
        moduli: Any = None,
        tau: Any = None,
        flux: Any = None,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        model: Any = None,
        flux_tol: int = 0,
        moduli_tol: float = 1e-6,
        tau_tol: float = 1e-6,
    ) -> Any:
        r"""
        **Description:**
        Find stored vacua that are similar to the given point within
        specified tolerances.

        Args:
            moduli: Complex array of length h12 (optional).
            tau: Complex scalar (optional).
            flux: Integer array (optional).  Compared with L∞ tolerance.
            h11, h12, ks_id, triang_id: Model identity filters.
            model: Alternative — extract identity from a model object.
            flux_tol (int): L∞ tolerance on flux vector (0 = exact match).
            moduli_tol (float): Relative tolerance on moduli.
            tau_tol (float): Relative tolerance on tau.

        Returns:
            pandas.DataFrame: Matching vacuum rows.
        """
        pd = _require_pandas()
        identity = _extract_model_identity(
            model=model, h11=h11, h12=h12, ks_id=ks_id, triang_id=triang_id,
        )

        self._ensure_vacua_catalog()
        cat = self._vacua_catalog
        if cat is None or len(cat) == 0:
            return pd.DataFrame()

        # Filter catalog to this model
        mask = np.ones(len(cat), dtype=bool)
        if identity["ks_id"] >= 0:
            mask &= (cat["ks_id"] == identity["ks_id"]).values
            mask &= (cat["triang_id"] == identity["triang_id"]).values
            if "h11" in cat.columns:
                mask &= (cat["h11"] == identity["h11"]).values
        elif identity["model_hash"]:
            mask &= (cat["model_hash"] == identity["model_hash"]).values

        matching_runs = cat[mask]
        if len(matching_runs) == 0:
            return pd.DataFrame()

        # Prepare query values
        flux_q = np.asarray(flux, dtype=np.int32) if flux is not None else None
        moduli_q = np.asarray(moduli, dtype=np.complex128) if moduli is not None else None
        tau_q = complex(tau) if tau is not None else None

        frames = []
        for _, run_row in matching_runs.iterrows():
            path = self._vacua_dir / run_row["file_path"]
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            keep = np.ones(len(df), dtype=bool)

            if flux_q is not None and "flux" in df.columns:
                for i, stored in enumerate(df["flux"]):
                    stored_arr = np.array(stored, dtype=np.int32)
                    if np.max(np.abs(stored_arr - flux_q)) > flux_tol:
                        keep[i] = False

            if moduli_q is not None and "moduli_re" in df.columns:
                for i in range(len(df)):
                    if not keep[i]:
                        continue
                    stored_z = (
                        np.array(df.iloc[i]["moduli_re"])
                        + 1j * np.array(df.iloc[i]["moduli_im"])
                    )
                    denom = np.abs(stored_z)
                    denom = np.where(denom > 0, denom, 1.0)
                    if np.max(np.abs(stored_z - moduli_q) / denom) > moduli_tol:
                        keep[i] = False

            if tau_q is not None and "tau_re" in df.columns:
                for i in range(len(df)):
                    if not keep[i]:
                        continue
                    stored_tau = df.iloc[i]["tau_re"] + 1j * df.iloc[i]["tau_im"]
                    denom = abs(stored_tau) if abs(stored_tau) > 0 else 1.0
                    if abs(stored_tau - tau_q) / denom > tau_tol:
                        keep[i] = False

            df = df[keep]
            if len(df) > 0:
                df["run_id"] = run_row["run_id"]
                frames.append(df)

        if not frames:
            return pd.DataFrame()
        return _safe_concat(frames, ignore_index=True)

    def vacua_info(self) -> None:
        r"""
        **Description:**
        Print a summary of stored vacua.
        """
        self._ensure_vacua_catalog()
        cat = self._vacua_catalog
        if cat is None or len(cat) == 0:
            print("No vacua stored.")
            return

        n_runs   = len(cat)
        n_vacua  = int(cat["n_vacua"].sum())
        n_models = len(cat.groupby(["ks_id", "triang_id", "model_hash"]))
        methods  = cat["method"].value_counts().to_dict()

        print(f"Vacua storage: {n_vacua} vacua across {n_runs} run(s), "
              f"{n_models} distinct model(s)")
        if methods:
            print(f"  Methods: {methods}")

        # Show per-model summary
        groups = cat.groupby(["h11", "ks_id", "triang_id"]).agg(
            n_runs=("run_id", "count"),
            n_vacua=("n_vacua", "sum"),
        ).reset_index()
        if len(groups) <= 20:
            print(f"\n{'h11':>4} {'ks_id':>6} {'trid':>5} {'runs':>5} {'vacua':>7}")
            for _, g in groups.iterrows():
                print(f"{g['h11']:4d} {g['ks_id']:6d} {g['triang_id']:5d} "
                      f"{g['n_runs']:5d} {g['n_vacua']:7d}")
        else:
            print(f"  (showing {len(groups)} models — use query_vacua() for details)")

    def delete_vacua(
        self,
        run_id: Optional[str] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
    ) -> int:
        r"""
        **Description:**
        Delete stored vacua matching the given filter.

        Args:
            run_id (str | None): Delete a specific run.
            ks_id (int | None): Delete all runs for this model.
            triang_id (int | None): Narrow by triangulation (with ks_id).

        Returns:
            int: Number of runs deleted.
        """
        pd = _require_pandas()
        self._ensure_vacua_catalog()
        cat = self._vacua_catalog
        if cat is None or len(cat) == 0:
            return 0

        mask = np.ones(len(cat), dtype=bool)
        if run_id is not None:
            mask &= (cat["run_id"] == run_id).values
        if ks_id is not None:
            mask &= (cat["ks_id"] == ks_id).values
        if triang_id is not None:
            mask &= (cat["triang_id"] == triang_id).values

        to_delete = cat[mask]
        n_deleted = 0
        for _, row in to_delete.iterrows():
            path = self._vacua_dir / row["file_path"]
            if path.exists():
                path.unlink()
            n_deleted += 1

        # Update catalog
        self._vacua_catalog = cat[~mask].reset_index(drop=True)
        catalog_path = self._vacua_dir / "vacua_catalog.parquet"
        if len(self._vacua_catalog) > 0:
            self._vacua_catalog.to_parquet(catalog_path, index=False)
        elif catalog_path.exists():
            catalog_path.unlink()

        return n_deleted

    # ------------------------------------------------------------------
    # Designated vacua — Tier 2 (permanent / curated solutions)
    # ------------------------------------------------------------------

    def validate_vacua(
        self,
        vacua_df: Any,
        model: Any = None,
        F_term_tol: float = 1e-8,
        check_tadpole: bool = True,
        check_physics: bool = True,
        check_duplicates: bool = True,
    ) -> List[dict]:
        r"""
        **Description:**
        Validate vacuum solutions before designation.  Each solution is
        checked for schema conformance, deduplication against already
        designated solutions, and (optionally) physics consistency.

        Can be called standalone or is invoked automatically inside
        :meth:`designate_vacua`.

        Args:
            vacua_df: ``pandas.DataFrame`` of vacuum rows (as returned by
                :meth:`query_vacua`).
            model: Optional ``flux_sector`` for physics re-verification.
                When provided, F-terms and tadpole are recomputed from the
                stored flux and moduli vectors.
            F_term_tol (float): Maximum allowed ``|F_i|`` for a solution to
                pass the physics check.  Defaults to ``1e-8``.
            check_tadpole (bool): Verify ``N_flux ≤ D3_tadpole``.
            check_physics (bool): Recompute F-terms from stored data
                (requires *model*).
            check_duplicates (bool): Check flux vectors against existing
                designated solutions.

        Returns:
            List[dict]: One report entry per row with keys ``"index"``,
            ``"passed"`` (bool), and ``"errors"`` (list of strings).
        """
        pd = _require_pandas()
        report: List[dict] = []

        # Required columns
        required = {"flux", "moduli_re", "moduli_im", "tau_re", "tau_im"}

        # Load existing designated catalog for dedup
        existing_fluxes: set = set()
        if check_duplicates:
            self._ensure_designated_catalog()
            dcat = self._designated_catalog
            if dcat is not None and len(dcat) > 0 and "flux" in dcat.columns:
                for fl in dcat["flux"]:
                    existing_fluxes.add(tuple(int(x) for x in fl))

        for idx in range(len(vacua_df)):
            row = vacua_df.iloc[idx]
            errors: List[str] = []

            # Schema check
            for col in required:
                if col not in vacua_df.columns:
                    errors.append(f"Missing required column: {col}")
                else:
                    val = row.get(col)
                    if val is None:
                        errors.append(f"Missing required column: {col}")
                    elif not hasattr(val, '__len__') and pd.isna(val):
                        errors.append(f"Missing required column: {col}")

            # Dedup check
            if check_duplicates and "flux" in vacua_df.columns:
                try:
                    flux_key = tuple(int(x) for x in row["flux"])
                    if flux_key in existing_fluxes:
                        errors.append("Flux vector already designated")
                except (TypeError, ValueError):
                    errors.append("Invalid flux vector format")

            # Physics check (requires model)
            if check_physics and model is not None and not errors:
                try:
                    flux = np.array(row["flux"], dtype=np.int32)
                    moduli = (np.array(row["moduli_re"])
                              + 1j * np.array(row["moduli_im"]))
                    tau = row["tau_re"] + 1j * row["tau_im"]
                    tau_c = np.conj(tau)
                    moduli_c = np.conj(moduli)

                    DW = np.asarray(
                        model.DW(moduli, moduli_c, tau, tau_c, flux),
                        dtype=np.complex128,
                    )
                    max_F = float(np.max(np.abs(DW)))
                    if max_F > F_term_tol:
                        errors.append(
                            f"F-term residual too large: max|F_i| = {max_F:.2e} "
                            f"> {F_term_tol:.0e}"
                        )
                except Exception as e:
                    errors.append(f"Physics check failed: {e}")

            # Tadpole check
            if check_tadpole and model is not None and not errors:
                try:
                    flux = np.array(row["flux"], dtype=np.int32)
                    N = float(model.tadpole(flux))
                    D3 = getattr(model, "D3_tadpole", None)
                    if D3 is not None and N > float(D3):
                        errors.append(
                            f"Tadpole violation: N_flux={N:.1f} > "
                            f"D3_tadpole={float(D3):.1f}"
                        )
                except Exception as e:
                    errors.append(f"Tadpole check failed: {e}")

            report.append({
                "index": idx,
                "passed": len(errors) == 0,
                "errors": errors,
            })

        return report

    def _validate_for_upload(
        self,
        vacua_df: Any,
        label: str,
        committed_by: str,
        model: Any = None,
        F_term_tol: float = 1e-6,
        check_remote: bool = True,
        repo_id: Optional[str] = None,
    ) -> dict:
        r"""
        **Description:**
        Upload-specific validation wrapper around :meth:`validate_vacua`.
        Runs the standard physics/dedup checks plus extras required for
        pushing to a HuggingFace dataset repo:

        - ``label`` matches the filesystem-safe slug regex.
        - ``committed_by`` is non-empty.
        - Schema version of this package matches the remote repo's
          ``schema.json`` (when *check_remote* is True and the package
          is online).
        - Optional remote duplicate check against the remote catalog.

        Args:
            vacua_df: Vacuum rows to upload.
            label (str): Dataset label (filesystem-safe).
            committed_by (str): Contributor identifier.
            model: Optional ``flux_sector`` for F-term re-verification.
            F_term_tol (float): F-term tolerance.  Defaults to ``1e-6``.
            check_remote (bool): Check schema + catalog on the remote
                repo.  Requires network; skipped automatically when
                ``self.offline`` is True.
            repo_id (str | None): Override target repo.  Defaults to the
                package-wide vault repo (see :func:`set_vault_repo`).

        Returns:
            dict: ``{"passed": bool, "errors": list, "warnings": list,
            "per_row_report": list}``.  Suitable for including in a PR
            summary comment.
        """
        import re
        errors: List[str] = []
        warnings_: List[str] = []

        # 1. Upload-specific bookkeeping
        if not label or not label.strip():
            errors.append("label must be non-empty")
        else:
            slug = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
            if not slug.match(label):
                errors.append(
                    f"label {label!r} is not a valid slug "
                    f"(must match {slug.pattern})"
                )
        if not committed_by or not committed_by.strip():
            errors.append("committed_by must be non-empty")

        # 2. Remote schema + catalogue checks
        if check_remote and not self.offline:
            try:
                repo = repo_id or _resolve_vault_repo()
                hf_download = _require_hf_hub()
                schema_path = hf_download(
                    repo_id=repo,
                    filename="schema.json",
                    repo_type="dataset",
                )
                with open(schema_path) as f:
                    remote_schema = json.load(f)
                remote_ver = int(remote_schema.get("schema_version", -1))
                if remote_ver != SCHEMA_VERSION:
                    errors.append(
                        f"Schema version mismatch: remote={remote_ver}, "
                        f"local={SCHEMA_VERSION}.  Upgrade your local "
                        f"`jaxvacua` package or coordinate with the repo "
                        f"maintainer."
                    )
            except Exception as e:
                # Remote schema may not exist yet (fresh repo) — warn only.
                warnings_.append(f"Could not fetch remote schema: {e}")

        # 3. Row-level validation (reuses existing validate_vacua)
        per_row = self.validate_vacua(
            vacua_df, model=model, F_term_tol=F_term_tol,
            check_tadpole=True,
            check_physics=(model is not None),
            check_duplicates=True,
        )
        n_failed = sum(1 for r in per_row if not r["passed"])
        if n_failed > 0:
            errors.append(
                f"{n_failed}/{len(per_row)} rows failed per-row validation"
            )

        return {
            "passed":          len(errors) == 0,
            "errors":          errors,
            "warnings":        warnings_,
            "per_row_report":  per_row,
        }

    def push_vacua_to_hub(
        self,
        vacua_df: Any,
        label: str,
        committed_by: str,
        model: Any,
        tags: Optional[list] = None,
        notes: str = "",
        repo_id: Optional[str] = None,
        create_pr: bool = True,
        partial_failure: str = "strict",
        if_exists: str = "error",
        check_remote: bool = True,
        token: Optional[str] = None,
        classification: Optional[dict] = None,
    ) -> dict:
        r"""
        **Description:**
        Upload a batch of vacuum solutions to the HuggingFace
        ``aschachner/vacua_vault`` dataset repository (or an override
        via *repo_id* / ``JAXVACUA_VAULT_REPO``).

        The upload lands inside the target model's ``community/``
        directory: ``tdf/h12_{N}/ks_{X}_tri_{Y}/community/{hf_username}_{label}.parquet``
        (or the analogous CICY path).  Filesystem-free-form labels
        (see §5.6 of the vacua-storage plan) mean classification
        (``susy``, ``method``, ``nmax``, ``qualifiers``) is stored as
        parquet-level key-value metadata and extracted by the remote
        catalogue rebuild on merge.

        Args:
            vacua_df: Rows to upload.  Must include at least ``flux``,
                ``moduli_re``, ``moduli_im``, ``tau_re``, ``tau_im``.
            label (str): Filesystem-safe slug for the output file.
            committed_by (str): Contributor identifier (ORCID preferred;
                HF username accepted).
            model: Required ``flux_sector`` for identity extraction and
                physics validation.
            tags (list | None): Free-form tags attached to the dataset.
            notes (str): Free-text annotation.
            repo_id (str | None): Target dataset repo.  Defaults to
                :func:`_resolve_vault_repo`.
            create_pr (bool): If ``True`` (default), open a pull
                request rather than pushing directly to ``main``.
            partial_failure (str): ``"strict"`` (default, refuse the
                upload if any rows fail validation) or ``"split"``
                (valid rows go to the main file; failing rows go to
                ``_rejected/{filename}.rejected.parquet``).
            if_exists (str): Behaviour on filename collision in the
                target repo.  ``"error"`` (default), ``"append"``, or
                ``"new_version"`` (auto-suffix ``_v2``, ``_v3``, ...).
                ``"append"`` is only supported with
                ``create_pr=False`` because it requires a read-modify-
                write round-trip.
            check_remote (bool): Perform remote schema check during
                validation.  Requires network access.
            token (str | None): Optional HF access token.  Falls back
                to the token cached by ``huggingface-cli login`` and
                then to ``HF_TOKEN``.
            classification (dict | None): Optional parquet-level
                metadata to attach to the uploaded file.  Expected
                keys: ``susy`` (``"SUSY"`` | ``"nonSUSY"`` | ``"mixed"``),
                ``method`` (str), ``nmax`` (int), and ``qualifiers``
                (dict).  When omitted, sensible defaults are inferred.

        Returns:
            dict: ``{"pr_url": str | None, "commit_url": str,
            "file_path": str, "n_uploaded": int, "n_rejected": int,
            "report": ValidationReport}``.

        Raises:
            RuntimeError: If ``self.offline=True``.
            ValidationError: If validation fails and
                ``partial_failure="strict"``.
            FileExistsError: If the target filename exists and
                ``if_exists="error"``.
        """
        if self.offline:
            raise RuntimeError(
                "push_vacua_to_hub requires network access "
                "(self.offline=True)."
            )
        if partial_failure not in ("strict", "split"):
            raise ValueError(
                f"partial_failure must be 'strict' or 'split', "
                f"got {partial_failure!r}"
            )
        if if_exists not in ("error", "append", "new_version"):
            raise ValueError(
                f"if_exists must be 'error', 'append', or 'new_version', "
                f"got {if_exists!r}"
            )

        pd = _require_pandas()
        pq = _require_pyarrow()
        try:
            import pyarrow as pa  # for Table-level metadata manipulation
        except ImportError as e:
            raise ImportError(
                "The 'pyarrow' package is required for push_vacua_to_hub."
            ) from e
        hf_api = _require_hf_api()
        hf_download = _require_hf_hub()
        repo = repo_id or _resolve_vault_repo()

        # 1. Client-side validation
        report = self._validate_for_upload(
            vacua_df, label=label, committed_by=committed_by,
            model=model, check_remote=check_remote, repo_id=repo,
        )

        # 2. Apply partial-failure policy
        n_total = len(vacua_df)
        passed_idx = [r["index"] for r in report["per_row_report"] if r["passed"]]
        failed_idx = [r["index"] for r in report["per_row_report"] if not r["passed"]]

        if failed_idx and partial_failure == "strict":
            msg = (
                f"{len(failed_idx)}/{n_total} rows failed validation. "
                f"Use partial_failure='split' to upload only the valid rows."
            )
            raise ValidationError(msg, report=report["per_row_report"])

        valid_df    = vacua_df.iloc[passed_idx].reset_index(drop=True)
        rejected_df = vacua_df.iloc[failed_idx].reset_index(drop=True) if failed_idx else None
        if rejected_df is not None and len(rejected_df) > 0:
            # Attach per-row error strings
            errs = []
            for r in report["per_row_report"]:
                if not r["passed"]:
                    errs.append("; ".join(r["errors"]))
            rejected_df = rejected_df.copy()
            rejected_df["error"] = errs

        # Abort if nothing to upload
        if len(valid_df) == 0:
            raise ValidationError(
                "No valid rows to upload.", report=report["per_row_report"],
            )

        # 3. Identify the authenticated HF user (for filename prefix)
        try:
            who = hf_api.whoami(token=token)
            hf_username = who.get("name") or who.get("username") or "anonymous"
        except Exception as e:
            raise RuntimeError(
                f"Could not identify HF user (check HF_TOKEN / huggingface-cli login): {e}"
            ) from e

        # 4. Resolve remote paths
        identity = _extract_model_identity(model=model)
        model_dir_remote = self._remote_model_dir(identity)
        if model_dir_remote is None:
            raise ValueError(
                "Could not derive a remote model directory from the "
                "given model.  Pass ks_id/triang_id or cicy_id "
                "explicitly."
            )

        base_name = f"{hf_username}_{label}"
        ext = ".parquet"
        target_rel = f"{model_dir_remote}/community/{base_name}{ext}"

        # 5. Handle filename collisions according to if_exists
        final_rel = self._resolve_remote_filename(
            hf_api, hf_download, repo, target_rel, if_exists=if_exists,
            token=token,
        )

        # 6. Write the valid rows (plus pyarrow metadata) to a scratch
        #    parquet file locally, then upload it.
        import tempfile, time
        cls_meta = classification or {}
        arrow_meta = {
            "schema_version": str(SCHEMA_VERSION),
            "susy":           str(cls_meta.get("susy", "unknown")),
            "method":         str(cls_meta.get("method", "unknown")),
            "nmax":           str(cls_meta.get("nmax", "")),
            "qualifiers":     json.dumps(cls_meta.get("qualifiers", {})),
            "label":          label,
            "committed_by":   committed_by,
            "notes":          notes or "",
            "uploaded_at":    datetime.datetime.now(
                                  datetime.timezone.utc).isoformat(),
        }
        if tags:
            arrow_meta["tags"] = json.dumps(list(tags))

        def _write_with_metadata(df, path, meta):
            table = pa.Table.from_pandas(df, preserve_index=False)
            existing = dict(table.schema.metadata or {})
            existing.update({k.encode(): v.encode() for k, v in meta.items()})
            table = table.replace_schema_metadata(existing)
            pq.write_table(table, path)

        result: Dict[str, Any] = {
            "pr_url":       None,
            "commit_url":   None,
            "file_path":    final_rel,
            "n_uploaded":   len(valid_df),
            "n_rejected":   len(rejected_df) if rejected_df is not None else 0,
            "report":       report,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid file
            local_main = os.path.join(tmpdir, os.path.basename(final_rel))
            _write_with_metadata(valid_df, local_main, arrow_meta)

            # Optional rejected file
            local_rejected = None
            rejected_rel   = None
            if rejected_df is not None and len(rejected_df) > 0:
                rej_name = (
                    os.path.splitext(os.path.basename(final_rel))[0]
                    + ".rejected" + ext
                )
                rejected_rel = f"{model_dir_remote}/community/_rejected/{rej_name}"
                local_rejected = os.path.join(tmpdir, rej_name)
                rej_meta = dict(arrow_meta)
                rej_meta["status"] = "rejected"
                _write_with_metadata(rejected_df, local_rejected, rej_meta)

            # Upload
            commit_msg = f"add: {final_rel} (label={label}, by={committed_by})"
            operations = []
            operations.append(dict(
                local_path=local_main, path_in_repo=final_rel,
            ))
            if local_rejected is not None:
                operations.append(dict(
                    local_path=local_rejected, path_in_repo=rejected_rel,
                ))

            uploaded_paths = []
            for op in operations:
                ci = hf_api.upload_file(
                    path_or_fileobj=op["local_path"],
                    path_in_repo=op["path_in_repo"],
                    repo_id=repo,
                    repo_type="dataset",
                    commit_message=commit_msg,
                    token=token,
                    create_pr=create_pr,
                )
                uploaded_paths.append(op["path_in_repo"])
                # huggingface_hub returns a CommitInfo for direct uploads
                # or a PR-related response when create_pr=True.
                if create_pr and result["pr_url"] is None:
                    pr_url = getattr(ci, "pr_url", None) or getattr(ci, "url", None)
                    result["pr_url"] = pr_url
                if not create_pr and result["commit_url"] is None:
                    result["commit_url"] = getattr(ci, "commit_url", None) or getattr(ci, "oid", None)

            result["file_path"]    = uploaded_paths[0]
            result["rejected_path"] = uploaded_paths[1] if len(uploaded_paths) > 1 else None

        return result

    def fetch_vacua_from_hub(
        self,
        model: Any = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        cicy_id: Optional[int] = None,
        label: Optional[str] = None,
        include_community: bool = False,
        include_retracted: bool = False,
        cache: bool = True,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Any:
        r"""
        **Description:**
        Download curated (and optionally community) vacuum solutions
        for a given model from the HuggingFace vacua repo.

        Args:
            model: Optional ``flux_sector`` for identity extraction.
            ks_id, triang_id, cicy_id: Explicit identity overrides.
            label (str | None): If given, match only files whose name
                starts with this label (before any ``_v{n}`` suffix).
            include_community (bool): Also fetch files under
                ``community/``.  Defaults to False.
            include_retracted (bool): Include rows flagged
                ``retracted=True`` in the returned DataFrame.
            cache (bool): Keep downloads in the HF cache (reserved for
                future use; ``hf_hub_download`` always caches).
            repo_id (str | None): Target dataset repo.  Defaults to
                :func:`_resolve_vault_repo`.
            token (str | None): Optional HF access token.

        Returns:
            pandas.DataFrame: Concatenated rows from all matching
            files, with an extra ``_source_path`` column identifying
            the origin file.  Empty DataFrame if nothing matches.
        """
        if self.offline:
            raise RuntimeError(
                "fetch_vacua_from_hub requires network access "
                "(self.offline=True)."
            )
        pd = _require_pandas()
        hf_download = _require_hf_hub()
        hf_api = _require_hf_api()
        repo = repo_id or _resolve_vault_repo()

        identity = _extract_model_identity(
            model=model, ks_id=ks_id, triang_id=triang_id,
            cicy_id=cicy_id,
        )
        model_dir = self._remote_model_dir(identity)
        if model_dir is None:
            raise ValueError(
                "Could not derive a remote model directory from the "
                "given model/identity."
            )

        # List remote files under the model directory
        try:
            all_files = hf_api.list_repo_files(
                repo_id=repo, repo_type="dataset", token=token,
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not list files in {repo}: {e}"
            ) from e

        def _matches(path: str) -> bool:
            if not path.startswith(model_dir + "/"):
                return False
            if not path.endswith(".parquet"):
                return False
            # Curated files live directly in model_dir; community files
            # live in model_dir/community/; _rejected/ always excluded.
            rel = path[len(model_dir) + 1:]
            if rel.startswith("_rejected/"):
                return False
            is_community = rel.startswith("community/")
            if is_community and not include_community:
                return False
            if not is_community and "/" in rel:
                return False  # nested file (shouldn't happen)
            # Label filter
            if label is not None:
                name = os.path.splitext(os.path.basename(path))[0]
                # Community files carry a "{hf_username}_" prefix; strip it
                if is_community and "_" in name:
                    # Find the first underscore; the rest is {label}[_v{n}]
                    name = name.split("_", 1)[1]
                # Accept exact label or label_v{n}
                import re
                if not re.match(rf"^{re.escape(label)}(_v\d+)?$", name):
                    return False
            return True

        matching = sorted(p for p in all_files if _matches(p))
        if not matching:
            return pd.DataFrame()

        dfs = []
        for rp in matching:
            try:
                local = hf_download(
                    repo_id=repo, filename=rp,
                    repo_type="dataset", token=token,
                )
            except Exception as e:
                print(f"[fetch_vacua_from_hub] failed to download {rp}: {e}")
                continue
            try:
                df = pd.read_parquet(local)
                df["_source_path"] = rp
                dfs.append(df)
            except Exception as e:
                print(f"[fetch_vacua_from_hub] failed to parse {rp}: {e}")

        if not dfs:
            return pd.DataFrame()
        out = _safe_concat(dfs, ignore_index=True)
        if not include_retracted and "retracted" in out.columns:
            out = out[~out["retracted"].fillna(False).astype(bool)].reset_index(drop=True)
        return out

    def list_hub_vacua(
        self,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        cicy_id: Optional[int] = None,
        label: Optional[str] = None,
        committed_by: Optional[str] = None,
        include_retracted: bool = False,
        limit: Optional[int] = None,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Any:
        r"""
        **Description:**
        Browse the remote vacua-vault catalogue.  Downloads
        ``catalog.parquet`` from the target repo and applies boolean
        masks following the same filter API as
        :meth:`query_designated`.

        Args:
            h11, h12, ks_id, triang_id, cicy_id: Identity filters.
            label (str | None): Filter by dataset label.
            committed_by (str | None): Filter by contributor.
            include_retracted (bool): Include retracted entries.
            limit (int | None): Return at most this many rows.
            repo_id (str | None): Target dataset repo.
            token (str | None): Optional HF access token.

        Returns:
            pandas.DataFrame: Catalog rows matching the filters.
            Empty DataFrame if the remote catalog is empty or missing.
        """
        if self.offline:
            raise RuntimeError(
                "list_hub_vacua requires network access (offline=True)."
            )
        pd = _require_pandas()
        hf_download = _require_hf_hub()
        repo = repo_id or _resolve_vault_repo()

        try:
            local = hf_download(
                repo_id=repo, filename="catalog.parquet",
                repo_type="dataset", token=token,
            )
        except Exception as e:
            # Catalog may not exist yet on a fresh repo.
            print(f"[list_hub_vacua] could not fetch remote catalog: {e}")
            return pd.DataFrame()

        cat = pd.read_parquet(local)
        filters = {
            "h11": h11, "h12": h12, "ks_id": ks_id,
            "triang_id": triang_id, "cicy_id": cicy_id,
            "label": label, "committed_by": committed_by,
        }
        for col, val in filters.items():
            if val is not None and col in cat.columns:
                cat = cat[cat[col] == val]

        if not include_retracted and "retracted" in cat.columns:
            cat = cat[~cat["retracted"].fillna(False).astype(bool)]

        if limit is not None:
            cat = cat.head(limit)

        return cat.reset_index(drop=True)

    @staticmethod
    def _remote_model_dir(identity: Dict[str, Any]) -> Optional[str]:
        r"""
        Derive the per-model subdirectory on the remote vault repo from
        an identity dict.  Returns ``None`` if the identity doesn't
        provide enough information.
        """
        ks  = identity.get("ks_id", -1)
        tr  = identity.get("triang_id", -1)
        cic = identity.get("cicy_id", -1)
        h12 = identity.get("h12")
        if ks >= 0 and tr >= 0 and h12 is not None:
            return f"tdf/h12_{h12}/ks_{ks}_tri_{tr}"
        if cic >= 0:
            return f"cicy/cicy_{cic}"
        return None

    @staticmethod
    def _resolve_remote_filename(
        hf_api: Any,
        hf_download: Any,
        repo: str,
        target_rel: str,
        if_exists: str,
        token: Optional[str] = None,
    ) -> str:
        r"""
        Implement the ``if_exists`` collision policy for a target remote
        filename.  Returns the final ``path_in_repo`` to use.

        - ``"error"``: raise if the file already exists.
        - ``"append"``: return the existing path (caller is responsible
          for appending — this helper just signals).
        - ``"new_version"``: auto-suffix ``_v2``, ``_v3``, ... until a
          free name is found.
        """
        # Probe existence by trying to download with a head-only call
        # (hf_hub_download returns file path if exists).  List files
        # via HfApi for a cleaner existence check.
        try:
            files = set(hf_api.list_repo_files(repo_id=repo, repo_type="dataset",
                                                token=token))
        except Exception:
            # Could be empty / unreachable repo; treat as "no collision".
            files = set()

        if target_rel not in files:
            return target_rel

        if if_exists == "error":
            raise FileExistsError(
                f"{target_rel!r} already exists in {repo!r}. "
                f"Pick a different label or use "
                f"if_exists='new_version' (auto-suffix) or "
                f"if_exists='append' (merge)."
            )
        if if_exists == "append":
            return target_rel
        # "new_version" → find the first free _vN suffix
        import re
        stem, ext = os.path.splitext(target_rel)
        base_stem = re.sub(r"_v\d+$", "", stem)  # strip any existing _vN
        n = 2
        while f"{base_stem}_v{n}{ext}" in files:
            n += 1
        return f"{base_stem}_v{n}{ext}"

    def designate_vacua(
        self,
        vacua_df: Any,
        label: str,
        committed_by: str,
        model: Any = None,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        conifold_id: int = -1,
        cicy_id: Optional[int] = None,
        model_name: Optional[str] = None,
        tags: Optional[Any] = None,
        notes: str = "",
        validate: bool = True,
        force: bool = False,
        F_term_tol: float = 1e-8,
    ) -> List[int]:
        r"""
        **Description:**
        Designate vacuum solutions as permanent / curated.  Solutions are
        written to the ``designated_vacua/`` directory with provenance
        metadata and added to the designated catalog.

        By default, all solutions are validated before writing.  Validation
        failures raise :class:`ValidationError` unless ``force=True``.

        The input *vacua_df* can be a ``DataFrame`` from
        :meth:`query_vacua`, or any DataFrame with the standard vacuum
        columns (``flux``, ``moduli_re``, ``moduli_im``, ``tau_re``,
        ``tau_im``).

        Args:
            vacua_df: ``pandas.DataFrame`` of vacuum rows.
            label (str): Human-readable label (e.g. ``"benchmark_ISD"``,
                ``"paper:2506.xxxxx"``).
            committed_by (str): Author identifier (name or ORCID).
            model: Optional ``flux_sector`` for identity extraction and
                physics validation.
            h11, h12, ks_id, triang_id, conifold_id, cicy_id: Explicit
                model identity overrides.
            model_name (str | None): Human-readable label for custom models.
            tags: String or list of tag strings applied to all solutions.
            notes (str): Free-text annotation.
            validate (bool): Run validation before designating.
            force (bool): Designate even if validation fails (failed
                solutions are skipped).
            F_term_tol (float): F-term tolerance for validation.

        Returns:
            List[int]: ``designated_id`` values assigned to the new
            solutions.

        Raises:
            ValidationError: If validation fails and ``force=False``.
            ValueError: If ``label`` or ``committed_by`` is empty.
        """
        pd = _require_pandas()

        if not label or not label.strip():
            raise ValueError("label must be non-empty")
        if not committed_by or not committed_by.strip():
            raise ValueError("committed_by must be non-empty")

        # Resolve model identity
        identity = _extract_model_identity(
            model=model, h11=h11, h12=h12, ks_id=ks_id,
            triang_id=triang_id, conifold_id=conifold_id,
            cicy_id=cicy_id, model_name=model_name,
        )

        # Validation
        if validate:
            report = self.validate_vacua(
                vacua_df, model=model, F_term_tol=F_term_tol,
            )
            n_failed = sum(1 for r in report if not r["passed"])
            if n_failed > 0 and not force:
                msg_lines = [f"{n_failed}/{len(report)} solutions failed validation:"]
                for r in report:
                    if not r["passed"]:
                        msg_lines.append(f"  row {r['index']}: {'; '.join(r['errors'])}")
                raise ValidationError("\n".join(msg_lines), report=report)
            # If force, keep only passing rows
            if force and n_failed > 0:
                keep_idx = [r["index"] for r in report if r["passed"]]
                vacua_df = vacua_df.iloc[keep_idx].reset_index(drop=True)

        if len(vacua_df) == 0:
            return []

        # Normalise tags
        if tags is None:
            tags_json = "[]"
        elif isinstance(tags, str):
            tags_json = json.dumps([tags])
        else:
            tags_json = json.dumps(list(tags))

        # Determine next designated_id
        self._ensure_designated_catalog()
        dcat = self._designated_catalog
        if dcat is not None and len(dcat) > 0:
            next_id = int(dcat["designated_id"].max()) + 1
        else:
            next_id = 0

        # Get jaxvacua version
        try:
            from jaxvacua import __version__ as _jv_version
        except ImportError:
            _jv_version = "unknown"

        now = datetime.datetime.now(datetime.timezone.utc)
        today = now.date().isoformat()

        # Build catalog rows and data rows
        catalog_rows = []
        data_rows = []
        assigned_ids = []

        for i in range(len(vacua_df)):
            row = vacua_df.iloc[i]
            did = next_id + i
            assigned_ids.append(did)

            # Merge tags: existing row tags + new tags
            existing_tags = []
            if "tags" in vacua_df.columns:
                try:
                    et = row["tags"]
                    if isinstance(et, str):
                        existing_tags = json.loads(et)
                except (json.JSONDecodeError, TypeError):
                    pass
            new_tags = json.loads(tags_json)
            merged_tags = list(dict.fromkeys(existing_tags + new_tags))

            # Source run_id
            source_run = str(row.get("run_id", "")) if "run_id" in vacua_df.columns else ""

            catalog_rows.append({
                "designated_id": did,
                "h11":           identity["h11"],
                "h12":           identity["h12"],
                "ks_id":         identity["ks_id"],
                "triang_id":     identity["triang_id"],
                "conifold_id":   identity["conifold_id"],
                "cicy_id":       identity["cicy_id"],
                "model_hash":    identity["model_hash"],
                "model_name":    identity.get("model_name"),
                "flux":          list(int(x) for x in row["flux"]),
                "N_flux":        row.get("N_flux"),
                "is_susy":       bool(row.get("is_susy", False)),
                "tags":          json.dumps(merged_tags),
                "label":         label,
                "notes":         notes,
                "committed_by":  committed_by,
                "commit_date":   today,
                "jaxvacua_version": _jv_version,
                "source_run_id": source_run,
                "retracted":     False,
                "retraction_reason": None,
            })

            # Data row — copy the solution data plus designated_id
            drow = {
                "designated_id": did,
                "flux":          list(int(x) for x in row["flux"]),
                "moduli_re":     list(row["moduli_re"]) if "moduli_re" in vacua_df.columns else None,
                "moduli_im":     list(row["moduli_im"]) if "moduli_im" in vacua_df.columns else None,
                "tau_re":        row.get("tau_re"),
                "tau_im":        row.get("tau_im"),
                "W_re":          row.get("W_re"),
                "W_im":          row.get("W_im"),
                "F_terms_re":    row.get("F_terms_re"),
                "F_terms_im":    row.get("F_terms_im"),
                "N_flux":        row.get("N_flux"),
                "residual":      row.get("residual"),
                "is_susy":       bool(row.get("is_susy", False)),
                "tags":          json.dumps(merged_tags),
                "extra_data":    row.get("extra_data"),
                "label":         label,
                "committed_by":  committed_by,
                "commit_date":   today,
            }
            data_rows.append(drow)

        # Determine shard path — per-model subdirectory inside vacua_vault/
        shard_dir = self._resolve_vacua_dir(model=model, **{
            k: identity[k] for k in (
                "h11", "h12", "ks_id", "triang_id",
                "conifold_id", "cicy_id", "model_name",
            ) if k in identity
        })
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shard_dir / "shard_0.parquet"

        # Append to shard
        data_df = pd.DataFrame(data_rows)
        if shard_path.exists():
            existing = pd.read_parquet(shard_path)
            data_df = _safe_concat([existing, data_df], ignore_index=True)
        data_df.to_parquet(shard_path, index=False)

        # Update catalog
        catalog_df = pd.DataFrame(catalog_rows)
        catalog_path = self._designated_dir / "designated_vacua_catalog.parquet"
        self._designated_dir.mkdir(parents=True, exist_ok=True)
        if dcat is not None and len(dcat) > 0:
            catalog_df = _safe_concat([dcat, catalog_df], ignore_index=True)
        catalog_df.to_parquet(catalog_path, index=False)
        self._designated_catalog = catalog_df

        return assigned_ids

    def query_designated(
        self,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        conifold_id: Optional[int] = None,
        cicy_id: Optional[int] = None,
        model_hash: Optional[str] = None,
        label: Optional[str] = None,
        committed_by: Optional[str] = None,
        is_susy: Optional[bool] = None,
        N_flux_max: Optional[float] = None,
        tags: Optional[Any] = None,
        include_retracted: bool = False,
        limit: Optional[int] = None,
    ) -> Any:
        r"""
        **Description:**
        Query designated (permanent) vacua from the catalog.

        All filtering is done at the catalog level (one row per solution),
        so this is fast even for large collections.

        Args:
            h11, h12, ks_id, triang_id, conifold_id, cicy_id: Model
                identity filters (exact match).
            model_hash (str | None): Filter by geometric fingerprint.
            label (str | None): Filter by label (exact match).
            committed_by (str | None): Filter by author.
            is_susy (bool | None): Filter by SUSY flag.
            N_flux_max (float | None): Maximum tadpole value.
            tags: String or list of strings.  Rows must contain **all**
                specified tags.
            include_retracted (bool): Include retracted solutions.
                Defaults to ``False``.
            limit (int | None): Maximum number of rows to return.

        Returns:
            pandas.DataFrame: Matching rows from the designated catalog.
        """
        pd = _require_pandas()
        self._ensure_designated_catalog()
        dcat = self._designated_catalog
        if dcat is None or len(dcat) == 0:
            return pd.DataFrame()

        mask = np.ones(len(dcat), dtype=bool)

        # Exclude retracted by default
        if not include_retracted and "retracted" in dcat.columns:
            mask &= ~dcat["retracted"].fillna(False).astype(bool).values

        if h11 is not None:
            mask &= (dcat["h11"] == h11).values
        if h12 is not None:
            mask &= (dcat["h12"] == h12).values
        if ks_id is not None:
            mask &= (dcat["ks_id"] == ks_id).values
        if triang_id is not None:
            mask &= (dcat["triang_id"] == triang_id).values
        if conifold_id is not None:
            mask &= (dcat["conifold_id"] == conifold_id).values
        if cicy_id is not None:
            mask &= (dcat["cicy_id"] == cicy_id).values
        if model_hash is not None:
            mask &= (dcat["model_hash"] == model_hash).values
        if label is not None:
            mask &= (dcat["label"] == label).values
        if committed_by is not None:
            mask &= (dcat["committed_by"] == committed_by).values
        if is_susy is not None:
            mask &= (dcat["is_susy"] == is_susy).values
        if N_flux_max is not None and "N_flux" in dcat.columns:
            mask &= dcat["N_flux"].fillna(float("inf")).values <= N_flux_max
        if tags is not None:
            if isinstance(tags, str):
                tags = [tags]
            def _has_all_tags(tags_json, required=tags):
                """Check whether tags_json contains all required tags."""
                try:
                    stored = json.loads(tags_json) if isinstance(tags_json, str) else []
                except (json.JSONDecodeError, TypeError):
                    stored = []
                return all(t in stored for t in required)
            if "tags" in dcat.columns:
                mask &= dcat["tags"].apply(_has_all_tags).values

        result = dcat[mask]
        if limit is not None:
            result = result.head(limit)
        return result.reset_index(drop=True)

    def load_designated(
        self,
        designated_ids: Optional[List[int]] = None,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        model_hash: Optional[str] = None,
    ) -> Any:
        r"""
        **Description:**
        Load full data (moduli, flux, derived quantities) for designated
        solutions.

        Either specify ``designated_ids`` directly, or use model identity
        filters to select all designated solutions for a model.

        Args:
            designated_ids (list[int] | None): Specific designated IDs.
            h11, h12, ks_id, triang_id: Model identity filters.
            model_hash (str | None): Filter by geometric fingerprint.

        Returns:
            pandas.DataFrame: Full solution data with ``designated_id``.
        """
        pd = _require_pandas()
        self._ensure_designated_catalog()
        dcat = self._designated_catalog
        if dcat is None or len(dcat) == 0:
            return pd.DataFrame()

        # Determine which shard directories to load
        if designated_ids is not None:
            subset = dcat[dcat["designated_id"].isin(designated_ids)]
        else:
            mask = np.ones(len(dcat), dtype=bool)
            if h11 is not None:
                mask &= (dcat["h11"] == h11).values
            if h12 is not None:
                mask &= (dcat["h12"] == h12).values
            if ks_id is not None:
                mask &= (dcat["ks_id"] == ks_id).values
            if triang_id is not None:
                mask &= (dcat["triang_id"] == triang_id).values
            if model_hash is not None:
                mask &= (dcat["model_hash"] == model_hash).values
            subset = dcat[mask]

        if len(subset) == 0:
            return pd.DataFrame()

        # Group by model to find shard paths
        target_ids = set(subset["designated_id"].values)
        frames = []

        # Scan all shard files under designated_vacua/
        if self._designated_dir.exists():
            for shard_path in self._designated_dir.rglob("shard_*.parquet"):
                try:
                    df = pd.read_parquet(shard_path)
                    if "designated_id" in df.columns:
                        matched = df[df["designated_id"].isin(target_ids)]
                        if len(matched) > 0:
                            frames.append(matched)
                except Exception:
                    continue

        if not frames:
            return pd.DataFrame()
        return _safe_concat(frames, ignore_index=True)

    def designated_info(self) -> None:
        r"""
        **Description:**
        Print a summary of designated (permanent) vacua.
        """
        self._ensure_designated_catalog()
        dcat = self._designated_catalog
        if dcat is None or len(dcat) == 0:
            print("No designated vacua stored.")
            return

        # Filter active (non-retracted)
        if "retracted" in dcat.columns:
            active = dcat[~dcat["retracted"].fillna(False).astype(bool)]
            n_retracted = len(dcat) - len(active)
        else:
            active = dcat
            n_retracted = 0

        n_total  = len(active)
        n_models = len(active.groupby(["ks_id", "triang_id", "model_hash"]))
        labels   = active["label"].value_counts().to_dict() if "label" in active.columns else {}
        authors  = active["committed_by"].nunique() if "committed_by" in active.columns else 0

        print(f"Designated vacua: {n_total} solution(s) across "
              f"{n_models} model(s), {authors} contributor(s)")
        if n_retracted > 0:
            print(f"  ({n_retracted} retracted, excluded from counts)")
        if labels:
            print(f"  Labels: {labels}")

        # Per-model summary
        groups = active.groupby(["h11", "ks_id", "triang_id"]).agg(
            n_vacua=("designated_id", "count"),
        ).reset_index()
        if len(groups) <= 20:
            print(f"\n{'h11':>4} {'ks_id':>6} {'trid':>5} {'vacua':>7}")
            for _, g in groups.iterrows():
                print(f"{g['h11']:4d} {g['ks_id']:6d} {g['triang_id']:5d} "
                      f"{g['n_vacua']:7d}")
        else:
            print(f"  ({len(groups)} models — use query_designated() for details)")

    def load_local_vacua(
        self,
        model: Any = None,
        label: Optional[str] = None,
        include_retracted: bool = False,
        **identity_kwargs: Any,
    ) -> Any:
        r"""
        **Description:**
        Load designated vacua stored in the local ``vacua_vault/`` for a
        given model.

        Convenience wrapper around :meth:`query_designated` that
        resolves the per-model vault subdirectory via
        :meth:`_resolve_vacua_dir` and returns any solutions previously
        designated for that model.

        Args:
            model: Optional ``flux_sector`` or ``lcs_tree`` used for
                identity extraction.
            label (str | None): If given, filter by designation label.
            include_retracted (bool): Include retracted entries.
                Defaults to False.
            **identity_kwargs: Explicit identity overrides
                (``h11``, ``h12``, ``ks_id``, ``triang_id``,
                ``cicy_id``, ``model_name``).

        Returns:
            pandas.DataFrame: Solutions with the standard vacuum
            columns.  Empty DataFrame if no vacua are stored.
        """
        ident = _extract_model_identity(model=model, **identity_kwargs)
        query = {}
        if ident.get("h11") is not None:     query["h11"]       = ident["h11"]
        if ident.get("h12") is not None:     query["h12"]       = ident["h12"]
        if ident.get("ks_id", -1) >= 0:      query["ks_id"]     = ident["ks_id"]
        if ident.get("triang_id", -1) >= 0:  query["triang_id"] = ident["triang_id"]
        if ident.get("cicy_id", -1) >= 0:    query["cicy_id"]   = ident["cicy_id"]
        if label is not None:                query["label"]     = label

        try:
            df = self.query_designated(**query)
        except TypeError:
            # Fallback for a reduced query_designated signature: query
            # by label (if any) and filter remaining columns manually.
            df = self.query_designated(label=label) if label else self.query_designated()
            if df is not None and len(df) > 0:
                for col, val in query.items():
                    if col in df.columns:
                        df = df[df[col] == val]
                df = df.reset_index(drop=True)

        if df is not None and len(df) > 0 and not include_retracted:
            if "retracted" in df.columns:
                df = df[~df["retracted"].fillna(False).astype(bool)].reset_index(drop=True)
        return df

    def retract_designated(
        self,
        designated_id: int,
        reason: str,
        retracted_by: Optional[str] = None,
    ) -> None:
        r"""
        **Description:**
        Retract a designated solution.  The solution is **not** deleted —
        it remains in the catalog and shard with ``retracted=True``,
        ``retraction_reason``, and ``retracted_at`` (timestamp) set.
        Retracted solutions are excluded from :meth:`query_designated` by
        default.

        A row is also appended to ``<vault>/retractions.parquet`` as a
        permanent audit trail that survives any future catalog rebuild.

        Args:
            designated_id (int): The ID of the solution to retract.
            reason (str): Explanation for the retraction (e.g.
                ``"Bug in jaxvacua v0.3.1 period computation"``).
            retracted_by (str | None): Optional identifier of the
                person/agent performing the retraction (ORCID, HF
                username, email).  Recorded in the audit trail only.

        Raises:
            KeyError: If the designated_id is not found.
            ValueError: If *reason* is empty.
        """
        pd = _require_pandas()
        if not reason or not reason.strip():
            raise ValueError("retraction reason must be non-empty")

        self._ensure_designated_catalog()
        dcat = self._designated_catalog
        if dcat is None or len(dcat) == 0:
            raise KeyError(f"Designated ID {designated_id} not found — catalog is empty.")

        mask = dcat["designated_id"] == designated_id
        if not mask.any():
            raise KeyError(f"Designated ID {designated_id} not found in catalog.")

        now = pd.Timestamp.now(tz="UTC")
        dcat.loc[mask, "retracted"]         = True
        dcat.loc[mask, "retraction_reason"] = reason
        dcat.loc[mask, "retracted_at"]      = now

        catalog_path = self._designated_dir / "designated_vacua_catalog.parquet"
        dcat.to_parquet(catalog_path, index=False)
        self._designated_catalog = dcat

        # Append to <vault>/retractions.parquet (audit trail)
        audit_row = {
            "designated_id": int(designated_id),
            "reason":        reason,
            "retracted_at":  now,
            "retracted_by":  retracted_by or "",
        }
        audit_path = _resolve_vault_dir() / "retractions.parquet"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        if audit_path.exists():
            existing = pd.read_parquet(audit_path)
            audit_df = _safe_concat([existing, pd.DataFrame([audit_row])],
                                    ignore_index=True)
        else:
            audit_df = pd.DataFrame([audit_row])
        audit_df.to_parquet(audit_path, index=False)

    def purge_retracted(
        self,
        older_than: Optional[str] = "30d",
        dry_run: bool = True,
        confirm: bool = False,
        archive: bool = False,
    ) -> list:
        r"""
        **Description:**
        Irreversibly delete retracted designated solutions.

        Selects rows in the designated catalog with ``retracted=True``
        and (if *older_than* is given) ``retracted_at`` older than the
        threshold.  Prints a summary of what would be deleted; only acts
        if both ``dry_run=False`` and ``confirm=True``.

        Args:
            older_than (str | None): Pandas timedelta string (e.g.
                ``"30d"``, ``"90d"``, ``"6h"``) or ``None`` to ignore
                age.  Defaults to ``"30d"``.
            dry_run (bool): If True (default), print what would be
                deleted and return the catalog subset — no files are
                touched.
            confirm (bool): Must be True alongside ``dry_run=False`` to
                actually delete.  Guardrail against accidental data
                loss.
            archive (bool): If True, copy each shard to
                ``<vault>/archive/`` before deletion.  Defaults to
                False.

        Returns:
            list: List of ``designated_id`` values selected for (or,
            actually, deleted in) the purge.
        """
        pd = _require_pandas()
        import shutil

        self._ensure_designated_catalog()
        dcat = self._designated_catalog
        if dcat is None or len(dcat) == 0:
            print("[purge_retracted] No designated catalog entries.")
            return []

        if "retracted" not in dcat.columns:
            print("[purge_retracted] Catalog has no 'retracted' column — nothing to purge.")
            return []

        mask = dcat["retracted"].fillna(False).astype(bool)
        if older_than is not None and "retracted_at" in dcat.columns:
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(older_than)
            ts = pd.to_datetime(dcat["retracted_at"], errors="coerce", utc=True)
            mask = mask & (ts < cutoff)

        selected = dcat[mask]
        if len(selected) == 0:
            print("[purge_retracted] No retracted entries match the criteria.")
            return []

        ids = [int(x) for x in selected["designated_id"].tolist()]
        print(f"[purge_retracted] {len(selected)} entries selected for purge:")
        for _, row in selected.iterrows():
            print(f"  id={row.get('designated_id')} label={row.get('label')!r} "
                  f"retracted_at={row.get('retracted_at')}")

        if dry_run:
            print("[purge_retracted] dry_run=True — no files touched. "
                  "Pass dry_run=False, confirm=True to actually delete.")
            return ids
        if not confirm:
            raise RuntimeError(
                "purge_retracted refuses to run with dry_run=False "
                "unless confirm=True is passed explicitly."
            )

        # Each shard may contain multiple designated rows; surgically
        # remove just the purged rows and rewrite (or delete) each shard.
        vault = _resolve_vault_dir()
        archive_dir = vault / "archive"
        if archive:
            archive_dir.mkdir(parents=True, exist_ok=True)

        shard_to_ids: Dict[Path, List[int]] = {}
        for _, row in selected.iterrows():
            try:
                shard_dir = self._resolve_vacua_dir(
                    h11       = row.get("h11"),
                    h12       = row.get("h12"),
                    ks_id     = row.get("ks_id", -1),
                    triang_id = row.get("triang_id", -1),
                    cicy_id   = row.get("cicy_id", -1),
                    model_name= row.get("model_name"),
                )
            except Exception as e:
                print(f"[purge_retracted] could not resolve shard for id={row['designated_id']}: {e}")
                continue
            shard_path = shard_dir / "shard_0.parquet"
            shard_to_ids.setdefault(shard_path, []).append(int(row["designated_id"]))

        deleted = []
        for shard_path, ids in shard_to_ids.items():
            if not shard_path.exists():
                continue
            if archive:
                rel = f"{shard_path.parent.name}__{shard_path.name}"
                shutil.copy2(shard_path, archive_dir / rel)
            shard_df = pd.read_parquet(shard_path)
            keep = ~shard_df["designated_id"].isin(ids)
            shard_df = shard_df[keep]
            if len(shard_df) > 0:
                shard_df.to_parquet(shard_path, index=False)
            else:
                shard_path.unlink()
            deleted.extend(ids)

        # Drop purged rows from catalog
        dcat = dcat[~mask].reset_index(drop=True)
        catalog_path = self._designated_dir / "designated_vacua_catalog.parquet"
        dcat.to_parquet(catalog_path, index=False)
        self._designated_catalog = dcat

        print(f"[purge_retracted] Purged {len(deleted)} entry(ies) across "
              f"{len(shard_to_ids)} shard(s). Audit trail remains in "
              f"{vault}/retractions.parquet")
        return deleted


# ---------------------------------------------------------------------------
# Convenience sub-classes
# ---------------------------------------------------------------------------

class TDFDatabase(CYDatabase):
    r"""
    **Description:**
    Convenience subclass of :class:`CYDatabase` pre-configured for the
    ``"tdf"`` sub-dataset (toric models from the Kreuzer-Skarke list).

    Models are identified by ``(ks_id, triang_id)``.

    Example usage::

        db = TDFDatabase()
        tree = db.load(ks_id=12345, triang_id=0)
    """

    def __init__(
        self,
        hf_repo: str = HF_REPO_ID,
        cache_dir: Optional[str] = None,
        offline: bool = False,
        cache_mode: str = "persistent",
    ) -> None:
        r"""
        **Description:**
        Initialise a :class:`TDFDatabase` instance.

        Args:
            hf_repo (str): HuggingFace repository identifier.
                Defaults to :data:`HF_REPO_ID`.
            cache_dir (str | None): Local data directory.  If ``None``,
                uses ``jaxvacua.data_dir``.
            offline (bool): Suppress all network access.  Defaults to ``False``.
            cache_mode (str): ``"persistent"`` or ``"none"``.
                See :class:`CYDatabase` for details.
        """
        super().__init__(
            dataset="tdf",
            hf_repo=hf_repo,
            cache_dir=cache_dir,
            offline=offline,
            cache_mode=cache_mode,
        )


class CICYDatabase(CYDatabase):
    r"""
    **Description:**
    Convenience subclass of :class:`CYDatabase` pre-configured for the
    ``"cicy"`` sub-dataset (Complete Intersection Calabi-Yau models).

    Models are identified by ``cicy_id``.

    Example usage::

        db = CICYDatabase()
        tree = db.load(cicy_id=7890)
    """

    def __init__(
        self,
        hf_repo: str = HF_REPO_ID,
        cache_dir: Optional[str] = None,
        offline: bool = False,
        cache_mode: str = "persistent",
    ) -> None:
        r"""
        **Description:**
        Initialise a :class:`CICYDatabase` instance.

        Args:
            hf_repo (str): HuggingFace repository identifier.
                Defaults to :data:`HF_REPO_ID`.
            cache_dir (str | None): Local data directory.  If ``None``,
                uses ``jaxvacua.data_dir``.
            offline (bool): Suppress all network access.  Defaults to ``False``.
            cache_mode (str): ``"persistent"`` or ``"none"``.
                See :class:`CYDatabase` for details.
        """
        super().__init__(
            dataset="cicy",
            hf_repo=hf_repo,
            cache_dir=cache_dir,
            offline=offline,
            cache_mode=cache_mode,
        )


# ---------------------------------------------------------------------------
# Vacua storage — helpers
# ---------------------------------------------------------------------------

def _compute_model_hash(tree: Any) -> str:
    r"""
    **Description:**
    Compute a deterministic SHA-256 hash of the geometric fingerprint of an
    ``lcs_tree``.  The fingerprint consists of ``(h11, h12, chi, intnums_coo,
    c2)`` — the minimal set of invariants that uniquely identify the geometry.

    Args:
        tree: An ``lcs_tree`` instance (or any object exposing ``h11``,
            ``h12``, ``chi``, ``intnums_coo``, ``c2`` attributes).

    Returns:
        str: Hex-encoded SHA-256 hash string.
    """
    h = hashlib.sha256()
    h.update(f"h11={tree.h11}".encode())
    h.update(f"h12={tree.h12}".encode())
    h.update(f"chi={tree.chi}".encode())
    if tree.intnums_coo is not None:
        h.update(np.asarray(tree.intnums_coo).tobytes())
    if tree.c2 is not None:
        h.update(np.asarray(tree.c2).tobytes())
    return h.hexdigest()


def _extract_model_identity(
    model: Any = None,
    h11: Optional[int] = None,
    h12: Optional[int] = None,
    ks_id: Optional[int] = None,
    triang_id: Optional[int] = None,
    conifold_id: int = -1,
    cicy_id: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    r"""
    **Description:**
    Extract or assemble model identity fields.  If *model* is provided
    (a ``flux_sector`` or ``lcs_tree``), identity is read from the tree;
    explicit keyword arguments override.

    Returns:
        Dict with keys ``h11``, ``h12``, ``ks_id``, ``triang_id``,
        ``conifold_id``, ``cicy_id``, ``model_hash``, ``model_name``.
    """
    tree = None
    if model is not None:
        # Accept either flux_sector (has .periods.lcs_tree) or lcs_tree
        tree = getattr(getattr(model, "periods", None), "lcs_tree", model)

    _h11 = h11 if h11 is not None else (getattr(tree, "h11", None) if tree else None)
    _h12 = h12 if h12 is not None else (getattr(tree, "h12", None) if tree else None)

    # Try to get ks_id/triang_id from tree.extra_data
    _extra = getattr(tree, "extra_data", None) or {}
    _ks  = ks_id if ks_id is not None else _extra.get("ks_id", -1)
    _tr  = triang_id if triang_id is not None else _extra.get("triang_id", -1)
    _cid = cicy_id if cicy_id is not None else _extra.get("cicy_id", -1)

    # conifold_id: check tree.limit and conifold metadata
    _cf = conifold_id
    if _cf == -1 and tree is not None:
        limit = getattr(tree, "limit", None)
        if limit is not None and "coni" in str(limit).lower():
            _cf = _extra.get("conifold_id", 0)

    _hash = _compute_model_hash(tree) if tree is not None else ""
    _name = model_name

    # For local models loaded via model_ID, auto-generate a model_name
    # so they are human-readable in queries.
    if _name is None and _ks == -1 and tree is not None:
        _mid = getattr(tree, "model_ID", None)
        if _mid is not None:
            _name = f"local_h12_{_h12}_ID_{_mid}"

    if _h11 is None or _h12 is None:
        raise ValueError(
            "Cannot determine h11/h12. Pass them explicitly or provide a model."
        )

    return {
        "h11": int(_h11),
        "h12": int(_h12),
        "ks_id": int(_ks),
        "triang_id": int(_tr),
        "conifold_id": int(_cf),
        "cicy_id": int(_cid),
        "model_hash": _hash,
        "model_name": _name,
    }


def _vacua_row_from_dict(
    result: dict,
    vacuum_id: int,
    tags: Optional[Any] = None,
    extra_data: Optional[dict] = None,
) -> dict:
    r"""
    Normalise a single result dict (as returned by ``enumerate_fluxes`` or
    ``sample_bounded_fluxes``) into a flat row matching the vacua parquet
    schema.

    Args:
        result: Dict with keys ``"flux"``, ``"moduli"``, ``"tau"``, and
            optionally ``"residual"``, ``"is_susy"``, ``"N_flux"``, ``"W"``,
            ``"F_terms"``.
        vacuum_id: 0-based index within the run.
        tags: String or list of strings.
        extra_data: Optional dict of arbitrary metadata.

    Returns:
        dict: Flat row dict ready for DataFrame construction.
    """
    flux = np.asarray(result["flux"], dtype=np.int32)
    moduli = np.asarray(result["moduli"], dtype=np.complex128)
    tau = complex(result["tau"])

    # Normalise tags
    if tags is None:
        tags_json = "[]"
    elif isinstance(tags, str):
        tags_json = json.dumps([tags])
    else:
        tags_json = json.dumps(list(tags))

    row = {
        "vacuum_id": vacuum_id,
        "flux":      flux.tolist(),
        "moduli_re": moduli.real.tolist(),
        "moduli_im": moduli.imag.tolist(),
        "tau_re":    tau.real,
        "tau_im":    tau.imag,
        "W_re":      None,
        "W_im":      None,
        "F_terms_re": None,
        "F_terms_im": None,
        "N_flux":    result.get("N_flux"),
        "residual":  result.get("residual"),
        "is_susy":   result.get("is_susy", False),
        "tags":      tags_json,
        "extra_data": json.dumps(extra_data) if extra_data else None,
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
    }

    # Optional pre-computed derived quantities
    W = result.get("W")
    if W is not None:
        W = complex(W)
        row["W_re"] = W.real
        row["W_im"] = W.imag

    F = result.get("F_terms")
    if F is not None:
        F = np.asarray(F, dtype=np.complex128)
        row["F_terms_re"] = F.real.tolist()
        row["F_terms_im"] = F.imag.tolist()

    return row


def _vacua_rows_from_batch(
    moduli: Any,
    tau: Any,
    fluxes: Any,
    tags: Optional[Any] = None,
    id_offset: int = 0,
    **extra_fields: Any,
) -> List[dict]:
    r"""
    **Description:**
    Convert batched arrays (as returned by ``initial_guesses_ISD``) into a
    list of row dicts matching the vacua parquet schema.

    Args:
        moduli: Array shape ``(N, h12)``, complex.
        tau: Array shape ``(N,)``, complex.
        fluxes: Array shape ``(N, 2*n_fluxes)``, int.
        tags: String or list of strings applied to all rows.
        id_offset: Starting vacuum_id.
        **extra_fields: Additional per-vacuum arrays of length N (e.g.
            ``residual``, ``N_flux``).

    Returns:
        List[dict]: One row dict per vacuum.
    """
    moduli_np = np.asarray(moduli, dtype=np.complex128)
    tau_np    = np.asarray(tau, dtype=np.complex128)
    fluxes_np = np.asarray(fluxes, dtype=np.int32)
    N = len(fluxes_np)

    # Normalise tags once
    if tags is None:
        tags_json = "[]"
    elif isinstance(tags, str):
        tags_json = json.dumps([tags])
    else:
        tags_json = json.dumps(list(tags))

    rows = []
    for i in range(N):
        row = {
            "vacuum_id": id_offset + i,
            "flux":      fluxes_np[i].tolist(),
            "moduli_re": moduli_np[i].real.tolist(),
            "moduli_im": moduli_np[i].imag.tolist(),
            "tau_re":    float(tau_np[i].real),
            "tau_im":    float(tau_np[i].imag),
            "W_re":      None,
            "W_im":      None,
            "F_terms_re": None,
            "F_terms_im": None,
            "N_flux":    float(extra_fields["N_flux"][i]) if "N_flux" in extra_fields else None,
            "residual":  float(extra_fields["residual"][i]) if "residual" in extra_fields else None,
            "is_susy":   bool(extra_fields["is_susy"][i]) if "is_susy" in extra_fields else False,
            "tags":      tags_json,
            "extra_data": None,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# VacuaWriter
# ---------------------------------------------------------------------------

class _VacuaWriter:
    r"""
    Context manager for buffered writing of vacuum solutions to local parquet
    storage.  Intended to be created via :meth:`CYDatabase.vacua_writer`.

    Usage::

        with db.vacua_writer(model=my_model) as writer:
            for result in results:
                writer.append(result, tags=["PFV"])
        # flushes to disk on __exit__

    Args:
        vacua_dir (Path): Root directory for vacua storage.
        identity (dict): Model identity fields (h11, h12, ks_id, ...).
        run_id (str): UUID of this run.
        method (str): Sampling method label.
        metadata (dict | None): Arbitrary run-level metadata (JSON-serialisable).
        model: Optional ``flux_sector`` for derived-quantity computation.
        compute_derived (bool): If True and *model* is set, compute W, DW,
            N_flux at flush time.
        filter_fn: Optional callable ``result_dict → bool``.
        deduplicate (bool): If True, skip flux vectors already stored for
            this model.
        map_to_fd (bool): If True and *model* has ``map_to_FD``, map each
            vacuum to the fundamental domain before storing and deduplicating.
    """

    def __init__(
        self,
        vacua_dir: Path,
        identity: Dict[str, Any],
        run_id: str,
        method: str = "manual",
        metadata: Optional[dict] = None,
        model: Any = None,
        compute_derived: bool = False,
        filter_fn: Any = None,
        deduplicate: bool = True,
        map_to_fd: bool = False,
    ) -> None:
        """Initialise the vacua writer for a given model identity and run."""
        self._vacua_dir = vacua_dir
        self._identity  = identity
        self._run_id    = run_id
        self._method    = method
        self._metadata  = metadata
        self._model     = model
        self._compute_derived = compute_derived
        self._filter_fn = filter_fn
        self._deduplicate = deduplicate
        self._map_to_fd = map_to_fd and model is not None and hasattr(model, 'map_to_FD')

        self._buffer: List[dict] = []
        self._flushed_count: int = 0
        self._n_duplicates_skipped: int = 0
        self._seen_fluxes: set = set()

        # Determine output path
        ks = identity["ks_id"]
        tr = identity["triang_id"]
        if ks >= 0 and tr >= 0:
            self._run_dir = (
                vacua_dir / f"ks_{ks}" / f"triang_{tr}"
            )
        else:
            mhash = identity["model_hash"]
            self._run_dir = vacua_dir / "custom" / mhash
        self._run_path = self._run_dir / f"run_{self._run_id}.parquet"

    def __enter__(self) -> "_VacuaWriter":
        """Enter context manager, loading existing fluxes if deduplicating."""
        if self._deduplicate:
            self._load_existing_fluxes()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, flushing buffered rows and updating the catalog."""
        if self._buffer:
            self.flush()
        # Update catalog regardless (even if 0 new vacua, to record the run)
        if self._flushed_count > 0:
            self._update_catalog()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        result: Any,
        tags: Optional[Any] = None,
        extra_data: Optional[dict] = None,
    ) -> None:
        r"""
        **Description:**
    Append a single vacuum solution.

        Args:
            result: Either a dict ``{"flux", "moduli", "tau", ...}`` (as
                returned by ``enumerate_fluxes``) or a tuple
                ``(moduli, tau, flux)``.
            tags: String or list of tag strings.
            extra_data: Optional dict of arbitrary per-vacuum metadata.
        """
        if isinstance(result, (tuple, list)) and not isinstance(result, dict):
            moduli, tau, flux = result
            result = {"moduli": moduli, "tau": tau, "flux": flux}

        # Map to fundamental domain (monodromy + SL(2,Z))
        if self._map_to_fd:
            import jax.numpy as jnp
            moduli_fd, tau_fd, fluxes_fd = self._model.map_to_FD(
                jnp.asarray(result["moduli"]),
                complex(result["tau"]),
                jnp.asarray(result["flux"]),
            )
            result = {**result, "moduli": moduli_fd, "tau": tau_fd, "flux": fluxes_fd}

        # Filter
        if self._filter_fn is not None and not self._filter_fn(result):
            return

        # Dedup
        if self._deduplicate:
            flux_key = tuple(int(x) for x in np.asarray(result["flux"]))
            if flux_key in self._seen_fluxes:
                self._n_duplicates_skipped += 1
                return
            self._seen_fluxes.add(flux_key)

        vid = self._flushed_count + len(self._buffer)
        self._buffer.append(_vacua_row_from_dict(result, vid, tags, extra_data))

    def append_batch(
        self,
        moduli: Any,
        tau: Any,
        fluxes: Any,
        tags: Optional[Any] = None,
        **extra_fields: Any,
    ) -> None:
        r"""
        **Description:**
    Append a batch of vacuum solutions from array data.

        Args:
            moduli: Array shape ``(N, h12)``, complex.
            tau: Array shape ``(N,)``, complex.
            fluxes: Array shape ``(N, 2*n_fluxes)``, int.
            tags: String or list applied to all rows.
            **extra_fields: Per-vacuum arrays of length N (``residual``,
                ``N_flux``, ``is_susy``).
        """
        fluxes_np = np.asarray(fluxes, dtype=np.int32)
        moduli_np = np.asarray(moduli, dtype=np.complex128)
        tau_np = np.asarray(tau, dtype=np.complex128)
        N = len(fluxes_np)

        # Map to fundamental domain (monodromy + SL(2,Z))
        if self._map_to_fd:
            import jax.numpy as jnp
            for i in range(N):
                m_fd, t_fd, f_fd = self._model.map_to_FD(
                    jnp.asarray(moduli_np[i]),
                    complex(tau_np[i]),
                    jnp.asarray(fluxes_np[i]),
                )
                moduli_np[i] = np.asarray(m_fd)
                tau_np[i] = complex(t_fd)
                fluxes_np[i] = np.asarray(f_fd, dtype=np.int32)

        # Build keep mask for dedup + filter
        keep = np.ones(N, dtype=bool)
        if self._deduplicate:
            for i in range(N):
                flux_key = tuple(int(x) for x in fluxes_np[i])
                if flux_key in self._seen_fluxes:
                    keep[i] = False
                    self._n_duplicates_skipped += 1
                else:
                    self._seen_fluxes.add(flux_key)

        if self._filter_fn is not None:
            for i in range(N):
                if keep[i]:
                    r = {"flux": fluxes_np[i], "moduli": moduli_np[i], "tau": tau_np[i]}
                    if not self._filter_fn(r):
                        keep[i] = False

        # Subset arrays
        idx = np.where(keep)[0]
        if len(idx) == 0:
            return

        sub_moduli = moduli_np[idx]
        sub_tau    = tau_np[idx]
        sub_fluxes = fluxes_np[idx]
        sub_extra  = {
            k: np.asarray(v)[idx]
            for k, v in extra_fields.items()
        }

        vid_offset = self._flushed_count + len(self._buffer)
        rows = _vacua_rows_from_batch(
            sub_moduli, sub_tau, sub_fluxes,
            tags=tags, id_offset=vid_offset, **sub_extra,
        )
        self._buffer.extend(rows)

    def flush(self) -> None:
        r"""Write buffered rows to the run parquet file."""
        if not self._buffer:
            return

        pd = _require_pandas()
        df = pd.DataFrame(self._buffer)

        # Optional derived-quantity computation
        if self._compute_derived and self._model is not None:
            df = self._compute_derived_quantities(df)

        # Write (append to existing file if present from prior flush)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        if self._run_path.exists():
            existing = pd.read_parquet(self._run_path)
            df = _safe_concat([existing, df], ignore_index=True)
        df.to_parquet(self._run_path, index=False)

        self._flushed_count += len(self._buffer)
        self._buffer.clear()

    @property
    def count(self) -> int:
        r"""Total number of vacua accepted (buffered + flushed)."""
        return self._flushed_count + len(self._buffer)

    @property
    def n_duplicates_skipped(self) -> int:
        r"""Number of duplicate flux vectors skipped."""
        return self._n_duplicates_skipped

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_existing_fluxes(self) -> None:
        r"""Load all flux vectors from prior runs for this model into the
        dedup set."""
        pd = _require_pandas()
        if not self._run_dir.exists():
            return
        for f in self._run_dir.glob("run_*.parquet"):
            try:
                df = pd.read_parquet(f, columns=["flux"])
                for flux_list in df["flux"]:
                    self._seen_fluxes.add(tuple(int(x) for x in flux_list))
            except Exception:
                pass  # skip corrupt files

    def _update_catalog(self) -> None:
        r"""Append this run to the vacua catalog."""
        pd = _require_pandas()
        catalog_path = self._vacua_dir / "vacua_catalog.parquet"

        new_row = {
            "run_id":       self._run_id,
            "h11":          self._identity["h11"],
            "h12":          self._identity["h12"],
            "ks_id":        self._identity["ks_id"],
            "triang_id":    self._identity["triang_id"],
            "conifold_id":  self._identity["conifold_id"],
            "cicy_id":      self._identity["cicy_id"],
            "model_hash":   self._identity["model_hash"],
            "model_name":   self._identity["model_name"],
            "n_vacua":      self._flushed_count,
            "method":       self._method,
            "created":      datetime.datetime.now(datetime.timezone.utc),
            "metadata":     json.dumps(self._metadata) if self._metadata else None,
            "file_path":    str(self._run_path.relative_to(self._vacua_dir)),
        }

        self._vacua_dir.mkdir(parents=True, exist_ok=True)
        if catalog_path.exists():
            existing = pd.read_parquet(catalog_path)
            catalog = _safe_concat([existing, pd.DataFrame([new_row])], ignore_index=True)
        else:
            catalog = pd.DataFrame([new_row])
        catalog.to_parquet(catalog_path, index=False)

    def _compute_derived_quantities(self, df: Any) -> Any:
        r"""Compute W, DW, N_flux for all rows in df using self._model."""
        model = self._model
        rows_updated = []
        for _, row in df.iterrows():
            row = dict(row)
            flux = np.array(row["flux"], dtype=np.int32)
            moduli = np.array(row["moduli_re"]) + 1j * np.array(row["moduli_im"])
            tau = row["tau_re"] + 1j * row["tau_im"]

            try:
                # Tadpole
                if row.get("N_flux") is None:
                    row["N_flux"] = float(model.tadpole(flux))

                # Superpotential
                W = complex(model.superpotential(moduli, tau, flux))
                row["W_re"] = W.real
                row["W_im"] = W.imag

                # F-terms
                moduli_c = np.conj(moduli)
                tau_c = np.conj(tau)
                DW = np.asarray(model.DW(moduli, moduli_c, tau, tau_c, flux),
                                dtype=np.complex128)
                row["F_terms_re"] = DW.real.tolist()
                row["F_terms_im"] = DW.imag.tolist()

                # is_susy: check if all F-terms are small
                if not row.get("is_susy"):
                    row["is_susy"] = bool(np.max(np.abs(DW)) < 1e-6)
            except Exception:
                pass  # leave derived fields as None if computation fails

            rows_updated.append(row)

        pd = _require_pandas()
        return pd.DataFrame(rows_updated)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def load_tdf_model(
    ks_id: int,
    triang_id: int,
    include_gv: bool = False,
    include_conifolds: Union[bool, str] = False,
    maximum_degree: Optional[int] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Any:
    r"""
    **Description:**
    Load a single toric Kreuzer-Skarke model as an
    :class:`~jaxvacua.lcs.lcs_tree` without instantiating a
    :class:`CYDatabase` explicitly.

    Args:
        ks_id (int): Kreuzer-Skarke polytope index.
        triang_id (int): Triangulation index.
        include_gv (bool): Attach GV invariants.  Defaults to ``False``.
        include_conifolds (bool | str): Attach conifold data.  Defaults to
            ``False``.
        maximum_degree (int | None): GV truncation degree.
        cache_dir (str): Local cache directory.

    Returns:
        lcs_tree: The requested model.
    """
    db = TDFDatabase(cache_dir=cache_dir)
    return db.load(
        ks_id=ks_id,
        triang_id=triang_id,
        include_gv=include_gv,
        include_conifolds=include_conifolds,
        maximum_degree=maximum_degree,
    )


def load_cicy_model(
    cicy_id: int,
    include_gv: bool = False,
    maximum_degree: Optional[int] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Any:
    r"""
    **Description:**
    Load a single CICY model as an :class:`~jaxvacua.lcs.lcs_tree` without
    instantiating a :class:`CYDatabase` explicitly.

    Args:
        cicy_id (int): CICY model index.
        include_gv (bool): Attach GV invariants.  Defaults to ``False``.
        maximum_degree (int | None): GV truncation degree.
        cache_dir (str): Local cache directory.

    Returns:
        lcs_tree: The requested model.
    """
    db = CICYDatabase(cache_dir=cache_dir)
    return db.load(
        cicy_id=cicy_id,
        include_gv=include_gv,
        maximum_degree=maximum_degree,
    )


def query_models(
    dataset: str = "tdf",
    h11: Optional[int] = None,
    h12: Optional[int] = None,
    has_conifolds: Optional[bool] = None,
    n_conifolds_min: Optional[int] = None,
    n_conifolds_max: Optional[int] = None,
    has_gv: Optional[bool] = None,
    D3_tadpole_max: Optional[int] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    **filters: Any,
) -> Any:
    r"""
    **Description:**
    Query the catalog of a sub-dataset and return matching rows as a
    ``pandas.DataFrame``, without instantiating a :class:`CYDatabase`
    explicitly.

    Args:
        dataset (str): Sub-dataset identifier (``"tdf"`` or ``"cicy"``).
        h11 (int | None): Filter by :math:`h^{1,1}`.
        h12 (int | None): Filter by :math:`h^{1,2}`.
        has_conifolds (bool | None): Filter by conifold availability.
        n_conifolds_min (int | None): Minimum number of conifold limits.
        n_conifolds_max (int | None): Maximum number of conifold limits.
        has_gv (bool | None): Filter by GV availability.
        D3_tadpole_max (int | None): Upper bound on the D3-tadpole.
        cache_dir (str): Local cache directory.
        **filters: Additional exact-match column filters.

    Returns:
        pandas.DataFrame: Matching rows from the catalog.
    """
    db = CYDatabase(dataset=dataset, cache_dir=cache_dir)
    return db.query(
        h11=h11, h12=h12,
        has_conifolds=has_conifolds,
        n_conifolds_min=n_conifolds_min,
        n_conifolds_max=n_conifolds_max,
        has_gv=has_gv,
        D3_tadpole_max=D3_tadpole_max,
        **filters,
    )


def load_catalog(
    path: Union[str, "Path", None] = None,
    dataset: str = "tdf",
    catalog: str = "catalog",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Any:
    r"""
    **Description:**
    Load a catalog as a ``pandas.DataFrame`` from either a local database
    directory or the default HuggingFace cache.

    This is a lightweight entry point — it reads only the catalog parquet
    file, not any geometry data.

    Args:
        path (str | Path | None): Path to a local database.  Can point to
            the root (containing the ``tdf/`` or ``cicy/`` sub-folder) or
            directly to the sub-dataset folder.  If ``None``, the catalog
            is loaded from (or downloaded to) the default HuggingFace cache.
        dataset (str): Sub-dataset identifier (``"tdf"`` or ``"cicy"``).
            Defaults to ``"tdf"``.
        catalog (str): Which catalog to load.  One of:

            - ``"catalog"`` — main model catalog (default)
            - ``"conifold_catalog"`` — conifold sub-catalog
            - ``"designated_vacua_catalog"`` — designated (Tier 2) solutions

        cache_dir (str): Cache directory when ``path`` is ``None``.

    Returns:
        pandas.DataFrame: The requested catalog.

    Example::

        # From a local build
        cat = load_catalog("/data/mydb")
        cf  = load_catalog("/data/mydb", catalog="conifold_catalog")

        # From the online / cached HuggingFace repo
        cat = load_catalog()
    """
    if path is not None:
        db = CYDatabase.from_local(path, dataset=dataset)
    else:
        db = CYDatabase(dataset=dataset, cache_dir=cache_dir)

    if catalog == "catalog":
        db._ensure_catalog()
        return db._catalog
    elif catalog == "conifold_catalog":
        db._ensure_conifold_catalog()
        return db._conifold_catalog
    elif catalog == "designated_vacua_catalog":
        db._ensure_designated_catalog()
        return db._designated_catalog
    else:
        raise ValueError(
            f"Unknown catalog '{catalog}'.  Choose one of: "
            "'catalog', 'conifold_catalog', 'designated_vacua_catalog'."
        )
