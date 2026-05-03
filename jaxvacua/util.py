# ==============================================================================
# This code is written by Andreas Schachner.
#
# If any questions arise, please feel free to reach out to me (Andreas) either at
# andreas.schachner@gmx.net or at as3475@cornell.edu .
# ==============================================================================
"""``jaxvacua.util`` — general-purpose utilities used across the package.

Contents are grouped into thematic sections separated by banners; see the
section headings below for the layout.  The companion Sphinx page is
``documentation/source/jaxvacua.util.rst``.
"""

# ==============================================================================
# Imports
# ==============================================================================

# --- Standard library ---------------------------------------------------------
import os
import sys
import gzip
import pickle
import threading
import itertools
from functools import partial, lru_cache
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

# Standard library `_thread` is exposed under the legacy name `thread` on
# Python < 3 — keep both branches in case of a stripped environment.
try:
    import thread
except ImportError:
    import _thread as thread

# --- Third-party --------------------------------------------------------------
import numpy as np
from flint import fmpz_mat                  # exact integer LLL for ``orthogonal_lattice``

# --- JAX ----------------------------------------------------------------------
import jax
import jax.numpy as jnp
from jax import jit, Array
from jax.typing import ArrayLike

# --- Module-level constants ---------------------------------------------------
DTYPE = jax.dtypes.canonicalize_dtype(float)


# ==============================================================================
# 1. PRNG / random sampling
# ==============================================================================

class PRNGSequence:
    r"""
    **Description:**
    Splittable JAX PRNG key generator.  Adopted from
    `CYJax <https://github.com/ml4physics/cyjax>`_.

    Each call to ``next(rns)`` splits the internal key and returns one of the
    halves, leaving the other as the new internal state.  Use this whenever a
    function needs a fresh subkey — guarantees deterministic, non-overlapping
    streams of random numbers across the whole process lifetime.
    """

    _key: Array = None

    def __init__(self, seed: Union[int, Array] = 42) -> None:
        r"""
        Args:
            seed (int | Array, optional): Either a Python ``int`` (used as a
                seed for ``jax.random.PRNGKey``) or an existing JAX PRNG key
                that becomes the initial internal state.  Defaults to ``42``.

        Example:
            ```python
            rns = PRNGSequence(42)
            key = next(rns)
            ```
        """
        if isinstance(seed, int):
            self._key = jax.random.PRNGKey(seed)
        else:
            self._key = seed

    def __next__(self) -> Array:
        r"""
        **Description:**
        Returns the next subkey in the sequence (and advances the internal
        state).

        Returns:
            Array: A fresh ``jax.random.PRNGKey``-shaped subkey.
        """
        k, self._key = jax.random.split(self._key)
        return k


def random_uniform(
    lower_bound: float,
    upper_bound: float,
    rns_key: Optional[Union["PRNGSequence", Array]] = None,
    seed: int = 42,
    shape: Tuple[int, ...] = (1,),
) -> Array:
    r"""
    **Description:**
    Sample uniformly distributed real numbers on ``[lower_bound, upper_bound)``.

    Args:
        lower_bound (float): Lower edge of the sampling interval.
        upper_bound (float): Upper edge of the sampling interval (exclusive).
        rns_key (PRNGSequence | Array, optional):  Key source.  If ``None``,
            a fresh ``PRNGSequence(seed)`` is created.  If a
            :class:`PRNGSequence`, ``next()`` is called to obtain a subkey.
            If a raw JAX PRNG key, it's used directly.
        seed (int, optional): Seed used when ``rns_key is None``.  Default ``42``.
        shape (tuple[int, ...], optional): Output shape.  Default ``(1,)``.

    Returns:
        Array: Uniform samples of shape ``shape`` and dtype ``DTYPE``.
    """
    if rns_key is None:
        key = next(PRNGSequence(seed))
    elif isinstance(rns_key, PRNGSequence):
        key = next(rns_key)
    else:
        key = rns_key                                           # raw JAX PRNGKey

    return jax.random.uniform(key, shape=shape, minval=lower_bound,
                              maxval=upper_bound, dtype=DTYPE)


def random_integer(
    lower_bound: int,
    upper_bound: int,
    rns_key: Optional[Union["PRNGSequence", Array]] = None,
    seed: int = 42,
    shape: Tuple[int, ...] = (1,),
) -> Array:
    r"""
    **Description:**
    Sample uniformly distributed integers on ``[lower_bound, upper_bound]``
    (inclusive on both ends, matching the convention of
    ``jax.random.randint(maxval=upper_bound + 1)``).

    Args:
        lower_bound (int):    Lower edge of the sampling interval.
        upper_bound (int):    Upper edge of the sampling interval (inclusive).
        rns_key (PRNGSequence | Array, optional): Key source — see
            :func:`random_uniform` for details.
        seed (int, optional): Seed used when ``rns_key is None``.  Default ``42``.
        shape (tuple[int, ...], optional): Output shape.  Default ``(1,)``.

    Returns:
        Array: Integer samples of shape ``shape``.
    """
    if rns_key is None:
        key = next(PRNGSequence(seed))
    elif isinstance(rns_key, PRNGSequence):
        key = next(rns_key)
    else:
        key = rns_key

    return jax.random.randint(key, shape=shape, minval=lower_bound,
                              maxval=upper_bound + 1)


@partial(jit, static_argnums=(3,))
def random_uniform_jit(
    rns_key: Array,
    lower_bound: float,
    upper_bound: float,
    shape: Tuple[int, ...] = (1,),
) -> Array:
    r"""
    **Description:**
    JIT-compiled version of :func:`random_uniform`.  ``rns_key`` is a
    JAX PRNG-key array; ``shape`` is treated as a static argument so the
    output shape is fixed at trace time.

    Args:
        rns_key (Array):     JAX PRNG-key array.
        lower_bound (float): Lower edge of the sampling interval.
        upper_bound (float): Upper edge of the sampling interval (exclusive).
        shape (tuple[int, ...], optional): Output shape.  Default ``(1,)``.

    Returns:
        Array: Uniform samples of shape ``shape`` and dtype ``DTYPE``.
    """
    return jax.random.uniform(rns_key, shape=shape, minval=lower_bound,
                              maxval=upper_bound, dtype=DTYPE)


@partial(jit, static_argnums=(3,))
def random_integer_jit(
    rns_key: Array,
    lower_bound: int,
    upper_bound: int,
    shape: Tuple[int, ...] = (1,),
) -> Array:
    r"""
    **Description:**
    JIT-compiled version of :func:`random_integer`.

    Args:
        rns_key (Array):    JAX PRNG-key array.
        lower_bound (int):  Lower edge of the sampling interval.
        upper_bound (int):  Upper edge of the sampling interval (inclusive).
        shape (tuple[int, ...], optional): Output shape.  Default ``(1,)``.

    Returns:
        Array: Integer samples of shape ``shape``.
    """
    return jax.random.randint(rns_key, shape=shape, minval=lower_bound,
                              maxval=upper_bound + 1)


# ==============================================================================
# 2. JIT / vmap helpers
# ==============================================================================

def vmapping_func(
    func: Callable,
    in_axes: Optional[Union[int, Tuple]] = None,
    **kwargs: Any,
) -> Callable:
    r"""
    **Description:**
    Build a JIT-compiled, vmapped wrapper around ``func`` with optional
    keyword arguments frozen inside the closure.

    .. note::
        Each call constructs a *fresh* ``jax.jit(jax.vmap(...))`` object;
        this defeats JAX's compilation cache and forces XLA recompilation.
        Use :func:`vmapping_func_cached` if you call the same combination
        repeatedly.

    Args:
        func (Callable): Function to be vmapped.
        in_axes (int | tuple, optional): Forwarded to ``jax.vmap``.
        **kwargs:        Keyword arguments to be bound inside the closure.

    Returns:
        Callable: JIT-compiled vmapped function.
    """
    def _tmp(*args):
        """Closure forwarding kwargs to func for vmap."""
        return func(*args, **kwargs)

    return jax.jit(jax.vmap(_tmp, in_axes=in_axes))


@lru_cache(maxsize=256)
def _build_vmap_jit(
    func: Callable,
    in_axes: Optional[Union[int, Tuple]],
    frozen_kwargs: Tuple[Tuple[str, Any], ...],
) -> Callable:
    r"""
    **Description:**
    Module-level LRU-cached factory that builds and stores a
    ``jax.jit(jax.vmap(...))`` kernel.  Called by
    :func:`vmapping_func_cached`; not intended for direct use.

    The cache key is ``(func, in_axes, frozen_kwargs)``.  Python bound
    methods are hashable (keyed on ``(method.__func__, method.__self__)``),
    so the same method on the same instance always resolves to the same
    cache entry.

    Args:
        func (Callable):           Hashable callable to be vmapped.
        in_axes (int | tuple | None): Batch-axis specification forwarded to
            ``jax.vmap``.
        frozen_kwargs (tuple):     Sorted tuple of ``(key, value)`` pairs
            representing keyword arguments to be bound inside the closure.
            All values must be hashable.

    Returns:
        Callable: JIT-compiled vmapped function.
    """
    kwargs = dict(frozen_kwargs)

    def _tmp(*args):
        """Closure forwarding frozen kwargs to func for vmap."""
        return func(*args, **kwargs)

    return jax.jit(jax.vmap(_tmp, in_axes=in_axes))


def vmapping_func_cached(
    func: Callable,
    in_axes: Optional[Union[int, Tuple]] = None,
    **kwargs: Any,
) -> Callable:
    r"""
    **Description:**
    Cached variant of :func:`vmapping_func`.  Returns a JIT-compiled vmapped
    function, **reusing the previously compiled XLA kernel** whenever
    ``(func, in_axes, kwargs)`` match a prior call.

    Args:
        func (Callable):     Function to be vmapped.  Must be hashable;
            Python bound methods satisfy this requirement.
        in_axes (int | tuple | None, optional): Forwarded to ``jax.vmap``.
            Must be hashable (tuples of ints or ``None``).  Default ``None``.
        **kwargs:            Keyword arguments to be bound inside the
            closure.  All values must be hashable.

    Returns:
        Callable: JIT-compiled vmapped function, reused from cache when
        possible.

    .. note::
        The backing LRU cache is module-level (capacity 256) and persists
        for the lifetime of the Python process.
    """
    frozen = tuple(sorted(kwargs.items()))
    return _build_vmap_jit(func, in_axes, frozen)


def jit_with_static_args(
    func: Callable,
    static_argnums: Tuple[int, ...] = (),
) -> Callable:
    r"""
    **Description:**
    Wrap ``func`` with ``jax.jit``, treating positional arguments at the
    indices in ``static_argnums`` as compile-time constants.

    Args:
        func (Callable):                The function to be JIT-compiled.
        static_argnums (tuple[int, ...]): Positions of arguments to mark as
            static.  Default ``()``.

    Returns:
        Callable: JIT-compiled wrapper around ``func``.

    Example:
        ```python
        def f(x, y, n): return x + y + n
        jit_f = jit_with_static_args(f, static_argnums=(2,))
        ```
    """
    @partial(jax.jit, static_argnums=static_argnums)
    def wrapped_func(*args, **kwargs):
        """JIT-compiled wrapper with static argument handling."""
        return func(*args, **kwargs)

    return wrapped_func


def is_static(arg: Any) -> bool:
    r"""
    **Description:**
    Heuristic test for whether ``arg`` should be treated as a JAX static
    argument: ``True`` iff ``arg`` is **not** a ``jnp.ndarray``.

    Args:
        arg (Any): The argument to test.

    Returns:
        bool: ``True`` if static, ``False`` if it should be traced.
    """
    return not isinstance(arg, jnp.ndarray)


def jit_with_dynamic_static_args(func: Callable) -> Callable:
    r"""
    **Description:**
    Build a wrapper that re-JITs ``func`` on each call, using
    :func:`is_static` to dynamically decide which positional arguments are
    static.  Convenient for prototyping; in production code prefer
    :func:`jit_with_static_args` so the trace cache hits.

    Args:
        func (Callable): The function to be JIT-compiled.

    Returns:
        Callable: Wrapper that rebuilds the JIT plan per call.
    """
    def wrapped_func(*args, **kwargs):
        """JIT-compiled wrapper with dynamically detected static arguments."""
        static_argnums = tuple(i for i, arg in enumerate(args) if is_static(arg))
        jit_func = jax.jit(func, static_argnums=static_argnums)
        return jit_func(*args, **kwargs)

    return wrapped_func


# ==============================================================================
# 3. Array / numerical helpers
# ==============================================================================

def subsets(
    iterable: Iterable,
    n: int,
    as_list: bool = True,
) -> Union[list, "itertools.combinations"]:
    r"""
    **Description:**
    All size-``n`` subsets of an iterable.  Thin wrapper around
    ``itertools.combinations`` with an optional list-eager flag.

    Args:
        iterable (Iterable): The source elements.
        n (int):             Size of each output subset.
        as_list (bool, optional): If ``True`` (default) materialise the result
            as a ``list``; otherwise return the ``itertools.combinations``
            object directly.

    Returns:
        list or itertools.combinations: All size-``n`` subsets.

    Example:
        ```python
        subsets(range(4), n=2)
        # [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        ```
    """
    if as_list:
        return list(itertools.combinations(iterable, n))
    return itertools.combinations(iterable, n)


def flatten(
    arr: Iterable,
    as_gen: bool = False,
    as_np_arr: bool = False,
) -> Union[list, np.ndarray, Iterable]:
    r"""
    **Description:**
    Recursively flatten an arbitrarily nested iterable of *any depth*.

    (Modified from `Stack Overflow #2158395
    <https://stackoverflow.com/q/2158395>`_.)

    Args:
        arr (Iterable):   The (possibly ragged, possibly nested) input.
        as_gen (bool, optional):    Return a generator instead of materialising.
        as_np_arr (bool, optional): Return a 1-D ``np.ndarray`` instead of a list.

    Returns:
        list | np.ndarray | generator: The flattened elements.

    Raises:
        ValueError: If both ``as_gen`` and ``as_np_arr`` are ``True``.

    Example:
        ```python
        A = np.asarray(range(2**3)).reshape(2, 2, 2)
        flatten(A)                    # [0, 1, 2, 3, 4, 5, 6, 7]
        flatten(A, as_np_arr=True)    # array([0, 1, 2, 3, 4, 5, 6, 7])
        ```
    """
    if as_gen and as_np_arr:
        raise ValueError("Either as_gen OR as_np_arr can be true...")

    def gen():
        """Generator yielding flattened elements recursively."""
        for ele in arr:
            if isinstance(ele, Iterable) and not isinstance(ele, (str, bytes)):
                yield from flatten(ele, as_gen=True)
            else:
                yield ele

    if as_gen:
        return gen()
    if as_np_arr:
        return np.asarray(list(gen()))
    return list(gen())


def flatten_top(
    arr: Iterable,
    as_list: bool = True,
    N: int = 1,
) -> Union[list, np.ndarray]:
    r"""
    **Description:**
    Flatten the top ``N`` levels (axis 0) of a nested iterable.

    Args:
        arr (Iterable):  The input (can be ragged).
        as_list (bool, optional): If ``True`` (default), return a list;
            otherwise return a ``np.ndarray``.
        N (int, optional): Number of top levels to flatten.  Default ``1``.

    Returns:
        list | np.ndarray:  ``arr`` with its top ``N`` levels flattened.

    Example:
        ```python
        A = np.asarray(range(2**3)).reshape(2, 2, 2)
        flatten_top(A.tolist())       # [[0, 1], [2, 3], [4, 5], [6, 7]]
        flatten_top(A.tolist(), N=2)  # [0, 1, 2, 3, 4, 5, 6, 7]
        ```
    """
    if N > 1:
        return flatten_top(flatten_top(arr, as_list=as_list, N=1),
                           as_list=as_list, N=N - 1)

    if isinstance(arr, np.ndarray):
        # `np.ndarray.reshape` is the right tool for ndarray flattening; this
        # branch exists only to handle accidental ndarray inputs.
        print("flatten_top: You really should use .reshape instead...")

    flattened = [
        ele.tolist() if isinstance(ele, np.ndarray) else ele
        for row in arr for ele in row
    ]
    return flattened if as_list else np.asarray(flattened)


def check_nan(x: ArrayLike) -> bool:
    r"""
    **Description:**
    ``True`` iff any element of ``x`` is ``NaN``.

    Args:
        x (ArrayLike): Array on which to check for ``NaN``.

    Returns:
        bool: ``True`` if any element of ``x`` is ``NaN``, else ``False``.
    """
    return jnp.any(jnp.isnan(x))


@jit
def compute_evs_hermitian(x: ArrayLike) -> Array:
    r"""
    **Description:**
    Eigenvalues of a Hermitian / symmetric matrix.

    Args:
        x (ArrayLike): Hermitian / symmetric matrix.

    Returns:
        Array: Real eigenvalues sorted in ascending order.
    """
    return jnp.linalg.eigvalsh(x)


@partial(jit, static_argnums=(1,))
def rank_matrix(x: ArrayLike, tolerance: float = 1e-10) -> Array:
    r"""
    **Description:**
    Matrix rank under a numerical tolerance.

    Args:
        x (ArrayLike):              Input matrix.
        tolerance (float, optional): Tolerance below which singular values
            are treated as zero.  Default ``1e-10``.

    Returns:
        Array: Scalar rank of ``x``.
    """
    return jnp.linalg.matrix_rank(x, tol=tolerance)


# ==============================================================================
# 4. Pickle I/O
# ==============================================================================

def load_pickle(filen: str) -> Any:
    r"""
    **Description:**
    Load and return the contents of a (plain, uncompressed) pickle file.

    Args:
        filen (str): Path to the pickle file.

    Returns:
        Any: The deserialised Python object.
    """
    with open(filen, "rb") as f:
        loaded_object = pickle.load(f)
    return loaded_object


def load_zipped_pickle(filen: str) -> Any:
    r"""
    **Description:**
    Load and return the contents of a gzip-compressed pickle file.

    Args:
        filen (str): Path to the ``.gz`` file.

    Returns:
        Any: The deserialised Python object.
    """
    with gzip.open(filen, "rb") as f:
        loaded_object = pickle.load(f)
    return loaded_object


def save_zipped_pickle(obj: Any, filen: str, protocol: int = -1) -> None:
    r"""
    **Description:**
    Pickle ``obj`` and write it gzip-compressed to ``filen``.

    Args:
        obj (Any):     Python object to serialise.
        filen (str):   Output path (``.gz`` recommended).
        protocol (int, optional): Pickle protocol; default ``-1`` (highest).
    """
    with gzip.open(filen, "wb") as f:
        pickle.dump(obj, f, protocol)


# ==============================================================================
# 5. Dict / DataFrame helpers
# ==============================================================================

def mergeDictionary(dict_1: dict, dict_2: dict) -> dict:
    r"""
    **Description:**
    Merge two dictionaries.  For keys present in both, values are
    concatenated along axis 0 via ``np.append``.

    Args:
        dict_1 (dict): First dictionary.
        dict_2 (dict): Second dictionary.

    Returns:
        dict: Combined dictionary.
    """
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = np.append(value, dict_1[key], axis=0)
    return dict_3


def is_outlier(
    data: Any,
    column: Optional[str] = None,
    percentile_cut: float = 5,
) -> np.ndarray:
    r"""
    **Description:**
    Boolean outlier mask for ``data`` based on a symmetric percentile cut.

    Args:
        data (ArrayLike | pandas.DataFrame): Input data.  If a DataFrame,
            ``column`` must be supplied.
        column (str, optional):              Column name (DataFrame inputs only).
        percentile_cut (float, optional):    Half-width of the cut, in
            percentile units.  Default ``5`` (drops the bottom 5% and top 5%).

    Returns:
        np.ndarray: Boolean mask, ``True`` where the corresponding sample is
        below the lower or above the upper percentile.

    Example:
        ```python
        cols = df.columns.to_list()
        all_outliers = np.logical_or.reduce(
            [is_outlier(df, c) for c in cols]
        )
        df_reduced = df.loc[~all_outliers]
        ```
    """
    # Lazy pandas import — keep `util` cheap to load when pandas is unused.
    try:
        import pandas as pd                                                 # noqa: WPS433
        is_df = isinstance(data, pd.DataFrame)
    except ImportError:
        is_df = False

    if is_df:
        if column is None:
            raise ValueError("Need to provide key for dataframe column!")
        values = data[column]
    else:
        values = data

    return np.logical_or(
        values < np.percentile(values, percentile_cut),
        values > np.percentile(values, 100 - percentile_cut),
    )


# ==============================================================================
# 6. Timeout / progress
# ==============================================================================

def progress_bar_jax(arg: Tuple[int, int, float], transforms: Any) -> int:
    r"""
    **Description:**
    JAX-host-callback progress printer.

    Designed to be used as the ``result_shape`` callback of
    ``jax.experimental.host_callback.id_tap`` (or its modern
    ``jax.debug.callback`` replacement).  Prints a single-line residual /
    iteration tracker via carriage return so successive calls overwrite each
    other on the terminal.

    Args:
        arg (tuple): ``(i, n_iter, residual)`` — current iteration index,
            total iteration count, and a scalar residual to display.
        transforms (Any): JAX-supplied trace metadata (unused here; required
            by the host-callback signature).

    Returns:
        int: ``i`` — the current iteration index, unchanged.
    """
    i, n_iter, res = arg
    print(f"Residual: {res}              Iteration: {int(i)}/{int(n_iter)}         ",
          end="\r")
    return i


def quit_function(fn_name: str) -> None:
    r"""
    **Description:**
    Hard-interrupt the main thread.  Used by :func:`exit_after` as the timer
    callback when the wrapped function exceeds its budget.

    Args:
        fn_name (str): Name of the function being aborted (printed to stderr).

    Raises:
        KeyboardInterrupt: Indirectly, via ``thread.interrupt_main()``.
    """
    print(f"{fn_name} ran out of time", file=sys.stderr)
    sys.stderr.flush()
    thread.interrupt_main()


def exit_after(s: int) -> Callable:
    r"""
    **Description:**
    Decorator factory: wrap a function to abort if its execution exceeds
    ``s`` seconds.  Internally arms a ``threading.Timer`` that calls
    :func:`quit_function` to interrupt the main thread on expiry.

    Args:
        s (int): Timeout in seconds.

    Returns:
        Callable: A decorator.

    Example:
        ```python
        @exit_after(5)
        def slow():
            ...
        ```
    """
    def outer(fn):
        """Wrap `fn` with a timeout timer."""
        def inner(*args, **kwargs):
            """Execute `fn` with a timeout guard."""
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer


# ==============================================================================
# 7. Model-data I/O
# ==============================================================================

# Default location for cached model data — overridable per-call by the
# downstream API (currently used by ``cytools_interface.save_model_data``).
home_dir = os.path.dirname(os.path.realpath(__file__))
files_dir = home_dir + "/models"


def save_model_data(
    data: Any,
    fname: str,
    model_ID: Union[int, str],
    h12: int,
) -> None:
    r"""
    **Description:**
    Write ``data`` to ``files_dir/h12_<h12>/<fname>`` as a gzip-compressed
    pickle.  Creates intermediate directories as needed.  Prompts the user
    on overwrite.

    Args:
        data (Any):                       Data to be stored.
        fname (str):                      Filename (relative to the per-h12
            subdirectory).
        model_ID (int | str):             Model identifier; required (used by
            callers as the cache key).
        h12 (int):                        Hodge number :math:`h^{1,2}` —
            determines the subdirectory layout.

    Raises:
        ValueError: If ``model_ID`` is ``None``.
    """
    if model_ID is None:
        raise ValueError("Please provide `model_ID` to save model data!")

    dir_h12 = files_dir + "/h12_" + str(h12) + "/"

    if not os.path.isdir(dir_h12):
        directory = files_dir + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        os.mkdir(dir_h12)

    filename = dir_h12 + fname

    if os.path.isfile(filename):
        print(f"Model ID already exists! File `{fname}` might be overwritten!")
        asking_input = input("Do You Want To Continue? [y/n]")
        if asking_input == "y":
            save_zipped_pickle(data, filename, protocol=-1)
    else:
        save_zipped_pickle(data, filename, protocol=-1)


# ==============================================================================
# 8. Pytree flatten / unflatten
#
# Generic flatten / unflatten functions used by ``register_pytree_node`` for
# the project's pytree-registered classes (``periods``, ``css``, ``FluxEFT``,
# ``Conifold``).  See https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.register_pytree_node.html
# ==============================================================================

# Class attributes that must travel as auxiliary (static / hashable) data
# rather than as JAX-traced children.  Anything in this set, plus any value
# whose Python type is ``str`` or ``bool``, is preserved verbatim across the
# pytree boundary.
_STATIC_KEYS: Tuple[str, ...] = (
    "h11", "h12", "model_ID", "dimension_H3", "_dimension_H3_tot",
    "model_type", "n_fluxes", "gauge_choice", "prange", "maximum_degree",
    "D3_tadpole", "nmax", "_active_conifold_idx",
    "prepotential_input", "period_input",
)


def flatten_func(obj: Any) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    r"""
    **Description:**
    Flatten ``obj`` for the JAX pytree protocol.

    Splits ``obj.__dict__`` into:
      * ``children`` — values that are JAX arrays / pytrees and should be
        traced;
      * ``aux_data`` — names of those children, plus a flat list of
        ``(key, value)`` pairs for static (non-traced) attributes.

    The classification is: a value is **static** iff its Python type is
    ``str`` / ``bool`` or its key appears in :data:`_STATIC_KEYS`.

    Args:
        obj (Any): Instance to flatten.  Any class registered with
            :func:`jax.tree_util.register_pytree_node`.

    Returns:
        tuple: ``(children, aux_data)`` — the standard pytree flatten output.
    """
    children: List[Any] = []
    aux_data: List[Any] = []
    static: List[Tuple[str, Any]] = []
    for key, value in obj.__dict__.items():
        if isinstance(value, (str, bool)) or key in _STATIC_KEYS:
            static.append((key, value))
        else:
            children.append(value)
            aux_data.append(key)

    aux_data.extend(static)
    return tuple(children), tuple(aux_data)


def unflatten_func_class(
    aux_data: Tuple[Any, ...],
    children: Tuple[Any, ...],
    myclass: type,
) -> Any:
    r"""
    **Description:**
    Inverse of :func:`flatten_func` for a specific class ``myclass``.

    Bypasses ``__init__`` (which often has side-effects) — restores the
    object via ``object.__new__`` + ``setattr`` of each saved attribute.

    Args:
        aux_data (tuple):   Auxiliary data from :func:`flatten_func`.
        children (tuple):   Children (traced values) from :func:`flatten_func`.
        myclass (type):     Class to reconstruct.

    Returns:
        myclass: A fresh instance with all flattened attributes restored.
    """
    obj = object.__new__(myclass)
    for i, attr in enumerate(aux_data):
        if isinstance(attr, tuple):
            object.__setattr__(obj, attr[0], attr[1])
        else:
            object.__setattr__(obj, attr, children[i])
    return obj


# ==============================================================================
# 9. Number-theoretic / lattice helpers
#
# General-purpose helpers used both inside the conifold subsystem
# (``jaxvacua/conifold/conifold_utils.py`` re-exports them) and by external
# callers like ``private/promotion/promotion.py``.
# ==============================================================================

def extended_euclidean(
    w: ArrayLike,
) -> Tuple[np.ndarray, int, np.ndarray]:
    r"""
    **Description:**
    Computes Bézout's identity and a unimodular integer basis transformation
    for an integer array :math:`w`.

    .. admonition:: Details
        :class: dropdown

        Given an integer array :math:`w = (w_1,\ldots,w_n)`, the function
        returns integers :math:`b_i` (Bézout coefficients) satisfying

        .. math::
            \sum_{i=1}^{n} b_i \, w_i = \gcd(w_1,\ldots,w_n) \,,

        together with a unimodular integer matrix
        :math:`\Lambda \in \mathrm{GL}(n,\mathbb{Z})` such that

        .. math::
            \Lambda \, w = \bigl(\gcd(w_1,\ldots,w_n),\; 0,\;\ldots,\; 0\bigr)^T \,.

        The algorithm iteratively reduces pairs of entries via the Euclidean
        algorithm, tracking the accumulated integer row operations in
        :math:`\Lambda`.  Edge cases (single non-zero entry, all-zero input)
        are handled explicitly.

    Args:
        w (ArrayLike): Integer input array of length :math:`n`.

    Returns:
        tuple: ``(Bezout, GCD, Lambda)`` where

        - ``Bezout`` (``np.ndarray``, shape ``(n,)``, dtype ``int``) —
          Bézout coefficients satisfying
          :math:`\sum_i \text{Bezout}_i \cdot w_i = \gcd(w)`.
        - ``GCD`` (``int``) — Greatest common divisor of all non-zero
          entries of :math:`w`.
        - ``Lambda`` (``np.ndarray``, shape ``(n, n)``, dtype ``int``) —
          Unimodular transformation satisfying
          :math:`\Lambda w = (\gcd(w), 0,\ldots,0)^T`.

    See also: :func:`orthogonal_lattice`, and the conifold-specific
    :func:`jaxvacua.conifold.conifold_utils.get_basis_change`.
    """
    # Ensure input is a NumPy array.
    w = np.asarray(w)

    # Identify non-zero / zero positions.
    nonvan_flag = (w != 0)
    van_flag = (w == 0)
    nonvan_pos = np.where(nonvan_flag)[0]
    van_pos = np.where(van_flag)[0]

    # Bézout coefficients (same shape as input).
    Bezout = np.zeros(len(w), dtype=int)

    # -----------------------------
    # Special case: only one non-zero entry.
    # -----------------------------
    if sum(nonvan_flag) == 1:
        # The gcd is just that entry.
        GCD = w[nonvan_flag][0]

        # Bézout coefficient is 1 for that entry.
        Bezout[nonvan_flag] = 1

        # Construct the transformation matrix as identity with a swap.
        Lambda_final = np.identity(len(w), dtype=int)
        Lambda_final[0][0] = 0
        Lambda_final[nonvan_pos[0]][nonvan_pos[0]] = 0
        Lambda_final[0][nonvan_pos[0]] = 1
        Lambda_final[nonvan_pos[0]][0] = 1

    else:
        # -----------------------------
        # General case: multiple non-zero entries.
        # -----------------------------

        # Extract non-zero entries.
        v = w[nonvan_flag]

        # Work with absolute values for the Euclidean reduction.
        acoeff = np.abs(v)

        # Sort entries in descending order (largest first).
        reordering = np.flip(np.argsort(acoeff))
        acoeffsorted = acoeff[reordering]

        # Initialise Lambda as the permutation matrix from the sort.
        Lambda = np.array([
            np.eye(1, len(reordering), i, dtype=int)[0]
            for i in reordering
        ])

        # Track how many dimensions have been reduced (zero-rows introduced).
        dim_red = 0

        # -----------------------------
        # Iterative Euclidean reduction.
        # -----------------------------
        while True:
            # Divide all but the last element by the smallest element.
            divs = acoeffsorted[:-1] / acoeffsorted[-1]

            # Integer quotients.
            qs = divs.astype(int)

            # Remainders (careful integer rounding).
            rs = np.rint(((divs - qs) * acoeffsorted[-1])).astype(int) \
                + np.arange(len(divs)) * 1e-10

            # Sort remainders in descending order.
            rssorted = np.flip(np.sort(rs))

            # Permutation mapping old remainders to sorted ones.
            perm = np.array([i == rs for i in rssorted], dtype=int)

            # Smallest becomes first, followed by the sorted remainders.
            acoeffsorted = np.rint(
                np.concatenate(([acoeffsorted[-1]], rssorted))
            ).astype(int)

            # Build the next transformation block.
            LambdaNext0 = np.block([
                [qs, np.transpose([[1]])],
                [perm, np.transpose([np.zeros(len(perm))])],
            ]).astype(int)

            # Expand to include previously eliminated dimensions.
            LambdaNext = np.block([
                [LambdaNext0, np.zeros([len(LambdaNext0), dim_red])],
                [np.zeros([dim_red, len(LambdaNext0)]), np.identity(dim_red)],
            ])

            # Accumulate the transformation.
            Lambda = LambdaNext @ Lambda

            # Identify non-zero / zero positions and shrink.
            posnonvan = np.where(acoeffsorted > 0)[0]
            posvan = np.where(acoeffsorted == 0)[0]
            acoeffsorted = acoeffsorted[posnonvan]
            dim_red = dim_red + len(posvan)

            # Stop when only the gcd remains.
            if len(acoeffsorted) == 1:
                break

        # -----------------------------
        # Recover Bézout coefficients.
        # -----------------------------

        # First row of inverse transformation gives the Bézout coefficients.
        Bezout0 = (
            np.rint(np.transpose(np.linalg.inv(Lambda))[0]) * np.sign(v)
        ).astype(int)

        # Full inverse transformation, with signs restored.
        Lambda0 = (
            np.rint(np.transpose(np.linalg.inv(Lambda))) * np.sign(v)
        ).astype(int)

        # Embed into the full ambient dimension (zeros included).
        Lambda_tilde = np.block([
            [np.zeros([len(Lambda0), len(w) - len(Lambda0)], dtype=int), Lambda0],
            [np.identity(len(w) - len(Lambda0), dtype=int),
             np.zeros([len(w) - len(Lambda0), len(Lambda0)], dtype=int)],
        ])

        # -----------------------------
        # Reassemble the full transformation.
        # -----------------------------
        Lambda_final = np.identity(len(w), dtype=int)

        # Rows for non-zero entries.
        Lambda_final[nonvan_pos] = Lambda_tilde.T[
            len(w) - len(Lambda0):len(Lambda_tilde)
        ]

        # Rows for zero entries.
        Lambda_final[van_pos] = Lambda_tilde.T[
            0:len(w) - len(Lambda0)
        ]

        # Transpose to get the final form.
        Lambda_final = Lambda_final.T

        # Compute gcd from the Bézout identity.
        GCD = np.rint(sum(Bezout0 * v)).astype(int)

        # Place Bézout coefficients back into their original positions.
        Bezout[nonvan_flag] = Bezout0

    return (Bezout, GCD, Lambda_final)


def orthogonal_lattice(gens_in: List[List[int]]) -> List[List[int]]:
    r"""
    **Description:**
    Returns generators of the integer lattice orthogonal to the lattice
    spanned by ``gens_in``.

    .. admonition:: Details
        :class: dropdown

        Given :math:`d` generators :math:`g_1,\ldots,g_d \in \mathbb{Z}^n`
        with :math:`d < n`, the function computes generators of the
        *orthogonal complement lattice*

        .. math::
            L^\perp = \bigl\{ v \in \mathbb{Z}^n \;:\;
                v \cdot g_i = 0 \;\;\forall\, i=1,\ldots,d \bigr\} \,,

        which has rank :math:`n - d`.

        The algorithm constructs the augmented matrix

        .. math::
            B = \begin{pmatrix} c\,G \\ I_n \end{pmatrix}
            \in \mathbb{Z}^{(d+n)\times n} \,,

        where :math:`G` is the :math:`d\times n` generator matrix and
        :math:`c` is an integer scale chosen so that LLL reduction on
        :math:`B^T` separates the null-space rows.  The last :math:`n-d`
        rows of the LLL-reduced matrix (extracted from the :math:`I_n`
        block) are the desired generators.  The LLL computation uses
        ``flint.fmpz_mat`` for exact integer arithmetic.

    Args:
        gens_in (list): List of :math:`d` integer generator vectors of
            length :math:`n`, with :math:`d < n`.

    Returns:
        list: List of :math:`n-d` integer generators of :math:`L^\perp`,
        each of length :math:`n`.

    See also: :func:`extended_euclidean`, and the conifold-specific
    :func:`jaxvacua.conifold.conifold_utils.get_basis_change`.
    """
    # Convert input list of generators to a NumPy array.
    gens = np.array(gens_in)

    # d = number of input generators, n = ambient dimension.
    d = len(gens)
    n = len(gens[0])

    # -----------------------------
    # Compute scaling factor c.
    # -----------------------------
    # The exponent comes from bounds ensuring LLL separates the orthogonal
    # complement correctly.
    exponent = (n - 1) / 2 + (n - d) * (n - d - 1) / 4

    # c scales the input generators so that LLL prioritises orthogonality
    # constraints over the identity rows.
    c = int(np.ceil(
        (2 ** exponent) * np.prod([np.linalg.norm(g) for g in gens])
    ))

    # -----------------------------
    # Build the augmented matrix B^T.
    # -----------------------------
    # B^T = [ c * G ]
    #       [  I_n  ]      (shape (d + n, n))
    b_T = np.concatenate((c * gens, np.identity(n, dtype=int)))

    # Convert to FLINT integer matrix for exact LLL reduction.
    b_T_mat = fmpz_mat(b_T.T.tolist())

    # -----------------------------
    # Perform LLL reduction.
    # -----------------------------
    # Apply LLL to B^T (transposed form expected by FLINT) and convert back.
    # The first (n - d) rows correspond to short vectors which encode the
    # orthogonal complement.
    b_T_lll = [
        [int(ii) for ii in row][-n:]    # extract last n entries (original coords)
        for row in np.array(b_T_mat.lll().tolist(), dtype=int)[:n - d]
    ]

    return b_T_lll
