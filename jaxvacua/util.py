# ==============================================================================
# This code is written by Andreas Schachner.
#
# If any questions arise, please feel free to reach out to me (Andreas) either at
# andreas.schachner@gmx.net or at as3475@cornell.edu .
# ==============================================================================
#
# ------------------------------------------------------------------------------
# This file holds functions for general utility purposes.
# ------------------------------------------------------------------------------

#Standard libraries
# Important standard libraries
import os, sys, warnings
import numpy as np
import itertools
from functools import partial, lru_cache

# To load pickle files
import pickle
import gzip

# Exact-integer lattice algebra (used by ``orthogonal_lattice``)
from flint import fmpz_mat

# For progress bar
import threading
try:
    import thread
except ImportError:
    import _thread as thread

# Import jax modules
import jax
import jax.numpy as jnp
from jax import jit
from jax import Array
from jax.typing import ArrayLike
from typing import Optional, Tuple, Any, List, Callable, Union, Iterable
DTYPE = jax.dtypes.canonicalize_dtype(float)

# Set random seed for reproducible results
class PRNGSequence:
    """
    **Description:**
    Random key sequence. Taken from CYJax (https://github.com/ml4physics/cyjax).
    """

    _key = None

    def __init__(self, seed: int = 42):
        """
        **Description:**
        Random key sequence. 

        Args:
            seed (int, optional): Seed to use for random number generation. Defaults to ``42``.

        Example usage:
        ```python
        # Example usage:
        rns = PRNGSequence(42)
        key = next(rns)
        ```
        """
        if isinstance(seed, int):
            self._key = jax.random.PRNGKey(seed)
        else:
            self._key = seed

    def __next__(self):
        r"""
        **Description:**
        Returns next random key in sequence.
        
        Returns:
            PRNG key: Next random key in sequence.
        """
        k, self._key = jax.random.split(self._key)
        return k


def random_uniform(lower_bound: float,upper_bound: float,
                    rns_key: PRNGSequence=None,seed: int=42,shape: Tuple=(1,)) -> Array:
    r"""
    **Description:**
    Returns random numbers from a uniform distributions.
    
    Args:
        lower_bound (float): Lower bound for the random number generation.
        upper_bound (float): Upper bound for the random number generation.
        rns_key (PRNGSequence, optional): Random key sequence. Defaults to ``None``.
        seed (int, optional): Seed to use for random number generation. Defaults to ``42``.
        shape (tuple, optional): Defaults to ``(1,)``.

    Returns:
        Uniformly distributed random numbers.
    
    """

    if rns_key is None:
        key = next(PRNGSequence(seed))
    elif isinstance(rns_key, PRNGSequence):
        key = next(rns_key)
    else:
        key = rns_key  # raw JAX PRNGKey array

    return jax.random.uniform(key, shape=shape, minval=lower_bound, maxval=upper_bound, dtype=DTYPE)

def random_integer(lower_bound: float, upper_bound: float,
                    rns_key: PRNGSequence=None, seed: int=42, shape: Tuple=(1,)) -> Array:
    r"""
    **Description:**
    Returns random integers from a uniform distributions.
    
    Args:
        lower_bound (float): Lower bound for the random integer generation.
        upper_bound (float): Upper bound for the random integer generation.
        rns_key (PRNGSequence, optional): Random key sequence. Defaults to ``None``.
        seed (int, optional): Seed to use for random number generation. Defaults to ``42``.
        shape (tuple, optional): Defaults to ``(1,)``.

    Returns:
        Uniformly distributed random integers.
    
    """

    if rns_key is None:
        key = next(PRNGSequence(seed))
    elif isinstance(rns_key, PRNGSequence):
        key = next(rns_key)
    else:
        key = rns_key  # raw JAX PRNGKey array

    return jax.random.randint(key, shape=shape, minval=lower_bound, maxval=upper_bound+1)


@partial(jit,static_argnums=(3,))
def random_uniform_jit(rns_key, lower_bound: float, upper_bound: float, shape: Tuple=(1,)) -> Array:
    r"""
    **Description:**
    Returns random numbers from a uniform distributions.
    
    Args:
        rns_key (PRNG key): Random key sequence. Defaults to ``None``.
        lower_bound (float): Lower bound for the random number generation.
        upper_bound (float): Upper bound for the random number generation.
        shape (tuple, optional): Defaults to ``(1,)``.

    Returns:
        Uniformly distributed random numbers.
    
    """

    return jax.random.uniform(rns_key,shape=shape, minval=lower_bound, maxval= upper_bound, dtype=DTYPE)

@partial(jit,static_argnums=(3,))
def random_integer_jit(rns_key, lower_bound: float, upper_bound: float, shape: Tuple=(1,)) -> Array:
    r"""
    **Description:**
    Returns random integers from a uniform distributions.
    
    Args:
        rns_key (PRNG key): Random key sequence. Defaults to ``None``.
        lower_bound (float): Lower bound for the random integer generation.
        upper_bound (float): Upper bound for the random integer generation.
        shape (tuple, optional): Defaults to ``(1,)``.

    Returns:
        Uniformly distributed random integers.
    
    """

    return jax.random.randint(rns_key,shape=shape, minval=lower_bound, maxval= upper_bound+1)


def vmapping_func(func,in_axes=None,**kwargs):
    r"""
    **Description:**
    Applies `jax.vmap` on input function for given optional input.
    
    Args:
        func (Callable): Function to be vmapped.
        in_axes (optional, Tuple): Axes for vmapping.
        **kwargs: Extra optional inputs to be fixed for the vmapped version of `func`.

    Returns:
        Callable: Vmapped function.
    
    """

    def _tmp(*args):
        """Closure forwarding kwargs to func for vmap."""
        return func(*args,**kwargs)
        
    return jax.jit(jax.vmap(_tmp,in_axes=in_axes))


@lru_cache(maxsize=256)
def _build_vmap_jit(func, in_axes, frozen_kwargs):
    r"""
    **Description:**
    Module-level LRU-cached factory that builds and stores a ``jax.jit(jax.vmap(...))``
    kernel.  Called by :func:`vmapping_func_cached`; not intended for direct use.

    The cache key is ``(func, in_axes, frozen_kwargs)``.  Python bound methods are
    hashable (keyed on ``(method.__func__, method.__self__)``), so the same method
    on the same instance always resolves to the same cache entry.

    Args:
        func (Callable): Hashable callable to be vmapped.
        in_axes (tuple | None): Batch-axis specification forwarded to ``jax.vmap``.
        frozen_kwargs (tuple): Sorted tuple of ``(key, value)`` pairs representing
            the keyword arguments to be bound inside the closure.  All values must
            be hashable (strings, bools, ints, tuples, ``None``).

    Returns:
        Callable: JIT-compiled vmapped function.

    """
    kwargs = dict(frozen_kwargs)

    def _tmp(*args):
        """Closure forwarding frozen kwargs to func for vmap."""
        return func(*args, **kwargs)

    return jax.jit(jax.vmap(_tmp, in_axes=in_axes))


def vmapping_func_cached(func, in_axes=None, **kwargs):
    r"""
    **Description:**
    Cached variant of :func:`vmapping_func`.  Returns a JIT-compiled vmapped
    function, **reusing the previously compiled XLA kernel** whenever
    ``(func, in_axes, kwargs)`` match a prior call.

    Unlike :func:`vmapping_func`, which builds a *new* closure and a *new*
    ``jax.jit(jax.vmap(...))`` object on every invocation — defeating JAX's
    internal compilation cache and forcing a full XLA recompilation — this
    variant delegates construction to the module-level LRU-cached helper
    :func:`_build_vmap_jit`.  Subsequent calls with identical arguments return
    the cached callable directly, with zero recompilation overhead.

    Args:
        func (Callable): Function to be vmapped.  Must be hashable; Python bound
            methods satisfy this requirement via ``(method.__func__, method.__self__)``.
        in_axes (optional, tuple | None): Axes for vmapping, forwarded to
            ``jax.vmap``.  Must be hashable (tuples of ints or ``None``).
            Default is ``None``.
        **kwargs: Extra keyword arguments to be fixed inside the vmapped closure.
            All values must be hashable (strings, bools, ints, tuples, ``None``).

    Returns:
        Callable: JIT-compiled vmapped function, reused from cache when possible.

    .. note::
        The backing LRU cache (:func:`_build_vmap_jit`) is module-level and
        persists for the lifetime of the Python process.  Its default capacity
        of 256 entries covers 256 distinct ``(func, in_axes, kwargs)``
        combinations before the least-recently-used entry is evicted.

    """
    frozen = tuple(sorted(kwargs.items()))
    return _build_vmap_jit(func, in_axes, frozen)


def progress_bar_jax(arg, transforms):
    r"""
    **Description:**
    Produces progress bar.
    
    Args:
        arg (Tuple): Tuple containing the index, number of iterations and residual.
        transforms (?): ?

    Returns:
        int: The current iteration index.
    """

    i, n_iter, res = arg
    print(f"Residual: {res}              Iteration: {int(i)}/{int(n_iter)}         ", end="\r")
    return i



def load_zipped_pickle(filen):
    r"""
    **Description:**
    Returns content of zipped pickle file.
    
    
    Args:
       filen (string): Filename of zipped file to be read.
        
    Returns:
       ArrayLike/dict: Data contained in file.
    
    """
    
    with gzip.open(filen, 'rb') as f:
        loaded_object = pickle.load(f)
            
    f.close()
            
    return loaded_object

def load_pickle(filen):
    r"""
    **Description:**
    Returns content of pickle file.
    
    Args:
       filen (string): Filename of file to be read.
        
    Returns:
       ArrayLike/dict: Data contained in file.
    
    """
    with open(filen, 'rb') as f:
        loaded_object = pickle.load(f)
            
    f.close()
            
    return loaded_object

def save_zipped_pickle(obj, filen, protocol=-1):
    r"""
    **Description:**
    Saves data in a zipped pickle file.
    
    
    Args:
       obj (array/dict): Data to be stored in file.
       filen (string): Filename of file to be read.
    """
    with gzip.open(filen, 'wb') as f:
        pickle.dump(obj, f, protocol)
        
    f.close()


def subsets(iterable, n, as_list=True):
    """
    **Description:**
    Returns all size=n collections of the iterable object. Really, just a
    wrapper for itertools.combinations

    Args:
        iterable (iterable): The iterable object of elements.
        n (int): The number of elements in the output.
        as_list (bool, optional): Whether to return as a list (True) or as
            an itertools.combinations object.

    Returns:
        list or itertools.combinations: The size=n collections of elements.

    Example usage:
        ```python
        # Example usage:
        A = range(4)
        subsets(A, n=1)  # [(0,), (1,), (2,), (3,)]
        subsets(A, n=2)  # [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        subsets(A, n=3)  # [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        subsets(A, n=4)  # [(0, 1, 2, 3)]
        ```
    """
    if as_list:
        return list(itertools.combinations(iterable, n))
    else:
        return itertools.combinations(iterable, n)

def flatten(arr, as_gen=False, as_np_arr=False):
    r"""
    **Description:**
    Totally flatten an array of *any depth*.

    (Modified from stackoverflow 2158395.)

    Args:
        arr (ArrayLike): The array to flatten. Can be ragged/have unequal
            depths.
        as_gen (bool, optional): Whether to return a generator.
        as_np_arr (bool, optional): Whether to return a np.array.
    
    Returns:
        generator or list or np.array: The elements, in the order that they
            appear in ``arr``.

    Example usage:
        ```python
        # Example usage:
        A = np.asarray(range(2**3)).reshape(2,2,2)
        flatten(A)                    # [0, 1, 2, 3, 4, 5, 6, 7]
        flatten(A, as_np_arr=True)    # array([0, 1, 2, 3, 4, 5, 6, 7])
        list(flatten(A, as_gen=True)) # [0, 1, 2, 3, 4, 5, 6, 7]
        ```
    """
    # input checking
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
    else:
        if as_np_arr:
            return np.asarray(list(gen()))
        else:
            return list(gen())
            
def flatten_top(arr, as_list=True, N=1):
    r"""
    **Description:**
    Flatten the top level (`axis=0`) of an array.

    Args:
        arr (ArrayLike): The array to flatten. Can be ragged/have unequal
            depths.
        as_list (bool, optional): Whether to return a list of elements
            (``True``) or a numpy array (``False``).
        N (int, optional): How many levels to flatten, from the top.
    
    Returns:
        List or ArrayLike: list, but with the top level flattened.

    Example usage:
        ```python
        # Example usage:
        A = np.asarray(range(2**3)).reshape(2,2,2)
        flatten_top(A.tolist())       # [[0, 1], [2, 3], [4, 5], [6, 7]]
        flatten_top(A.tolist(), N=2)  # [0, 1, 2, 3, 4, 5, 6, 7]
        ```
    """
    if N>1:
        return flatten_top( flatten_top(arr, as_list=as_list, N=1),\
                                                as_list=as_list, N=N-1)
    else:
        if isinstance(arr, np.ndarray):
            print("flatten_top: You really should use .reshape instead...")

        # we convert elements to lists if they are np arrays
        flattened = [ele.tolist() if isinstance(ele, np.ndarray) else ele\
                                                for row in arr for ele in row]
        if as_list:
            return flattened
        else:
            return np.asarray(flattened)


def check_nan(x):
    r"""
    **Description:**
    Checks whether an array contains ``NaN``.
    
    Args:
       x (ArrayLike): Array on which to run check for ``NaN``.
        
    Returns:
       bool: ``True`` if any of the values in ``x`` are ``NaN``, otherwise ``False``.
    
    """
    return jnp.any(jnp.isnan(x))
    
@jit
def compute_evs_hermitian(x):
    r"""
    **Description:**
    Returns eigenvalues of a hermitian/symmetric matrix.
    
    
    Args:
        x (ArrayLike): Hermitian/symmetric matrix for which eigenvalues are computed.
        
    Returns:
        ArrayLike: Eigenvalues of input matrix.
    
    """
    
    return jnp.linalg.eigvalsh(x)
    
@partial(jit, static_argnums = (1,))
def rank_matrix(x,tolerance=1e-10):
    r"""
    **Description:**
    Returns rank of a matrix for a given tolerance matrix.
    
    Args:
        x (ArrayLike): Input matrix.
        tolerance (float, optiona): Tolerance used for computing the rank.
        
    Returns:
        r (ArrayLike): Rank of matrix.
    
    """
    
    return jnp.linalg.matrix_rank(x,tol=tolerance)


def mergeDictionary(dict_1, dict_2):
    r"""
    
    **Description:**
    Merge two dictionaries.
    
    
    Args:
        dict_1 (dict): First dictionary.
        dict_2 (dict): Second dictionary.
        
    Returns:
        dict (dict): Combined dictionary.
    
    """
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = np.append(value , dict_1[key],axis=0)
            
    return dict_3

def is_outlier(data, column = None,percentile_cut=5):
    r"""
    **Description:**
    Remove outliers from data.

    Args:
        data (ArrayLike, pd.DataFrame): Input data.
        column (str, optional): Column of dataframe.
        percentile_cut (float, optional): Percentile cut.

    Returns:
        ArrayLike: Boolean values for outliers for given percentile.

    Example usage:
    ```python
    # Example usage:
    cols = df.columns.to_list()
    all_outliers = np.logical_or.reduce([is_outlier(df, column) for column in cols])
    df_reduced = df.loc[np.logical_not(all_outliers)]
    ```
    """
    if type(data)==pd.core.frame.DataFrame:

        if column is None:
            raise ValueError("Need to provide key for dataframe column!")

        values = data[column]

    else:

        values = data
    
    return np.logical_or(
                        values < np.percentile(values, percentile_cut),
                        values > np.percentile(values, 100-percentile_cut)
                        )



def jit_with_static_args(func, static_argnums=()):
    r"""
     **Description:**
    A wrapper function to JIT-compile a function while treating some arguments as static.

    Args:
        func (Callable): The function to be JIT-compiled.
        static_argnums (tuple): A tuple of integers specifying the positions of arguments to treat as static.

    Returns:
        Callable: A JIT-compiled version of the input function.

    Example usage:
    ```python
    # Example usage:
    def my_function(x, y, static_arg):
        # Some computation
        return x + y + static_arg

    # JIT-compile my_function, treating the third argument as static
    jit_func = jit_with_static_args(my_function, static_argnums=(2,))

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    static_arg = 10
    print(jit_func(x, y, static_arg))
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
    Determine if an argument should be treated as static.
    
    Args:
        arg: The argument to check.
    
    Returns:
        bool: True if the argument is static, False otherwise.
    """
    # Define criteria for static arguments
    return not isinstance(arg, jnp.ndarray)

def jit_with_dynamic_static_args(func):
    r"""
    **Description:**
    A wrapper function to JIT-compile a function, dynamically detecting static arguments.
    
    Args:
        func (Callable): The function to be JIT-compiled.

    Returns:
        Callable: A JIT-compiled version of the input function with dynamically detected static arguments.

    Example usage:
    ```python
    # Example usage:
    def my_function(x, y, static_arg):
        # Some computation
        return x + y + static_arg

    # JIT-compile my_function, dynamically detecting static arguments
    jit_func = jit_with_dynamic_static_args(my_function)

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    static_arg = 10
    print(jit_func(x, y, static_arg))
    ```
    """
    
    def wrapped_func(*args, **kwargs):
        """JIT-compiled wrapper with dynamically detected static arguments."""
        # Detect which arguments are static
        static_argnums = tuple(i for i, arg in enumerate(args) if is_static(arg))
        
        # Apply JIT compilation with the detected static arguments
        jit_func = jax.jit(func, static_argnums=static_argnums)
        
        # Call the JIT-compiled function
        return jit_func(*args, **kwargs)
    
    return wrapped_func



# ------------------------------------------------------------------------------
# Lattice / number-theory helpers (general-purpose; relocated from the old
# ``conifold_utils.py`` during the Phase 2 conifold-subpackage split).
#
# These two functions don't depend on any conifold-specific state and are also
# used by ``private/promotion/promotion.py`` and elsewhere, so they belong
# alongside the other general-purpose helpers.  Conifold-specific lattice
# wrappers (``get_basis_change``, ``getAMatrix``, ``get_projection``) live in
# ``jaxvacua/conifold/conifold_utils.py``.
# ------------------------------------------------------------------------------

def extended_euclidean(w):
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
        w (Array): Integer input array of length :math:`n`.

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

    # Ensure input is a NumPy array
    w = np.asarray(w)

    # Identify non-zero and zero entries
    nonvan_flag = (w != 0)   # Boolean mask for non-zero entries
    van_flag = (w == 0)      # Boolean mask for zero entries

    # Get indices of non-zero and zero entries
    nonvan_pos = np.where(nonvan_flag)[0]
    van_pos = np.where(van_flag)[0]

    # Initialize Bézout coefficients (same size as input)
    Bezout = np.zeros(len(w), dtype=int)

    # -----------------------------
    # Special case: only one non-zero entry
    # -----------------------------
    if sum(nonvan_flag) == 1:
        # The gcd is just that entry
        GCD = w[nonvan_flag][0]

        # Bézout coefficient is 1 for that entry
        Bezout[nonvan_flag] = 1

        # Construct transformation matrix
        Lambda_final = np.identity(len(w), dtype=int)

        # Swap first row with the non-zero position
        Lambda_final[0][0] = 0
        Lambda_final[nonvan_pos[0]][nonvan_pos[0]] = 0
        Lambda_final[0][nonvan_pos[0]] = 1
        Lambda_final[nonvan_pos[0]][0] = 1

    else:
        # -----------------------------
        # General case: multiple non-zero entries
        # -----------------------------

        # Extract non-zero entries
        v = w[nonvan_flag]

        # Work with absolute values for Euclidean algorithm
        acoeff = np.abs(v)

        # Sort entries in descending order (largest first)
        reordering = np.flip(np.argsort(acoeff))
        acoeffsorted = acoeff[reordering]

        # Initialize Lambda as permutation matrix corresponding to sorting
        Lambda = np.array([
            np.eye(1, len(reordering), i, dtype=int)[0]
            for i in reordering
        ])

        # Track how many dimensions have been reduced (zeros introduced)
        dim_red = 0

        # -----------------------------
        # Iterative Euclidean reduction
        # -----------------------------
        while True:
            # Divide all but last element by smallest element
            divs = acoeffsorted[:-1] / acoeffsorted[-1]

            # Integer quotients
            qs = divs.astype(int)

            # Remainders (careful rounding for integer stability)
            rs = np.rint(((divs - qs) * acoeffsorted[-1])).astype(int) \
                 + np.arange(len(divs)) * 1e-10

            # Sort remainders in descending order
            rssorted = np.flip(np.sort(rs))

            # Build permutation matrix mapping old remainders → sorted ones
            perm = np.array([i == rs for i in rssorted], dtype=int)

            # Update coefficients: smallest becomes first, followed by remainders
            acoeffsorted = np.rint(
                np.concatenate(([acoeffsorted[-1]], rssorted))
            ).astype(int)

            # Build next transformation block
            LambdaNext0 = np.block([
                [qs, np.transpose([[1]])],
                [perm, np.transpose([np.zeros(len(perm))])]
            ]).astype(int)

            # Expand transformation to include previously eliminated dimensions
            LambdaNext = np.block([
                [LambdaNext0, np.zeros([len(LambdaNext0), dim_red])],
                [np.zeros([dim_red, len(LambdaNext0)]), np.identity(dim_red)]
            ])

            # Update accumulated transformation
            Lambda = LambdaNext @ Lambda

            # Identify non-zero and zero positions
            posnonvan = np.where(acoeffsorted > 0)[0]
            posvan = np.where(acoeffsorted == 0)[0]

            # Remove zeros (dimension reduction)
            acoeffsorted = acoeffsorted[posnonvan]
            dim_red = dim_red + len(posvan)

            # Stop when only one value remains (the gcd)
            if len(acoeffsorted) == 1:
                break

        # -----------------------------
        # Recover Bézout coefficients
        # -----------------------------

        # First row of inverse transformation gives Bézout coefficients
        Bezout0 = (
            np.rint(np.transpose(np.linalg.inv(Lambda))[0]) * np.sign(v)
        ).astype(int)

        # Full inverse transformation (with signs restored)
        Lambda0 = (
            np.rint(np.transpose(np.linalg.inv(Lambda))) * np.sign(v)
        ).astype(int)

        # Embed into full dimension (including zeros)
        Lambda_tilde = np.block([
            [np.zeros([len(Lambda0), len(w) - len(Lambda0)], dtype=int), Lambda0],
            [np.identity(len(w) - len(Lambda0), dtype=int),
             np.zeros([len(w) - len(Lambda0), len(Lambda0)], dtype=int)]
        ])

        # -----------------------------
        # Reassemble full transformation
        # -----------------------------
        Lambda_final = np.identity(len(w), dtype=int)

        # Fill rows corresponding to non-zero entries
        Lambda_final[nonvan_pos] = Lambda_tilde.T[
            len(w) - len(Lambda0):len(Lambda_tilde)
        ]

        # Fill rows corresponding to zero entries
        Lambda_final[van_pos] = Lambda_tilde.T[
            0:len(w) - len(Lambda0)
        ]

        # Transpose to get final form
        Lambda_final = Lambda_final.T

        # Compute gcd from Bézout identity
        GCD = np.rint(sum(Bezout0 * v)).astype(int)

        # Place Bézout coefficients back into original positions
        Bezout[nonvan_flag] = Bezout0

    return (Bezout, GCD, Lambda_final)


def orthogonal_lattice(gens_in):
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

    # Convert input list of generators into a NumPy array
    gens = np.array(gens_in)

    # d = number of input generators, n = ambient dimension
    d = len(gens)
    n = len(gens[0])

    # -----------------------------
    # Compute scaling factor c
    # -----------------------------
    # The exponent comes from bounds ensuring LLL separates
    # the orthogonal complement correctly
    exponent = (n - 1) / 2 + (n - d) * (n - d - 1) / 4

    # c scales the input generators so that LLL reduction
    # prioritizes orthogonality constraints over identity rows
    c = int(np.ceil(
        (2 ** exponent) *
        np.prod([np.linalg.norm(g) for g in gens])
    ))

    # -----------------------------
    # Build augmented matrix B^T
    # -----------------------------
    # Stack scaled generators on top of identity matrix:
    #   B^T = [ c * G ]
    #         [  I_n  ]
    #
    # Shape: (d + n) x n
    b_T = np.concatenate((c * gens, np.identity(n, dtype=int)))

    # Convert to FLINT integer matrix for exact LLL reduction
    b_T_mat = fmpz_mat(b_T.T.tolist())

    # -----------------------------
    # Perform LLL reduction
    # -----------------------------
    # Apply LLL to B^T (transposed form expected by FLINT)
    # Convert result back to NumPy array
    #
    # The first (n - d) rows correspond to short vectors,
    # which encode the orthogonal complement
    b_T_lll = [
        [int(ii) for ii in row][-n:]   # Extract last n entries (original coordinates)
        for row in np.array(b_T_mat.lll().tolist(), dtype=int)[:n - d]
    ]

    # -----------------------------
    # Return orthogonal lattice generators
    # -----------------------------
    return b_T_lll


def quit_function(fn_name):
    r"""
    **Description:**
    Quits a function if it takes too long.
    
    Args:
       fn_name (str): Name of function to be quit.

    Raises:
        KeyboardInterrupt: Raised in the main thread when the function runs out of time.
    """
    
    print('{0} ran out of time'.format(fn_name), file=sys.stderr)
    sys.stderr.flush() 
    # raise KeyboardInterrupt
    thread.interrupt_main() 

def exit_after(s):
    r"""
    **Description:**
    Decorator to exit a function if it takes longer than ``s`` seconds.
    
    Args:
        s (int): Number of seconds after which function is quit.    
    
    Returns:
        Decorator function.
    """
    def outer(fn):
        """Wrap fn with a timeout timer."""
        def inner(*args, **kwargs):
            """Execute fn with a timeout guard."""
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer





    




