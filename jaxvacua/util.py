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
from functools import partial

# To load pickle files
import pickle
import gzip

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
jax.config.update("jax_enable_x64", True)

DTYPE='float64'

# Set random seed for reproducible results
class PRNGSequence:
    """
    **Description:**
    Random key sequence. Taken from CYJax (https://github.com/ml4physics/cyjax).
    """

    _key = None

    def __init__(self, seed: 42):
        """
        **Description:**
        Random key sequence. 

        Args:
            seed (int, optional): Seed to use for random number generation. Defaults to :c:var:`42`.

        Use as follows:
        >>> rns = PRNGSequence(42)
        >>> key = next(rns)
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
        rns_key (PRNGSequence, optional): Random key sequence. Defaults to :c:var:`None`.
        seed (int, optional): Seed to use for random number generation. Defaults to :c:var:`42`.
        shape (tuple, optional): Defaults to :c:var:`(1,)`.

    Returns:
        Uniformly distributed random numbers.
    
    """

    if rns_key is None:
        rns_key = PRNGSequence(seed)

    return jax.random.uniform(next(rns_key),shape=shape, minval=lower_bound, maxval= upper_bound, dtype=DTYPE)

def random_integer(lower_bound: float, upper_bound: float,
                    rns_key: PRNGSequence=None, seed: int=42, shape: Tuple=(1,)) -> Array:
    r"""
    **Description:**
    Returns random integers from a uniform distributions.
    
    Args:
        lower_bound (float): Lower bound for the random integer generation.
        upper_bound (float): Upper bound for the random integer generation.
        rns_key (PRNGSequence, optional): Random key sequence. Defaults to :c:var:`None`.
        seed (int, optional): Seed to use for random number generation. Defaults to :c:var:`42`.
        shape (tuple, optional): Defaults to :c:var:`(1,)`.

    Returns:
        Uniformly distributed random integers.
    
    """

    if rns_key is None:
        rns_key = PRNGSequence(seed)
        
    return jax.random.randint(next(rns_key),shape=shape, minval=lower_bound, maxval= upper_bound+1)


@partial(jit,static_argnums=(3,))
def random_uniform_jit(rns_key, lower_bound: float, upper_bound: float, shape: Tuple=(1,)) -> Array:
    r"""
    **Description:**
    Returns random numbers from a uniform distributions.
    
    Args:
        rns_key (PRNG key): Random key sequence. Defaults to :c:var:`None`.
        lower_bound (float): Lower bound for the random number generation.
        upper_bound (float): Upper bound for the random number generation.
        shape (tuple, optional): Defaults to :c:var:`(1,)`.

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
        rns_key (PRNG key): Random key sequence. Defaults to :c:var:`None`.
        lower_bound (float): Lower bound for the random integer generation.
        upper_bound (float): Upper bound for the random integer generation.
        shape (tuple, optional): Defaults to :c:var:`(1,)`.

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
        return func(*args,**kwargs)
        
    return jax.jit(jax.vmap(_tmp,in_axes=in_axes))


def progress_bar_jax(arg, transforms):
    r"""
    **Description:**
    Produces progress bar.
    
    Args:
        arg (Tuple): Tuple containing the index, number of iterations and residual. 
        transforms (?): ?
    
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
        
    Returns:
        
    
    """
    with gzip.open(filen, 'wb') as f:
        pickle.dump(obj, f, protocol)
        
    f.close()


def subsets(iterable, n, as_list=True):
    """
    **Description:**
    Returns all size=n collections of the iterable object. Really, just a
    wrapper for itertools.combinations

    **Arguments:**
    - `iterable` *(iterable)*: The iterable object of elements.
    - `n` *(integer)*: The number of elements in the output.
    - `as_list` *(boolean,optional)*: Whether to return as a list (True) or as
    an itertools.combinations object
    
    **Returns:**
    *(list or itertools.combinations)* The size=n collections of elements

    **Examples:**
    >>> A = range(4)
    >>> subsets(A, n=1)
    [(0,), (1,), (2,), (3,)]
    >>> subsets(A, n=2)
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    >>> subsets(A, n=3)
    [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    >>> subsets(A, n=4)
    [(0, 1, 2, 3)]
    >>> subsets(A, n=5)
    []
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
            appear in :c:var:`arr`.

    Examples:
        >>> A = np.asarray(range(2**3)).reshape(2,2,2)
        >>> flatten(A)
        [0, 1, 2, 3, 4, 5, 6, 7]
        >>> flatten(A, as_np_arr=True)
        array([0, 1, 2, 3, 4, 5, 6, 7])
        >>> type(flatten(A, as_gen=True))
        <class 'generator'>
        >>> list(flatten(A, as_gen=True))
        [0, 1, 2, 3, 4, 5, 6, 7]
        >>> flatten(A, as_gen=True, as_np_arr=True)
        Traceback (most recent call last):
          ...
        ValueError: Either as_gen OR as_np_arr can be true...
    """
    # input checking
    if as_gen and as_np_arr:
        raise ValueError("Either as_gen OR as_np_arr can be true...")

    def gen():  # the generator giving the elements
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
            (:c:var:`True`) or a numpy array (:c:var:`False`).
        N (int, optional): How many levels to flatten, from the top.
    
    Returns:
        List or ArrayLike: list, but with the top level flattened.

    Examples:
        >>> A = np.asarray(range(2**3)).reshape(2,2,2)
        >>> flatten_top(A)
        flatten_top: You really should use .reshape instead...
        [[0, 1], [2, 3], [4, 5], [6, 7]]
        >>> flatten_top(A.tolist())
        [[0, 1], [2, 3], [4, 5], [6, 7]]
        >>> flatten_top(A.tolist(), N=2)
        [0, 1, 2, 3, 4, 5, 6, 7]
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
    Checks whether an array contains :c:var:`NaN`.
    
    Args:
       x (ArrayLike): Array on which to run check for :c:var:`NaN`.
        
    Returns:
       bool: :c:var:`True` if any of the values in :var:`x` are :c:var:`NaN`, otherwise :c:var:`False`.
    
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

    Example:

    ```
        cols=df.columns.to_list()
        all_outliers = np.logical_or.reduce([is_outlier(df, column) for column in cols])
        df_reduced = df.loc[np.logical_not(all_outliers)]
    ```
    
    Args:
        data (ArrayLike, pd.DataFrame): Input data.
        column (str, optional): Column of dataframe.
        percentile_cut (float, optional): Percentile cut.
        
    Returns:
         ArrayLike: Boolean values for outliers for given percentile.
    
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

    ```
    # Example usage:
    def my_function(x, y, static_arg):
        # Some computation
        return x + y + static_arg

    # JIT-compile my_function, treating the third argument as static
    jit_func = jit_with_static_args(my_function, static_argnums=(2,))

    # Test the function
    import jax.numpy as jnp

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    static_arg = 10

    print(jit_func(x, y, static_arg))
    ```

    Parameters:
    func: The function to be JIT-compiled.
    static_argnums: A tuple of integers specifying the positions of arguments to treat as static.

    Returns:
    A JIT-compiled version of the input function.
    """
    
    @partial(jax.jit, static_argnums=static_argnums)
    def wrapped_func(*args, **kwargs):
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
    
    ```
    # Example usage:
    def my_function(x, y, static_arg):
        # Some computation
        return x + y + static_arg

    # JIT-compile my_function, dynamically detecting static arguments
    jit_func = jit_with_dynamic_static_args(my_function)

    # Test the function
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    static_arg = 10

    print(jit_func(x, y, static_arg))
    ```

    Parameters:
        func: The function to be JIT-compiled.
    
    Returns:
        A JIT-compiled version of the input function with dynamically detected static arguments.
    """
    
    def wrapped_func(*args, **kwargs):
        # Detect which arguments are static
        static_argnums = tuple(i for i, arg in enumerate(args) if is_static(arg))
        
        # Apply JIT compilation with the detected static arguments
        jit_func = jax.jit(func, static_argnums=static_argnums)
        
        # Call the JIT-compiled function
        return jit_func(*args, **kwargs)
    
    return wrapped_func



def quit_function(fn_name):
    r"""
    **Description:**
    Quits a function if it takes too long.
    
    Args:
       fn_name (str): Name of function to be quit.
       
    Returns:
       Raises KeyboardInterrupt.
    """
    
    print('{0} ran out of time'.format(fn_name), file=sys.stderr)
    sys.stderr.flush() 
    # raise KeyboardInterrupt
    thread.interrupt_main() 

def exit_after(s):
    r"""
    **Description:**
    Decorator to exit a function if it takes longer than :c:var:`s` seconds.
    
    Args:
        s (int): Number of seconds after which function is quit.    
    
    Returns:
        Decorator function.
    """
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer





    




