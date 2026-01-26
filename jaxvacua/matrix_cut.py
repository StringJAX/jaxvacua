# =============================================================================
# Unless this comment is removed, this code is meant solely for members of Liam
# McAllister's group or other people associated with it.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Simple utility functions for matrix operations.
# -----------------------------------------------------------------------------

# standard imports
import warnings
import functools
import numpy as np
from collections.abc import Iterable
import multiprocessing as mp
import copy

# CYTools imports
# (for utilities, there should be ~no such imports...)

# repo imports
from .basics import to_base10

# basics
# ------
def is_matrix_integral(mat,eps=1e-4):
    mat = np.asarray(mat)
    return (abs(mat.round()-mat)<eps).all()

# classes
# =======
class LIL():
    """
    This class describes a 2D LIL matrix. This has the same/less functionality
    as scipy.sparse.lil_array, but it is sometimes (much) quicker

    **Arguments:**
    - `width` *(int, optional)*: The width of the matrix.
    - `nworkers` *(integer, optional)*: Number of processors to use for
        multithreaded calculations.
    """
    def __init__(self, dtype, width=None, nworkers=1, iter_densely=False):
        self.arr = []
        self.dtype = dtype
        self.arr_dense = None
        self._len = None
        self.width = width
        self.default_val = 0

        self.iter_densely = iter_densely
        self.nworkers = nworkers

        self._sum_all = None
        self._sum_0 = None
        self._sum_0_dense = None
        self._sum_1 = None

    # basic interface
    # ---------------
    def __repr__(self):
        # piggy-back printing from list
        return self.arr.__repr__()
    
    def __str__(self):
        # piggy-back string conversion from list
        return self.arr.__str__()

    def __iter__(self):
        # iterator
        if self.iter_densely:
            return iter(self.dense())
        else:
            return iter(self.arr)

    def __setitem__(self, idx, value):
        # item assignment
        if not isinstance(idx,tuple):
            raise ValueError(f"Index must be tuple but was {type(idx)}...")

        self.arr[idx[0]][idx[1]] = value

    def __getitem__(self, idx):
        # indexing
        if isinstance(idx,tuple):
            # get element self.arr[i][j]
            if self.width is None:
                print("LIL: Width not set. Inferring from non-zero values...")
                self.width = self.infer_width()

            if idx[1]>=self.width:
                raise IndexError("list index out of range")
            else:
                return self.arr[idx[0]].get(idx[1],0)
        else:
            # get element self.arr[i]
            return self.arr[idx]

    def __len__(self):
        # length
        if self._len is None:
            self._len = len(self.arr)
        return self._len

    def __array__(self, dtype=None):
        # np.array
        return np.array(self.dense(), dtype=dtype)

    @property
    def shape(self):
        return (len(self),self.width)

    def __add__(self, other):
        # addition
        out = LIL(dtype=self.dtype, width=self.width, nworkers=self.nworkers)
        out.arr = self.arr+other.arr
        return out

    # basic methods
    # --------------
    def infer_width(self):
        """
        **Description:**
        Find the minimum width necessary to hold array

        **Arguments:**
        None.
        
        **Returns:**
        Nothing
        """
        return 1+max([max(row.keys()) for row in self.arr])

    def new_row(self):
        """
        **Description:**
        Append an empty row to the dict.

        **Arguments:**
        None.
        
        **Returns:**
        Nothing
        """
        self.arr.append(dict())

    def append(self, toadd, tocopy=True):
        """
        **Description:**
        Append (a) row(s) to the array.

        **Arguments:**
        - `toadd` *(dict or LIL-like)*: Row(s) to add.
        - `tocopy` *(bool,optional)*: Whether to append a copy of toadd.
        
        **Returns:**
        *(LIL)* Itself.
        """
        if len(toadd)==0:
            return self

        # convert to list of dicts
        if isinstance(toadd,dict):
            toadd = [toadd]
        elif isinstance(toadd[0],type(self)):
            toadd = flatten_top(toadd)

        if tocopy:
            self.arr += copy.copy(toadd)
        else:
            self.arr += toadd

        # reset length
        self._len = None
        
        return self

    def col_inds(self):
        return set().union(*[r.keys() for r in self.arr])

    def reindex(self, f=None):
        """
        **Description:**
        Reindex the ith column to be the f(i)-th one.

        **Arguments:**
        - `f` *(dict)*: Dictionary mapping old column indices to new ones.
        
        **Returns:**
        Nothing
        """
        self.arr_dense = None

        # default map is contiguous from 0 to N_cols-1
        if f is None:
            f = {v:k for (k,v) in enumerate(self.col_inds())}

        if self.nworkers<=1:
            for i,row in enumerate(self.arr):
                self.arr[i] = {(f[j] if j in f else j):v for j,v in row.items()}
        else:
            raise NotImplementedError("Multithreading not yet implemented...")

    def unique_rows(self, allow_shuffle=False):
        """
        **Description:**
        Delete repeated rows. Maybe re-orders rows...

        **Arguments:**
        None.
        
        **Returns:**
        Nothing
        """
        self.arr_dense = None

        if allow_shuffle:
            self.arr = [dict(t) for t in {tuple(d.items()) for d in self.arr}]
        else:
            raise NotImplementedError("LIL.unique_rows: Not yet implemented...")

    def dense(self, tocopy=False):
        """
        **Description:**
        Return a dense version of the array

        **Arguments:**
        - `copy` *(bool,optional)*: Whether to return a copy of self.arr_dense.
        
        **Returns:**
        *(np.array)* The dense array
        """
        if self.arr_dense is None:
            # build empty dense array
            height = len(self.arr)

            if self.default_val==0:
                self.arr_dense = np.zeros((height,self.width),dtype=self.dtype)
            else:
                self.arr_dense = self.default_val*np.ones((height,self.width),\
                                                            dtype=self.dtype)

            # fill in output
            if self.nworkers<=1:
                for i,row in enumerate(self.arr):
                    for j,v in row.items():
                        self.arr_dense[i,j] = v
            else:
                raise NotImplementedError("Multithreading not yet "\
                                            "implemented...")

        # return
        if tocopy:
            return self.arr_dense.copy()
        else:
            return self.arr_dense

    def tolist(self):
        return self.dense().tolist()

    def sum(self, axis=None, dense=True):
        if axis is None:
            if self._sum_all is None:
                self._sum_all = np.sum(self.sum(axis=1))
            return self._sum_all
        elif axis==1:
            if self._sum_1 is None:
                self._sum_1 = np.asarray([sum(r.values()) for r in self.arr])
            return self._sum_1
        elif axis==0:
            if dense:
                if self._sum_0_dense is None:
                    self._sum_0_dense = np.asarray([sum(r.get(i,0) for r in\
                                        self.arr) for i in range(self.width)])
                return self._sum_0_dense
            else:
                if self._sum_0 is None:
                    self._sum_0 = {i: sum(r.get(i,0) for r in self.arr) for i\
                                                            in self.col_inds()}
                return self._sum_0

class lazy_tuple:
    """
    A tuple class whose components are only lazily calculated

    **Arguments:**
    - `data` *(misc)*: Tuple elements (or functions to calculate them).
    """
    def __init__(self, *data):
        self._data = tuple(data)
    
    def __repr__(self):
        # piggy-back printing from tuple
        return self._data.__repr__()
    
    def __str__(self):
        # piggy-back string conversion from tuple
        return self._data.__str__()

    def __len__(self):
        # length
        return len(self._data)

    def __getitem__(self, key):
        item = self._data[key]
        
        if (item is None) or callable(item):
            self._data = list(self._data)
            self._data[key] = item()
            self._data = tuple(self._data)
            item = self._data[key]
            
        return item

class LIL_stack():
    """
    This class describes a stack of LIL objects. One could just manually stack the
    rows but this implementation is quicker.

    The stack is organized as a list of options,
        options = [ [top_block_option1, top_block_option2, ...],
                    [next_block_option1, next_block_option2, ...],
                    ...
                    [bot_block_option1, bot_block_option2, ...]]
    and a list of choices
        choices = [i_top_block, i_next_block, ..., i_bot_block]
    E.g., if choices were [7,2,...,6], the stack would look like:
        stack = [top_block_option7;
                 next_block_option2;
                 ...
                 bot_block_option6]

    **Arguments:**
    - `options`: The possible arrays to stack. The entry options[i] is a list
        of all possible blocks you can put in the ith entry
    - `choices`: The selection of which blocks (from options) to stack.
    - `choice_bounds`: The number of possible choices for each block. I.e.,
        [len(opts) for opts in options]
    - `iter_densely`: Whether to iterate densely over the array or sparsely.
    """
    def __init__(self,
                 options: [["ArrayLike"]],
                 choices: [int],
                 choice_bounds: [int],
                 iter_densely: bool = False):

        self._options = options
        if isinstance(choices,int):
            self._choices = choices
        else:
            self._choices = to_base10(choices,choice_bounds)
        self._choice_bounds = choice_bounds
        self.iter_densely = iter_densely

    # basic interfaces
    def __repr__(self):
        # piggy-back printing from list
        return self.arr.__repr__()
    
    def __str__(self):
        # piggy-back string conversion from list
        return self.arr.__str__()

    def __getitem__(self, idx):
        # indexing

        if isinstance(idx,tuple):
            # get element self.arr[i][j]

            if idx[0]>=len(self):
                raise IndexError("0th list index out of range")
            elif idx[0]<0:
                raise IndexError("negative indexing not currently allowed")

            for block in self._blocks():
                L = len(block)
                if idx[0]<L:
                    return block[idx]
                else:
                    idx = (idx[0]-L,idx[1])
        else:
            # get element self.arr[i]
            if idx>=len(self):
                raise IndexError("list index out of range")
            elif idx<0:
                raise IndexError("negative indexing not currently allowed")

            for block in self._blocks():
                L = len(block)
                if idx<L:
                    return block[idx]
                else:
                    idx -= L

    def __len__(self):
        # length
        if not hasattr(self, '_len'):
            self._len = sum(len(opts[i]) for i,opts in
                                            zip(self.choices,self._options))
        return self._len

    def __iter__(self):
        # iterator
        return self._rows(self.iter_densely)

    def __array__(self, dtype=None):
        # np.array
        return np.array(self.dense(), dtype=dtype)

    # properties
    @property
    def choices(self):
        return from_base10(self._choices, self._choice_bounds)
    
    @property
    def dtype(self):
        return self._options[0][0].dtype
    
    @property
    def width(self):
        return self._options[0][0].width

    @property
    def shape(self):
        if not hasattr(self, '_shape'):
            #self._shape = (len(self),self.width) # slow
            self._shape = lazy_tuple(self.__len__,self.width)
        return self._shape
    
    @property
    def is_empty(self):
        if not hasattr(self, '_is_empty'):
            self._is_empty = True
            
            for block in self._blocks():
                if len(block)>0:
                    self._is_empty = False
                    break

        return self._is_empty
    
    def _blocks(self):
        for i,opts in zip(self.choices,self._options):
            yield opts[i]

    def _rows(self,dense=True):
        if dense:
            row_iter = lambda r:r.dense()
        else:
            row_iter = lambda r:r

        for block in self._blocks():
            for h in row_iter(block):
                yield h

    # getter
    @property
    def arr(self):
        if not hasattr(self, '_arr'):
            self._arr = [row for row in self._rows(False)]

        return self._arr

    def dense(self, tocopy=False):
        """
        **Description:**
        Return a dense version of the array

        **Arguments:**
        - `copy` *(bool,optional)*: Whether to return a copy of self.arr_dense.
        
        **Returns:**
        *(np.array)* The dense array
        """
        if not hasattr(self, '_arr_dense'):
            # build empty dense array
            self._arr_dense = np.zeros(self.shape, dtype=self.dtype)

            # fill in output
            for i,row in enumerate(self.arr):
                for j,v in row.items():
                    self._arr_dense[i,j] = v

        # return
        if tocopy:
            return self._arr_dense.copy()
        else:
            return self._arr_dense

    def tolist(self):
        return self.dense().tolist()

    # basic methods
    def sum(self, axis=None, dense=True):
        if axis is None:
            if not hasattr(self, '_sum_all'):
                self._sum_all = np.sum(self.sum(axis=1))
            return self._sum_all
        elif axis==1:
            if not hasattr(self, '_sum_1'):
                self._sum_1 = flatten_top(\
                                    [M.sum(axis=1) for M in self._blocks()],\
                                    as_list=False)
            return self._sum_1
        elif axis==0:
            if dense:
                if not hasattr(self, '_sum_0_dense'):
                    self._sum_0_dense = np.sum(M.sum(axis=0,dense=True) for M\
                                                            in self._blocks())
                return self._sum_0_dense
            else:
                raise NotImplementedError("sparse sum not yet implemented (shouldn't be hard though...)")

# trivial methods
# ---------------
def all_even(arr):
    """
    **Description:**
    Returns whether every element of arr is even.

    **Arguments:**
    - `arr` *(array-like)*: The array of interest
    
    **Returns:**
    *(bool)* Whether every element is even

    **Examples:**
    >>> A = [1,2,3,4,5,6]
    >>> all_even(A)
    False
    >>> B = [-2, 0, 2, 10]
    >>> all_even(B)
    True
    >>> all_even(2*np.asarray(A))
    True
    """
    return np.all(np.mod(arr,2)==0)

# array manipulation
# ------------------
def flatten_top(arr, as_list=True, N=1):
    """
    **Description:**
    Flatten the top level (axis=0) of an array.

    **Arguments:**
    - `arr` *(array-like)*: The array to flatten. Can be ragged/have unequal
        depths.
    - `as_list` *(boolean, optional)*: Whether to return a list of elements
        (True) or a numpy array (False).
    - `N` *(int, optional)*: How many levels to flatten, from the top.
    
    **Returns:**
    *(list or np.array)* lis, but with the top level flattened.

    **Examples:**
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

def flatten(arr, as_gen=False, as_np_arr=False):
    """
    **Description:**
    Totally flatten an array of *any depth*.

    (Modified from stackoverflow 2158395.)

    **Arguments:**
    - `arr` *(array-like)*: The array to flatten. Can be ragged/have unequal
        depths.
    - `as_gen` *(boolean, optional)*: Whether to return a generator.
    - `as_np_arr` *(boolean, optional)*: Whether to return a np.array.
    
    **Returns:**
    *(generator or list or np.array)* The elements, in the order that they
        appear in arr.

    **Examples:**
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

def extend(arr, extend_by):
    """
    **Description:**
    (if extend_by is 1d): Replaces each element, e, of arr with [e,*extend_by].
    Returns output. This increases the dimension of arr by 1.

    (if extend_by is 2d): Returns [extend(arr, row) for row in extend_by]. We
    don't call it recursively, but the output is the same
    
    **Arguments:**
    - `arr` *(array-like)*: An arbitrary array
    - `extend_by` *(array-like)*: A 1d or 2d array.
    
    **Returns:**
    (See description)

    **Examples:**
    >>> A = np.asarray(range(2**3)).reshape(2,2,2)
    >>> B = [-1,-2,-3]
    >>> C = [-4,-5,-6]
    >>> print(extend(A,B).tolist())
    [[[[0, -1, -2, -3], [1, -1, -2, -3]], [[2, -1, -2, -3], [3, -1, -2, -3]]],\
 [[[4, -1, -2, -3], [5, -1, -2, -3]], [[6, -1, -2, -3], [7, -1, -2, -3]]]]
    >>> print(extend(A,[B,C]).tolist())
    [[[[[0, -1, -2, -3], [1, -1, -2, -3]], [[2, -1, -2, -3],\
 [3, -1, -2, -3]]], [[[4, -1, -2, -3], [5, -1, -2, -3]], [[6, -1, -2, -3],\
 [7, -1, -2, -3]]]],\
 [[[[0, -4, -5, -6], [1, -4, -5, -6]], [[2, -4, -5, -6],\
 [3, -4, -5, -6]]], [[[4, -4, -5, -6], [5, -4, -5, -6]], [[6, -4, -5, -6],\
 [7, -4, -5, -6]]]]]
    """
    # ensure inputs are numpy arrays
    arr = np.asarray(arr); extend_by = np.asarray(extend_by)

    # grab shapes
    arr_shape = arr.shape; extend_shape = extend_by.shape

    # check shape of extend_by
    extend_1d = (len(extend_shape)==1)

    if extend_1d:
        extend_by = extend_by[np.newaxis,:] # convert to 2d
        extend_shape = extend_by.shape
    elif len(extend_shape)!=2:
        raise ValueError("extend_by must be 1/2d, but it had shape"
                                                        f"{extend_shape}...")

    # extend arr by extend_by
    num_extensions = extend_shape[0]
    arr_extended = arr.reshape(1, *arr_shape, 1)
    arr_extended = np.repeat(arr_extended, num_extensions, axis=0)

    extensions = np.tile(extend_by, (1, arr.size))
    extensions = extensions.reshape(*(arr_extended.shape[:-1]), -1)

    arr_extended = np.concatenate((arr_extended, extensions), axis=-1)

    # return
    if extend_1d:
        return arr_extended[0]
    else:
        return arr_extended
    
# matrix element/row comparisons
# ------------------------------
def eles_in(A, B, check_all=False):
    """
    **Description:**
    Returns flags indicating if each element of A appears in B.

    **Notes:**
    Identical, in output, to rows_in if A and B are seen as Nx1 arrays.
    
    **Arguments:**
    - `A` *(list-like)*: A 1d array
    - `B` *(list-like)*: A 1d array
    - `check_all` *(bool, optional)*: Whether to return a bool indicating if
        all elements of A are also elements of B.
    
    **Returns:**
    *(np.array)* 1D array (length n_A) with a 1 in the ith indix iff the
        ith element of A appears as a row in B.

    **Examples:**
    >>> A = [1,0.32,'hi',-3.2]
    >>> B = ['hello',-3.2,1.0]
    >>> C = [0.32,-3.2,1,'hi']
    >>> eles_in(A,B)
    array([False, False, False,  True])
    >>> eles_in(A,C)
    array([ True,  True,  True,  True])
    >>> eles_in(A,C,check_all=True)
    True
    """
    eles_in = (np.asarray(A)[:,None]==np.asarray(B)).any(axis=1)

    if check_all:
        return all(eles_in)
    else:
        return eles_in

def rows_in(A, B):
    """
    **Description:**
    Returns flags indicating if each row of A appears in B.
    
    **Arguments:**
    - `A` *(array-like)*: An (n_A by m) array
    - `B` *(array-like)*: An (n_B by m) array
    
    **Returns:**
    *(np.ndarray)* 1D array (length n_A) with a 1 in the ith indix iff the
        ith row of A appears as a row in B.

    **Examples:**
    >>> A = [[1,2,3],[4,5,6]]
    >>> B = [[-2,-1,0],[1,2,3],[4,5,6]]
    >>> C = [[1,2],[3,4],[5,6]]
    >>> D = np.array(A,dtype=float)
    >>> rows_in(A,B)
    array([ True,  True])
    >>> rows_in(B,A)
    array([False,  True,  True])
    >>> rows_in(A,C)
    Traceback (most recent call last):
      ...
    ValueError: A and B must have the same row-length... A has shape (2, 3)\
 while B has shape (3, 2)...
    >>> rows_in(A,D)
    Traceback (most recent call last):
      ...
    TypeError: A and B must have the same data type... A has type int64 while\
 B has type float64...
    """
    # ensure inputs are numpy arrays
    A = np.asarray(A); B = np.asarray(B)

    # check input array shapes/types
    if not (A.shape[1] == B.shape[1]):
        raise ValueError("A and B must have the same row-length... "
                f"A has shape {A.shape} while B has shape {B.shape}...")

    if not (A.dtype == B.dtype):
        raise TypeError("A and B must have the same data type... "
                f"A has type {A.dtype} while B has type {B.dtype}...")

    # create 'row-data-type'
    row_size = A.dtype.itemsize*A.shape[1]
    dtype_row = np.dtype((np.void, row_size))
    
    # ensure contiguous data (since np.array.view works on memory)
    A_raw = np.ascontiguousarray(A)
    B_raw = np.ascontiguousarray(B)
    
    # get raw arrays
    A_raw_rows = A_raw.view(dtype_row).ravel()
    B_raw_rows = B_raw.view(dtype_row).ravel()
      
    return np.isin(A_raw_rows, B_raw_rows)

def row_sub(A, B):
    """
    **Description:**
    Deletes any rows of A that appear in B.
    
    **Arguments:**
    - `A` *(np.ndarray)*: An (n_A by m) array
    - `B` *(np.ndarray)*: An (n_B by m) array
    
    **Returns:**
    *(np.ndarray)* The array, A, after deleting each row that occurs in B.

    **Examples:**
    >>> A = [[1,2,3],[4,5,6]]
    >>> B = [[-2,-1,0],[1,2,3],[4,5,6]]
    >>> row_sub(A,B)
    array([], shape=(0, 3), dtype=int64)
    >>> row_sub(B,A)
    array([[-2, -1,  0]])
    """
    return np.asarray(A)[~rows_in(A,B)]

def row_perm(A, B, check=True):
    """
    **Description:**
    Find the permutation, perm, such that A[perm]==B. Unique iff A/B has no
    repeated rows.

    (Modified from stackoverflow 35234571.)

    **Arguments:**
    - `A` *(np.array)*: A 2d array
    - `B` *(np.array)*: A 2d array
    - `check` *(np.array)*: Basic check if B is a row permutation of A.
        (Doesn't check multiplicities, if A/B has repeated rows. Ill-defined
        in that case regardless, so this isn't too bad...)

    **Returns:**
    *(list of ints)* The array, perm, s.t. A[perm] == B

    **Examples:**
    >>> A = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
    >>> B = [[10,11,12],[7,8,9],[4,5,6],[1,2,3]]
    >>> C = [[1,2,3],[4,5,6],[7,8,9],[-1,-2,-3]]
    >>> D = [[1,2,3],[4,5,6],[7,8,9]]
    >>> row_perm(A,B)
    array([3, 2, 1, 0])
    >>> np.asarray(A)[row_perm(A,B)].tolist()
    [[10, 11, 12], [7, 8, 9], [4, 5, 6], [1, 2, 3]]
    >>> np.asarray(A)[row_perm(A,B)].tolist()==B
    True
    >>> row_perm(A,C)
    Traceback (most recent call last):
      ...
    ValueError: A has a row that B doesn't...
    >>> row_perm(A,D)
    Traceback (most recent call last):
      ...
    ValueError: A and B must have the same shape... A has shape (4, 3) while B has shape (3, 3)...
    """
    # ensure inputs are numpy arrays
    A = np.asarray(A); B = np.asarray(B)

    # check that input arrays have same shapes/types
    if not (A.shape == B.shape):
        raise ValueError("A and B must have the same shape... "
                f"A has shape {A.shape} while B has shape {B.shape}...")

    if not (A.dtype == B.dtype):
        raise TypeError("A and B must have the same data type... "
                f"A has type {A.dtype} while B has type {B.dtype}...")

    # create 'row-data-type'
    row_size = A.dtype.itemsize*A.shape[1]
    dtype_row = np.dtype((np.void, row_size))

    # ensure contiguous data (since np.array.view works on memory)
    A_raw = np.ascontiguousarray(A)
    B_raw = np.ascontiguousarray(B)
    
    # get raw arrays
    A_raw_rows = A_raw.view(dtype_row).ravel()
    B_raw_rows = B_raw.view(dtype_row).ravel()

    # check that input arrays have same rows
    if check:
        if (False in np.isin(A_raw_rows, B_raw_rows)):
            raise ValueError("A has a row that B doesn't...")
        if (False in np.isin(B_raw_rows, A_raw_rows)):
            raise ValueError("B has a row that A doesn't...")

    # find permutation mapping A to B
    A_sort_inds = np.argsort(A_raw_rows)
    B_insert_inds = A_raw_rows.searchsorted(B_raw_rows, sorter=A_sort_inds)

    return A_sort_inds[B_insert_inds]

# fancier matrix calculations
# ---------------------------
def det(M):
    """
    **Description:**
    Calculate the determinant of a matrix, M. If M is an integer array,
    preserve the integer nature.
    
    **Arguments:**
    - `M` *(np.ndarray)*: An np.array
    
    **Returns:**
    *(numeric)* The determinant

    **Examples:**
    >>> A = [[1,2,3],[4,5,7],[7,8,123]]
    >>> det(A)
    -336
    >>> det(np.asarray(A,dtype=float))
    -336.0
    """
    M = np.asarray(M)

    if not np.issubdtype(M.dtype, np.integer):
        # standard (non-integer) case
        return np.linalg.det(M)
    else:
        # integer-case -> use Bareiss algorithm
        # (Copied from stackoverflow 66192894.)
        M = [row[:] for row in M] # make a copy to keep original M unmodified
        N, sign, prev = len(M), 1, 1
        for i in range(N-1):
            if M[i][i] == 0: # swap with another row having nonzero i's elem
                swapto= next( (j for j in range(i+1,N) if M[j][i] != 0), None )
                if swapto is None:
                    return 0 # all M[*][i] are zero => zero determinant
                M[i], M[swapto], sign = M[swapto], M[i], -sign
            for j in range(i+1,N):
                for k in range(i+1,N):
                    assert (M[j][k] * M[i][i] - M[j][i] * M[i][k] ) % prev == 0
                    M[j][k] = ( M[j][k] * M[i][i] - M[j][i] * M[i][k] ) // prev
            prev = M[i][i]
        return sign * M[-1][-1]

def basis(M):
    """
    **Description:**
    Lazily construct a basis out of the rows of a matrix, M.
    
    **Arguments:**
    - `M` *(np.ndarray)*: An np.array.
    
    **Returns:**
    *(list of ints)* The row indices of the basis

    **Examples:**
    >>> A = [[1,2,3],[-1,-2,-3],[0,0,0],[4,5,7],[3,3,4],[7,8,123]]
    >>> basis(A)
    [0, 3, 5]
    """
    M = np.asarray(M)
    dim = M.shape[0]

    # start of basis
    warnings.warn("NM: It'd be nice to have exact rank calc for integer matrices...")
    basis = [0]; rank = np.linalg.matrix_rank(M[basis])

    # lazily constructs
    for i in range(len(M)):
        # see if row #i increases rank
        if np.linalg.matrix_rank(M[basis+[i]])>rank:
            basis = basis+[i]; rank += 1

        if rank==dim:
            break   # full rank... quit...

    return basis

def kron(*mats):
    """
    **Description:**
    Calculate the Kronecker product of the input matrices.

    Basically, just a wrapper of np.kron, except it accepts an arbitrary count
    of matrices
    
    **Arguments:**
    - `mats` *(2+ matrix-like)*: The matrices
    
    **Returns:**
    *(np.array)* The kronecker product

    **Examples:**
    >>> A=[[1,2],[3,4]]
    >>> B=[[5,6]]
    >>> kron(A).tolist()
    [[1, 2], [3, 4]]
    >>> kron(A,B).tolist()
    [[5, 6, 10, 12], [15, 18, 20, 24]]
    >>> kron(A,B,A).tolist()
    [[5, 10, 6, 12, 10, 20, 12, 24], [15, 20, 18, 24, 30, 40, 36, 48],\
 [15, 30, 18, 36, 20, 40, 24, 48], [45, 60, 54, 72, 60, 80, 72, 96]]
    """
    mats = [np.asarray(M) for M in mats]

    if True:    # despite iterative calculations seeming slow, this is quicker
        return functools.reduce(np.kron, mats)
    else:       # this is a non-iterative method
        N_mats = len(mats)

        shapes = np.asarray([M.shape for M in mats])

        # we tile each matrix and then take the kronecker product with an
        # appropriately-sized identity matrix
        I_shapes = [np.prod(shapes[i+1:],axis=0) for i in range(N_mats)]
        tile_shapes = np.prod(shapes,axis=0)//np.multiply(shapes,I_shapes)

        shaped = [np.kron(np.tile(mats[i],tile_shapes[i]),\
                        np.ones(I_shapes[i],dtype=int)) for i in range(N_mats)]
        return functools.reduce(np.multiply,shaped)

# testing
# =======
if __name__ == "__main__":
    import doctest
    doctest.testmod()
