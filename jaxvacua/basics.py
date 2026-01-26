# =============================================================================
# Unless this comment is removed, this code is meant solely for members of Liam
# McAllister's group or other people associated with it.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Very basic utilities
# -----------------------------------------------------------------------------

# standard imports
from collections.abc import Iterable
import gzip
import json
import math
#from numba import njit
from numbers import Number
import numpy as np
import os
import pickle
import sqlite3
import warnings

# CYTools imports
# (for utilities, there should be ~no such imports...)

# repo imports
# (for basic utilities, there should be ~no such imports...)

# path to data directory
data_path = os.path.dirname(__file__)
data_path = os.path.join(data_path, 'data')

if not os.path.isdir(data_path):
    os.mkdir(data_path )

# UI
# --
_last_ints = -1
def progress_bar(val, val_range, n_ints=100):
    """
    **Description:**
    Prints a string 'progress bar' indicating the status of a calculation.

    **Arguments:**
    - `val` *(numeric)*: Current value
    - `val_range` *(list of 2 numerics)*: Ranges of val: [val_low, val_high]
    - `n_ints` *(int, optional)*: The number of intervals.

    **Returns:**
    Nothing.

    **Examples:**
    (unfortunately, these don't really work due to the '\r' end..)
    """
    warnings.warn("We should be able to use some standard library...")
    global _last_ints

    # calculate exact progress in count of intervals
    progress = (val - val_range[0])/(val_range[1] - val_range[0])
    prog_ints = math.floor(progress*n_ints)

    # print
    if prog_ints > _last_ints:
        _last_ints = prog_ints

        prog_bars = "="*prog_ints + " "*(n_ints-prog_ints)
        prog_str = f"Progress = [{prog_bars}]... ({100*progress:.2f}%)"
        print(prog_str, end='\r', flush=True)

# intervals
# ---------
def split_range(low: Number, up: Number) -> ["list | None", "list | None"]:
    """
    **Description:**
    Split a range [low, up] into a positive and negative part. For use in
    linear/logarithmic spacing functions.

    **Arguments:**
    - `low`: The lower bound of the range.
    - `up`: The upper bound of the range.

    **Returns:**
    A length-2 list [negative_range, positive_range]. If one range is null
    (e.g., if low>0), then said range is set to None,
    """
    if low<0:
        if up<0:
            # purely negative
            return [[low,up],None]
        elif up==0:
            # purely non-positive
            return [[low,0],None]
        else:
            # split b/t positive and negative
            return [[low,0],[0,up]]
    elif low==0:
        # purely non-negative
        return [None,[0,up]]
    else:
        # purely positive
        return [None,[low,up]]

def linspace0(low: Number, high: Number, N: int,
              enforce_0: bool = True) -> np.array:
    """
    **Description:**
    A wrapper to numpy.linspace that allows enforcing that the points 0 arises
    in the range (if it is in the range).

    WARNING: spacing may NOT be even!!! It'll be linear in the negative and
    positive regions, but the spacing might not match between the two.

    **Arguments:**
    - `low`: The lower bound of the range.
    - `high`: The upper bound of the range.
    - `N`: The number of points to place.

    **Returns:**
    The linearly spaced points, as a numpy array.
    """
    # if we don't enforce 0 or our range doesn't include 0, this just
    # trivially calls np.linspace
    if (not enforce_0) or (low>0) or (high<0):
        return np.linspace(low,high,N)

    # enforcing 0...
    # --------------
    assert N>=3
    
    # split the range into negative and positive parts
    neg, pos = split_range(low, high)
    
    # determine number of points to allocate to each range
    if neg is None:
        # range is like [None, [0, up]]...
        num_neg, num_pos = 0, N
    elif pos is None:
        # range is like [[low, 0], None]...
        num_neg, num_pos = N, 0
    else:
        # calculate length of ranges
        lengths = [-neg[0], pos[1]]
    
        # distribute N+1 points to the ranges
        # (N+1) b/c we'll accidentally duplicate 0
        # We throw out the duplicate
        num_neg = int((N+1)*lengths[0]//sum(lengths))
        num_neg = min(N, max(2, num_neg))
        num_pos = (N+1)-num_neg

    # allocate the points
    if num_neg: neg_pts = np.linspace(neg[0],0,num_neg).tolist()
    else:       neg_pts = []
    
    if num_pos: pos_pts = np.linspace(0,pos[1],num_pos).tolist()
    else:       pos_pts = []

    # check for duplicated 0
    if (num_neg and num_pos) and (neg_pts[-1] == pos_pts[0]):
        # duplicated 0... delete one copy!
        neg_pts = neg_pts[:-1]
    
    return np.asarray(neg_pts+pos_pts)

def symlogspace(low: Number, high: Number, N: int,
                zero: Number = 1e-2) -> np.array:
    """
    **Description:**
    Place N points in a range logarithmically. Allow positive and negative
    values.

    WARNING: spacing may NOT be even!!! It'll be logarithmic in the negative
    and positive regions, but the spacing might not match between the two.

    **Arguments:**
    - `low`: The lower bound of the range.
    - `high`: The upper bound of the range.
    - `N`: The number of points to place.
    - `zero`: A small number to treat as approximately 0.

    **Returns:**
    The logarithmically spaced points, as a numpy array.
    """
    # if our range is all non-negative or non-positive, then this is trivial
    if low==0:
        low = zero
    elif high==0:
        high = -zero

    if (low>0) or (high<0):
        return np.geomspace(low,high,N)

    # the range spans negative and positive values
    # --------------------------------------------
    # split the range into negative and positive parts
    neg, pos = split_range(low, high)
    neg[1] = -zero
    pos[0] = +zero
    
    # determine number of points to allocate to each range
    # (N-1) b/c we'll enforce that 0 arises
    lengths = [np.log(-neg[0])-np.log(-neg[1]),
               np.log(+pos[1])-np.log(+pos[0])]

    num_neg = int((N-1)*lengths[0]//sum(lengths))
    num_pos = (N-1)-num_neg
    
    # allocate the points
    neg_pts = np.geomspace(neg[0],neg[1],num_neg).tolist()
    pos_pts = np.geomspace(pos[0],pos[1],num_pos).tolist()
    
    # return with 0 input
    return np.asarray(neg_pts+[0]+pos_pts)

# misc
# ----
def unit(i, dim, as_np_arr=False):
    """
    **Description:**
    Calculates the ith unit vector in dim space.

    **Arguments:**
    - `i` *(int)*: The location of the 1.
    - `dim` *(int)*: The dimension of the space (length of vector).

    **Returns:**
    *(list)*: The ith unit vector

    **Examples:**
    >>> unit(4, 10)
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    >>> unit(4, 10, as_np_arr=True).tolist()
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    """
    vec = [1 if i==j else 0 for j in range(dim)]

    if as_np_arr:
        return np.asarray(vec)
    else:
        return vec

def midpt(pt1, pt2):
    """
    **Description:**
    Calculates the mid point between two input points.

    **Arguments:**
    - `pt1` *(array_like)*: One of the points.
    - `pt2` *(array_like)*: The other point.

    **Returns:**
    *(array_like)*: (pt1+pt2)/2

    **Examples:**
    >>> midpt([1,2,3],[1,1,1])
    [1.0, 1.5, 2.0]
    """
    return [(x+y)/2 for x,y in zip(pt1,pt2)]

def to_base10(c, B):
    """
    **Description:**
    Converts a number given in components w.r.t. some bases to a number in base
    10.

    **Arguments:**
    - `c` *(array_like)*: A list of the components.
    - `B` *(array_like)*: A list of the bases.

    **Returns:**
    *(numeric)*: The number in base-10
    """
    result = 0
    multiplier = 1
    for c_i, B_i in zip(reversed(c), reversed(B)):
        result += int(c_i) * multiplier
        multiplier *= B_i
    return result

#@njit
#def to_base10_numba(c, B):
#    """
#    Optimized version of to_base10 using Numba.
#
#    Needs numpy inputs.
#
#    Arguments:
#    - `c` (array_like): A list of the components.
#    - `B` (array_like): A list of the bases.
#    
#    Returns:
#    (numeric): The number in base-10
#    """
#    result = 0
#    multiplier = 1
#    for i in range(len(c) - 1, -1, -1):
#        result += c[i] * multiplier
#        multiplier *= B[i]
#    return result

def from_base10(n, B):
    """
    **Description:**
    Split a number in base 10 to components components w.r.t. some bases.

    **Arguments:**
    - `n` *(numeric)*: The number in base 10.
    - `B` *(array_like)*: A list of the bases.

    **Returns:**
    *(list)*: The bases
    """
    c = []
    for B_i in reversed(B):
        c.append(n % B_i)
        n //= B_i
    return list(reversed(c))

# sets
# ----
def add_return(the_set, the_ele):
    """
    **Description:**
    Add an element to a set *and return the set*.

    **Arguments:**
    - `the_set` *(set)*: The set.
    - `the_ele` *(any)*: The element

    **Returns:**
    *(set)* the_set, with the_ele added

    **Examples:**
    >>> add_return(set(),5)
    {5}
    """
    the_set.add(the_ele)
    return the_set

# loading/saving zipped pickle files
# ----------------------------------
def load_zipped_pickle(fname,path=data_path):
    r"""
    **Description:**
    Loads zipped pickle files.
    
    Custom/atypical classes may fail to load.

    Args:
        filen (string): Filename.

    Returns:
        _ (_): Data from file.
    """
    if '.' not in fname:
        fname += '.p'
    file = os.path.join(path,fname)
    
    if os.path.isfile(file):
        with gzip.open(file, 'rb') as f:
            loaded_object = pickle.load(f)

        return loaded_object
    else:
        return None

def save_zipped_pickle(obj, fname, path=data_path, protocol=pickle.DEFAULT_PROTOCOL):
    r"""
    **Description:**
    Saves zipped pickle files.
    
    Note the default path!

    Args:
        obj (any): Object to be saved.
        filen (string): Filename.
        protocol (int, optional): Protocol to use for saving the file. Defaults to -1.
        
    """
    if '.' not in fname:
        fname += '.p'
    file = os.path.join(path,fname)

    with gzip.open(file, 'wb') as f:
        pickle.dump(obj, f, protocol)

# loading/saving from SQL databases
# ---------------------------------
class NpEncoder(json.JSONEncoder):
    # from stackoverflow 50916422
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def create_connection(fname, path=data_path):
    """
    **Description:**
    Create an SQL connection.

    **Arguments:**
    - `fname` *(string)*: The database file name. NO EXTENSION.
    - `path` *(string, optional)*: The path to the database file.

    **Returns:**
    Nothing.
    """
    file = os.path.join(path,fname+'.db')
    return sqlite3.connect(file)

def create_db(connection, dbname):
    """
    **Description:**
    Create an SQL database file if it doesn't exist

    **Arguments:**
    - `connection` *(sqlite3.Connection)*: Connection to db file.
    - `dbname` *(string)*: The database file name.

    **Returns:**
    Nothing.
    """
    with connection:
        connection.execute('CREATE TABLE IF NOT EXISTS '+dbname+'(key UNIQUE, value)')

def get_from_db(connection, dbname, key):
    """
    **Description:**
    Get the value associated to a given key in the database.

    **Arguments:**
    - `connection` *(sqlite3.Connection)*: Connection to db file.
    - `dbname` *(string)*: The database file name.
    - `key` *(misc)*: The key. Can be mutable.

    **Returns:**
    *(misc)* The item stored under key. If no such item, returns None.
    """
    serialized_key = json.dumps(key, cls=NpEncoder)
    
    with connection:
        row = connection.execute('SELECT value FROM '+dbname+' WHERE key = ?', (serialized_key,)).fetchone()
        return None if row is None else json.loads(row[0])

def put_into_db(connection, dbname, key, value):
    """
    **Description:**
    Store a value associated indexed by a given key in the database.

    **Arguments:**
    - `connection` *(sqlite3.Connection)*: Connection to db file.
    - `dbname` *(string)*: The database file name.
    - `key` *(misc)*: The key. Can be mutable.
    - `value` *(misc)*: The value. Can be mutable.

    **Returns:**
    Nothing
    """
    serialized_key = json.dumps(key, cls=NpEncoder)
    serialized_value = json.dumps(value, cls=NpEncoder)

    with connection:
        connection.execute('INSERT OR REPLACE INTO '+dbname+'(key, value) VALUES (?, ?)', (serialized_key, serialized_value))

def close_db(connection):
    """
    **Description:**
    Close an SQL connection.

    **Arguments:**
    - `connection` *(sqlite3.Connection)*: Connection to db file.

    **Returns:**
    Nothing.
    """
    connection.close()

# testing
# =======
if __name__ == "__main__":
    import doctest
    doctest.testmod()
