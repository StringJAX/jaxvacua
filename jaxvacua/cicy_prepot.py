# ==============================================================================
# This code is written by Andreas Schachner.
#
# If any questions arise, please feel free to reach out to me (Andreas) either at
# andreas.schachner@gmx.net or at as3475@cornell.edu .
# ==============================================================================
#
# ------------------------------------------------------------------------------
# This file holds functions for obtaining topological data for CICYs.
# ------------------------------------------------------------------------------


import os, time, sys, shutil, h5py
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Other miscellaneous libraries
import numpy as np
from math import pi as Pi
import sympy
from sympy import sympify,diff,var,expand
from itertools import *
import pandas as pd
from scipy.special import zeta

# import packages
import jax.numpy as jnp
from jax import jit
import jax
from jax import Array
#from jax.typing import ArrayLike
from numpy.typing import ArrayLike
from functools import partial

#Necessary to have the right precision below!!!
from jax import config
config.update("jax_enable_x64", True)

from .util import *


CICY_dir=os.path.dirname(os.path.realpath(__file__))+"/models/CICY"


def load_cicy_data():
    r"""
    **Description:**
    Loads CICY data from pickled dataframe.
    
    Args:
        None
        
    Returns:
        (pd.DataFrame): Dataframe with CICY data.
    """

    return load_zipped_pickle(CICY_dir+"/CICY_dataframe.p")


def read_CICY_input():
    r"""
    
    **Description:**
    Reads CICY configuration matrices from h5 file.
    
    Args:
        None
        
    Returns:
        (np.array): Array with CICY configuration matrices.
    
    """

    f=h5py.File(CICY_dir+"/AllCICYsdim3_v02.h5",'r')
    
    a_group_key = list(f.keys())[0]
    data=[list(f[a_group_key])]
    ya=np.array(data[0])
    
    return ya
    

def sliced(t):
    r"""
    **Description:**
    Slices configuration matrix to remove zero rows and columns.
    
    Args:
        (np.array): Configuration matrix.
        
    Returns:
        (np.array): Sliced configuration matrix.
    
    """
    # slices list of degrees for input
    index2 = 12
    index1 = 16
    for i in range(len(t[0,:])):
        collum = np.sum(t[:,i])
        if collum == 0:
            index1 = i
            break
    for j in range(len(t)):
        row = np.sum(t[j,:])
        if row == 0:
            index2 = j
            break
    return t[:index2,:index1]

def deg(nr,ya):
    r"""
    **Description:**
    Returns degrees of CICY number.
    
    Args:
        (int): CICY number.
        (np.array): Array with CICY configuration matrices.
        
    Returns:
        (list): List of degrees.
    
    """
    # returns configuration matrix for CICY number
    Am = ya[nr-1].astype(int)
    Am = sliced(Am)
    Dl = Am[:,1:]
    d = Dl.tolist()
    return d



def prepot(nr):
    r"""
    **Description:**
    Reads prepotential from .txt file and returns sympy function.
    
    Args:
        (int): CICY number.
        
    Returns:
        (sympy expression): Prepotential as sympy expression.
    
    """
    # reads prepotential from .txt file and returns sympy function
    path = CICY_dir+"/DATA/CICYs/CICY_"+str(nr)+".txt"
    with open(path) as f:
        contents = f.read()
        
    return sympify(contents)


def extract_cpl(F,H):
    r"""
    **Description:**
    Extracts triple intersection numbers from prepotential.
    
    Args:
        (sympy expression): Prepotential as sympy expression.
        (int): Hodge number :math:`h^{1,1}`.
        
    Returns:
        (np.array): Triple intersection numbers as numpy array.
    
    """
    # extracts coupling from prepotential
    tt = list(var('t_%d' % i) for i in (range(H)))
    
    K = [[[0 for k in range(H)] for j in range(H)] for i in range(H)]
    for i in range(H):
        for j in range(H):
            for k in range(H):
                K[i][j][k] = float((-1)*diff(diff(diff(F,tt[i]),tt[j]),tt[k]))
    return np.array(K)

def extract_a(F,H):
    r"""
    
    **Description:**
    Extracts a-matrix from prepotential.
    
    Args:
        (sympy expression): Prepotential as sympy expression.
        (int): Hodge number :math:`h^{1,1}`.
        
    Returns:
        (np.array): a-matrix as numpy array.
    
    """
    tt = list(var('t_%d' % i) for i in (range(H)))
    a = [[0 for j in range(H)] for j in range(H)]
    for i in range(H):
        for j in range(H):
            a[i][j] = float(2*diff(diff(F,tt[i]),tt[j]).subs([(tt[k],0) for k in range(H)]))
    return np.array(a)
    

def extract_euler(F,H):
    r"""
    **Description:**
    Extracts Euler number from prepotential.
    
    Args:
        (sympy expression): Prepotential as sympy expression.
        (int): Hodge number :math:`h^{1,1}`.
        
    Returns:
        (float): Euler number as float.
    
    """
    
    # extracts Euler number from prepotential
    tt = list(var('t_%d' % i) for i in (range(H)))
    
    euler = F.subs([(tt[i],0) for i in range(H)])

    euler = float(expand(euler * 2 * (2*sympy.pi*sympy.I)**3 ))
    #zeta = Function('zeta')(3)
    
    if euler == 0.:
        return float(0)
    else:
        euler = euler/zeta(3)

        if np.abs(euler-np.rint(euler).astype(int))>1e-5:
            raise ValueError("Euler number for CICY not integer!")

        return np.rint(euler).astype(float)
    
def extract_c2(F,H):
    r"""
    **Description:**
    Extracts second Chern class from prepotential.
    
    Args:
        (sympy expression): Prepotential as sympy expression.
        (int): Hodge number :math:`h^{1,1}`.
        
    Returns:
        (np.array): Second Chern class as numpy array.
    
    """
    # extract c2 component
    tt = list(var('t_%d' % i) for i in (range(H)))
    c2 = []
    for j in range(H):
        val = float(diff(F,tt[j],1).subs([(tt[i],0) for i in range(H)]))
        c2.append(val)
        
    return np.array(c2)

def cicy_input(nr):
    r"""
    **Description:**
    Prepares input data for CICY number.
    
    Args:
        (int): CICY number.
        
    Returns:
        (dict): Dictionary with CICY data.
        (dict or None): Dictionary with instanton data or None if not available.
    """
    
    ya = read_CICY_input()
    
    F = prepot(nr)
    
    H = len(deg(nr,ya))
    
    input_dic = {
      "intersection numbers": (-1)*extract_cpl(F,H),
      "a_matrix": extract_a(F,H),
      "second chern": extract_c2(F,H)*24.,
      "chi": extract_euler(F,H),
      "h12": H
       }

    path = CICY_dir+"/DATA/"
    model_ID = nr
    h12 = H

    # 7447 -> Calabi–Yau threefold of Hulek–Verrill. See https://arxiv.org/pdf/2404.12422.
    if nr==7447:
        input_dic["name"] = "Hulek–Verrill (2404.12422)"
    elif nr==14:
        input_dic["name"] = "Schoen (1708.07907)"
    elif nr==7890:
        input_dic["name"] = "mirror quintic"
    else:
        input_dic["name"] = ""

    gv_file = path+f"CICY_GVs/CICY-H11={h12}/{model_ID}.p"
    kahler_file = path+f"CICY_Kahler_cones/CICY-H11={h12}/{model_ID}.p"
    mori_file = path+f"CICY_Mori_cones/CICY-H11={h12}/{model_ID}.p"


    if os.path.isfile(kahler_file):
        input_dic["generators KC"] = load_zipped_pickle(kahler_file)
    else:
        input_dic["generators KC"] = np.array([])

    if os.path.isfile(mori_file):
        input_dic["generators MC"] = load_zipped_pickle(mori_file)
    else:
        input_dic["generators MC"] = np.array([])

    gws = {tuple(np.zeros(h12)):0}

    if os.path.isfile(gv_file):
        gvs = load_zipped_pickle(gv_file)
        
        grading_vector = np.ones(h12)
        gv_charges = np.array(list(gvs.keys()))
        gv_degrees = gv_charges@grading_vector

        if np.all(gv_degrees>0):
            grading_vector = np.ones(h12)
        else:
            warnings.warn("Could not determine grading vector for CICY!")
            grading_vector = None
    else:
        gvs = None
        grading_vector = None

    if gvs is None:
        instanton_data = None
    else:
        instanton_data = {"gvs":gvs,"gws":gws,"grading_vector":grading_vector}   

    cicy_data = load_zipped_pickle(CICY_dir+"/CICY_dataframe.p")

    input_dic["extra_data"] = cicy_data[cicy_data["CICY ID"]==model_ID]
    
    return input_dic, instanton_data
 



