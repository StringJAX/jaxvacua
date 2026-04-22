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

from .util import save_zipped_pickle, load_zipped_pickle



# Some global variables
home_dir=os.path.dirname(os.path.realpath(__file__))
files_dir=home_dir+"/models"

def save_model_data(data,fname,model_ID,h12):
    r"""
    **Description:**
    Saves model data in a zipped pickle file.
    
    Args:
       data (array/dict): Data to be stored in file.
       fname (string): Filename of file to be read.
       model_ID (int): Model ID used to create subdirectory for saving data.
       h12 (int): Hodge number h^(1,2) of the model used to create subdirectory for saving data.
       
    Returns:
        None

    """

    if model_ID is None:
        raise ValueError("Please provide `model_ID` to save model data!")

    dir_h12 = files_dir+'/KS/h12_'+str(h12)+'/'

    if not os.path.isdir(dir_h12):
        directory = files_dir+'/KS/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        os.mkdir(dir_h12)

    file= dir_h12+'Model_'+str(model_ID)+'/'

    if not os.path.isdir(file):
        os.mkdir(file)
        
    filename = file+fname

    if os.path.isfile(filename):
        print(f"Model ID already exists! File `{fname}` might be overwritten!")

        asking_input = input("Do You Want To Continue? [y/n]")
        if asking_input  == "y":
            save_zipped_pickle(data, filename, protocol=-1)
    else:
        save_zipped_pickle(data, filename, protocol=-1)
        
def flatten_func(obj):
    r"""
    **Description:**
    Flattens the object for use in JAX transformations.
    This function is used by JAX to convert the object into a tuple of children and auxiliary data.
    The children contain all arrays and pytrees, while the auxiliary data contains static, hashable data.
    
    Args:
        obj (myclass): Instance of the object to be flattened.
        
    Returns:
        tuple: A tuple containing the children and auxiliary data.
    """
    # Based on https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.register_pytree_node.html
    
    children = tuple(list(obj.__dict__.values()))
    #children = (obj.h11, obj.intnums2)  # children must contain arrays & pytrees
    
    # Using aux_data to store model attribtues
    aux_data = tuple(list(obj.__dict__.keys()))  # aux_data must contain static, hashable data.
    
    static_keys = ["h11","h12","model_ID","dimension_H3","_dimension_H3_tot","model_type","n_fluxes","gauge_choice","prange","maximum_degree","D3_tadpole","conifold_limits","n_conifolds","ncf","nmax"]
    
    children = []
    aux_data = []
    static = []
    for key, value in obj.__dict__.items():
        if type(value) in [str,bool] or key in static_keys:
            static.append((key, value))
        else:
            children.append(value)
            aux_data.append(key)
            
    for x in static:
        aux_data.append(x)
    
    children = tuple(children)
    aux_data = tuple(aux_data)
    
    return (children, aux_data)

def unflatten_func_class(aux_data, children, myclass):
    r"""
    **Description:**
    Unflattens the object for use in JAX transformations.
    This function is used by JAX to reconstruct the object from a tuple of children and auxiliary data.
    The children contain all arrays and pytrees, while the auxiliary data contains static, hashable data.
    
    Args:
        aux_data (tuple): Auxiliary data containing static, hashable data.
        children (tuple): Children containing all arrays and pytrees.
        myclass (type): The class of the object to be reconstructed.
        
    Returns:
        myclass: Reconstructed object of the specified class.
    """
    # Based on https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.register_pytree_node.html
    
    # Here we avoid `__init__` because it has extra logic we don't require:
    obj = object.__new__(myclass)
    
    # Using aux_data to store model attribtues
    for i,attr in enumerate(aux_data):
        if type(attr) in [tuple]:
            object.__setattr__(obj,attr[0],attr[1])
        else:
            object.__setattr__(obj,attr,children[i])
        
    return obj