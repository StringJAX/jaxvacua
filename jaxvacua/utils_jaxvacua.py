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