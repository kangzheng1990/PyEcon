# -*- coding:utf-8 -*-
# Import 3rd party Packages for common use -------------------------------------

# Scipy and Numpy
import scipy
from scipy import *
from scipy import linalg
from scipy import optimize
from scipy import stats
from scipy import integrate
from scipy import fftpack
from scipy import sparse
import numpy as np



# Matplotlib and Image
from PIL import Image
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm


# Pandas
import pandas
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import autocorrelation_plot

from openpyxl import load_workbook


# Statsmodels
import statsmodels.formula.api as smf
import statsmodels.tsa.api as tsa
import statsmodels.stats.api as sts

from patsy import dmatrices, dmatrix


# General file management
import pickle
from os import path
import csv



# Dictionary List manuplation ------------------------------------------------------

def dict_sum(lst_of_dict,com_keys=None):
    """
    return a summed dict of common keys in multiple dictionaries
    Each element of key must be able to numerically sum up
    """
    if com_keys==None:
        com_keys=lst_of_dict[0].keys()
            
    newdict={}
    for key in com_keys:
        val=0
        for Dict in lst_of_dict:
            val=val+Dict[key]
        newdict[key]=val
        
    return newdict



def dict_sumprod(lst_of_dict,weight,com_keys=None):
    """
    return a sumproduct dict of common keys in multiple dictionaries
    Each element of key must be able to numerically sum and prod
    """
    if com_keys==None:
        com_keys=lst_of_dict[0].keys()
        
    newdict={}
    Nlst = len(lst_of_dict)
    for key in com_keys:
        val=0
        for i in range(Nlst):
            Dict=lst_of_dict[i]
            val=val+weight[i]*Dict[key]
        newdict[key]=val
        
    return newdict



def dict_mean(lst_of_dict,com_keys=None):
    """
    return a mean dict of common keys in multiple dictionaries
    Each element of key must be able to numerically mean
    """
    if com_keys==None:
        com_keys=lst_of_dict[0].keys()
    
    newdict={}
    Nlst = len(lst_of_dict)
    for key in com_keys:
        val=0
        for Dict in lst_of_dict:
            val=val+(1.0/Nlst)*Dict[key]
        newdict[key]=val
        
    return newdict


    
    

def dict_std(lst_of_dict,com_keys=None):
    """
    return a mean dict of common keys in multiple dictionaries
    Each element of key must be able to numerically mean
    """
    if com_keys==None:
        com_keys=lst_of_dict[0].keys()
    
    newdict={}
    Nlst = len(lst_of_dict)
    for key in com_keys:
        temp=[]
        for Dict in lst_of_dict:
            temp.append(Dict[key])
        newdict[key]=std(array(temp),axis=0)
    
    return newdict




def dict_median(lst_of_dict,com_keys=None):
    """
    return a mean dict of common keys in multiple dictionaries
    Each element of key must be able to numerically mean
    """
    if com_keys==None:
        com_keys=lst_of_dict[0].keys()
    
    newdict={}
    Nlst = len(lst_of_dict)
    for key in com_keys:
        temp=[]
        for Dict in lst_of_dict:
            temp.append(Dict[key])
        newdict[key]=median(array(temp),axis=0)
    
    return newdict










