import csv
import numpy as np
import pandas as pd
# import geopandas as gpd
from IPython.display import Image
# from shapely.geometry import Point, Polygon
from math import factorial
import scipy
import scipy.signal
import os, os.path

from datetime import date
import datetime
import time

from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
from patsy import cr

# from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sb

import sys
import io

# to move files from one directory to another
import shutil


import yfinance as yf
from nasdaq_stock import nasdaq_stock as nasdaq_stock
import requests

from pylab import rcParams
##########################################################################################


def reconstruction(Y, C, M, h, w, image_index):
    #
    # https://towardsdatascience.com/eigenfaces-recovering-humans-from-ghosts-17606c328184
    #
    n_samples, n_features = Y.shape
    weights = np.dot(Y, C.T)
    centered_vector=np.dot(weights[image_index, :], C)
    recovered_image=(M+centered_vector).reshape(h, w)
    return recovered_image


def pca(array_of_images, n_pc):
    #
    # https://towardsdatascience.com/eigenfaces-recovering-humans-from-ghosts-17606c328184
    #
    n_samples, h, w = array_of_images.shape

    X = array_of_images.reshape(n_samples, h*w)
    n_samples, n_features = X.shape
    mean = np.mean(X, axis=0)
    centered_data = X-mean
    U, S, V = np.linalg.svd(centered_data)
    components = V[:n_pc]
    projected = U[:,:n_pc]*S[:n_pc]
    
    return projected, components, mean, centered_data

##########################################################################################

def convert_matrix_to_tall_Vector(matrix):
    """
    Convert a matrix (2D numpy array) to a long 1D array.
    Where we go row by row.
    
    input:  2D numpy array
    output: 1D numpy array
    
    """
    long_vect = np.asarray(matrix).reshape(-1)
    # long_vect = matrix.ravel() # does the same thing.
    return(long_vect)

##########################################################################################

def find_jump(noisy_jump_vect):
    H = create_upperTri_matrix_Ones(row_number = len(noisy_jump_vect))
    H_dot_Noise = np.dot(H, noisy_jump_vect)
    jump_idx = np.where(H_dot_Noise == np.amax(H_dot_Noise))[0][0]
    return (jump_idx)



##########################################################################################

def create_upperTri_matrix_Ones(row_number):
    """
    input:   row_number: Number of rows/columns of a square matrix

    output:  an upper triangular matrix with all entries being 1 and 0.
    """
    upper = np.triu(np.ones(row_number), k=0)
    return(upper)

##########################################################################################

def create_noisy_stepFunc(dim = 20000, jump = 100, variance = 10):
    """
    input: 
             dim : Length of the vector
             jump: the entry at which the jump occurs
             variance: variance of a normal distribution to choose noise from.
             
    output:  A vector of length [dim] whose first [jump] entries are -1s.
    """

    first_part_size = jump
    second_part_size = dim - jump
    
    first_part = -1 * np.ones(first_part_size)
    second_part = np.ones(second_part_size)
    
    first_part = first_part  + np.random.normal(loc = -1.0, scale = variance, size = first_part_size)
    second_part= second_part + np.random.normal(loc = 1.0 , scale = variance, size = second_part_size)
    
    noisy_signal = np.concatenate((first_part, second_part))
    return(noisy_signal)

##########################################################################################

def create_true_stepFunc(dim=20000, jump=100):
    """
    input: 
             dim : Length of the vector
             jump: the entry at which the jump occurs.
             
    output: A vector of length [dim] whose first [jump] entries are -1s.
    """
    true_signal = np.ones(dim)
    true_signal[0:jump] = -1
    return(true_signal)


