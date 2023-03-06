import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

data_npy_file = 'data/PB_data.npy'
trained_GMM_file_1 = 'data/GMM_params_phoneme_01_k_06.npy'
trained_GMM_file_2 = 'data/GMM_params_phoneme_02_k_06.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.column_stack((f1, f2))

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong
# to phoneme 2.
X_phonemes_1_2 = X_full[np.logical_or(phoneme_id==1, phoneme_id==2)]

# as dataset X, we will use only the samples of phoneme 1 and 2
X = X_phonemes_1_2.copy()

min_f1, max_f1 = X[:,0].min().astype(int), X[:,0].max().astype(int)
min_f2, max_f2 = X[:,1].min().astype(int), X[:,1].max().astype(int)
N_f1 = max_f1 - min_f1 + 1
N_f2 = max_f2 - min_f2 + 1

print(f'f1 range: {min_f1}-{max_f1} | {N_f1} points')
print(f'f2 range: {min_f2}-{max_f2} | {N_f2} points')

# Create a custom grid of shape N_f1 x N_f2
# The grid will span all the values of (f1, f2) pairs, between [min_f1, max_f1] on f1 axis, and between [min_f2, max_f2]
# on f2 axis
custom_grid = np.transpose([np.tile(np.arange(min_f1, max_f1+1), len(np.arange(min_f2, max_f2+1))),
                              np.repeat(np.arange(min_f2, max_f2+1), len(np.arange(min_f1, max_f1+1)))])

# load GMM params
trainedGMM1 = np.load(trained_GMM_file_1, allow_pickle=True)
trainedGMM2 = np.load(trained_GMM_file_2, allow_pickle=True)

# classify each point [i.e., each (f1, f2) pair] of the custom grid, to either phoneme 1, or phoneme 2,
# using the two trained GMMs
# Do predictions, using GMM trained on phoneme 1, on custom grid
gmm1 = trainedGMM1[0]
pi1 = trainedGMM1[1]
mu1 = trainedGMM1[2]
sigma1 = trainedGMM1[3]
likelihood1 = np.zeros((custom_grid.shape[0], k))

for i in range(k):
    likelihood1[:, i] = pi1[i] * multivariate_normal.pdf(custom_grid, mean=mu
