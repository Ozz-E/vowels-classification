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

# File that contains the data
data_npy_file = 'data/PB_data.npy'
trained_GMM_file_1 = 'data/GMM_params_phoneme_01_k_03.npy'
trained_GMM_file_2 = 'data/GMM_params_phoneme_02_k_03.npy'
# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)
trainedGMM1 = np.load(trained_GMM_file_1, allow_pickle=True)
trainedGMM1 = np.ndarray.tolist(trainedGMM1)
trainedGMM2 = np.load(trained_GMM_file_2, allow_pickle=True)
trainedGMM2 = np.ndarray.tolist(trainedGMM2)
# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
for i in range(X_full.shape[0]):
    X_full[i, 0] = f1[i]
    X_full[i, 1] = f2[i]
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 3

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and
# samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each
# column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full
X_phonemes_1_2 = np.zeros((np.sum(np.add(phoneme_id == 1, phoneme_id == 2)), 2))
m1 = np.where(phoneme_id == 1)
m2 = np.where(phoneme_id == 2)
ids = np.append(m1[0], m2[0], axis=0)

for n in range(X_phonemes_1_2.shape[0]):
    X_phonemes_1_2[n] = X_full[ids[n]]
# X_phonemes_1_2 = ...

########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar
# variable named "accuracy"

mu1 = trainedGMM1['mu']
mu2 = trainedGMM2['mu']

s1 = trainedGMM1['s']
s2 = trainedGMM2['s']

p1 = trainedGMM1['p']
p2 = trainedGMM2['p']


Z1 = get_predictions(mu1, s1, p1, X_phonemes_1_2)
Z2 = get_predictions(mu2, s2, p2, X_phonemes_1_2)
accuracy = 0
true = 0
length = int(len(X_phonemes_1_2) / 2)

for i in range(length):
    if(np.mean(Z1[i]) > np.mean(Z2[i])):
        true += 1

for j in range(length, len(X_phonemes_1_2)):
    if(np.mean(Z1[j]) < np.mean(Z2[j])):
        true += 1

accuracy = (true) / (len(X_phonemes_1_2))*100
########################################/

print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()