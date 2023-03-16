import numpy as np
import os
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

data_npy_file = 'data/PB_data.npy'

trained_GMM_file_1 = 'data/GMM_params_phoneme_01_k_06.npy'
trained_GMM_file_2 = 'data/GMM_params_phoneme_02_k_06.npy'

data = np.load(data_npy_file, allow_pickle=True)
phoneme_id = data['phoneme_id']
f1 = data['f1']
f2 = data['f2']

X_full = np.column_stack((f1, f2))

k = 6

m1 = np.where(phoneme_id == 1)[0]
m2 = np.where(phoneme_id == 2)[0]
ids = np.concatenate((m1, m2))
X_phonemes_1_2 = X_full[ids]

X = X_phonemes_1_2.copy()

min_f1, max_f1 = int(np.min(X[:,0])), int(np.max(X[:,0]))
min_f2, max_f2 = int(np.min(X[:,1])), int(np.max(X[:,1]))
N_f1, N_f2 = max_f1 - min_f1, max_f2 - min_f2
print(f'f1 range: {min_f1}-{max_f1} | {N_f1} points')
print(f'f2 range: {min_f2}-{max_f2} | {N_f2} points')

grid = np.zeros((N_f2, N_f1))
M = np.zeros((N_f2, N_f1))

mu1, mu2 = np.load(trained_GMM_file_1, allow_pickle=True)['mu'], np.load(trained_GMM_file_2, allow_pickle=True)['mu']
s1, s2 = np.load(trained_GMM_file_1, allow_pickle=True)['s'], np.load(trained_GMM_file_2, allow_pickle=True)['s']
p1, p2 = np.load(trained_GMM_file_1, allow_pickle=True)['p'], np.load(trained_GMM_file_2, allow_pickle=True)['p']

grid_i, grid_j = np.arange(min_f1, max_f1), np.arange(min_f2, max_f2)

for i in range(N_f1):
    for j in range(N_f2):
        point = np.array([grid_i[i], grid_j[j]]).reshape(1, -1)
        p1x = get_predictions(point, k, mu1, s1, p1)
        p2x = get_predictions(point, k, mu2, s2, p2)
        M[j, i] = 1.0 if p2x > p1x else 0.0

plot_gaussians(X, mu1, s1, p1, mu2, s2, p2)
plt.imshow(M, origin='lower', extent=[min_f1, max_f1, min_f2, max_f2], cmap='jet', alpha=0.5)
plt.xlabel('f1')
plt.ylabel('f2')
plt.title('Decision boundary for phoneme classification')
plt.colorbar()
plt.savefig(os.path.join('figures', 'phoneme_classification.png'))
plt.show()
