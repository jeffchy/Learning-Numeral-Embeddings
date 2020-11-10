import pickle
import random
import numpy as np
import os
from sklearn.mixture import GaussianMixture
import sys
sys.path.append("../")
from utils.number_handler import to_numeral

if __name__ == '__main__':

    # nc path
    nc = pickle.load(open('../data/wikipedia/preprocess0.05Bnotable/NumeralAsNumeral/nc.dat', 'rb'))
    gmm_save_dir = 'gmm'
    if not os.path.exists(gmm_save_dir):
        os.makedirs(gmm_save_dir)

    random.seed(100)
    # unfold and shuffle nc data
    data = []
    for k, v in nc.items():
        if to_numeral(k) == None:
            print('invalid numeral {}'.format(k))
        else:
            data += [[to_numeral(k)]] * v

    print('total number of different numerals: ', len(nc))
    print('total number of numeral samples: ', len(data))

    random.shuffle(data)
    data = np.array(data).reshape(-1, 1)

    prototypes = pickle.load(open('../data/wikipedia/preprocess0.05Bnotable/NumeralAsNumeral/som/prototypes-50-0.6-1.0.dat', 'rb'))

    print(prototypes.shape)

    mus = prototypes
    min_sigma = 1e-6

    diff = np.abs(data.reshape(len(data)) - mus[:, np.newaxis])
    amin = np.argmin(diff, axis=0)

    K = len(prototypes)
    clusters = [[0] for i in range(K)]
    for i in range(len(data)):
        clusters[amin[i]].append(data[i])

    means = np.array([np.mean(i) for i in clusters]).reshape(-1, 1)

    covs = np.array([np.std(i) if len(i) > 1 else min_sigma for i in clusters]).reshape(-1, 1, 1)
    precision = np.linalg.inv(covs)

    weights = np.array([len(c) for c in clusters])
    weights = weights / np.sum(weights)

    gmm = GaussianMixture(K, max_iter=100, n_init=1, verbose=10, means_init=means, precisions_init=precision, weights_init=weights)

    gmm.fit(data)

    pickle.dump(gmm, open(os.path.join(gmm_save_dir, 'gmm-{}-fp.dat'.format(K)), 'wb'))
    print('means: {} \n sigma: {}\n, weights: {}'.format(gmm.means_, gmm.covariances_, gmm.weights_))
    data_points = np.array(list(nc.keys()), dtype=np.float).reshape(-1, 1)
    posterior = gmm.predict_proba(data_points)
    pickle.dump(posterior, open(os.path.join(gmm_save_dir, 'gmm_posterior-{}-fp.dat'.format(K)), 'wb'))
    print('...Saving trained GMMs objects')