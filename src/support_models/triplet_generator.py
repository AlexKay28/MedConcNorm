import random
import pandas as pd
import numpy as np

from multiprocessing import Pool

from scipy.spatial import distance
from scipy.spatial.distance import cdist

N_RANDOM_OBS = 200
N_POTENTIAL_EL = 3
choose_type = 'neg_hard'
DISTANCE_TYPE = 'euclidean'


class TripletGenerator:

    def __init__(self, n_jobs=5):
        self.n_jobs = n_jobs
        self.paired_nodes = []

    def choose_pos_x_hard(self, X, y, anchor_x, anchor_y, n_random_objects=N_RANDOM_OBS, distance_type=DISTANCE_TYPE):
        """
        choose the pos label with attention on the most remote examples
        """
        N_RANDOM_OBS = 500
        X = X[y==anchor_y]
        y = y[y==anchor_y]
        n_random_objects = n_random_objects if n_random_objects < X.shape[0] else X.shape[0]
        indices = np.random.choice(X.shape[0], n_random_objects, replace=False)
        X, y = X[indices], y[indices]
        y = np.array(y)
        if distance_type == 'euclidean':
            d = map(lambda ex: distance.euclidean(anchor_x, ex), X)
        elif distance_type == 'cosine':
            d = map(lambda ex: distance.cosine(anchor_x, ex), X)
        elif distance_type == 'minkowski':
            d = map(lambda ex: distance.minkowski(anchor_x, ex), X)
        elif distance_type == 'chebyshev':
            d = map(lambda ex: distance.chebyshev(anchor_x, ex), X)
        elif distance_type == 'cityblock':
            d = map(lambda ex: distance.cityblock(anchor_x, ex), X)
        else:
            raise KeyError('Unknown distance metric!')
        #print('pos', d.shape, X.shape)
        pos_x = X[np.argmax(d)]
        return pos_x

    def choose_neg_x_hard(self, X, y, anchor_x, anchor_y, n_random_objects=N_RANDOM_OBS, distance_type=DISTANCE_TYPE):
        """
        choose the neg label with attention on the closest exaples
        """
        X = X[y!=anchor_y]
        y = y[y!=anchor_y]
        n_random_objects = n_random_objects if n_random_objects < X.shape[0] else X.shape[0]
        indices = np.random.choice(X.shape[0], n_random_objects, replace=False)
        X, y = X[indices], y[indices]
        y = np.array(y)
        if distance_type == 'euclidean':
            d = map(lambda ex: distance.euclidean(anchor_x, ex), X)
        elif distance_type == 'cosine':
            d = map(lambda ex: distance.cosine(anchor_x, ex), X)
        elif distance_type == 'minkowski':
            d = map(lambda ex: distance.minkowski(anchor_x, ex), X)
        elif distance_type == 'chebyshev':
            d = map(lambda ex: distance.chebyshev(anchor_x, ex), X)
        elif distance_type == 'cityblock':
            d = map(lambda ex: distance.cityblock(anchor_x, ex), X)
        else:
            raise KeyError('Unknown distance metric!')
        #print('neg', d.shape, X.shape)
        neg_x = X[np.argmin(d)]
        return neg_x

    def get_triplet(self, X, y):
        # choose random class
        probs = np.array([y[y==cls].shape[0]/y.shape[0] for cls in y])
        probs = probs/sum(probs)
        anchor_y = np.random.choice(y, p=probs)
        anchor_x_idx = np.random.choice(X[y==anchor_y].shape[0])
        anchor_x = X[y==anchor_y][anchor_x_idx]
        if y[y==anchor_y].shape[0] == 1:
            pos_x = anchor_x
        else:
            pos_x = self.choose_pos_x_hard(X, y, anchor_x, anchor_y)
        neg_x = self.choose_neg_x_hard(X, y, anchor_x, anchor_y)
        return anchor_x, pos_x, neg_x

    def generate_triplets(self, X, y, batch_size):
        while 1:
            list_a = []
            list_p = []
            list_n = []

            for i in range(batch_size):
                a, p, n = self.get_triplet(X, y)
                list_a.append(a)
                list_p.append(p)
                list_n.append(n)

            A = np.array(list_a, dtype='float32')
            P = np.array(list_p, dtype='float32')
            N = np.array(list_n, dtype='float32')

            # a "dummy" label which will come in to our identity loss
            # function below as y_true. We'll ignore it.
            label = np.ones(batch_size)
            yield [A, P, N], label
