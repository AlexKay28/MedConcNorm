import random
import pandas as pd
import numpy as np
from multiprocessing import Pool

from scipy.spatial import distance
from scipy.spatial.distance import cdist
from src.configs import GENERAL, PREPROCESSING, MODELING

N_RANDOM_OBS = None
N_POTENTIAL_EL = 3
DISTANCE_TYPE = MODELING['distance_type']


class TripletGenerator:

    def __init__(self, n_jobs=5):
        self.n_jobs = n_jobs
        self.paired_nodes = []

    @staticmethod
    def map_parallel(func, iterable_args, n_jobs=1):
        if n_jobs==1:
            return map(func, iterable_args)
        with Pool(n_jobs) as pool:
            result = pool.starmap(func, iterable_args)
        return result

    @staticmethod
    def corrected_cosine(x, y, corr):
        x, y, corr = np.array(x), np.array(y), np.array(corr)
        corrected_x = x - corr
        corrected_y = y - corr
        return distance.cosine(corrected_x, corrected_y)

    def choose_pos_x_hard(self, X, y, anchor_x, anchor_y, n_random_objects=N_RANDOM_OBS, distance_type=DISTANCE_TYPE):
        """
        choose the pos label with attention on the most remote examples
        """
        X = X[y==anchor_y]
        y = y[y==anchor_y]

        if n_random_objects is not None:
            n_random_objects = n_random_objects if n_random_objects < X.shape[0] else X.shape[0]
        else:
            n_random_objects = X.shape[0]

        indices = np.random.choice(X.shape[0], n_random_objects, replace=False)
        X, y = X[indices], y[indices]
        y = np.array(y)
        if distance_type == 'euclidean':
            d = self.map_parallel(
                lambda x, y: distance.euclidean(x, y)/distance.cosine(x, y),
                [(anchor_x, ex) for ex in X])
        elif distance_type == 'cosine':
            d = self.map_parallel(distance.cosine, [(anchor_x, ex) for ex in X])
        elif distance_type == 'minkowski':
            d = self.map_parallel(
                lambda x, y: distance.minkowski(x, y)/distance.cosine(x, y),
                [(anchor_x, ex) for ex in X])
        elif distance_type == 'chebyshev':
            d = self.map_parallel(
                lambda x, y: distance.chebyshev(x, y)/distance.cosine(x, y),
                [(anchor_x, ex) for ex in X])
        elif distance_type == 'cityblock':
            d = self.map_parallel(
                lambda x, y: distance.cityblock(x, y)/distance.cosine(x, y),
                [(anchor_x, ex) for ex in X])
        else:
            raise KeyError('Unknown distance metric!')
        #print('pos', d.shape, X.shape)
        pos_x = X[np.argmax(d)]
        return pos_x

    def choose_neg_x_hard(self, X, y, anchor_x, pos_x, anchor_y, n_random_objects=N_RANDOM_OBS, distance_type=DISTANCE_TYPE):
        """
        choose the neg label with attention on the closest exaples
        """
        X = X[y!=anchor_y]
        y = y[y!=anchor_y]

        if n_random_objects is not None:
            n_random_objects = n_random_objects if n_random_objects < X.shape[0] else X.shape[0]
        else:
            n_random_objects = X.shape[0]

        indices = np.random.choice(X.shape[0], n_random_objects, replace=False)
        X, y = X[indices], y[indices]
        y = np.array(y)
        if distance_type == 'euclidean':
            d = self.map_parallel(
                lambda x, y: distance.euclidean(x, y)/self.corrected_cosine(pos_x, y, anchor_x),
                [(anchor_x, ex) for ex in X])
        elif distance_type == 'cosine':
            d = self.map_parallel(distance.cosine, [(anchor_x, ex) for ex in X])
        elif distance_type == 'minkowski':
            d = self.map_parallel(
                lambda x, y: distance.minkowski(x, y)/self.fixed_cosine(pos_x, y, anchor_x),
                [(anchor_x, ex) for ex in X])
        elif distance_type == 'chebyshev':
            d = self.map_parallel(
                lambda x, y: distance.chebyshev(x, y)/self.fixed_cosine(pos_x, y, anchor_x),
                [(anchor_x, ex) for ex in X])
        elif distance_type == 'cityblock':
            d = self.map_parallel(
                lambda x, y: distance.cityblock(x, y)/self.fixed_cosine(pos_x, y, anchor_x),
                [(anchor_x, ex) for ex in X])
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
        neg_x = self.choose_neg_x_hard(X, y, anchor_x, pos_x, anchor_y)
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
