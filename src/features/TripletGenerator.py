import random
import pandas as pd
import numpy as np
from scipy.spatial import distance


N_RANDOM_OBS = 200
N_POTENTIAL_EL = 5
choose_type = 'neg_hard'


class TripletGenerator:
    
    def __init__(self, n_jobs=5):
        self.n_jobs = n_jobs
        
    def get_observation(self, X, y, label):
        y_labels_indices = np.array(list(range(y.shape[0])))
        y_label = y_labels_indices[y==label]
        idx = random.choice(y_label)
        return X[idx]
    
    def choose_neg_from_n(self, X, y, anchor_x, anchor_y):
        """
        choose the neg label with attention on the closest exaples
        """
        
        X = X[y!=anchor_y]
        y = y[y!=anchor_y]
        indices = np.random.choice(X.shape[0], N_RANDOM_OBS)
        X, y = X[indices], y[indices]
        
        y = np.array(y)
        d = np.array([distance.euclidean(anchor_x, ex) for ex in X])
        candidates = y[np.argpartition(d, N_POTENTIAL_EL)[:N_POTENTIAL_EL]]
        
        random_list = [random.random() for i in range(N_POTENTIAL_EL)]
        s = sum(random_list)
        random_list = [i/s for i in random_list ]
        choose_probabilities = sorted(random_list)[::-1]
        
        neg_y = np.random.choice(candidates, 1, p=choose_probabilities)
        return neg_y
    
    def choose_neg_hard(self, X, y, anchor_x, anchor_y):
        """
        choose the neg label with attention on the closest exaples
        """
        
        X = X[y!=anchor_y]
        y = y[y!=anchor_y]
        indices = np.random.choice(X.shape[0], N_RANDOM_OBS)
        X, y = X[indices], y[indices]
        
        y = np.array(y)
        d = np.array([distance.euclidean(anchor_x, ex) for ex in X])
        neg_y = y[np.argmin(d)]
        return neg_y

    def get_triplet(self, X, y, n_classes=10):
        # choose random class
        anchor_y = random.choice(y)
        anchor_x = self.get_observation(X, y, anchor_y)
        pos_x    = self.get_observation(X, y, anchor_y)
        
        if choose_type=='neg_hard':
            neg_y = self.choose_neg_hard(X, y, anchor_x, anchor_y)
        elif choose_type=='neg_worst_rand':
            neg_y = self.choose_neg_from_n(X, y, anchor_x, anchor_y)
            
        neg_x = self.get_observation(X, y, neg_y)
        
        return anchor_x, pos_x, neg_x        
    
    def generate_triplets(self, X, y, batch_size):
        while 1:
            list_a = []
            list_p = []
            list_n = []

            for i in range(batch_size):
                a, p, n = self.get_triplet(X, y, n_classes=len(set(y)))
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