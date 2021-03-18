import numpy as np


class TripletGenerator:
    
    def __init__(self, n_jobs=5):
        self.n_jobs = n_jobs
        
    def get_observation(self, X, y, label):
        idx = np.random.randint(len(y))
        while y[idx] != label:
            # keep searching randomly
            idx = np.random.randint(len(y))
        return X[idx]

    def get_triplet(self, X, y, n_classes=10):
        # choose random class
        neg = anchor = np.random.randint(n_classes)
        
        # n - negative, should be different class
        while neg == anchor:
            # keep searching randomly
            neg = np.random.randint(n_classes)
        
        # p - positive class, will be example of the same class
        anchor_get = self.get_observation(X, y, anchor)
        pos_get    = self.get_observation(X, y, anchor)
        neg_get    = self.get_observation(X, y, neg)
        return anchor_get, pos_get, neg_get        
    
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