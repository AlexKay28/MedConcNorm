import numpy as np
from sklearn.neighbors import LocalOutlierFactor

class NoveltyDetector:

    def __init__(self, n_neighbors=20, contamination=0.1, novelty=True):
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=novelty)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def score_samples(self, X):
        return self.model.score_samples(X)

    def select_novelties(self, X):
        novpred = self.predict(X)
        X_novelties = X[novpred == -1]
        X_not_novelties = X[novpred != -1]
        return X_novelties, X_not_novelties, novpred
