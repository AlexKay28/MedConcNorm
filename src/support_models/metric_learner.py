import numpy as np
import pandas as pd

from src.support_models import ClassicMetricLearner
from src.support_models import SiameseMetricLearner

class MetricLearner:

    def __init__(self, learner_name, n_jobs=5):
        self.learner_name = learner_name
        if self.learner_name == 'siamese':
            self.learner = SiameseMetricLearner(n_jobs=n_jobs)
        elif self.learner_name in ClassicMetricLearner.available_mltools():
            self.learner = ClassicMetricLearner(mltool_name=self.learner_name)
        else:
            raise KeyError('Unknown learner name!')
        self.n_jobs = n_jobs

    def fit(self, X, y, X_test=None, y_test=None):
        if self.learner_name == 'siamese':
            self.learner.fit(X, y, X_test, y_test)
        else:
            self.learner.fit(X, y)

    def transform(self, X):
        return self.learner.transform(X)

    def fit_transform(self, X, y):
        return self.learner.fit_transform(X, y)
