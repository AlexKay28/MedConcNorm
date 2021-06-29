import numpy as np

from scipy.stats import randint as sp_randInt
from scipy.stats import uniform as sp_randFloat

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV

import warnings
warnings.filterwarnings("ignore")

CV = 5
N_ITER = 80
RANDOM_SEED = 32


class SGD_model:

    def __init__(self, n_jobs=15):
        self.n_jobs = n_jobs
        self.model = CalibratedClassifierCV(SGDClassifier())
        self.metric_learner = None

    def add_metric_learner(self, metric_learner):
        self.metric_learner = metric_learner

    def get_best_model_configuration(self, X, y):
        estimator = CalibratedClassifierCV(SGDClassifier())
        parameters = {
            'base_estimator__loss': ['hinge', 'modified_huber', 'squared_hinge'],
            'base_estimator__penalty': ['l2', 'l1', 'elasticnet'],
            'base_estimator__max_iter': [500, 1000]
            }
        self.model = RandomizedSearchCV(estimator=estimator,
                                      param_distributions=parameters,
                                      cv=CV,
                                      n_iter=N_ITER,
                                      n_jobs=self.n_jobs,
                                      verbose=1,
                                      scoring=accuracy_score,
                                      random_state=RANDOM_SEED)
        try:
            self.model.fit(X, y)
        except Exception as e:
            self.model = SGDClassifier()
            self.model.fit(X, y)
        self._best_model_params = self.model.best_params_

    def fit(self, X, y, X_test, y_test):
        self.classes_ = np.unique(y)
        if self.metric_learner:
            self.metric_learner.fit(X, y, X_test, y_test)
            X = self.metric_learner.transform(X)
        self.get_best_model_configuration(X, y)

    def predict_proba(self, X):
        if self.metric_learner:
            X = self.metric_learner.transform(X)
        return self.model.predict_proba(X)

    def get_params(self):
        return self._best_model_params
