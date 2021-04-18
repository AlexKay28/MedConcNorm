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
N_ITER = 100
RANDOM_SEED = 32


class SGD_model:

    def __init__(self):
        self.model = CalibratedClassifierCV(SGDClassifier())
        self.metric_learner = None

    def add_metric_learner(self, metric_learner):
        self.metric_learner = metric_learner

    def get_best_model_configuration(self, X, y):
        estimator = CalibratedClassifierCV(SGDClassifier())
        if self.metric_learner:
            self.metric_learner.fit(X, y)

        parameters = {
            'base_estimator__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
            'base_estimator__penalty': ['l2', 'l1', 'elasticnet'],
            'base_estimator__max_iter': [500, 1000]
            }

        decision = RandomizedSearchCV(estimator=estimator,
                                      param_distributions=parameters,
                                      cv=CV,
                                      n_iter=N_ITER,
                                      n_jobs=10,
                                      verbose=1,
                                      scoring=accuracy_score,
                                      random_state=RANDOM_SEED)
        X = self.metric_learner.transform(X) if self.metric_learner else X
        decision.fit(X, y)
        if self.metric_learner:
            decision = Pipeline([
                ('metric_learner', self.metric_learner),
                #('svc', estimator)
                ('sgd', decision)
            ])
            self._best_model_params = decision['sgd'].get_params()
        else:
            self._best_model_params = decision.get_params()
        return decision

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model = self.get_best_model_configuration(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self):
        return self._best_model_params
