import numpy as np

from scipy.stats import randint as sp_randInt
from scipy.stats import uniform as sp_randFloat

from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV

import warnings
warnings.filterwarnings("ignore")

CV = 5
N_ITER = 150
RANDOM_SEED = 32


class LGBM_model:

    def __init__(self):
        self._best_model_params = None
        self.model = LGBMClassifier(random_state=RANDOM_SEED, silent=True,)
        self.metric_learner = None

    def add_metric_learner(self, metric_learner):
        self.metric_learner = metric_learner

    def get_best_model_configuration(self, X, y):
        estimator = LGBMClassifier(random_state=RANDOM_SEED, silent=True,)
        if self.metric_learner:
            self.metric_learner.fit(X, y)

        parameters = {'max_depth': sp_randInt(2, 8),
                      'learning_rate': sp_randFloat(),
                      'num_leaves': sp_randInt(2, 15)}

        decision = RandomizedSearchCV(estimator=estimator,
                                      param_distributions=parameters,
                                      cv=CV,
                                      n_iter=N_ITER,
                                      n_jobs=15,
                                      verbose=1,
                                      scoring=accuracy_score,
                                      random_state=RANDOM_SEED)
        X = self.metric_learner.transform(X) if self.metric_learner else X
        decision.fit(X, y)
        if self.metric_learner:
            decision = Pipeline([
                ('metric_learner', self.metric_learner),
                #('svc', estimator)
                ('lgbm', decision)
            ])
            self._best_model_params = decision['lgbm'].best_params_
        else:
            self._best_model_params = decision.best_params_
        return decision

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model = self.get_best_model_configuration(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self):
        return self._best_model_params
