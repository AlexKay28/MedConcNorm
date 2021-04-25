import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from metric_learn import NCA, LMNN
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import NearestNeighbors


from src.features.metrics import accuracy_top_K_pobs
from src.configs import GENERAL, PREPROCESSING, MODELING
from src.purpose_models.model_svc import SVC_model
from src.purpose_models.model_knn import kNN_model
from src.purpose_models.model_sgd import SGD_model
from src.purpose_models.model_lgbm import LGBM_model
from src.support_models.metric_learner import MetricLearner

import mlflow
import mlflow.sklearn

MODEL_NAME = MODELING['model_name']
METRIC_LEARNER_NAME = MODELING['metric_learner_name']
USE_MLALG = MODELING['use_metric_learning']


class Trainer:

    _models = {
        'SVC': SVC_model,
        'SGD': SGD_model,
        'kNN': kNN_model,
        'LGBM': LGBM_model
    }

    def __init__(self):
        pass

    def fscore(self, X, y):
        pass

    def predict_proba(self, X):
        X = np.nan_to_num(X)
        return self.model.predict_proba(X)

    def accuracy(self, X, y, k=1):
        X = np.nan_to_num(X)
        y_hat = self.model.predict_proba(X)
        acc = accuracy_top_K_pobs(y, y_hat, self.model.classes_, k=k)
        return acc

    def get_classes(self):
        return self.model.classes_

    def train_model(self, X, y, mlalg=USE_MLALG, model_name=MODEL_NAME):
        self.model = self._models[model_name]()
        if mlalg:
            print('USE MERTRIC LEARNING')
            metric_learner = MetricLearner(METRIC_LEARNER_NAME)
            self.model.add_metric_learner(metric_learner)
        self.model.fit(X, y)
