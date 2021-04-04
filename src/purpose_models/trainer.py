import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from metric_learn import NCA
from sklearn.linear_model import SGDClassifier
from src.features.metrics import accuracy_top_K_pobs

import mlflow
import mlflow.sklearn

from sklearn.calibration import CalibratedClassifierCV

class Trainer:

    def __init__(self):
        pass

    def fscore(self, X, y):
        pass

    def accuracy(self, X, y, k=1):
        y_hat = self.model.predict_proba(X)
        acc = accuracy_top_K_pobs(y, y_hat, self.model.classes_, k=k)
        return acc

    def train_model(self, X, y, mlalg=False):
        self.model = SVC(kernel='poly', gamma='scale', probability=True) #, class_weight='balanced')
        self.model = CalibratedClassifierCV(self.model)
#        self.model = SGDClassifier(max_iter=1000, tol=1e-3, loss='log')
        if mlalg:
            print('USE MERTRIC LEARNING')
            self.model = make_pipeline(NCA(), self.model)
        self.model.fit(X, y)
