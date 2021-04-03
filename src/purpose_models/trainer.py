import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from metric_learn import NCA
import mlflow
import mlflow.sklearn


class Trainer:

    def __init__(self):
        pass

    def evaluate_model(self, X, y):
        return self.model.score(X, y)

    def train_model(self, X, y, mlalg=False):
        # TRAIN AND EVAL MODEL
        self.model = SVC(kernel='poly', gamma='scale')
        if mlalg:
            print('USE MERTRIC LEARNING')
            self.model = make_pipeline(NCA(), self.model)
        self.model.fit(X, y)
