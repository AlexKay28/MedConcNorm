import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from src.data.class_balancing import class_sampler
from src.support_models.triplet_generator import TripletGenerator
from src.support_models.loss_functions import triplet_loss, identity_loss
from src.support_models.base_model import base_model, base_model_lstm
from src.support_models.siamese_model_architecture import siamese_model, siamese_model_lstm

from src.configs import GENERAL, PREPROCESSING, MODELING

LR = MODELING['siamese_params']['lr']
BATCH_SIZE = MODELING['siamese_params']['batch_size']
EPOCHS = MODELING['siamese_params']['epochs']
STEPS_PER_EPOCH = MODELING['siamese_params']['steps_per_epoch']


class SiameseMetricLearner:

    def __init__(self, n_jobs=5):
        self.n_jobs = n_jobs

    def summary(self):
        return self.learner.summary()

    def fit(self, X, y, epochs=EPOCHS):
        self.sent_emb = X.shape[1]
        tgen = TripletGenerator()
        train_generator = tgen.generate_triplets(X, y, BATCH_SIZE)
        self.emb_model = base_model(self.sent_emb)
        self.learner = siamese_model(
            self.emb_model,
            self.sent_emb ,
            triplet_loss,
            identity_loss,
            learning_rate=LR
            )

        early_stopping_callback = EarlyStopping(
            monitor="loss",
            min_delta=0,
            patience=20,
            verbose=1,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
        history = self.learner.fit_generator(train_generator,
                                             epochs=epochs,
                                             verbose=1,
                                             workers=self.n_jobs,
                                             use_multiprocessing=True,
                                             steps_per_epoch=STEPS_PER_EPOCH,
                                             callbacks=[early_stopping_callback]
                                         )
        del self.learner
        del tgen
        return history

    def transform(self, X):
        return self.emb_model.predict(X.reshape(-1, self.sent_emb , 1))

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
