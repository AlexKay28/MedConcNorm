import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from src.data.class_balancing import class_sampler
from src.support_models.triplet_generator import TripletGenerator
from src.support_models.loss_functions import triplet_loss, identity_loss
from src.support_models.base_model import base_model
from src.support_models.siamese_model_architecture import siamese_model


#SENT_EMB = 768
BATCH_SIZE = 256
lr = 1e-3
EPOCHS = 15
alpha = 0.2
#monitor = "val_loss"
patience = 15
test_size = 0.2

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
            learning_rate=lr
            )
        history = self.learner.fit_generator(train_generator,
                                             epochs=epochs,
                                             verbose=1,
                                             workers=10,
                                             use_multiprocessing=True,
                                             steps_per_epoch=20
                                         )
        del self.learner
        del tgen
        return history



    def transform(self, X):
        return self.emb_model.predict(X.reshape(-1, self.sent_emb , 1))

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
