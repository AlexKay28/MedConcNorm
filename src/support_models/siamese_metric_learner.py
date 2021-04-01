import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from src.data.class_balancing import class_sampler
from src.support_models.triplet_generator import TripletGenerator
from src.support_models.loss_functions import triplet_loss, identity_loss
from src.support_models.base_model import base_model
from src.support_models.siamese_model_architecture import siamese_model


sent_emb = 768
batch_size = 256
lr = 1e-3
EPOCHS = 100
alpha = 0.2
#monitor = "val_loss"
patience = 15
test_size = 0.2

class SiameseMetricLearner:

    def __init__(self, n_jobs=5):
        self.n_jobs = n_jobs
        self.emb_model = base_model(sent_emb)
        self.learner = siamese_model(self.emb_model, sent_emb, triplet_loss, identity_loss, learning_rate=lr)


    def summary(self):
        return self.learner.summary()


    def fit(self, X, y):
        tgen = TripletGenerator()
        train_generator = tgen.generate_triplets(X, y, batch_size)

        history = self.learner.fit_generator(train_generator,
                                             #validation_data=test_generator,
                                             epochs=EPOCHS,
                                             verbose=1,
                                             workers=10,
                                             use_multiprocessing=True,
                                             steps_per_epoch=20,
                                            # validation_steps=10,
                                             #callbacks=[early_stopping_callback]
                                         )
        del self.learner
        return history

    def transform(self, X):
        return self.emb_model.predict(X.reshape(-1, sent_emb, 1))

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
