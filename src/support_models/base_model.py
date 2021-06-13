import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling1D

from src.configs import GENERAL, PREPROCESSING, MODELING

base_model_output_dim = MODELING['siamese_params']['base_model_output_dim']
base_model_layers = MODELING['siamese_params']['base_model_layers']
choosed_dropout = MODELING['siamese_params']['choosed_dropout']


def base_model(sent_emb):
    """
    Base configuration of model which is implemented
    as main part of siamese model
    """
    model = Sequential()
    for iteration in range(base_model_layers):
        model.add(Dense(base_model_output_dim*4, activation='relu'))
        model.add(Dropout(choosed_dropout))
    for iteration in range(math.ceil(base_model_layers/2)):
        model.add(Dense(base_model_output_dim*2, activation='relu'))
        model.add(Dropout(choosed_dropout))
    model.add(Flatten())
    model.add(Dense(base_model_output_dim))
    return model


def base_model_lstm(embedding_matrix):
    model = tf.keras.Sequential()
    model.add(Embedding(nb_words, embed_dim, input_length=max_seq_len,
                        weights=[embedding_matrix],trainable=False))
    model.add(Bidirectional(LSTM(32, return_sequences= True)))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Flatten())
    #model.summary()
    return model
