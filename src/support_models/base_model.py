from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling1D


def base_model(sent_emb):
    model = Sequential()
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32))
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
