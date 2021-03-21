from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling1D


def base_model(sent_emb):
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))    
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(sent_emb))
    return model