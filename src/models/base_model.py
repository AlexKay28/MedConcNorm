from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D


def base_model():
    # Simple convolutional model 
    # used for the embedding model.
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu',
                        input_shape=(28,28,1)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    return model


def base_model2():
    model = Sequential()
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3))
    return model