from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Lambda


def siamese_model(base_model, imsize, triple_loss_function, identity_loss_function, learning_rate=0.001):
    """
    Create the complete model with three
    embedding models and minimize the loss 
    between their output embeddings
    """
    input_1 = Input((imsize, imsize, 1))
    input_2 = Input((imsize, imsize, 1))
    input_3 = Input((imsize, imsize, 1))
        
    A = base_model(input_1)
    P = base_model(input_2)
    N = base_model(input_3)
   
    loss = Lambda(triple_loss_function)([A, P, N]) 
    model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
    model.compile(loss=identity_loss_function, optimizer=Adam(learning_rate))
    return model


def siamese_model2(base_model, imsize, triple_loss_function, identity_loss_function, learning_rate=0.001):
    """
    Create the complete model with three
    embedding models and minimize the loss 
    between their output embeddings
    """
    input_1 = Input((imsize, 1))
    input_2 = Input((imsize, 1))
    input_3 = Input((imsize, 1))
        
    A = base_model(input_1)
    P = base_model(input_2)
    N = base_model(input_3)
   
    loss = Lambda(triple_loss_function)([A, P, N]) 
    model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
    model.compile(loss=identity_loss_function, optimizer=Adam(learning_rate))
    return model