import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.features.retrofitting import vectorize_mention, vectorize_concept

def prepare_data(concepts, terms_train, terms_test, use_case='train_codes'):

    print(f"TRAIN COLUMNS: {terms_train.columns}")
    print(f"TEST COLUMNS: {terms_test.columns}")

    print("Prepare train columns")
    terms_vecs_train = terms_train.progress_apply(lambda row: vectorize_mention(row), axis=1)
    terms_vecs_train = pd.DataFrame(terms_vecs_train.dropna().values.tolist()).dropna().values

    print("Prepare test columns")
    terms_vecs_test = terms_test.progress_apply(lambda row: vectorize_mention(row), axis=1)
    terms_vecs_test = pd.DataFrame(terms_vecs_test.dropna().values.tolist()).dropna().values

    if use_case=='train_codes':
        # все из трейна
        concepts = concepts[concepts['code'].isin(terms_train['code'])
            ].reset_index(drop=True).reset_index().set_index('code')
    elif use_case=='fully_filtered':
        concepts = concepts[concepts['code'].isin(codes)
            ].reset_index(drop=True).reset_index().set_index('code')
    elif use_case=='all_available':
        concepts = concepts.reset_index(drop=True).reset_index().set_index('code')
    else:
        raise KeyError('Unknown use case')

    print("Prepare concept columns")
    codes = concepts['index'].to_numpy()
    concepts_vecs = concepts.progress_apply(lambda row: vectorize_concept(row), axis=1)
    concepts_vecs = pd.DataFrame(concepts_vecs.values.tolist()).dropna().values

    terms_codes_train = terms_train['code'].apply(
        lambda code: concepts.loc[code]['index'])
    #OOV CODE
    terms_codes_test = terms_test['code'].apply(
        lambda code: concepts.loc[code]['index'] if code in concepts.index else len(codes))

    terms_codes_train = tf.keras.utils.to_categorical(
        terms_codes_train, num_classes=len(codes)+1, dtype='float32'
    )
    terms_codes_test = tf.keras.utils.to_categorical(
        terms_codes_test, num_classes=len(codes)+1, dtype='float32'
    )
    set_ = terms_vecs_train, terms_codes_train, terms_vecs_test,  terms_codes_test, concepts_vecs, codes
    return set_


def data_generator(terms_vecs, terms_codes, concepts_vecs, batch_size=50):
    while 1:
        concepts_vecs_choosed = []
        index_to_choose = np.random.choice(terms_vecs.shape[0]-batch_size)
        concepts_vecs_choosed.append(concepts_vecs)
        terms_vecs_choosed = terms_vecs[index_to_choose:index_to_choose+batch_size]
        terms_codes_choosed = terms_codes[index_to_choose:index_to_choose+batch_size]
        concepts_vecs_choosed = np.array(concepts_vecs_choosed, dtype='float32')
        train = [terms_vecs_choosed, concepts_vecs_choosed]
        test = [terms_codes_choosed]
        yield train, test

def cosine_mul_fun(inputs):
    "Calc cosine distance between vectors"
    vec1, vec2 = inputs[0], inputs[1]
    normalize_vec1 = tf.nn.l2_normalize(vec1, 1)
    normalize_vec2 = tf.nn.l2_normalize(vec2, 1)
    distance = 1 - tf.matmul([normalize_vec1], normalize_vec2, transpose_b=True)
    return distance[0]

def get_syn_model(n_concepts, embedding_size, dn_layers=1):
    inputs1 =    tf.keras.layers.Input(shape=(embedding_size), name='term')
    dn =         tf.keras.layers.Dense(512)(inputs1)
    for i in range(1, dn_layers+1):
        dn =     tf.keras.layers.Dense(512)(dn)
        dn =     tf.keras.layers.Dropout(0.2)(dn)
    dn =         tf.keras.layers.Dense(embedding_size)(dn)
    term_hat =   tf.keras.layers.Flatten()(dn)
    inputs2 =    tf.keras.layers.Input(shape=(n_concepts, embedding_size), name='concepts')
    cosine_mul = tf.keras.layers.Lambda(cosine_mul_fun)([term_hat, inputs2])
    activation = tf.keras.layers.ReLU()(cosine_mul)
    output =     tf.keras.layers.Dense(n_concepts+1, activation='softmax')(activation)

    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=[output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.0005),
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.CategoricalAccuracy(name='cat_acc')],
    )
    return model

def get_data_gens(terms_vecs_train,
                  terms_codes_train,
                  terms_vecs_test,
                  terms_codes_test,
                  concepts_vecs, batch_size=256):

    train_gen = data_generator(
        terms_vecs_train, terms_codes_train, concepts_vecs, batch_size=batch_size)
    test_gen = data_generator(
        terms_vecs_test,  terms_codes_test,  concepts_vecs, batch_size=batch_size)

    return train_gen, test_gen

def fit_synonimer(train_gen, test_gen, n_concepts, embedding_size, dn_layers=1,
                  verbose=2,
                  epochs=80,
                  steps_per_epoch=200,
                  validation_steps=50,
                  patience=15,
                  show_model_info=False):

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_cat_acc",
            patience=patience,
            verbose=0,
            restore_best_weights=True,
        )

    model = get_syn_model(n_concepts, embedding_size, dn_layers=dn_layers)
    if show_model_info:
        model.summary()
    history = model.fit_generator(train_gen,
              epochs=epochs,
              verbose=verbose,
              steps_per_epoch=steps_per_epoch,
              validation_data=test_gen,
              validation_steps=validation_steps,
              callbacks=[early_stopping_callback])

    max_train = np.array(history.history['cat_acc']).max()
    max_val   = np.array(history.history['val_cat_acc']).max()

    return max_train, max_val
