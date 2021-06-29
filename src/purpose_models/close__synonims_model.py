import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from src.features.retrofitting import vectorize_mention, vectorize_concept

vec_model_name = os.environ.get('vec_model_name', 'fasttext')

def prepare_data(folder, train_corp, test_corp, terms_train_name, retro_iters, use_case='train_codes'):
    le = preprocessing.LabelEncoder()
    concepts = pd.read_csv('data/interim/used_codes_big.csv')[['code', 'STR', 'SNMS']]
    terms_train = pd.read_csv(f'data/interim/{train_corp}/{folder}/{terms_train_name}')
    terms_test = pd.read_csv(f'data/interim/{test_corp}/{folder}/test.csv')

    train_path = f'data/processed/indian_net/train_{folder}_{train_corp}_{vec_model_name}_{terms_train_name}'
    test_path = f'data/processed/indian_net/test_{folder}_{test_corp}_{vec_model_name}_{terms_train_name}'
    concepts_path = f'data/processed/indian_net/concept_{folder}_{train_corp}_{vec_model_name}_{use_case}_{terms_train_name}'

    print(terms_train.shape, terms_test.shape)
    terms_train = terms_train[terms_train['code'].isin(concepts['code'])]
    terms_test = terms_test[terms_test['code'].isin(concepts['code'])]
    print(terms_train.shape, terms_test.shape)

    print("Prepare train columns")
    if os.path.exists(train_path):
        print("Load:", train_path)
        terms_vecs_train = pd.read_csv(train_path)
    else:
        print("Create:", train_path)
        terms_vecs_train = terms_train.progress_apply(lambda row: vectorize_mention(row), axis=1)
        terms_vecs_train = pd.DataFrame(terms_vecs_train.dropna().values.tolist())
        terms_vecs_train.to_csv(train_path, index=False)
    terms_vecs_train = terms_vecs_train.dropna().values

    print("Prepare test columns")
    if os.path.exists(test_path):
        print("Load:", test_path)
        terms_vecs_test = pd.read_csv(test_path)
    else:
        print("Create:", test_path)
        terms_vecs_test = terms_test.progress_apply(lambda row: vectorize_mention(row), axis=1)
        terms_vecs_test = pd.DataFrame(terms_vecs_test.dropna().values.tolist())
        terms_vecs_test.to_csv(test_path, index=False)
    terms_vecs_test = terms_vecs_test.dropna().values

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
    if os.path.exists(concepts_path):
        print("Load:", concepts_path)
        concepts_vecs = pd.read_csv(concepts_path)
    else:
        print("Create:", concepts_path)
        concepts_vecs = concepts.progress_apply(lambda row: vectorize_concept(row, retro_iters), axis=1)
        concepts_vecs = pd.DataFrame(concepts_vecs.values.tolist())
        concepts_vecs.to_csv(concepts_path, index=False)
    concepts_vecs = concepts_vecs.dropna().values

    codes = terms_train['code'].unique()
    le.fit(codes)
    terms_codes_train = terms_train['code'].apply(
        lambda code: le.transform([code])[0])
    #OOV CODE
    terms_codes_test = terms_test['code'].apply(
        lambda code: le.transform([code])[0] if code in le.classes_ else len(codes))

    terms_codes_train = terms_codes_train.tolist()
    terms_codes_test = terms_codes_test.tolist()

    num_classes = len(set(terms_codes_train + terms_codes_test)) + 1
    terms_codes_train_cat = tf.keras.utils.to_categorical(
        terms_codes_train, num_classes=num_classes, dtype='float32'
    )
    terms_codes_test_cat = tf.keras.utils.to_categorical(
        terms_codes_test, num_classes=num_classes, dtype='float32'
    )
    set_ = terms_vecs_train, terms_codes_train_cat, terms_vecs_test,  terms_codes_test_cat, concepts_vecs, codes
    return set_


def data_generator(terms_vecs, terms_codes, concepts_vecs, batch_size=64):
    while 1:
        concepts_vecs_choosed = np.array([concepts_vecs], dtype='float32')
        index_to_choose = np.random.choice(terms_codes.shape[0], size=batch_size)
        terms_vecs_choosed = terms_vecs[index_to_choose]
        terms_codes_choosed = terms_codes[index_to_choose]
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

def get_syn_model(n_concepts,
                  embedding_size,
                  learning_rate=0.0005,
                  layer_size=512,
                  dn_layers=1):

    inputs1 =    tf.keras.layers.Input(shape=(embedding_size), name='term')
    dn =         tf.keras.layers.Dense(layer_size)(inputs1)
    for i in range(1, dn_layers+1):
        dn =     tf.keras.layers.Dense(layer_size)(dn)
        dn =     tf.keras.layers.Dropout(0.2)(dn)
    dn =         tf.keras.layers.Dense(embedding_size)(dn)
    term_hat =   tf.keras.layers.Flatten()(dn)
    inputs2 =    tf.keras.layers.Input(shape=(n_concepts, embedding_size), name='concepts')
    cosine_mul = tf.keras.layers.Lambda(cosine_mul_fun)([term_hat, inputs2])
    activation = tf.keras.layers.ReLU()(cosine_mul)
    output =     tf.keras.layers.Dense(n_concepts+1, activation='softmax')(activation)

    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=[output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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

def save_history_plot(history):
    # summarize history for accuracy
    plt.plot(history.history['cat_acc'])
    plt.plot(history.history['val_cat_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("/home/kaigorodov/myprojects/MedConcNorm/reports/run2_plot.pdf")

def fit_synonimer(train_gen,
                  test_gen,
                  n_concepts,
                  embedding_size,
                  learning_rate,
                  layer_size,
                  dn_layers=1,
                  verbose=1,
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

    model = get_syn_model(
        n_concepts,
        embedding_size,
        learning_rate=learning_rate,
        layer_size=layer_size,
        dn_layers=dn_layers,
    )

    if show_model_info:
        model.summary()
    history = model.fit(train_gen,
              epochs=epochs,
              verbose=verbose,
              steps_per_epoch=steps_per_epoch,
              validation_data=test_gen,
              validation_steps=validation_steps,
              callbacks=[early_stopping_callback]
              )
    plt.tight_layout()
    save_history_plot(history)
    max_train = np.array(history.history['cat_acc']).max()
    max_val   = np.array(history.history['val_cat_acc']).max()
    return max_train, max_val
