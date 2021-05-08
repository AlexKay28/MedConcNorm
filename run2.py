import os
import pandas as pd
import numpy as np
import multiprocessing
import mlflow
import mlflow.sklearn
from copy import copy

import tensorflow as tf
from gensim.models import FastText
from src.data.sentence_vectorizer import SentenceVectorizer
from src.features.retrofitting import retrofitting, vectorize_sent, retrofit_row
from src.purpose_models.close__synonims_model import *


MIN_BATCH_SIZE = 32
EPOCHS = 200
VALIDATION_STEPS = 50
N_ITERATIONS = 6

def main():
    # configure mlflow
    mlflow.set_experiment('Synonims_net')
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name('Synonims_net').experiment_id
    notes = """
        SMM4H17: 67.2, 87-90, 91.73
        SMM4H21(20): 36-37, 42-44, 43-45
        CADEC: 63.23, 70-83. 86.94
        PsyTar: 60.15, 77-82, 85.76
    """
    client.set_experiment_tag(experiment_id, "mlflow.note.content", notes)

    print('loading fasttext')
    model = FastText.load_fasttext_format(
        'data/external/embeddings/cc.en.300.bin')

    for corp in ['smm4h21','smm4h17','cadec','psytar']:
        print(f'loading datasets {corp}')
        terms_train = pd.read_csv(f'data/interim/{corp}/train_pure.csv')
        terms_test = pd.read_csv(f'data/interim/{corp}/test.csv')
        concepts = pd.read_csv('data/interim/used_codes_big.csv')[['code', 'STR', 'SNMS']]

        print('preparing and creating generators')
        terms_vecs_train, terms_codes_train, terms_vecs_test,  terms_codes_test, concepts_vecs, codes = prepare_data(model, concepts, terms_train, terms_test)
        n_concepts = len(codes)
        assert terms_vecs_train.shape[1]==concepts_vecs.shape[1]
        embedding_size = terms_vecs_train.shape[1]

        print('start fitting')
        for iter in tqdm(range(N_ITERATIONS)):
            batch_size = MIN_BATCH_SIZE*(iter+1)
            steps_per_epoch = terms_train.shape[0]//batch_size * 10
            for dn_layers in [1, 3, 5]:
                with mlflow.start_run(run_name=f"run{iter}") as run:
                    train_gen, test_gen = get_data_gens(terms_vecs_train,
                                      terms_codes_train,
                                      terms_vecs_test,
                                      terms_codes_test,
                                      concepts_vecs, batch_size=batch_size)
                    mlflow.log_param('corpus', corp)
                    mlflow.log_param('batch_size', batch_size)
                    mlflow.log_param('epochs', EPOCHS)
                    mlflow.log_param('steps_per_epoch', steps_per_epoch)
                    mlflow.log_param('validation_steps', VALIDATION_STEPS)
                    mlflow.log_param('dn_layers', dn_layers)
                    max_train, max_val = fit_synonimer(
                        train_gen, test_gen, n_concepts, embedding_size,
                        epochs=EPOCHS,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=VALIDATION_STEPS,
                        dn_layers=dn_layers)
                    print(max_train, max_val)
                    mlflow.log_metric(f"max_train", max_train)
                    mlflow.log_metric(f"max_val", max_val)


if __name__ == "__main__":
    main()
