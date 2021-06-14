import os
import tensorflow as tf
import pandas as pd
import numpy as np
import multiprocessing
import mlflow
import mlflow.sklearn
from copy import copy

MIN_BATCH_SIZE = 64
EPOCHS = 1500
VALIDATION_STEPS = 50
N_ITERATIONS = 6


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--experiment_name", default="default", type=str)
parser.add_argument("--run_name", default="default", type=str)
args = parser.parse_args()


vec_model_name = 'SRoBERTa' # fasttext / endr-bert / SRoBERTa
os.environ["vec_model_name"] = vec_model_name
experiment_name = args.experiment_name
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from src.purpose_models.close__synonims_model import *
from src.data.sentence_vectorizer import SentenceVectorizer


def main():
    # configure mlflow
    mlflow.set_experiment(experiment_name)
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    notes = """
        SMM4H17: 67.2, 87-90, 91.73
        SMM4H21(20): 36-37, 42-44, 43-45
        CADEC: 63.23, 70-83. 86.94
        PsyTar: 60.15, 77-82, 85.76
    """
    client.set_experiment_tag(experiment_id, "mlflow.note.content", notes)

    # ['smm4h21','smm4h17','cadec','psytar']
    train_courpuses = ['smm4h21','smm4h17','cadec','psytar']
    test_courpuses = ['smm4h21','smm4h17','cadec','psytar']
    # train_courpuses = np.random.choice(train_courpuses, size=1)
    # test_courpuses = np.random.choice(test_courpuses, size=1)
    for train_corp in train_courpuses:
        for test_corp in test_courpuses:
            print(f'loading datasets {train_corp} / {test_corp}')

            terms_train_dataset = [
                'train_pure.csv',
                'train_aug.csv',
                'train_aug_wdnt.csv',
                #'train_aug_ppdb.csv'
            ]
            for terms_train_name in terms_train_dataset:
                retro_iters = np.random.choice([1, 5, 10, 25, 50])
                print('preparing and creating generators')
                terms_vecs_train, terms_codes_train, terms_vecs_test,  terms_codes_test, concepts_vecs, codes \
                    = prepare_data(train_corp, test_corp, terms_train_name, retro_iters)
                n_concepts = len(codes)


                print("TRAIN   VEC SHAPE:", terms_vecs_train.shape[1])
                print("TEST    VEC SHAPE:", terms_vecs_test.shape[1])
                print("CONCEPT VEC SHAPE:", concepts_vecs.shape[1])

                assert terms_vecs_train.shape[1]==concepts_vecs.shape[1]
                embedding_size = terms_vecs_train.shape[1]
            
                print('start fitting')
                for iter in tqdm(range(N_ITERATIONS)):
                    batch_size = MIN_BATCH_SIZE*(iter+1)
                    steps_per_epoch = terms_vecs_train.shape[0]//batch_size * 10
                    for dn_layers in [1, 2, 3]:
                        for patience in [15, 30, 60, 180, 500]:
                            learning_rate = np.random.choice([1e-2, 1e-3, 5e-4, 1e-4])
                            layer_size = np.random.choice([256, 512, 768, 1024])
                            with mlflow.start_run(run_name=f"run{iter}") as run:
                                print(f"INFO: terms:{terms_train_name}, train_corp:{train_corp}, test_corp:{test_corp}")
                                train_gen, test_gen = get_data_gens(terms_vecs_train,
                                                  terms_codes_train,
                                                  terms_vecs_test,
                                                  terms_codes_test,
                                                  concepts_vecs,
                                                  batch_size=batch_size)
                                mlflow.log_param('terms_train', terms_train_name)
                                mlflow.log_param('corpus_train', train_corp)
                                mlflow.log_param('corpus_test', test_corp)
                                mlflow.log_param('batch_size', batch_size)
                                mlflow.log_param('epochs', EPOCHS)
                                mlflow.log_param('steps_per_epoch', steps_per_epoch)
                                mlflow.log_param('validation_steps', VALIDATION_STEPS)
                                mlflow.log_param('dn_layers', dn_layers)
                                mlflow.log_param('patience', patience)
                                mlflow.log_param('learning_rate', learning_rate)
                                mlflow.log_param('layer_size', layer_size)
                                mlflow.log_param('retro_iters', retro_iters)
                                max_train, max_val = fit_synonimer(
                                    train_gen, test_gen, n_concepts, embedding_size,
                                    epochs=EPOCHS,
                                    patience=patience,
                                    steps_per_epoch=steps_per_epoch,
                                    validation_steps=VALIDATION_STEPS,
                                    dn_layers=dn_layers,
                                    learning_rate=learning_rate,
                                    layer_size=layer_size)
                                print(max_train, max_val)
                                mlflow.log_metric(f"max_train", max_train)
                                mlflow.log_metric(f"max_val", max_val)
                                try:
                                    mlflow.log_artifact('reports/run2_plot.pdf')
                                except Exception as e:
                                    print(e)

if __name__ == "__main__":
    main()
