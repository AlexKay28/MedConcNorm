import os
import pandas as pd
import numpy as np
import multiprocessing
import mlflow
import mlflow.sklearn
from argparse import ArgumentParser

import logging
import traceback

from tqdm import tqdm
tqdm.pandas()

parser = ArgumentParser()
parser.add_argument("--run_name", default="default", type=str)
parser.add_argument("--experiment_name", default="test baseline run", type=str)
args = parser.parse_args()
print(args)
RUN_NAME = args.run_name
os.environ["RUN_NAME"] = RUN_NAME
EXPERIMENT_NAME = args.experiment_name
os.environ["EXPERIMENT_NAME"] = EXPERIMENT_NAME

from src.data.sentence_vectorizer import SentenceVectorizer
from src.purpose_models.trainer import Trainer
from src.configs import GENERAL, PREPROCESSING, AVAILABLE_CONFIGURATIONS, MODELING
from src.configs import create_random_configuration, delete_process_configuration_file

USE_MLALG = MODELING['use_metric_learning']
MODEL_NAME = MODELING['model_name']
VECTORIZER_NAME = PREPROCESSING['sentence_vectorizer']

# create log file
log_file = f'logs/{EXPERIMENT_NAME}_{RUN_NAME}_logs.log'
logging.basicConfig(
    filename=log_file,
    level=logging.ERROR,
    filemode="w+",
)

def run_pipe(sv, meddra_labels, name_train, corpus_train, name_test, corpus_test):
    """
    Runs pipeline with defined settings and log it
    in MLflow Tracker node
    """
    # initalize particular vectorizer method
    print(f"vectorizer: {VECTORIZER_NAME}")

    with mlflow.start_run(run_name=RUN_NAME) as run:
        # PREPROCESSING, AVAILABLE_CONFIGURATIONS, MODELING
        mlflow.log_param('GENERAL', GENERAL)
        mlflow.log_param('PREPROCESSING', PREPROCESSING)
        mlflow.log_param('MODELING', MODELING)
        try:
            mlflow.log_artifact(f'src/configs/temp/run_config_{RUN_NAME}.yml')

            train_file = f'data/processed/train_ex_{name_train}_{VECTORIZER_NAME}.pkl'
            print(train_file)
            if f"train_ex_{name_train}_{VECTORIZER_NAME}.pkl" in os.listdir('data/processed'):
                print('Use cached train data')
                train = pd.read_pickle(train_file)
            else:
                # PREPARE TRAIN SETS
                print(f"Work with {name_train} ", end='.')
                train = pd.read_csv(corpus_train)
                print(train.shape)
                print(f'Vectorize train by {VECTORIZER_NAME} ', end='.')
                train = sv.vectorize(train, vectorizer_name=VECTORIZER_NAME)
                train.to_pickle(train_file)
            train = train.dropna()
            X_train, y_train = train['term_vec'], train['code']
            X_train = pd.DataFrame([pd.Series(x) for x in X_train]).to_numpy()
            y_train = y_train.progress_apply(lambda x: int(meddra_labels[x])).to_numpy()

            # FIT MODEL
            print('Fit model ')
            trainer = Trainer()
            trainer.train_model(X_train, y_train)

            # PREPARE TEST SETS
            mlflow.set_tag("mlflow.note.content","<my_note_here>")
            test_file = f'data/processed/test_ex_{name_train}_{VECTORIZER_NAME}.pkl'
            if f"test_ex_{name_train}_{VECTORIZER_NAME}.pkl" in os.listdir('data/processed'):
                print('Use cached test data')
                test = pd.read_pickle(test_file)
            else:
                test = pd.read_csv(corpus_test)
                test = sv.vectorize(test, vectorizer_name=VECTORIZER_NAME)
                test.to_pickle(test_file)
            test = test.dropna()
            X_test, y_test = test['term_vec'], test['code']
            X_test = pd.DataFrame([pd.Series(x) for x in X_test]).to_numpy()
            y_test = y_test.progress_apply(lambda x: int(meddra_labels[x])).to_numpy()

            # MLFLOW LOG PARAMS
            mlflow.log_param('train corpus', name_train)
            mlflow.log_param('test corpus', name_test)
            mlflow.log_param('vectorizer', VECTORIZER_NAME)
            mlflow.log_param("use_metric_learning", USE_MLALG)
            mlflow.log_param('model_name', MODEL_NAME)
            mlflow.log_param('model_params', trainer.model.get_params())

            # MLFLOW LOG METRICS
            for k in range(1, 4):
                score = trainer.accuracy(X_test, y_test, k=k)
                print(f'\ttest with {name_test} acc@{k}:', score)
                mlflow.log_metric(f"accuracy_{k}", score)
            mlflow.set_tag("exp_name", 'first')
            mlflow.log_artifact(log_file)

            acc1 = trainer.accuracy(X_test, y_test, k=1)
            if any([
                name_train=='smm4h17' and acc1 >= 65.0,
                name_train=='smm4h21' and acc1 >= 37.0,
                name_train=='psytar' and acc1 >= 74.0,
                name_train=='cadec' and acc1 >= 72.0,
            ]):
                mlflow.set_tag("QUALITY", 'HIGH')
                mlflow.sklearn.log_model(trainer.model, f"model_for_{name_train}")
                mlflow.log_artifact(train_file)
                mlflow.log_artifact(test_file)
            else:
                mlflow.set_tag("QUALITY", 'LOW')

        except Exception as e:
            logging.error(f"\nERROR FOR RUN: {run.info.run_id}")
            logging.error(e, exc_info=True)
            mlflow.log_artifact(log_file)
            mlflow.set_tag("LOG_STATUS", "FAILED RUN")
            mlflow.end_run(status='FAILED')


def main():
    """
    Set parameters and run pipeline with defined settings
    """
    print('READ LABELS!')
    labels = pd.read_csv('data/interim/meddra_codes_terms_synonims.csv')
    labels = labels['CODE']
    meddra_labels = {v:k for k, v in enumerate(labels.unique())}

    # configure mlflow
    mlflow.set_experiment(args.experiment_name)
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(args.experiment_name).experiment_id
    notes = """
        SMM4H17: 65, 87-90, 97.73
        SMM4H21(20): 36-37, 42-44, 43-45
        CADEC: 72.72, 70-83. 86.4, 86.93
        PsyTar: 74.39, 77-82, 85.04, 87.7
    """
    client.set_experiment_tag(experiment_id, "mlflow.note.content", notes)

    create_random_configuration(AVAILABLE_CONFIGURATIONS)
    sv = SentenceVectorizer()
    path = 'data/interim/'
    for name_folder_train in os.listdir(path):
        if name_folder_train not in ['smm4h17', 'smm4h21', 'psytar', 'cadec']:
            continue
        # PREPARE TRAIN SETS
        folder = os.path.join(path, name_folder_train)
        corpus_train = folder + '/train_ex.csv'
        for name_folder_test in os.listdir(path):
            if name_folder_test not in ['smm4h17', 'smm4h21', 'psytar', 'cadec'] \
                                       or name_folder_test!=name_folder_train:
                continue
            # PREPARE TEST SETS
            folder = os.path.join(path, name_folder_test)
            corpus_test = folder + '/test.csv'

            print(name_folder_train, name_folder_test)
            run_pipe(
                sv, meddra_labels,
                name_folder_train, corpus_train,
                name_folder_test, corpus_test
            )

    delete_process_configuration_file()
    print('DONE')

if __name__ == "__main__":
    main()
