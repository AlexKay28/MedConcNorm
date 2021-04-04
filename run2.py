import os
import pandas as pd
import numpy as np
import multiprocessing
from src.data.sentence_vectorizer import SentenceVectorizer

from src.purpose_models.trainer import Trainer
from argparse import ArgumentParser

import mlflow
import mlflow.sklearn

def run_pipe(sv, vectorizer_name, meddra_labels, name_train, corpus_train, name_test, corpus_test, mlalg):
    """
    Runs pipeline with defined settings and log it
    in MLflow Tracker node
    """
    # initalize particular vectorizer method
    print(f"vectorizer: {vectorizer_name}")

    # PREPARE TRAIN SETS
    print(f"Work with {name_train} ", end='.')
    train = pd.read_csv(corpus_train)

    print(f'Vectorize train by {vectorizer_name} ', end='.')
    train = sv.vectorize(train, vectorizer_name=vectorizer_name)
    X_train, y_train = train['term_vec'], train['code']
    X_train = pd.DataFrame([pd.Series(x) for x in X_train])
    y_train = y_train.apply(lambda x: int(meddra_labels[x]))

    # FIT MODEL
    print('Fit model ')
    trainer = Trainer()
    trainer.train_model(X_train, y_train, mlalg=mlalg)


    with mlflow.start_run() as run:
        # PREPARE TEST SETS
        mlflow.set_tag("mlflow.note.content","<my_note_here>")
        test = pd.read_csv(corpus_test)
        test = sv.vectorize(test, vectorizer_name=vectorizer_name)
        X_test, y_test = test['term_vec'], test['code']
        X_test = pd.DataFrame([pd.Series(x) for x in X_test])
        y_test = y_test.apply(lambda x: int(meddra_labels[x]))

        # MLFLOW LOG PARAMS
        mlflow.log_param('vectorizer', vectorizer_name)
        mlflow.log_param('train corpus', name_train)
        mlflow.log_param('test corpus', name_test)
        mlflow.log_param("use_metric_learning", mlalg)

        # MLFLOW LOG METRICS
        for k in range(1, 4):
            score = trainer.accuracy(X_test, y_test, k=k)
            print(f'\ttest with {name_test} acc@{k}:', score)
            mlflow.log_metric(f"accuracy_{k}", score)
        mlflow.set_tag("exp_name", 'first')
        mlflow.sklearn.log_model(trainer.model, "model")


def main():
    """
    Set parameters and run pipeline with defined settings
    """
    parser = ArgumentParser()
    parser.add_argument("--n_jobs", default="2", type=int)
    parser.add_argument("--experiment_name", default="Best baseline", type=str)
    args = parser.parse_args()
    print(args)

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

    params_sets_list = []
    sv = SentenceVectorizer()
    for vectorizer_name in sv.get_availables_vectorizers():
        # initalize particular vectorizer method
        path = 'data/interim/'
        for name_train in os.listdir(path):
            if name_train not in ['smm4h17', 'smm4h21', 'psytar', 'cadec']:
                continue
            # PREPARE TRAIN SETS
            folder = os.path.join(path, name_train)
            corpus_train = folder + '/train.csv'
            for mlalg in [False, True]:
                # FIT MODEL
                for name_test in os.listdir(path):
                    if name_test not in ['smm4h17', 'smm4h21', 'psytar', 'cadec']:
                        continue
                    # PREPARE TEST SETS
                    folder = os.path.join(path, name_test)
                    corpus_test = folder + '/test.csv'
                    params_set = (
                        sv, vectorizer_name, meddra_labels,
                        name_train, corpus_train,
                        name_test, corpus_test, mlalg
                     )
                    params_sets_list.append(params_set)

    # run pipeline in multiple processes
    with multiprocessing.Pool(processes=args.n_jobs) as pool:
        pool.starmap(run_pipe, params_sets_list)
    print('DONE')

if __name__ == "__main__":
    main()
