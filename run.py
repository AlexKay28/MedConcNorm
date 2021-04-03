import os
import pandas as pd
import numpy as np

from src.data.sentence_vectorizer import SentenceVectorizer

from src.purpose_models.trainer import Trainer
from argparse import ArgumentParser

import mlflow
import mlflow.sklearn

def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", default="Best baseline", type=str)
    parser.add_argument("--use_ml", default=False, type=bool)
    args = parser.parse_args()
    print(args)

    labels = pd.read_csv('data/interim/meddra_codes_terms_synonims.csv')
    labels = labels['CODE']
    meddra_labels = {v:k for k, v in enumerate(labels.unique())}
    results = {
        'vectorizer': [], 'train_model': [],
        'smm4h21': [],    'smm4h17': [],
        'psytar': [],     'cadec': [],
    }

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

    sv = SentenceVectorizer()
    for vectorizer_name in sv.get_availables_vectorizers():
        print(f"vectorizer: {vectorizer_name}")
        results['vectorizer'] += [vectorizer_name] * 4
        path = 'data/interim/'
        for name_train in os.listdir(path):

            if name_train not in ['smm4h17', 'smm4h21', 'psytar', 'cadec']:
                continue
            print(f"work with {name_train}")
            results['train_model'].append(name_train)

            folder = os.path.join(path, name_train)
            corpus_train = folder + '/train.csv'
            train = pd.read_csv(corpus_train)
            train = sv.vectorize(train, vectorizer_name=vectorizer_name)
            X_train, y_train = train['term_vec'], train['code']
            X_train = pd.DataFrame([pd.Series(x) for x in X_train])
            y_train = y_train.apply(lambda x: int(meddra_labels[x]))

            trainer = Trainer()
            if args.use_ml:
                trainer.train_model(X_train, y_train, mlalg=True)
            else:
                trainer.train_model(X_train, y_train, mlalg=False)

            for name_test in os.listdir(path):
                if name_test not in ['smm4h17', 'smm4h21', 'psytar', 'cadec']:
                    continue

                with mlflow.start_run() as run:

                    mlflow.set_tag("mlflow.note.content","<my_note_here>")
                    folder = os.path.join(path, name_test)
                    corpus_test = folder + '/test.csv'
                    test = pd.read_csv(corpus_test)
                    test = sv.vectorize(test, vectorizer_name=vectorizer_name)
                    X_test, y_test = test['term_vec'], test['code']
                    X_test = pd.DataFrame([pd.Series(x) for x in X_test])
                    y_test = y_test.apply(lambda x: int(meddra_labels[x]))

                    score = trainer.evaluate_model(X_test, y_test)
                    print(f'\ttest with {name_test} score:', score)
                    results[name_test].append(score)

                    # MLFLOW LOGS
                    mlflow.log_param('vectorizer', vectorizer_name)
                    mlflow.log_param('train corpus', name_train)
                    mlflow.log_param('test corpus', name_test)
                    mlflow.log_param("use_metric_learning", args.use_ml)

                    mlflow.log_metric("accuracy", score)
                    mlflow.set_tag("exp_name", 'first')

                    mlflow.sklearn.log_model(trainer.model, "model")


if __name__ == "__main__":
    main()
