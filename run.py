import os
import pandas as pd
import numpy as np
import multiprocessing
import mlflow
import mlflow.sklearn
from argparse import ArgumentParser
from copy import copy
from sklearn.model_selection import train_test_split

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
EXPERIMENT_NAME = args.experiment_name
os.environ["RUN_NAME"] = RUN_NAME
os.environ["EXPERIMENT_NAME"] = EXPERIMENT_NAME
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from src.data.sentence_vectorizer import SentenceVectorizer
from src.purpose_models.trainer import Trainer
from src.features.metrics import accuracy_top_K_pobs, precision, recall, f1score
from src.configs import GENERAL, PREPROCESSING, AVAILABLE_CONFIGURATIONS, MODELING
from src.configs import create_random_configuration, delete_process_configuration_file
from src.features.novelty_detector import NoveltyDetector


#calc_subset = 'pure' # ['pure', 'concept', 'concept_retro', 'all_internal', 'big']
USE_MLALG = MODELING['use_metric_learning']
MODEL_NAME = MODELING['model_name']
distance_type = MODELING['distance_type']
VECTORIZER_NAME = PREPROCESSING['sentence_vectorizer']
AGG_TYPE = PREPROCESSING['agg_type']
tokenizer_name = PREPROCESSING['tokenizer_name']
ask_select_novelties = PREPROCESSING['ask_select_novelties']

# create log file
log_file = f'logs/{EXPERIMENT_NAME}_{RUN_NAME}_logs.log'
logging.basicConfig(
    filename=log_file,
    level=logging.ERROR,
    filemode="w+",
)

def log_dict_to_mlflow(dictionary):
    """
    This function logs nested dicts to mlflow
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            log_dict_to_mlflow(value)
        else:
            mlflow.log_param(str(key), value)

def run_pipe(folder, sv, meddra_labels, name_train, corpus_train, name_test, corpus_test, calc_subset):
    """
    Runs pipeline with defined settings and log it
    in MLflow Tracker node
    """
    # initalize particular vectorizer method
    print(f"vectorizer: {VECTORIZER_NAME}")

    with mlflow.start_run(run_name=RUN_NAME) as run:
        # PREPROCESSING, AVAILABLE_CONFIGURATIONS, MODELING
        log_dict_to_mlflow(GENERAL)
        log_dict_to_mlflow(PREPROCESSING)
        log_dict_to_mlflow(MODELING)

        # MLFLOW LOG PARAMS
        mlflow.log_param('train corpus', name_train)
        mlflow.log_param('test corpus', name_test)
        mlflow.log_param('vectorizer', VECTORIZER_NAME)
        mlflow.log_param("use_metric_learning", USE_MLALG)
        mlflow.log_param('model_name', MODEL_NAME)
        mlflow.log_param('ask_select_novelties', ask_select_novelties)
        mlflow.log_param('tokenizer_name', tokenizer_name)
        mlflow.log_param('distance_type', distance_type)
        mlflow.log_param('calc_subset', calc_subset)
        mlflow.log_param('folder', folder)

        try:
            mlflow.log_artifact(f'src/configs/temp/run_config_{RUN_NAME}.yml')
            if calc_subset == 'pure':
                train_file = f"train_pure_{folder}_{name_train}_{VECTORIZER_NAME}_{tokenizer_name}.pkl"

            elif calc_subset == 'augmented_textaugment_wdnt':
                train_file = f"train_aug_textaugment_wdnt_{folder}_{name_train}_{VECTORIZER_NAME}_{tokenizer_name}.pkl"
            elif calc_subset == 'augmented_nlpaug_wdnt':
                train_file = f"train_aug_nlpaug_wdnt_{folder}_{name_train}_{VECTORIZER_NAME}_{tokenizer_name}.pkl"
            elif calc_subset == 'augmented_nlpaug_ppdb':
                train_file = f"train_aug_nlpaug_ppdb_{folder}_{name_train}_{VECTORIZER_NAME}_{tokenizer_name}.pkl"

            elif calc_subset == 'augmented_textaugment_wdnt_insrt':
                train_file = f"train_aug_textaugment_wdnt_insrt_{folder}_{name_train}_{VECTORIZER_NAME}_{tokenizer_name}.pkl"
            elif calc_subset == 'augmented_textaugment_wdnt_2_repl':
                train_file = f"train_aug_textaugment_wdnt_2_rpl_{folder}_{name_train}_{VECTORIZER_NAME}_{tokenizer_name}.pkl"
            elif calc_subset == 'augmented_textaugment_wdnt_3_repl':
                train_file = f"train_aug_textaugment_wdnt_3_rpl_{folder}_{name_train}_{VECTORIZER_NAME}_{tokenizer_name}.pkl"
            elif calc_subset == 'augmented_textaugment_wdnt_insrt_retro':
                train_file = f"train_aug_textaugment_wdnt_insrt_retro_{folder}_{name_train}_{VECTORIZER_NAME}_{tokenizer_name}.pkl"

            elif calc_subset == 'extended':
                train_file = f"train_concept_ex_{folder}_{name_train}_{VECTORIZER_NAME}_{tokenizer_name}.pkl"
            elif calc_subset == 'extended_aug_ppdb':
                train_file = f"train_concept_ex_aug_pbdb_{folder}_{name_train}_{VECTORIZER_NAME}_{tokenizer_name}.pkl"
            elif calc_subset == 'concept':
                train_file = f"train_concept_{folder}_{name_train}_{VECTORIZER_NAME}_{tokenizer_name}.pkl"
            elif calc_subset == 'concept_retro':
                train_file = f"train_concept_retro_{folder}_{name_train}_{VECTORIZER_NAME}_{tokenizer_name}.pkl"
            elif calc_subset == 'all_internal':
                train_file = f"train_all_internal_{folder}_{name_train}_{VECTORIZER_NAME}_{tokenizer_name}.pkl"
            elif calc_subset == 'big':
                train_file = f"train_big_{folder}_{name_train}_{VECTORIZER_NAME}_{tokenizer_name}.pkl"
            else:
                raise KeyError(f"Such set as {name_train} isnt exists")

            file_path = 'data/processed/ml_classifier/' + train_file
            print("TRAIN FILE", train_file)
            if train_file in os.listdir('data/processed/ml_classifier'):
                logging.critical('Use cached train data')
                print('Use cached train data')
                train = pd.read_pickle(file_path)
            else:
                # PREPARE TRAIN SETS
                logging.critical(f'Creating new train pkl data file {file_path}')
                print('Creating new train pkl data file')
                print(f"Work with {name_train} ", end='.')
                train = pd.read_csv(corpus_train)
                print(train.shape)
                print(f'Vectorize train by {VECTORIZER_NAME} ', end='.')
                train = sv.vectorize(train, vectorizer_name=VECTORIZER_NAME)
                train.to_pickle('data/processed/ml_classifier/' + train_file)

            # PREPARE TEST SETS
            mlflow.set_tag("mlflow.note.content","<my_note_here>")
            test_file = f'data/processed/ml_classifier/test_{folder}_{name_test}_{VECTORIZER_NAME}_{tokenizer_name}.pkl'
            print("TEST FILE", test_file)
            if f"test_{folder}_{name_test}_{VECTORIZER_NAME}_{tokenizer_name}.pkl" in os.listdir('data/processed/ml_classifier'):
                logging.critical('Use cached test data')
                print('Use cached test data')
                test = pd.read_pickle(test_file)
            else:
                logging.critical('Creating new test pkl data file')
                print('Creating new test pkl data file')
                test = pd.read_csv(corpus_test)
                test = sv.vectorize(test, vectorizer_name=VECTORIZER_NAME)
                test.to_pickle(test_file)



            train['SNMS'] = train['SNMS'].apply(lambda x: [] if x is np.nan else x)
            test['SNMS'] = test['SNMS'].apply(lambda x: [] if x is np.nan else x)

            train = train[train['code'].isin(meddra_labels.keys())]
            test = test[test['code'].isin(meddra_labels.keys())]
            print(train.shape)
            print(test.shape)
            # train = train.dropna()
            # test  = test.dropna()

            # labels_codes = labels.unique()
            # corpus_train['code'] = corpus_train['code'].astype(int)
            # corpus_test['code'] = corpus_test['code'].astype(int)
            # corpus_train = corpus_train[corpus_train['code'].isin(labels_codes)]
            # corpus_test = corpus_test[corpus_test['code'].isin(labels_codes)]


            logging.info(f'train shape: {train.shape}')
            X_train, y_train = train['term_vec'], train['code']
            X_train = pd.DataFrame([pd.Series(x) for x in X_train]).to_numpy()
            y_train = y_train.progress_apply(lambda x: int(meddra_labels[x])).to_numpy()
            #test = test.dropna()
            logging.info(f'test shape: {test.shape}')
            X_test, y_test = test['term_vec'], test['code']
            X_test = pd.DataFrame([pd.Series(x) for x in X_test]).to_numpy()
            y_test = y_test.progress_apply(lambda x: int(meddra_labels[x])).to_numpy()

            # FIT Novelty Detector
            if ask_select_novelties:
                novdet = NoveltyDetector()
                novdet.fit(X_train)

            # FIT MODEL
            logging.critical('fitting model')
            print('Fit model ')
            trainer = Trainer()
            trainer.train_model(X_train, y_train, X_test, y_test)

            # Select Novelties
            if ask_select_novelties:
                X_test_novelties, X_test, novpred = novdet.select_novelties(X_test)
                y_test_novelties = y_test[novpred == -1]
                y_test = y_test[novpred != -1]

            mlflow.log_param('model_params', trainer.model.get_params())

            # MLFLOW LOG METRICS
            classes = trainer.get_classes()
            y_hat_train = trainer.predict_proba(X_train)
            y_hat_test =  trainer.predict_proba(X_test)

            # Set OOV Class
            if ask_select_novelties:
                y_test = np.concatenate(
                    [y_test, y_test_novelties])
                y_hat_test = np.concatenate(
                    [y_hat_test, np.ones((y_test_novelties.shape[0], y_hat_test.shape[1]))*-1])
                mlflow.log_metric(f"novelties", len(y_test_novelties))
                X_test = np.concatenate(
                    [X_test, X_test_novelties])

            mlflow.log_metric(f"precision", precision(y_test,  y_hat_test))
            mlflow.log_metric(f"recall", recall(y_test,  y_hat_test))
            mlflow.log_metric(f"f1", f1score(y_test,  y_hat_test))
            for k in [1, 3, 5]:
                score_train = accuracy_top_K_pobs(y_train, y_hat_train, classes, k=k)
                score_test  = accuracy_top_K_pobs(y_test,  y_hat_test,  classes, k=k)
                print(f'\tEVAL: with {name_test} acc@{k}: {score_train}/{score_test}')
                mlflow.log_metric(f"acc{k}train", score_train)
                mlflow.log_metric(f"acc{k}test", score_test)
            mlflow.set_tag("exp_name", 'first')
            mlflow.log_artifact(log_file)

            logging.critical(f'test shape {y_test.shape}, {X_test.shape}')
            seen_data_mask = np.isin(y_test, y_train)
            y_hat_test_seen = trainer.predict_proba(X_test[seen_data_mask])
            score_test_seen  = accuracy_top_K_pobs(
                y_test[seen_data_mask],  y_hat_test_seen,  classes, k=1)
            mlflow.log_metric(f"acc{1}test_seen", score_test_seen)
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
    labels = pd.read_csv('data/interim/used_codes_big.csv')
    labels = labels['code']
    meddra_labels = {v:k for k, v in enumerate(labels.unique())}

    # configure mlflow
    mlflow.set_experiment(args.experiment_name)
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(args.experiment_name).experiment_id
    notes = """
        SMM4H17: 67.2, 87-90, 91.73
        SMM4H21(20): 36-37, 42-44, 43-45
        CADEC: 63.23, 70-83. 86.94
        PsyTar: 60.15, 77-82, 85.76
    """
    client.set_experiment_tag(experiment_id, "mlflow.note.content", notes)

    create_random_configuration(AVAILABLE_CONFIGURATIONS)
    sv = SentenceVectorizer()
    path = 'data/interim/'

    #['smm4h17', 'smm4h21', 'psytar', 'cadec', 'combined']
    data_sets_train = ['smm4h21', 'smm4h17', 'psytar', 'cadec', "cadec_custom"]
    data_sets_train = ['cadec_custom']
    #['smm4h17', 'smm4h21', 'psytar', 'cadec', 'combined']
    #data_sets_test = ['smm4h17', 'smm4h21', 'psytar', 'cadec', 'combined']

    # data_sets_train = np.random.choice(data_sets_train, size=1)
    # data_sets_test = np.random.choice(data_sets_test, size=1)
    #for calc_subset in ['pure', 'concept', 'concept_retro']: #, 'all_internal', 'big']:
    #for calc_subset in ['augmented_textaugment_wdnt', 'augmented_nlpaug_wdnt', 'augmented_nlpaug_ppdb']:
    #for calc_subset in ['augmented_textaugment_wdnt_insrt', 'augmented_textaugment_wdnt_2_repl', 'augmented_textaugment_wdnt_3_repl']:
    #for calc_subset in ['augmented_textaugment_wdnt_insrt_retro']:
    calc_subsets = [
        'pure',
        'extended',
        #'all_internal',
        #'big',
        'concept',
        #'concept_retro',
        'extended_aug_ppdb',
        'augmented_textaugment_wdnt',
        #'augmented_nlpaug_wdnt',
        #'augmented_nlpaug_ppdb',
        #'augmented_textaugment_wdnt_insrt_retro',
        'augmented_textaugment_wdnt_insrt',
        #'augmented_textaugment_wdnt_2_repl',
        #'augmented_textaugment_wdnt_3_repl'
    ]


    data_sets_train = np.random.choice(data_sets_train, size=1)
    data_sets_test = data_sets_train
    calc_subsets = np.random.choice(calc_subsets, size=1)

    path_to_orig = 'data/interim/'

    folders = [f"folder_{i}" for i in [1, 2, 3, 4, 5]]
    #folders = np.random.choice(folders, size=1)
    #for folder in [f"folder_{i}" for i in [1, 2, 3, 4, 5]]:
    for folder in folders:
        for calc_subset in calc_subsets:
            for name_folder_train in os.listdir(path):
                if name_folder_train not in data_sets_train:
                    continue
                print(folder)
                # PREPARE TRAIN SETS
                train_data_folder = os.path.join(path, name_folder_train, folder)

                if calc_subset == 'pure':
                    corpus_train = train_data_folder + '/train_pure.csv'
                elif calc_subset == 'augmented_textaugment_wdnt':
                    corpus_train = train_data_folder + '/train_aug.csv'
                elif calc_subset == 'augmented_nlpaug_wdnt':
                    corpus_train = train_data_folder + '/train_aug_wdnt.csv'
                elif calc_subset == 'augmented_nlpaug_ppdb':
                    corpus_train = train_data_folder + '/train_aug_ppdb.csv'

                elif calc_subset == 'augmented_textaugment_wdnt_insrt':
                    corpus_train = train_data_folder + '/train_textaug_wdnt_insrt.csv'
                elif calc_subset == 'augmented_textaugment_wdnt_2_repl':
                    corpus_train = train_data_folder + '/train_textaug_wdnt_2_repl.csv'
                elif calc_subset == 'augmented_textaugment_wdnt_3_repl':
                    corpus_train = train_data_folder + '/train_textaug_wdnt_3_repl.csv'
                elif calc_subset == 'augmented_textaugment_wdnt_insrt_retro':
                    corpus_train = train_data_folder + '/train_textaug_wdnt_insrt_retro.csv'

                elif calc_subset == 'extended':
                    corpus_train = train_data_folder + '/train_ex.csv'
                elif calc_subset == 'extended_aug_ppdb':
                    corpus_train = train_data_folder + '/train_ex_aug_ppdb.csv'
                elif calc_subset == 'concept':
                    corpus_train = train_data_folder + '/train_concept.csv'
                elif calc_subset == 'concept_retro':
                    corpus_train = train_data_folder + '/train_concept_retro.csv'
                elif calc_subset == 'all_internal':
                    corpus_train = train_data_folder + '/train_all_internal.csv'
                elif calc_subset == 'big':
                    corpus_train = train_data_folder + '/train_big.csv'


                for name_folder_test in os.listdir(path):
                    if name_folder_test not in data_sets_test or \
                       (name_folder_test!=name_folder_train and name_folder_train != 'combined'):
                        continue
                    # PREPARE TEST SETS
                    test_data_folder = os.path.join(path, name_folder_test, folder)
                    corpus_test = test_data_folder + '/test.csv'

                    print(f"TRAIN FOLDER: {train_data_folder} | TEST FOLDER: {test_data_folder}")

                    run_pipe(
                        folder,
                        sv, meddra_labels,
                        name_folder_train, corpus_train,
                        name_folder_test, corpus_test,
                        calc_subset
                    )
    delete_process_configuration_file()
    print('DONE')

if __name__ == "__main__":
    main()
