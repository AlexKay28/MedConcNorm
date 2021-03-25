import os
import numpy as np
import pandas as pd
from src.data.sentence_vectorizer import SentenceVectorizer
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
DATA_TYPE = 'pure'
REPRESENTATION_TYPE = 'vectorize_sent_bert'

class DataPreparator:
    """Data preparing class"""

    def __init__(self, n_jobs=5):
        self.n_jobs = n_jobs

    def get_prepared_data(self):
        if DATA_TYPE == 'pure':
            data = self.prepare_pure_data()
        elif DATA_TYPE == 'enriched':
            data = self.prepare_enriched_data()

        pass

        return X_train, y_train, X_test, y_test

    def hand_craft(self):
        pass #TODO

    def vectorize_data(self):
        pass #TODO

    def prepare_pure_data(self):
        """
        Prepare only from SMM4H datasets 2017 and 2021
        """
        # smm4h17
        path_smm4h17 = '../data/external/smm4h_2017/'
        files = []
        for file in os.listdir(path_smm4h17):
            files.append(pd.read_csv(path_smm4h17 + file, sep='\t', header=None))
        smm4h17 = pd.concat(files)
        smm4h17 = smm4h17.rename(columns={
            1: 'text',
            2: 'code'
        })[['text', 'code']].drop_duplicates(subset=['text'])

        # smm4h21
        smm4h21 = pd.read_csv('../data/external/smm4h_2021/SMM4H_2021_train_spans.tsv', sep='\t', header=None)
        smm4h21 = smm4h21.rename(columns={
            4: 'text',
            5: 'code'
        })[['text', 'code']].drop_duplicates(subset=['text'])
        smm4h21_tweets = pd.read_csv('../data/external/smm4h_2021/SMM4H_2021_train_tweets.tsv',
            sep='\t', header=None)



        pure_data = pd.concat([smm4h17, smm4h21]).drop_duplicates('text').reset_index(drop=True)

        labels = {v:k for k, v in enumerate(pure_data['code'].unique())}
        pure_data['label'] = pure_data['code'].apply(lambda x: int(labels[x]))

        pure_data_train, pure_data_test = train_test_split(pure_data, test_size=TEST_SIZE)

        vectorizer = SentenceVectorizer()
        if REPRESENTATION_TYPE == 'vectorize_sent_bert':
            print('Creating BERT vectores...')
            pure_data_train = vectorizer.vectorize_sent_bert(pure_data_train, text_column="text")
            pure_data_test = vectorizer.vectorize_sent_bert(pure_data_test, text_column="text")
        elif REPRESENTATION_TYPE == 'vectorize_sent_bow':
            print('Creating BOW vectores...')
            pure_data_train = vectorizer.vectorize_sent_bow(pure_data_train, text_column="text")
            pure_data_test = vectorizer.vectorize_sent_bert(pure_data_test, text_column="text")
        elif REPRESENTATION_TYPE == 'vectorize_sent_tfidf':
            print('Creating TFIDF vectores...')
            pure_data_train = vectorizer.vectorize_sent_tfidf(pure_data_train, text_column="text")
            pure_data_test = vectorizer.vectorize_sent_bert(pure_data_test, text_column="text")
        elif REPRESENTATION_TYPE == 'vectorize_sent_w2v':
            print('Creating word2vec vectores...')
            pure_data_train = vectorizer.vectorize_sent_w2v(pure_data_train, text_column="text")
            pure_data_test = vectorizer.vectorize_sent_bert(pure_data_test, text_column="text")
        elif REPRESENTATION_TYPE == 'vectorize_sent_ft':
            print('Creating FastText vectores...')
            pure_data_train = vectorizer.vectorize_sent_ft(pure_data_train, text_column="text")
            pure_data_test = vectorizer.vectorize_sent_bert(pure_data_test, text_column="text")
        else:
            raise KeyError(f"""
                Such type of representations isnt available. choose from: {vectorizer.get_availables_vectorizers()}
                """)

        # now i must drop na values. capacity of shelf bert isnt enough
        # i should find bigger model
        pure_data_train = pure_data_train.drop(columns=['text', 'code']).dropna()
        pure_data_test = pure_data_test.drop(columns=['text', 'code']).dropna()

        target_columns = ['label']
        feature_columns = [i for i in pure_data_train if i not in target_columns]

        X_train = pure_data_train[feature_columns] #.to_numpy()
        y_train = pure_data_train[target_columns] #.to_numpy()

        X_test = pure_data_test[feature_columns] #.to_numpy()
        y_test = pure_data_test[target_columns] #.to_numpy()

        return X_train, y_train, X_test, y_test

    def prepare_enriched_data(self):
        """
        Concatenate terms from all datasets. Cheating?
        """
        pass #TODO
