import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

from tensorflow.keras.preprocessing.text import Tokenizer
from sent2vec.vectorizer import Vectorizer


class SentenceVectorizer:

    __available_tokenizers = {
        'tf_keras': Tokenizer
    }
    __available_vectorizers = {
        'sent2vec': Vectorizer
    }

    def __init__(self, n_jobs=5):
        self.n_jobs = n_jobs
        self.tfidf_vectorizer = None

    def tokenize_sent(self, data, model_tok, text_column="text", num_words=5000, oov_token="<OOV>"):
        self.model_tok = self.__available_tokenizers[model_tok]()
        pass

    def vectorize_sent_bow(self, data, text_column="text"):
        pass #TODO

    def vectorize_sent_tfidf(self, data_train, data_test, feat_col='term', text_columns=None):
        min_df = 10
        max_df = 500
        max_features = 400

        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=min_df, max_df=max_df, max_features=max_features)

        texts = []
        for col in text_columns:
            texts += data_train[col].to_list()

        self.tfidf_vectorizer.fit(texts)

        emb = self.tfidf_vectorizer.transform(data_train['term']).toarray()
        data_train['term_vec'] = [[i] for i in emb]
        emb = self.tfidf_vectorizer.transform(data_test['term']).toarray()
        data_test['term_vec'] = [[i] for i in emb]

        return data_train, data_test

    def vectorize_sent_w2v(self, data, text_column="text"):
        pass #TODO

    def vectorize_sent_ft(self, data, text_column="text"):
        pass #TODO

    def vectorize_sent_bert(self, data, text_column="text"):
        model_vec = self.__available_vectorizers['sent2vec']()
        model_vec.bert(data[text_column])
        vectors = model_vec.vectors
        return pd.concat([data, pd.DataFrame(vectors)], axis=1)

    def vectorize_span_bert(self, data, text_col='text', term_col='term', bert_type='bert-base-uncased', agg_type='mean'):
        tokenizer = BertTokenizer.from_pretrained(bert_type)
        model = TFBertModel.from_pretrained(bert_type)

        def get_vecors_from_context(text, span):
            span_vecs = []
            text_tokens = tokenizer.tokenize(text)
            span_tokens = tokenizer.tokenize(span)
            word_ids = tokenizer.encode(text_tokens)
            words_tokenized = tokenizer.decode(word_ids)
            word_ids_tf = tf.constant(word_ids)[None, :]  # Batch size 1
            outputs = model(word_ids_tf)
            vectors = outputs[0][0]  # The last hidden-state is the first element of the output tuple
            for word, code, vector in zip(text_tokens, word_ids, vectors):
                if word in span_tokens:
                    span_vecs.append(vector.numpy())
            return span_vecs

        def aggregation_type(numpy_array):
            agg = {
                'sum': np.sum,
                'mean': np.mean
            }
            return agg['mean'](numpy_array, axis=0)

        data['term_vec'] = data.progress_apply(lambda x: aggregation_type(get_vecors_from_context(x[text_col], x[term_col])), axis=1)
        return data

    def get_availables_vectorizers(self):
        return self.__available_vectorizers.keys()
