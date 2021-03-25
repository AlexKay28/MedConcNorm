import numpy as np
import pandas as pd

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

    def tokenize_sent(self, data, model_tok, text_column="text", num_words=5000, oov_token="<OOV>"):
        self.model_tok = self.__available_tokenizers[model_tok]()
        pass

    def vectorize_sent_bow(self, data, text_column="text"):
        pass #TODO

    def vectorize_sent_tfidf(self, data, text_column="text"):
        pass #TODO

    def vectorize_sent_w2v(self, data, text_column="text"):
        pass #TODO

    def vectorize_sent_ft(self, data, text_column="text"):
        pass #TODO

    def vectorize_sent_bert(self, data, text_column="text"):
        model_vec = self.__available_vectorizers['sent2vec']()
        print('Creating BERT vectores...')
        model_vec.bert(data[text_column])
        vectors = model_vec.vectors
        return pd.concat([data, pd.DataFrame(vectors)], axis=1)

    def get_availables_vectorizers(self):
        return self.__available_vectorizers.keys()
