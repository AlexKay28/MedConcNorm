import re
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

from tensorflow.keras.preprocessing.text import Tokenizer
from sent2vec.vectorizer import Vectorizer

from gensim.models import FastText
from gensim.test.utils import common_texts

VEC_SIZE = 100

class SentenceVectorizer:

    __available_tokenizers = {
        'tf_keras': Tokenizer
    }
    __available_vectorizers = {
        # 'tfidf': 1,
        'fasttext': 1,
        'sent2vec': 1,
        'bert': 1
    }

    def __init__(self, n_jobs=5):
        self.n_jobs = n_jobs
#        self.tfidf_vectorizer = None

    def tokenize_sent(self, data, model_tok, text_column="text", num_words=5000, oov_token="<OOV>"):
        self.model_tok = self.__available_tokenizers[model_tok]()
        pass

    def vectorize_sent_bow(self, data, text_column="text"):
        pass #TODO

    def vectorize_sent_sent2vec(self, data, feat_col='term'):
        vectorizer = Vectorizer()
        vectorizer.bert(data[feat_col])
        vectors = vectorizer.vectors
        data[feat_col+'_vec'] = [[i][0] for i in vectors]
        return data

    def vectorize_sent_w2v(self, data, text_column="text"):
        pass #TODO

    def vectorize_sent_ft(self, data, feat_col='term', text_columns=None, size=VEC_SIZE):
        model = FastText(size=size, window=5, min_count=1)
        model.build_vocab(sentences=common_texts)
        model.train(sentences=common_texts, total_examples=len(common_texts), epochs=10)
        def sent2vec(sent, model=model):
            def aggregate(vecs, agg_type):
                if agg_type == 'avg':
                    return vecs.mean(axis=0)
            sent = sent.split(' ')
            sent = np.array([model.wv[word] for word in sent])
            sent_vec = aggregate(sent, 'avg')
            return sent_vec
        data[feat_col+'_vec'] = data[feat_col].apply(lambda x: sent2vec(x))
        return data

    def vectorize_sent_bert(self, data, text_column="text"):
        model_vec = self.__available_vectorizers['sent2vec']()
        model_vec.bert(data[text_column])
        vectors = model_vec.vectors
        return pd.concat([data, pd.DataFrame(vectors)], axis=1)

    def vectorize_span_bert(self, data, feat_col='term', text_col='text', bert_type='bert-base-uncased', agg_type='mean'):
        tokenizer = BertTokenizer.from_pretrained(bert_type)
        model = TFBertModel.from_pretrained(bert_type)

        def get_vecors_from_context(text, span):
            if len(text) > 512:
                try:
                    #print('BIG TEXT')
                    start = re.search(span.lower(), text.lower())
                    if start is None:
                        text = span
                    else:
                        start = start.span()[0]
                        start = 0 if start-255 < 0 else start-255
                        text = text[start:start+256]
                except Exception as e:
                    print(e)
                    print(text)
                    text = span
            span_vecs = []
            #print('the text:', text)
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

        if text_col in data.columns:
            data[feat_col+'_vec'] = data.progress_apply(
                lambda x: aggregation_type(get_vecors_from_context(x[text_col], x[feat_col])), axis=1)
        else:
            data[feat_col+'_vec'] = data.progress_apply(
                lambda x: aggregation_type(get_vecors_from_context(x[feat_col], x[feat_col])), axis=1)
        return data.dropna()

    def vectorize(self, data, vectorizer_name='fasttext'):

        if vectorizer_name=='fasttext':
            data = self.vectorize_sent_ft(data)
        elif vectorizer_name=='bert':
            data = self.vectorize_span_bert(data)
        elif vectorizer_name=='sent2vec':
            data = self.vectorize_sent_sent2vec(data)
        else:
            raise KeyError('Unknown vectorizer!')
        return data


    def get_availables_vectorizers(self):
        return self.__available_vectorizers.keys()
