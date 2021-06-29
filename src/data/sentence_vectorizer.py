import re
import numpy as np
import pandas as pd
import pickle
import torch
from functools import reduce
from tqdm import tqdm
#import keras
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from sent2vec.vectorizer import Vectorizer
from transformers import BertTokenizer, TFBertModel, BertModel
from gensim.models import KeyedVectors
from gensim.models import FastText
from gensim.test.utils import common_texts
from gensim.utils import tokenize

from src.configs import GENERAL, PREPROCESSING
from src.data.tokenizer import Tokenizer
from sentence_transformers import SentenceTransformer

tqdm.pandas()
tf.executing_eagerly()

N_JOBS = GENERAL['n_jobs']
FT_EPOCHS = None # DEPRECATED: PREPROCESSING['fasttext_model']['epochs']
FT_WINDOW = None # DEPRECATED: PREPROCESSING['fasttext_model']['window']
AGG_TYPE = PREPROCESSING['agg_type']
TOKENIZER_NAME = PREPROCESSING['tokenizer_name']

#ft_model_path = 'data/external/embeddings/cc.en.300.bin'


class SentenceVectorizer:

    __available_vectorizers = [
        'fasttext_default_100',
        'fasttext_default_300',
        'fasttext_cadec_100',
        'fasttext_cadec_300',
        'fasttext_facebook',
        'bert-base-uncased',
        'bert-PubMed',
        "bertweet-base"
        'word2vec'
    ]

    def __init__(self, tokenizer_name=TOKENIZER_NAME, n_jobs=5):
        self.n_jobs = n_jobs
        self.word_tokenizer = Tokenizer(tokenizer_name)


    def vectorize_sent_sent2vec(self, data, feat_col='term'):
        vectorizer = Vectorizer()
        vectorizer.bert(data[feat_col])
        vectors = vectorizer.vectors
        data[feat_col+'_vec'] = [[i][0] for i in vectors]
        return data

    def pretrain_ft__model(self, corpus='default', size=300, epochs=FT_EPOCHS, window=FT_WINDOW):
        model = FastText(size=size, window=window, min_count=1)
        if corpus=='default':
            sentences = common_texts
            model.build_vocab(sentences=sentences)
            model.train(sentences=sentences, total_examples=len(sentences), epochs=epochs)
        elif corpus=='cadec':
            sentences = pd.read_csv('data/interim/cadec/test.csv')['text'].apply(
                lambda x: x.split('<SENT>')).explode().apply(lambda x: list(tokenize(x))).to_list()
            model.build_vocab(sentences=sentences)
            model.train(sentences=sentences, total_examples=len(sentences), epochs=epochs)
        else:
            raise KeyError('Unknown corpus name! (Kay)')
        return model

    def vectorize_sent_ft(self, data, feat_col='term', text_columns=None,
                                corpus='default'):
        print('LOADING MODEL')
        model = FastText.load_fasttext_format("data/external/embeddings/cc.en.300.bin")

        def sent2vec(sent, model=model):
            def aggregate(vecs, agg_type):
                if agg_type == 'mean':
                    return vecs.mean(axis=0)
                elif agg_type == 'max':
                    return vecs.max(axis=0)

            sent = self.word_tokenizer.tokenize(sent)
            sent = np.array([model.wv[word] for word in sent])
            sent_vec = aggregate(sent, AGG_TYPE)
            return sent_vec
        data[feat_col+'_vec'] = data[feat_col].progress_apply(lambda x: sent2vec(x) if len(x)>1 else x)
        return data

    def vectorize_sent_w2v(self, data, feat_col='term'):
        print('LOADING MODEL')
        model = KeyedVectors.load_word2vec_format(
            'data/external/embeddings/pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin', binary=True)

        def sent2vec(sent, model=model):
            def aggregate(vecs, agg_type):
                if agg_type == 'mean':
                    return vecs.mean(axis=0)
                elif agg_type == 'max':
                    return vecs.max(axis=0)

            sent = self.word_tokenizer.tokenize(sent)
            sent = np.array([model[word] for word in sent if word in model.vocab])
            sent_vec = aggregate(sent, AGG_TYPE)
            return sent_vec
        data[feat_col+'_vec'] = data[feat_col].progress_apply(lambda x: sent2vec(x) if len(x)>1 else x)
        return data

    def vectorize_sent_bert(self, data, text_column="text"):
        model_vec = self.__available_vectorizers['sent2vec']()
        model_vec.bert(data[text_column])
        vectors = model_vec.vectors
        return pd.concat([data, pd.DataFrame(vectors)], axis=1)

    def vectorize_span_bert(self, data, feat_col='term', text_col='sent',
                                  bert_type='bert-base-uncased', agg_type='mean'):
        # DEFINE FUNCS
        def get_vecors_from_context_TF(text, span):
            """ Get sent vec with TF model """
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

        def get_vecors_from_context_TORCH(text, span):
            """ Get sent vec with TOCH model """
            try:
                tokenized_text = tokenizer.tokenize(text)
                text_ids = tokenizer.encode(text)
            except Exception as e:
                print(text, span, e)
                tokenized_text = tokenizer.tokenize(span)
                text_ids = tokenizer.encode(span)
            tokenized_span = tokenizer.tokenize(span)

            # get tokens vecs
            text_ids = torch.tensor(text_ids).unsqueeze(0)
            outputs = model(text_ids)
            last_hidden_states = outputs[0]
            # define which vecs are related to span
            span_words_indices = [
                i+1 for i in range(len(tokenized_text)) if tokenized_text[i] in tokenized_span
            ][:len(tokenized_span)]
            span_vec = np.array(last_hidden_states[0][span_words_indices].tolist())
            return span_vec

        def aggregation_type(numpy_array, agg_type=AGG_TYPE):
            agg = {
                'sum': np.sum,
                'mean': np.mean
            }
            return agg[agg_type](numpy_array, axis=0)

        tokenizer = BertTokenizer.from_pretrained(bert_type)
        if bert_type in ['cimm-kzn/endr-bert']:
            model = BertModel.from_pretrained(bert_type)
            get_vecors_from_context = get_vecors_from_context_TORCH
        else:
            model = TFBertModel.from_pretrained(bert_type)
            get_vecors_from_context = get_vecors_from_context_TF

        if text_col in data.columns:
            data[feat_col+'_vec'] = data.progress_apply(
                lambda x: aggregation_type(get_vecors_from_context(x[text_col], x[feat_col])), axis=1)
        else:
            data[feat_col+'_vec'] = data.progress_apply(
                lambda x: aggregation_type(get_vecors_from_context(x[feat_col], x[feat_col])), axis=1)
        return data.dropna()

    def vectorize_encoder(self, data, feat_col='term', max_seq_len=10):
        with open('models/encoder/tokenizer.pickle', 'rb') as tok:
            tokenizer = pickle.load(tok)
        #encoder = keras.models.load_model('models/encoder/encoder_cadec.h5')
        tokenized = tokenizer.texts_to_sequences(data[feat_col])
        tokenized = sequence.pad_sequences(tokenized, maxlen=max_seq_len)
        data[feat_col+'_vec'] = tokenizer.texts_to_sequences(data[feat_col])
        data[feat_col+'_vec'] = sequence.pad_sequences(
            data[feat_col+'_vec'],
            maxlen=max_seq_len).tolist()
        data[feat_col+'_vec'] = data[feat_col+'_vec'].progress_apply(
            lambda seq: encoder.predict([seq])[0])
        return data

    def vectorize_tfidf(self, data, feat_col='term'):
        with open('models/tfidfvectorizer/vectorizer.pickle', 'rb') as tok:
            vectorizer = pickle.load(tok)
        lemmatized = data[feat_col].apply(
            lambda text: ' '.join(self.word_tokenizer.tokenize(text)))
        data[feat_col+'_vec'] = vectorizer.transform(lemmatized).todense().tolist()
        return data

    def vectorize_SRoBERTa(self, data, feat_col='term'):
        model = SentenceTransformer('cimm-kzn/endr-bert')
        sentence_embeddings = model.encode(data[feat_col])
        data[feat_col+'_vec'] = sentence_embeddings.tolist()
        return data

    def vectorize(self, data, vectorizer_name='fasttext'):
        print(f'Used vectorizer: {vectorizer_name}')
        if vectorizer_name=='fasttext_facebook':
            data = self.vectorize_sent_ft(data)
        elif vectorizer_name=='bert-base-uncased':
            data = self.vectorize_span_bert(data, bert_type='bert-base-uncased')
        elif vectorizer_name=='endr-bert':
            data = self.vectorize_span_bert(data, bert_type='cimm-kzn/endr-bert')
        elif vectorizer_name=='bert-PubMed':
            data = self.vectorize_span_bert(data, bert_type='cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
        elif vectorizer_name=='bertweet-base':
            data = self.vectorize_span_bert(data, bert_type="vinai/bertweet-base")
        elif vectorizer_name=='sent2vec':
            data = self.vectorize_sent_sent2vec(data)
        elif vectorizer_name=='encoder':
            data = self.vectorize_encoder(data)
        elif vectorizer_name=='tfidf':
            data = self.vectorize_tfidf(data)
        elif vectorizer_name=='word2vec':
            data = self.vectorize_sent_w2v(data)
        elif vectorizer_name=='SRoBERTa':
            data = self.vectorize_SRoBERTa(data)
        else:
            raise KeyError('Unknown vectorizer!')
        return data


    def get_availables_vectorizers(self):
        return self.__available_vectorizers
