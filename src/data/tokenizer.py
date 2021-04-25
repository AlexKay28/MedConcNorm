import numpy as np
import pandas as pd
from gensim.utils import lemmatize
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

class tokenizer_NLTK:

    def __init__(self):
        pass

    def tokenize(self, text):
        text_tokens = word_tokenize(text)
        tokens_without_sw = [word.lower() for word in text_tokens if not word in stopwords.words()]
        return tokens_without_sw

class tokenizer_gensim:
    from gensim.utils import tokenize

    def __init__(self):
        pass

    def tokenize(self, text):
        return [wd.split('/')[0] for wd in lemmatize(text)]
        #return list(tokenize(sent, lower=True))

class tokenizer_spacy:
    def __init__(self):
        self.spacy_model = spacy.load('en_core_web_sm')

    def tokenize(self, text):
        text_tokens = word_tokenize(text)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
        text = ' '.join(tokens_without_sw)
        return [word.lemma_.lower() for word in self.spacy_model(text)]

class Tokenizer:

    _tokenizers = {
        'nltk': tokenizer_NLTK,
        'gensim': tokenizer_gensim,
        'spacy': tokenizer_spacy
    }

    def __init__(self, tokenizer_name):
        self.tokenizer = self._tokenizers[tokenizer_name]()

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)
