import os
import numpy as np
import torch
from copy import deepcopy
from gensim.models import FastText
from transformers import BertTokenizer, TFBertModel, BertModel
from sentence_transformers import SentenceTransformer


vec_model_name = os.environ.get('vec_model_name', 'fasttext')
if vec_model_name == 'fasttext':
    print('loading fasttext')
    vec_model = FastText.load_fasttext_format(
        'data/external/embeddings/cc.en.300.bin')
elif vec_model_name == 'endr-bert':
    print('loading ENDR-BERT')
    tokenizer = BertTokenizer.from_pretrained('cimm-kzn/endr-bert')
    vec_model = BertModel.from_pretrained('cimm-kzn/endr-bert')
elif vec_model_name == 'SRoBERTa':
    print('loading SRoBERTa')
    vec_model = SentenceTransformer('xlm-roberta-base')
else:
    raise KeyError('Unknown vectorizer model')


def get_vecors_from_context_TORCH(text, span):
    """ Get sent vec with TOCH model """
    tokenized_text = tokenizer.tokenize(text)
    tokenized_span = tokenizer.tokenize(span)
    text_ids = tokenizer.encode(text)
    # get tokens vecs
    text_ids = torch.tensor(text_ids).unsqueeze(0)
    outputs = vec_model(text_ids)
    last_hidden_states = outputs[0]
    # define which vecs are related to span
    span_words_indices = [
        i+1 for i in range(len(tokenized_text)) if tokenized_text[i] in tokenized_span
    ][:len(tokenized_span)]
    span_vec = np.array(last_hidden_states[0][span_words_indices].tolist())
    return span_vec

def vectorize_sent_direct(sent):
    if vec_model_name == 'fasttext':
        return np.array([vec_model.wv[w] for w in sent.split()]).mean(axis=0)
    elif vec_model_name == 'endr-bert':
        return get_vecors_from_context_TORCH(sent, sent).mean(axis=0)
    elif vec_model_name == 'SRoBERTa':
        return vec_model.encode(sent)

def vectorize_sent_context(text, sent):
    if vec_model_name == 'fasttext':
        return np.array([vec_model.wv[w] for w in sent.split()]).mean(axis=0)
    elif vec_model_name == 'endr-bert':
        return get_vecors_from_context_TORCH(text, sent).mean(axis=0)
    elif vec_model_name == 'SRoBERTa':
        return vec_model.encode(sent)

def retrofitting(word_vec, lexicons_vecs, iters):
    if len(lexicons_vecs) == 0:
        return word_vec
    new_word_vec = deepcopy(word_vec)
    for iteration in range(iters):
        new_word_vec *= len(lexicons_vecs)
        for lexicons_vec in lexicons_vecs:
            new_word_vec += lexicons_vec
        new_word_vec /= 2 * len(lexicons_vecs)
    return new_word_vec

def vectorize_mention(row, term_row='term', text_row='sent'):
    if vec_model_name == 'fasttext':
        return vectorize_sent_direct(row[term_row])
    elif vec_model_name == 'endr-bert':
        if text_row in row:
            return vectorize_sent_context(row[text_row], row[term_row])
        elif 'text' in row:
            return vectorize_sent_context(row[term_row], row['text'])
        else:
            return vectorize_sent_context(row[term_row], row[term_row])
    elif vec_model_name == 'SRoBERTa':
        return vectorize_sent_direct(row[term_row])
    else:
        raise KeyError('Unknown vectorizer model')

def vectorize_concept(row, retro_iters, concept_row='STR', synonim_row='SNMS'):
    concept = row[concept_row]
    synonims = set(eval(row[synonim_row]))
    concept_vec = vectorize_sent_direct(concept)
    lexicons_vecs = [vectorize_sent_direct(syn) for syn in synonims]
    concept_vec = retrofitting(concept_vec, lexicons_vecs, iters=retro_iters)
    return concept_vec
