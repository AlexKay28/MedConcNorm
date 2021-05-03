import numpy as np
from copy import deepcopy

def vectorize_sent(vec_model, sent):
    return np.array([vec_model.wv[w] for w in sent.split()]).mean(axis=0)

def retrofitting(word_vec, lexicons_vecs, iters=1):
    if len(lexicons_vecs) == 0:
        return word_vec
    new_word_vec = deepcopy(word_vec)
    for iteration in range(iters):
        new_word_vec *= len(lexicons_vecs)
        for lexicons_vec in lexicons_vecs:
            new_word_vec += lexicons_vec
        new_word_vec /= 2 * len(lexicons_vecs)
    return new_word_vec

def retrofit_row(vec_model, row, concept_row='STR', synonim_row='SNMS'):
    concept = row[concept_row]
    synonims = set(eval(row[synonim_row]))
    concept_vec = vectorize_sent(vec_model, concept)
    lexicons_vecs = [vectorize_sent(vec_model, syn) for syn in synonims]
    concept_vec = retrofitting(concept_vec, lexicons_vecs)
    return concept_vec
