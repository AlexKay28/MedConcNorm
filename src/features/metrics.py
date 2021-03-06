import numpy as np
from scipy.integrate import simps

from sklearn.metrics import precision_score, recall_score, f1_score


def accuracy_top_K_pobs(y_true, y_score, classes, k=1):
    true_counter = 0
    top_k_preds = y_score.argsort()[:, -k:]
    top_k_preds = map(lambda x: classes[x], top_k_preds)
    for real_class, pred_classes in zip(y_true, top_k_preds):
        if real_class in pred_classes:
            true_counter += 1
    return true_counter/y_true.shape[0]


def sensitivity_area_K_range(y_true, y_score, k_range=None):
    if k_range is None:
        k_range = [1, 11]
    assert len(k_range) == 2
    k_estimates = [accuracy_top_K_pobs(y_true, y_score, k=k) for k in range(k_range[0], k_range[1])]
    return np.trapz(k_estimates)

def precision(y_true, y_pred):
    y_pred = [np.argmax(p) for p in y_pred]
    return precision_score(y_true, y_pred, average='micro')

def recall(y_true, y_pred):
    y_pred = [np.argmax(p) for p in y_pred]
    return recall_score(y_true, y_pred, average='micro')

def f1score(y_true, y_pred):
    y_pred = [np.argmax(p) for p in y_pred]
    return f1_score(y_true, y_pred, average='micro')
