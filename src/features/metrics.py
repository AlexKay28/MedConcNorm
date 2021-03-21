import numpy as np
from scipy.integrate import simps


def accuracy_top_K_pobs(y_true, y_score, k=1):
    true_counter = 0
    for real_case, pred_case in zip(y_true, y_score):
        top_probs = pred_case.argsort()[-k:]
        if real_case in top_probs:
            true_counter += 1
    return true_counter/y_score.shape[0]


def sensitivity_area_K_range(y_true, y_score, k_range=None):
    if k_range is None:
        k_range = [1, 11]
    assert len(k_range) == 2
    k_estimates = [accuracy_top_K_pobs(y_true, y_score, k=k) for k in range(k_range[0], k_range[1])]
    return np.trapz(k_estimates)
    