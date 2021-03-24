import numpy as np
import random
from tqdm import tqdm
from imblearn.over_sampling import SMOTE

def class_sampler(X, y):
    uniq, counts = np.unique(y, return_counts=True)
    max_index = np.argmax(counts)
    max_value = uniq[max_index]
    max_counts = counts[max_index]
    
    for cls in tqdm(uniq):
        if cls == max_value:
            continue
        
#         for i in range(5):
#             X = np.vstack((X, random.choice(X[y==cls])))
#             y = np.append(y, cls)
            
        for i in range(2):
            n = X[y==cls].shape[0]
            X = np.vstack((X, X[y==cls]))
            for j in range(n):
                y = np.append(y, cls)

#     strategy = {k:500 for k in uniq}
#     oversample = SMOTE(k_neighbors=2, sampling_strategy=strategy)

    oversample = SMOTE(k_neighbors=2)
    X, y = oversample.fit_resample(X, y)

    return X, y

