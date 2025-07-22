# utils/data_split.py
import numpy as np

def train_test_split(X, y, test_size=0.2, seed=None):
    if seed:
        np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    test_size = int(len(X) * test_size)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
