## Dataloader for mlp

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def load_data(dir_name):
    data = []
    with open(dir_name, "r") as f:
        for line in f:
            split_line = np.array(line.split(',')).astype(np.float32)
            data.append(split_line)

    data = np.asarray(data)
    return data[:, 1:], data[:, 0]

def load_mnist_data(train_path, test_path, val_split=0.1, seed=42):
    # Load raw data
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)

    # Normalize
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # One-hot encode
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

    # Split training into train + validation
    if val_split > 0:
        X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(
            X_train, y_train_onehot,
            test_size=val_split,
            random_state=seed
        )
    else:
        X_val = np.array([])
        y_val_onehot = np.array([])

    return X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot

