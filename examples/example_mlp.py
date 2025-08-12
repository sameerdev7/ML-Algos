import numpy as np
from algos.mlp import MLP 
from utils.data_loader import load_mnist_data

train_path = "data/mnist_data/mnist_train.csv"
test_path = "data/mnist_data/mnist_test.csv"

X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot = load_mnist_data(train_path, test_path)

# Train MLP 
model = MLP(d_in=784, d_h=128, d_out=10)
model.train(X_train, y_train_onehot, X_val, y_val_onehot, epochs=10, batch_size=32, lr=0.01)

# Evaluate 
y_test_pred, _, _, _ = model.forward(X_test.T)
test_acc = np.mean(np.argmax(y_test_pred, axis=0) == np.argmax(y_test_onehot, axis=1))

print(f"Test accuracy: {test_acc:.4f}")
