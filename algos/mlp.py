import numpy as np

class MLP:
    def __init__(self, d_in, d_h, d_out):
        # Store dimensions
        self.d_in = d_in  # 784
        self.d_h = d_h    # 128
        self.d_out = d_out  # 10
        # Initialize weights and biases
        self.weights_inp_hid = np.random.randn(d_h, d_in) * np.sqrt(2.0 / d_in)  # He initialization
        self.bias_hidden = np.zeros((d_h,))
        self.weights_hid_out = np.random.randn(d_out, d_h) * np.sqrt(2.0 / d_h)
        self.bias_out = np.zeros((d_out,))

    def ReLU(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        # x shape: (d_out, batch_size) or (d_out,)
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # Subtract max for stability
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def forward(self, x):
        # x shape: (d_in, batch_size) or (d_in,)
        # Input to hidden
        z_one = self.weights_inp_hid @ x + self.bias_hidden[:, np.newaxis]
        act = self.ReLU(z_one)
        # Hidden to output
        z_two = self.weights_hid_out @ act + self.bias_out[:, np.newaxis]
        act_two = self.softmax(z_two)
        return act_two, z_one, act, z_two

    def backward(self, x, y_true, y_pred, z_one, act, z_two):
        # x: (d_in, batch_size), y_true: (d_out, batch_size), y_pred: (d_out, batch_size)
        batch_size = x.shape[1] if x.ndim == 2 else 1
        # Output layer gradients
        delta_out = y_pred - y_true  # (d_out, batch_size)
        grad_w_out = (delta_out @ act.T) / batch_size  # (d_out, d_h)
        grad_b_out = np.sum(delta_out, axis=1) / batch_size  # (d_out,)
        # Hidden layer gradients
        delta_hidden = (self.weights_hid_out.T @ delta_out) * (z_one > 0)  # ReLU derivative
        grad_w_inp_hid = (delta_hidden @ x.T) / batch_size  # (d_h, d_in)
        grad_b_hidden = np.sum(delta_hidden, axis=1) / batch_size  # (d_h,)
        return grad_w_inp_hid, grad_b_hidden, grad_w_out, grad_b_out

    def cross_entropy_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.sum(y_true * np.log(y_pred), axis=0)
        return np.mean(loss)

    def train(self, X_train, Y_train, X_val, Y_val, epochs=10, batch_size=32, lr=0.01):
        n_samples = X_train.shape[0]
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            Y_train = Y_train[indices]
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_train[i:i+batch_size].T  # (d_in, batch_size)
                Y_batch = Y_train[i:i+batch_size].T  # (d_out, batch_size)
                # Forward
                y_pred, z_one, act, z_two = self.forward(X_batch)
                # Loss
                loss = self.cross_entropy_loss(Y_batch, y_pred)
                # Backward
                grad_w_inp_hid, grad_b_hidden, grad_w_out, grad_b_out = self.backward(
                    X_batch, Y_batch, y_pred, z_one, act, z_two
                )
                # Update parameters
                self.weights_inp_hid -= lr * grad_w_inp_hid
                self.bias_hidden -= lr * grad_b_hidden
                self.weights_hid_out -= lr * grad_w_out
                self.bias_out -= lr * grad_b_out
            # Validation accuracy
            y_val_pred, _, _, _ = self.forward(X_val.T)
            acc = np.mean(np.argmax(y_val_pred, axis=0) == np.argmax(Y_val, axis=1))
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Val Accuracy: {acc:.4f}")
