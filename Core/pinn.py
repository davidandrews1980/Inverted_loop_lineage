import numpy as np

class TinyPINN:
    def __init__(self, hidden=16, weights=None):
        if weights is not None:
            self.W1, self.b1, self.W2, self.b2 = weights
        else:
            self.W1 = np.random.randn(3, hidden) * 0.1
            self.b1 = np.zeros(hidden)
            self.W2 = np.random.randn(hidden, 1) * 0.1
            self.b2 = np.zeros(1)

    def predict(self, t, pos, vel):
        x = np.array([t, pos, vel])
        h = np.tanh(x @ self.W1 + self.b1)
        return float(h @ self.W2 + self.b2)

    def get_weights(self):
        return (self.W1, self.b1, self.W2, self.b2)

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        weights = (data['W1'], data['b1'], data['W2'], data['b2'])
        return cls(weights=weights)
