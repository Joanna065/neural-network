import numpy as np

from net.layers.layer import Layer


class ReLU(Layer):
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x, dy):
        dx = np.where(x > 0, dy, 0)
        return dict(dx=dx)


class Sigmoid(Layer):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x, dy):
        sigm = self.forward(x)
        dx = dy * sigm * (1 - sigm)
        return dict(dx=dx)


class Tanh(Layer):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x, dy):
        dx = (1 - np.tanh(x) ** 2) * dy
        return dict(dx=dx)


class LeakyReLU(Layer):
    def forward(self, x):
        pass

    def backward(self, x, dy):
        pass
