import numpy as np

from net import learning_phase
from net.layers.layer import Layer


class BatchNormalization(Layer):
    def __init__(self):
        super().__init__()
        self._mean = None
        self._gamma = None
        self._variance = None
        self._beta = None

    def _build(self):
        self._gamma = np.ones((self._input_shape[3],))
        self._beta = np.zeros((self._input_shape[3],))
        self._mean = np.zeros((self._input_shape[3],))
        self._variance = np.zeros((self._input_shape[3],))

    def load_variables(self, vars):
        self._mean = vars["mean"]
        self._variance = vars["variance"]
        self._gamma = vars["gamma"]
        self._beta = vars["beta"]

    def get_variables(self):
        return dict(
            mean=self._mean,
            variance=self._variance,
            gamma=self._gamma,
            beta=self._beta
        )

    def forward(self, x):
        eps = 1e-3
        if not learning_phase():
            if len(x.shape) == 2:
                return ((x - self._mean) / np.sqrt(self._variance + eps)) * self._gamma + self._beta
            if len(x.shape) == 4:
                index = [np.newaxis, np.newaxis, np.newaxis, slice(None)]
                return ((x - self._mean[index]) / np.sqrt(self._variance[index] + eps)) * \
                       self._gamma[index] + self._beta[index]
        else:
            return x

    def apply_gradients(self, grads):
        pass

    def backward(self, x, dy):
        # TODO implement backward
        return dict(dx=dy)
