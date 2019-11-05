import numpy as np

from net import dtype
from net.layers.layer import Layer


class Dense(Layer):
    """
    Dense layer performing the following operation:
      y = xW + b
      x - input of shape (N, M)
      W - weights of shape (M, units)
      b - bias of shape (units,)
      y - output of shape (N, units)
    """

    def __init__(self, units, use_bias=True, scale=1e-3):
        super().__init__()
        self.__units = units
        self.__scale = scale
        self.__use_bias = use_bias

    def load_variables(self, vars):
        assert self.__W.shape == vars["W"].shape
        self.__W = vars["W"]
        if self.__use_bias:
            assert self.__bias.shape == vars["bias"].shape
            self.__bias = vars["bias"]

    def get_variables(self):
        return dict(W=self.__W, bias=self.__bias) if self.__use_bias else dict(W=self.__W)

    def apply_gradients(self, grads):
        self.__W += grads["dW"]
        if self.__use_bias:
            self.__bias += grads["db"]

    def _build(self):
        assert len(self._input_shape) == 2, "input array to dense layer should be two-dimensional"
        W_shape = self._input_shape[1], self.__units
        self.__W = self._initializer(W_shape)

        if self.__use_bias:
            self.__bias = np.zeros((self.__units,), dtype=dtype())

    def output_shape(self):
        return None, self.__units

    def forward(self, x):
        return np.dot(x, self.__W) + self.__bias if self.__use_bias else np.dot(x, self.__W)

    def backward(self, x, dy):
        """
        :param x: input of shape (N, M)
        :param dy: upstream gradient of shape (N, units)
        :return: dict with gradients:
                  - dx - gradient w.r.t. input x of shape (N, M)
                  - dW - gradient w.r.t. weights W of shape (M, units)
                  - db - gradient w.r.t. bias b of shape (units, )
        """

        dx = np.dot(dy, self.__W.T)
        dW = np.dot(x.T, dy)
        if self.__use_bias:
            db = dy.sum(axis=0)
            return dict(dx=dx, dW=dW, db=db)
        return dict(dx=dx, dW=dW)
