import numpy as np


class Batcher(object):
    def __init__(self, x, y):
        """
        :param x: array of an arbitrary shape with first dimension is used to code samples. (N, D)
        :param y: sample labels N
        """
        self.__x = x
        self.__y = y
        self.__num_samples = len(y)
        self.__init()
        self.__epoch = 0

    def __init(self):
        self.__permutation = np.random.permutation(self.__num_samples)
        self.__consumed = 0

    def __call__(self, size):
        if self.__consumed + size <= self.__num_samples:
            indices = self.__permutation[self.__consumed:self.__consumed + size]
            x = self.__x[indices]
            y = self.__y[indices]
            self.__consumed += size
        else:
            indices = self.__permutation[self.__consumed:]
            x = self.__x[indices]
            y = self.__y[indices]
            remaining = size - x.shape[0]
            self.__init()
            rem_x, rem_y = self(remaining)
            x = np.concatenate([x, rem_x])
            y = np.concatenate([y, rem_y])
            self.__epoch += 1
        return x, y

    def epoch(self):
        return self.__epoch + float(self.__consumed) / self.__num_samples
