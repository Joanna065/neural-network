import os
import pickle

from net import *
from net.initializers import Xavier
from net.losses import softmax, categorical_cross_entropy
from net.optimizers import SGDMomentum


class Model(object):
    def __init__(self, optimizer=None, initializer=None):
        self.__layers = []
        self.__optimizer = optimizer
        self.__initializer = initializer if initializer is not None else Xavier()

    def add(self, layer):
        if len(self.__layers) > 0:
            layer.build(self.__layers[-1].output_shape(), self.__initializer, self.__optimizer)
        self.__layers.append(layer)

    def predict(self, x):
        """
        :param x: N x D
        :return: N x classes
        """
        set_learning_phase(False)
        y = x
        for l in self.__layers:
            y = l.forward(y)
        return softmax(y)

    def predict_classes(self, x):
        # take max prob classification and reshape to (N, 1)
        return self.predict(x).argmax(axis=1)[:, np.newaxis]

    def setup_train(self, learning_rate):
        # setup optimizer
        if self.__optimizer is not None:
            self.__optimizer.set_learning_rate(learning_rate)
        else:
            self.__optimizer = SGDMomentum(learning_rate=learning_rate)
            # self.__optimizer = Adam(learning_rate=0.001)

    def train(self, x, y):
        set_learning_phase(True)

        # forward pass
        cache = []
        for l in self.__layers:
            cache.append(x)
            x = l.forward(x)

        # loss computation
        loss, dscores = categorical_cross_entropy(x, y)

        # backpropagation
        updates = {}
        dy = dscores
        for idx in reversed(range(len(self.__layers))):
            # g - dict of grad values from each layer
            x = cache[idx]

            g = self.__layers[idx].backward(x, dy)
            # dx from prev layer is dy for earlier layer
            dy = g.pop("dx")
            updates["layer{}".format(idx)] = g

        updates = self.__optimizer.compute_update(updates)
        for idx, l in enumerate(self.__layers):
            l.apply_gradients(updates["layer{}".format(idx)])

        return loss

    def load_variables(self, filename):
        with open(filename, "rb") as f:
            vars = pickle.load(f)
        for idx, layer in enumerate(self.__layers):
            layer.load_variables(vars["layer{}".format(idx)])

    def save_variables(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self.get_variables(), f)

    def get_variables(self):
        vars = {}
        for idx, layer in enumerate(self.__layers):
            vars["layer{}".format(idx)] = layer.get_variables()
        return vars

    def param_count(self):
        count = 0
        for l in self.__layers:
            for var in l.get_variables().values():
                count += np.prod(var.shape)
        return count

    def output_shape(self):
        return self.__layers[-1].output_shape()

    def model_dump(self, filename):
        with open(filename, "w") as f:
            f.write("initializer: {}\n".format(self.__initializer.__str__()))
            f.write("optimizer: {}\n".format(self.__optimizer.__str__()))

            for layer, vars in self.get_variables().items():
                desc = "{}: ".format(layer)
                if not vars:
                    desc += "no variables"
                else:
                    for name, value in vars.items():
                        desc += "{} {}, ".format(name, value.shape)
                f.write("{}\n".format(desc))
