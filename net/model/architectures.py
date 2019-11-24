from net.layers import *
from net.layers.conv import Conv2D
from net.layers.flatten import Reshape, Flatten
from net.layers.pooling import MaxPool2d
from net.model.model import Model


class SimpleNet(Model):
    """
    simple architecture for flatten neural network with only dense hidden layers
    """

    def __init__(self, optimizer=None, initializer=None, metrics=None, loss_fun=None,
                 activation=None, hidden_units=None):
        Model.__init__(self, optimizer, metrics, loss_fun)
        self.activation = activation
        self.hidden_size = hidden_units

        self.add(Input(shape=(None, 28 * 28)))
        for units in hidden_units:
            self.add(Dense(units=units, weights_initializer=initializer))
            self.add(activation())
        self.add(Dense(units=10))


class SimpleConv(Model):
    def __init__(self, optimizer=None, metrics=None, loss_fun=None):
        Model.__init__(self, optimizer, metrics, loss_fun)

        self.add(Input(shape=(None, 28 * 28)))
        self.add(Reshape(output_shape=(None, 28, 28, 1)))
        self.add(Conv2D(filters=8, kernel_size=3, stride=1))    # 26 x 26
        self.add(ReLU())
        self.add(MaxPool2d())   # 13 x 13
        # self.add(Flatten())
        self.add(Reshape(output_shape=(None, 8 * 13 * 13)))
        self.add(Dense(units=500))
        self.add(ReLU())
        self.add(Dense(units=10))
