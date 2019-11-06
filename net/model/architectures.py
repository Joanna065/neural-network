from net.layers import *
from net.model.model import Model


class SimpleNet(Model):
    """
    simple architecture for flatten neural ntwork with only dense hidden layers
    """

    def __init__(self, optimizer=None, initializer=None, metrics=None, loss_fun=None,
                 activation=None, hidden_units=None):
        Model.__init__(self, optimizer, initializer, metrics, loss_fun)
        self.activation = activation
        self.hidden_size = hidden_units

        self.add(Input(shape=(None, 28 * 28)))
        for units in hidden_units:
            self.add(Dense(units=units))
            self.add(activation())
        self.add(Dense(units=10))
