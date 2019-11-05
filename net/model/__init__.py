from net.initializers import *
from net.layers.activations import *
from net.optimizers import *
from .architectures import *


def get_model(model_name, optimizer='SGDMomentum', initializer='Xavier', activation='sigmoid',
              hidden_units=(512,)):
    optimizer = get_optimizer(optimizer)
    initializer = get_initializer(initializer, activation)

    models = dict(
        SimpleNet=SimpleNet(optimizer, initializer, activation, hidden_units)
    )

    m = models[model_name]
    print("Creating model {} with parameters count: {}".format(model_name, m.param_count()))
    return m


def get_initializer(initializer_name, activation):
    initializers = dict(
        RandomInit=RandomInit(activation),
        Randomization=Randomization(),
        Xavier=Xavier(),
        KaimingHe=KaimingHe()
    )

    initializer = initializers[initializer_name]
    print("Using {} initializer".format(initializer_name))
    return initializer


def get_optimizer(optimizer_name):
    optimizers = dict(
        SGD=SGD(),
        SGDMomentum=SGDMomentum(),
        NAG=NAG(),
        Adagrad=Adagrad(),
        Adadelta=Adadelta(),
        RMSprop=RMSprop(),
        Adam=Adam(),
    )

    optimizer = optimizers[optimizer_name]
    print("Using {} optimizer".format(optimizer_name))
    return optimizer


def get_activation(activation_name):
    activations = dict(
        relu=ReLU(),
        sigmoid=Sigmoid(),
        tanh=Tanh()
    )
    activation = activations[activation_name]
    print("Using {} activation function in hidden layers".format(activation_name))
    return activation
