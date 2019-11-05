import pickle

import numpy as np

from net.callbacks import ModelDump, SaveBestModel
from net.initializers import Xavier
from net.losses import categorical_cross_entropy
from net.metrics import LabelAccuracy
from net.model import SimpleNet
from net.optimizers import SGD
from training import Trainer

FILENAME = "/home/joanna/Desktop/SIECI NEURONOWE/Sieci neuronowe/Laboratorium/neural_net/data/mnist.pkl"


def load_data(filename):
    with open(filename, "rb") as f:
        train_data, val_data, test_data = pickle.load(f, encoding="latin1")
    x_train, y_train = train_data
    x_val, y_val = val_data
    x_test, y_test = test_data
    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == "__main__":
    np.random.seed(3)

    x_train, y_train, x_val, y_val, x_test, y_test = load_data(FILENAME)

    model_dictionary = {
        'optimizer': SGD(),
        'initializer': Xavier(),
        'metrics': [LabelAccuracy()],
        'loss_fun': categorical_cross_entropy,
        'activation': 'sigmoid',
        'hidden_units': (500, )
    }

    train_dictionary = {
        'train_data': (x_train, y_train),
        'val_data': (x_val, y_val),
        'epochs': 30,
        'batch_size': 50,
        'callbacks': [
            ModelDump(),
            SaveBestModel(output_dir='my_nets/simple_net')
        ]
    }

    model = SimpleNet(**model_dictionary)

    trainer = Trainer(model, **train_dictionary)
    trainer.train_loop()

