import pickle

import numpy as np

from net.callbacks import ModelDump, SaveBestModel
from net.initializers import Xavier
from net.losses import categorical_cross_entropy
from net.metrics import LabelAccuracy
from net.model import SimpleNet
from net.optimizers import SGD
from settings import DATA_PATH
from training import Trainer


def load_data(filename):
    with open(filename, "rb") as f:
        train_data, val_data, test_data = pickle.load(f, encoding="latin1")
    return train_data, val_data, test_data


if __name__ == "__main__":
    np.random.seed(3)

    train_data, val_data, test_data = load_data(DATA_PATH)

    model_dictionary = {
        'optimizer': SGD(),
        'initializer': Xavier(),
        'metrics': [LabelAccuracy()],
        'loss_fun': categorical_cross_entropy,
        'activation': 'sigmoid',
        'hidden_units': (500,)
    }

    train_dictionary = {
        'train_data': train_data,
        'val_data': val_data,
        'epochs': 30,
        'batch_size': 50,
        'callbacks': [
            ModelDump(output_dir='my_nets/simple_net'),
            SaveBestModel(output_dir='my_nets/simple_net')
        ]
    }

    model = SimpleNet(**model_dictionary)

    trainer = Trainer(model, **train_dictionary)
    trainer.train_loop()
