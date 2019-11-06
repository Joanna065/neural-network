import os

import numpy as np

from net.callbacks import ModelDump, SaveBestModel, LoggerUpdater
from net.initializers import Xavier
from net.losses import categorical_cross_entropy
from net.metrics import LabelAccuracy
from net.model import SimpleNet
from net.optimizers import SGD
from settings import DATA_PATH, PROJECT_PATH
from training import Trainer
from utils import load_data, ensure_dir_path_exists

if __name__ == "__main__":
    np.random.seed(3)

    train_data, val_data, test_data = load_data(DATA_PATH)

    out_dir = 'my_nets/simple_net'
    ensure_dir_path_exists(os.path.join(PROJECT_PATH, out_dir))

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
        'epochs': 10,
        'batch_size': 50,
        'callbacks': [
            ModelDump(output_dir=out_dir),
            SaveBestModel(output_dir=out_dir),
            LoggerUpdater()
        ]
    }

    model = SimpleNet(**model_dictionary)

    trainer = Trainer(model, **train_dictionary)
    trainer.train_loop()

    # check on test data
    acc_metric = LabelAccuracy()
    x_test, y_test = test_data
    accuracy = acc_metric(model.predict_classes(x_test), y_test)
    print('Accuracy on test data: {}'.format(accuracy))
