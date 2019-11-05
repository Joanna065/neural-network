import os
import pickle
import sys

import numpy as np
from sklearn.metrics import confusion_matrix

from net.batcher import Batcher
from net.model import get_model

BEST_MODEL_FILENAME = "best_model.pkl"
BEST_VAL_ACCURACY_FILENAME = "best_val_accuracy.txt"
DUMP_FILENAME = "model_dump.txt"


def write_accuracy(output_dir, accuracy, epoch):
    """
    Writes and saves best achieved accuracy during training
    :param output_dir: directory where to save
    :param accuracy: accuracy on validation data
    :param epoch: epoch number in which that accuracy occurred
    """
    with open(os.path.join(output_dir, BEST_VAL_ACCURACY_FILENAME), "w") as f:
        f.write("accuracy = %f\n" % accuracy)
        f.write("epoch = %d\n" % epoch)


def load_data():
    FILENAME = "data/mnist.pkl"
    with open(FILENAME, "rb") as f:
        train_data, val_data, test_data = pickle.load(f, encoding="latin1")
    x_train, y_train = train_data
    x_val, y_val = val_data
    x_test, y_test = test_data
    return x_train, y_train, x_val, y_val, x_test, y_test


def validate(model, x_val, y_val):
    """
    Calculates accuracy on validaton dataset
    :param model:
    :param x_val: validation input data vector
    :param y_val: validation labels
    :return: accuracy
    """
    y_pred = model.predict_classes(x_val)
    labels = np.arange(10)
    cm = confusion_matrix(y_val, y_pred, labels=labels)
    return np.diag(cm).sum() / cm.sum()


def train_loop(model, input_file_weights, mini_batch_size, stop_epoch,
               optimizer, initializer, activation, output_dir):
    m = get_model(model, optimizer, initializer, activation)

    # load checkpoint weights if specified
    if input_file_weights:
        print("Loading weights from %s" % input_file_weights)
        m.load_variables(input_file_weights)

    x_train, y_train, x_val, y_val, _, _ = load_data()
    print("Training data shape: {}".format(x_train.shape))
    print("Validation data shape: {}".format(x_val.shape))

    next_val_epoch = 0
    best_val_epoch = 0
    best_val_accuracy = 0
    last_val_accuracy = 0

    b = Batcher(x_train, y_train)
    stop_reason = None
    try:
        while not stop_reason:
            x, y = b(mini_batch_size)
            loss = m.train(x, y)

            if b.epoch() > next_val_epoch:
                next_val_epoch += 1
                last_val_accuracy = validate(m, x_val, y_val)

                if last_val_accuracy > best_val_accuracy:
                    best_val_accuracy = last_val_accuracy
                    best_val_epoch = b.epoch()
                    m.save_variables(os.path.join(output_dir, BEST_MODEL_FILENAME))
                    write_accuracy(output_dir, best_val_accuracy, best_val_epoch)

            if 0 < stop_epoch < b.epoch():
                stop_reason = "stop epoch achieved"

            sys.stdout.write(
                "[epoch = %.2f] loss = %.5f, best val acc (epoch=%d) = %.3f, last val acc = %.3f     \r" %
                (b.epoch(), loss, best_val_epoch, best_val_accuracy, last_val_accuracy))
            sys.stdout.flush()
            m.save_variables(os.path.join(output_dir, BEST_MODEL_FILENAME))

    except KeyboardInterrupt:
        stop_reason = "requested by the user"
    print()

    m.save_variables(os.path.join(output_dir, BEST_MODEL_FILENAME))
    m.model_dump(os.path.join(output_dir, DUMP_FILENAME))

    return stop_reason, best_val_accuracy


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Start neural net training.")
    parser.add_argument("-m", "--model", type=str, default="SimpleNet")
    parser.add_argument("-in", "--initializer", type=str, default="Xavier")
    parser.add_argument("-opt", "--optimizer", type=str, default="SGD")
    parser.add_argument("-l", "--loss", type=str, default="cross-entropy")
    parser.add_argument("-act", "--activation", type=str, default="sigmoid")
    parser.add_argument("-mbs", "--mini-batch-size", type=int, default=100)
    parser.add_argument("-se", "--stop-epoch", type=int, default=30)
    parser.add_argument("-iw", "--input-file-weights", type=str)
    parser.add_argument("output_dir")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    stop_reason, best_accuracy = train_loop(args.model,
                                            args.input_file_weights,
                                            args.mini_batch_size,
                                            args.stop_epoch,
                                            args.optimizer,
                                            args.initializer,
                                            args.activation,
                                            args.output_dir)
    print("Training stopped (%s)" % stop_reason)
