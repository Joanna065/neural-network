import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from settings import SAVE_PATH


def load_data(filename):
    with open(filename, "rb") as f:
        train_data, val_data, test_data = pickle.load(f, encoding="latin1")
    return train_data, val_data, test_data


def ensure_dir_path_exists(path):
    """Checks if path is an existing directory and if not, creates it."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def plot_bar(y_data, labels, x_label='', y_label='', filename=None, dirname=None):
    """
    :param y_data: list of scalar values
    :param labels: list of labels for each bar
    :param x_label: name of x axis
    :param y_label: name of y axis
    :param filename: name of file to save
    :param dirname: directory to save
    """
    plt.tight_layout()
    plt.figure(figsize=(10, 5))
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    x = np.arange(len(labels))
    plt.bar(x, y_data)
    plt.xticks(x, labels, fontsize=5, rotation=30)

    if filename is not None:
        if dirname is not None:
            ensure_dir_path_exists(os.path.join(SAVE_PATH, dirname))
            plt.savefig(os.path.join(SAVE_PATH, dirname, filename) + '.png')
        else:
            plt.savefig(os.path.join(SAVE_PATH, filename) + '.png')
    plt.show()


def plot_data(data, legend_labels, x_label='', y_label='', filename=None, dirname=None):
    plt.tight_layout()
    plt.figure(figsize=(10, 5))
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for y_data, label in zip(data, legend_labels):
        x = np.arange(1, len(y_data) + 1)  # epoch amount
        plt.plot(x, y_data, label=label)

    plt.legend(loc='best')

    if filename is not None:
        if dirname is not None:
            ensure_dir_path_exists(os.path.join(SAVE_PATH, dirname))
            plt.savefig(os.path.join(SAVE_PATH, dirname, filename) + '.png')
        else:
            plt.savefig(os.path.join(SAVE_PATH, filename) + '.png')
    plt.show()


def plot_val_loss(results, dirname=None):
    val_loss_data = []
    for log_data in results['log_data']:
        val_loss_data.append(log_data['loss_epoch']['val'])

    filename = 'loss_val'
    plot_data(val_loss_data, results['label'], x_label='epoka', y_label='funkcja kosztu',
              filename=filename, dirname=dirname)


def plot_val_accuracy(results, dirname=None):
    val_acc_data = []
    for log_data in results['log_data']:
        val_acc_data.append(log_data['accuracy']['val'])

    filename = 'acc_val'
    plot_data(val_acc_data, results['label'], x_label='epoka', y_label='funkcja kosztu',
              filename=filename, dirname=dirname)


def plot_val_vs_train_acc(results, dirname):
    for (log_data, exp_label) in zip(results['log_data'], results['label']):
        val_acc = log_data['accuracy']['val']
        train_acc = log_data['accuracy']['train']
        legend_labels = ['dane treningowe', 'dane walidacyjne']

        filename = exp_label + '_val_train_acc'
        plot_data([train_acc, val_acc], legend_labels, x_label='epoka', y_label='skuteczność',
                  filename=filename, dirname=dirname)


def plot_time_bar(results, dirname):
    times = results['time']
    labels = results['label']

    filename = 'time'

    plot_bar(times, labels, x_label='', y_label='czas trwania uczenia',
             filename=filename, dirname=dirname)


def plot_val_loss_per_batch(data, filename, dirname):
    plot_data(data=[data], legend_labels=['dane treningowe'], x_label='batch',
              y_label='funkcja kosztu', filename=filename, dirname=dirname)
