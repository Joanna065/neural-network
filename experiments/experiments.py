from collections import defaultdict
from time import time

from net.callbacks import LoggerUpdater
from net.initializers import *
from net.layers.activations import *
from net.losses import categorical_cross_entropy
from net.metrics import LabelAccuracy
from net.model import SimpleNet
from net.optimizers import *
from settings import DATA_PATH
from training import Trainer
from utils import plot_val_loss_per_batch, load_data, plot_val_loss, plot_val_accuracy, \
    plot_time_bar, plot_val_vs_train_acc


def batch_size_experiment(model_dict, train_dict):
    def exp_generator():
        for batch_size in [5000, 2000, 500, 100, 30, 1]:
            train_dict['batch_size'] = batch_size
            yield model_dict, train_dict, 'batch_size', batch_size

    return exp_generator


def initializer_experiment(model_dict, train_dict):
    def exp_generator():
        for initializer in [
            ZeroInit(),
            NormalInit(),
            RandomInit(model_dict['activation']),
            Randomization(),
            Xavier(),
            KaimingHe()
        ]:
            model_dict['initializer'] = initializer
            yield model_dict, train_dict, 'initializer', initializer.name

    return exp_generator


def hidden_layer_experimnt(model_dict, train_dict):
    def exp_generator():
        for hidden_units in [(10,), (100,), (300,), (500,)]:
            train_dict['hidden_units'] = hidden_units
            yield model_dict, train_dict, 'hidden_layer', hidden_units

    return exp_generator


def activation_fun_experiment(model_dict, train_dict):
    def exp_generator():
        for act_fun in [Sigmoid, ReLU, Tanh]:
            model_dict['activation'] = act_fun
            yield model_dict, train_dict, 'activation', act_fun.name

    return exp_generator


def optimizer_experiment(model_dict, train_dict):
    def exp_generator():
        for optimizer, params in [
            (SGD(), ': lr=0.1'),
            (SGD(), ': lr=0.01'),
            (SGDMomentum(), ': lr=0.01'),
            (NAG(), ': lr=0.01'),
            (Adagrad(), ': rho=0.01'),
            (Adadelta(), ': rho=0.95'),
            (RMSprop(), ': rho=0.9, eta=0.001'),
            (Adam(), ': lr=0.001')
        ]:
            model_dict['optimizer'] = optimizer
            yield model_dict, train_dict, 'optimizer', optimizer.name + params

    return exp_generator


def run_experiment(experiment_generator, out_dir, test_data, plot_loss_batch=False):
    np.random.seed(12345)
    results = defaultdict(list)

    for i, (model_dict, train_dict, exp_name, value) in enumerate(experiment_generator()):
        model = SimpleNet(**model_dict)
        trainer = Trainer(model, **train_dict)

        label = exp_name + str(value)
        print(f'{i}. {label}')

        start_time = time()
        trainer.train_loop()
        time_period = time() - start_time

        log_data = trainer.logger.logging_data

        if plot_loss_batch:
            # plot train loss per batch in first epoch
            filename = exp_name + str(value) + '_loss_one_batch'
            plot_val_loss_per_batch(log_data['loss_batch']['train'], filename, out_dir)

        results['model_dict'].append(model_dict)
        results['train_dict'].append(train_dict)
        results['time'].append(time_period)
        results['label'].append(label)
        results['log_data'].append(log_data)

        # calculate accuracy on test data
        acc_metric = LabelAccuracy()
        x_test, y_test = test_data
        accuracy = acc_metric(model.predict_classes(x_test), y_test)
        print('Accuracy on test data: {}'.format(accuracy))

    return results


if __name__ == "__main__":
    train_data, val_data, test_data = load_data(DATA_PATH)

    out_dir = 'experiments'

    model_dict = {
        'optimizer': SGD(),
        'initializer': Xavier(),
        'metrics': [LabelAccuracy()],
        'loss_fun': categorical_cross_entropy,
        'activation': Sigmoid,
        'hidden_units': (500,)
    }

    train_dict = {
        'train_data': train_data,
        'val_data': val_data,
        'epochs': 30,
        'batch_size': 50,
        'callbacks': [
            # ModelDump(output_dir=out_dir),
            # SaveBestModel(output_dir=out_dir),
            LoggerUpdater()
        ]
    }

    # Experiment - BATCH SIZE
    results = run_experiment(batch_size_experiment(model_dict, train_dict), out_dir='batch_size',
                             test_data=test_data, plot_loss_batch=True)
    plot_val_loss(results, dirname='batch_size')
    plot_val_accuracy(results, dirname='batch_size')
    plot_time_bar(results, dirname='batch_size')

    # Experiment - HIDDEN LAYER SIZE
    results = run_experiment(hidden_layer_experimnt(model_dict, train_dict), out_dir='hidden_layer',
                             test_data=test_data)
    plot_val_loss(results, dirname='hidden_layer')
    plot_val_accuracy(results, dirname='hidden_layer')
    plot_val_vs_train_acc(results, dirname='hidden_layer')

    # Experiment - initializers
    results = run_experiment(initializer_experiment(model_dict, train_dict), out_dir='initializer',
                             test_data=test_data)
    plot_val_loss(results, dirname='initializer')
    plot_val_accuracy(results, dirname='initializer')

    # Experiment - ACTIVATION FUNCTIONS
    results = run_experiment(activation_fun_experiment(model_dict, train_dict),
                             out_dir='activation', test_data=test_data)
    plot_val_loss(results, dirname='activation')
    plot_val_accuracy(results, dirname='activation')

    # Experiment - OPTIMIZERS
    results = run_experiment(optimizer_experiment(model_dict, train_dict), out_dir='optimizer',
                             test_data=test_data)
    plot_val_loss(results, dirname='optimizer')
    plot_val_accuracy(results, dirname='optimizer')
    plot_time_bar(results, dirname='optimizer')
