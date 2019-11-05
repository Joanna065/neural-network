import os
import re

BEST_MODEL_FILENAME = "best_model.pkl"
BEST_VAL_ACCURACY_FILENAME = "best_val_accuracy.txt"
DUMP_FILENAME = "model_dump.txt"
PROJECT_PATH = '/home/joanna/Desktop/SIECI NEURONOWE/Sieci neuronowe/Laboratorium/neural_net'


class Callback:
    def __init__(self):
        self.cb_name = camel2snake(type(self).__name__)
        self.trainer = None
        self.params = None

    def set_params(self, params):
        self.params = params

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_epoch_begin(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
        pass

    def on_batch_begin(self, **kwargs):
        pass

    def on_batch_end(self, **kwargs):
        pass

    def on_train_batch_begin(self, **kwargs):
        pass

    def on_train_batch_end(self, **kwargs):
        pass

    def on_predict_batch_begin(self, **kwargs):
        pass

    def on_predict_batch_end(self, **kwargs):
        pass


def camel2snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class ModelDump(Callback):
    def on_train_begin(self, **kwargs):
        self.trainer.model.dump(os.path.join(PROJECT_PATH, DUMP_FILENAME))


class SaveBestModel(Callback):
    """
    Saves best model params
    """

    def __init__(self, output_dir):
        super().__init__()
        self._output_dir = output_dir
        self._best_accuracy = 0

    def on_epoch_end(self, **kwargs):
        epoch = kwargs.get("epoch")

        last_val_accuracy = self.trainer.logger.logging_data['accuracy']['val'][-1]

        if last_val_accuracy > self._best_accuracy:
            self._best_accuracy = last_val_accuracy
            self.trainer.model.save_variables(os.path.join(self._output_dir, BEST_MODEL_FILENAME))
            self._write_accuracy(epoch)

    def _write_accuracy(self, epoch):
        """
        Writes and saves best achieved accuracy during training
        :param epoch: epoch number in which that accuracy occurred
        """
        with open(os.path.join(PROJECT_PATH, self._output_dir, BEST_VAL_ACCURACY_FILENAME),
                  "w") as f:
            f.write("accuracy = %f\n" % self._best_accuracy)
            f.write("epoch = %d\n" % epoch)
