from collections import defaultdict


class LossAccLogger(object):
    def __init__(self, dirs=('loss', 'accuracy')):
        self._dirs = dirs
        self._tags = ['train', 'val']
        self.logging_data = defaultdict()
        self._init_logging_data()

    def _init_logging_data(self):
        for directory in self._dirs:
            self.logging_data[directory] = {tag: [] for tag in self._tags}

    def add_loss(self, tag, loss):
        self.logging_data['loss'][tag].append(loss)

    def add_accuracy(self, tag, accuracy):
        self.logging_data['accuracy'][tag].append(accuracy)


