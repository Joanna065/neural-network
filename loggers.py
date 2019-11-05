from collections import defaultdict


class LossAccLogger(object):
    def __init__(self, dirs=('loss', 'accuracy')):
        self._dirs = dirs
        self._tags = ['train', 'val']
        self._logging_data = defaultdict()
        self._init_logging_data()

    def _init_logging_data(self):
        for directory in self._dirs:
            self._logging_data[directory] = {tag: defaultdict(list) for tag in self._tags}

    def add_loss(self, tag, loss):
        self._logging_data['loss'][tag].append(loss)

    def add_accuracy(self, tag, accuracy):
        self._logging_data['accuracy'][tag].append(accuracy)


