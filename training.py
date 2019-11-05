import numpy as np

from loggers import *
from net.batcher import Batcher


class Trainer:

    def __init__(self, model, train_data, val_data, epochs, batch_size, callbacks):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.epochs_amount = epochs
        self.batch_size = batch_size
        self.logger = LossAccLogger()

        self.callbacks = callbacks
        for cb in self.callbacks:
            cb.set_trainer(self)

    def update_log(self, tag, loss, accuracy):
        self.logger.add_loss(tag, loss)
        self.logger.add_accuracy(tag, accuracy)

    def validate(self):
        x_val, y_val = self.val_data
        out, _ = self.model.forward_pass(x_val, learning_phase=False)
        loss_val, _ = self.model.compute_loss(out, y_val)

        y_pred = self.model.predict_classes(x_val)
        metric_measures = self.model.eval_metrics(y_pred, y_val)

        return metric_measures, loss_val

    def train_loop(self):
        x_train, y_train = self.train_data
        b = Batcher(x_train, y_train)

        self._callback('on_train_begin')
        for epoch in range(self.epochs_amount):
            self._callback('on_epoch_begin', epoch=epoch + 1)
            loss_accum = []
            train_preds = []
            y_shuffle = []

            while b.epoch() < epoch + 1:
                x, y = b(self.batch_size)
                y_shuffle.extend(y)
                self._callback('on_batch_begin', batch=(x, y))
                train_loss = self.model.train(x, y)
                loss_accum.append(train_loss)
                train_pred = self.model.predict_classes(x)
                train_preds.extend(train_pred)
                self._callback('on_batch_end')

            train_loss = np.mean(loss_accum)
            train_metrics = self.model.eval_metrics(train_preds, y_shuffle)
            train_acc = train_metrics['label_accuracy']

            val_metrics, val_loss = self.validate()
            val_acc = val_metrics['label_accuracy']

            self.update_log('train', train_loss, train_acc)
            self.update_log('val', val_loss, val_acc)

            print(
                "[epoch = %d] train_loss = %.5f, train_acc = %.3f,  val_loss = %.5f, val acc = %.3f     \r" %
                (b.epoch(), float(train_loss), train_acc, val_loss, val_acc), flush=True)

            self._callback('on_epoch_end', epoch=epoch + 1)
            continue

        self._callback('on_train_end')

    def _callback(self, func, *args, **kwargs):
        kwargs['trainer'] = self
        ls = []
        for cb in self.callbacks:
            result = getattr(cb, func)(*args, **kwargs)
            if result is not None:
                ls.append(result)
        return ls
