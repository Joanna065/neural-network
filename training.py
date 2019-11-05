import sys

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
        out = self.model.predict(x_val)
        loss_val = self.model.compute_loss(out, y_val)

        y_pred = self.model.predict_classes(x_val)
        metric_measures = self.model.eval_metrics(y_pred, y_val)

        return metric_measures, loss_val

    def train_loop(self):
        x_train, y_train = self.train_data
        loss_accum = []
        train_preds = []

        best_epoch = 0
        best_val_accuracy = 0
        last_val_accuracy = 0
        b = Batcher(x_train, y_train)

        self._callback('on_train_begin')

        for epoch in range(self.epochs_amount):
            self._callback('on_epoch_begin', epoch=epoch + 1)
            train_pred = []

            while b.epoch() < epoch + 1:
                x, y = b(self.batch_size)
                self._callback('on_batch_begin', batch=(x, y))
                train_loss = self.model.train(x, y)
                loss_accum.append(train_loss)
                train_pred = self.model.predict_classes()
                train_pred.extend(train_pred)

                self._callback('on_batch_end')

            train_loss = int(np.mean(loss_accum))
            train_metrics = self.model.eval_metrics(train_pred, y_train)
            train_acc = train_metrics['label_accuracy']
            self.update_log('train', train_loss, train_acc)
            val_metrics, val_loss = self.validate()
            val_acc = val_metrics['label_accuracy']
            self.update_log('val', val_loss, val_acc)

            sys.stdout.write(
                "[epoch = %.2f] train_loss = %.5f, train_acc = %.3f,  val_loss = %.5f, val acc = %.3f     \r" %
                (b.epoch(), train_loss, train_acc, val_loss, val_acc))
            sys.stdout.flush()

            self._callback('on_epoch_end', epoch=epoch + 1)
            continue

        self._callback('on_train_end')

        #  #################################
        # stop_reason = None
        # try:
        #     while not stop_reason:
        #         x, y = b(mini_batch_size)
        #         loss = m.train(x, y)
        #
        #         if b.epoch() > next_val_epoch:
        #             next_val_epoch += 1
        #             last_val_accuracy = validate(m, x_val, y_val)
        #
        #             if last_val_accuracy > best_val_accuracy:
        #                 best_val_accuracy = last_val_accuracy
        #                 best_val_epoch = b.epoch()
        #                 m.save_variables(os.path.join(output_dir, BEST_MODEL_FILENAME))
        #                 write_accuracy(output_dir, best_val_accuracy, best_val_epoch)
        #
        #         if 0 < stop_epoch < b.epoch():
        #             stop_reason = "stop epoch achieved"
        #
        #         sys.stdout.write(
        #             "[epoch = %.2f] loss = %.5f, best val acc (epoch=%d) = %.3f, last val acc = %.3f     \r" %
        #             (b.epoch(), loss, best_val_epoch, best_val_accuracy, last_val_accuracy))
        #         sys.stdout.flush()
        #         m.save_variables(os.path.join(output_dir, BEST_MODEL_FILENAME))
        #
        # except KeyboardInterrupt:
        #     stop_reason = "requested by the user"
        # print()

    def _callback(self, func, *args, **kwargs):
        kwargs['trainer'] = self
        ls = []
        for cb in self.callbacks:
            result = getattr(cb, func)(*args, **kwargs)
            if result is not None:
                ls.append(result)
        return ls
