import re


class Callback:
    #
    # def __init__(self):
    #     self.cb_name = camel2snake(type(self).__name__)
    #     # Trainer is assigned in trainer's init
    #     self.trainer = None

    def __init__(self):
        self.cb_name = camel2snake(type(self).__name__)
        self._validation_data = None
        self._model = None

    def set_trainer(self, trainer):
        self._trainer = trainer

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

    def on_forward_begin(self, **kwargs):
        pass

    def on_forward_end(self, **kwargs):
        pass

    def on_backward_begin(self, **kwargs):
        pass

    def on_backward_end(self, **kwargs):
        pass


def camel2snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


# class Sa
