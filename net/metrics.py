import numpy as np
from sklearn.metrics import confusion_matrix


class Metric(object):

    def __call__(self, y_pred, y_true, **kwargs):
        raise NotImplementedError


class LabelAccuracy(Metric):
    def __call__(self, y_pred, y_true, **kwargs):
        labels_amount = kwargs.get('labels_amount')

        labels = np.arange(labels_amount)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return np.diag(cm).sum() / cm.sum()
