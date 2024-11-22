import numpy as np
from tf_keras.callbacks import Callback

class CustomMultitaskMetricCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):
        #print(logs.keys())
        metric = self._calculate_metric_on_val(logs)
        logs['val_multitask_f1_distance'] = metric

    def on_train_end(self, logs=None):
        #print(logs.keys())
        metric = self._calculate_metric_on_val(logs)
        logs['val_multitask_f1_distance'] = metric

    def _calculate_metric_on_val(self, logs):
        satd_f1 = logs['val_satd_f1']
        vulnerable_f1 = logs['val_vulnerable_f1']
        metric = self._calculate_metric(satd_f1, vulnerable_f1)
        return metric

    def _calculate_metric(self, satd_f1, vulnerable_f1):
        f1s = np.array([ satd_f1, vulnerable_f1 ])
        best = np.array([ 1.0, 1.0 ])
        metric = np.linalg.norm(best - f1s)
        return metric