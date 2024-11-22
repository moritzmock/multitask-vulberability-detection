import tensorflow as tf
from tf_keras.metrics import Metric
from tensorflow.python.util.tf_export import keras_export
from tf_keras.src.dtensor import utils as dtensor_utils
from tf_keras.src.utils import metrics_utils
from tf_keras import backend
import numpy as np

@keras_export("keras.metrics.F1")
class F1(Metric):
 

    @dtensor_utils.inject_mesh
    def __init__(
        self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold
        )
        self._thresholds_distributed_evenly = (
            metrics_utils.is_evenly_distributed_thresholds(self.thresholds)
        )

        self.true_positives = self.add_weight(
            "true_positives", shape=(len(self.thresholds),), initializer="zeros"
        )
        self.true_negatives = self.add_weight(
            "true_negatives", shape=(len(self.thresholds),), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            "false_positives", shape=(len(self.thresholds),), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            "false_negatives", shape=(len(self.thresholds),), initializer="zeros"
        )


    def update_state(self, y_true, y_pred, sample_weight=None):

        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,  # noqa: E501
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight,
        )

    def result(self):
        result = tf.math.divide_no_nan(
            self.true_positives,
            tf.math.add(self.true_positives, 0.5 * tf.math.add(self.false_positives, self.false_negatives)),
        )
        return result[0] if len(self.thresholds) == 1 else result

    def reset_state(self):
        num_thresholds = len(self.thresholds)
        confusion_matrix_variables = (
            self.true_positives,
            self.true_negatives,
            self.false_positives,
            self.false_negatives,
        )
        backend.batch_set_value(
            [
                (v, np.zeros((num_thresholds,)))
                for v in confusion_matrix_variables
            ]
        )

    def get_config(self):
        config = {
            "thresholds": self.init_thresholds,
            "top_k": self.top_k,
            "class_id": self.class_id,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))