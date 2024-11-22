import tf_keras.backend as backend
import tensorflow as tf
from tf_keras.src.losses import LossFunctionWrapper
from tf_keras.src.utils import losses_utils
from tf_keras.losses import BinaryCrossentropy

def weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        bce = BinaryCrossentropy()
        loss = bce(y_true, y_pred)

        weight_vector = y_true * one_weight + (1 - y_true) * zero_weight
        weighted_loss = weight_vector * loss

        return backend.mean(weighted_loss)

    return weighted_binary_crossentropy

class WeightedBinaryCrossEntropy(LossFunctionWrapper):

    def __init__(self, 
                 zero_weight=0.5, 
                 one_weight=0.5,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name="weighted_binary_crossentropy"):
        self.zero_weight = zero_weight
        self.one_weight = one_weight
        super().__init__(weighted_binary_crossentropy(zero_weight=zero_weight, one_weight=one_weight),
                         name=name,
                         reduction=reduction)
        
    def get_config(self):
        return {"zero_weight": self.zero_weight, "one_weight": self.one_weight, "reduction": self.reduction, "name": self.name}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
