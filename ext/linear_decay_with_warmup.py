import tensorflow as tf
from tf_keras.optimizers.schedules import LearningRateSchedule

class LinearDecayWithWarmup(LearningRateSchedule):

    def __init__(
        self,
        initial_learning_rate,
        warmup_steps,
        decay_steps,
        end_learning_rate=0,
        name=None,
    ):
        super().__init__()

        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "LinearDecayWithWarmup") as name:
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            end_learning_rate = tf.cast(self.end_learning_rate, dtype)

            global_step_recomp = tf.cast(step, dtype)
            warmup_steps_recomp = tf.cast(self.warmup_steps, dtype)
            decay_steps_recomp = tf.cast(self.decay_steps, dtype)

            global_step_recomp = tf.minimum(
                    global_step_recomp, decay_steps_recomp
                )
            
            def warming(): return tf.multiply(initial_learning_rate, tf.divide(global_step_recomp, warmup_steps_recomp))
            def decaying(): 
                p = tf.divide(global_step_recomp - warmup_steps_recomp, decay_steps_recomp - warmup_steps_recomp)
                return tf.add(
                    tf.multiply(
                        initial_learning_rate - end_learning_rate,
                        1 - p,
                    ),
                    end_learning_rate,
                    name=name,
                )

            return tf.cond(tf.less(global_step_recomp, warmup_steps_recomp), warming, decaying)
                
        
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "end_learning_rate": self.end_learning_rate,
            "name": self.name,
        }