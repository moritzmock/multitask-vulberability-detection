from transformers import TFRobertaModel
from tf_keras import Model
from tf_keras.layers import Dense, Input, Dropout
from tf_keras.regularizers import L2
from tf_keras.metrics import Precision, Recall
from tf_keras.losses import BinaryCrossentropy

from tf_keras.optimizers import AdamW

from ext.weighted_binary_crossentropy import WeightedBinaryCrossEntropy
from f1 import F1
from ext.linear_decay_with_warmup import LinearDecayWithWarmup

class SingleTaskModelFactory():

    def __init__(self, name):
        self.name = name

    def build_model(self,  
                 learning_rate, 
                 dropout_prob, 
                 l2_reg_lambda, 
                 shared_layer, 
                 kernel_initializer,
                 weight_decay,
                 max_grad_norm,
                 adam_epsilon,
                 warmup_steps,
                 decay_steps,
                 gamma=None,
                 class_weight=None):

        print(locals())

        transformer_model = TFRobertaModel.from_pretrained("microsoft/codebert-base")
        input_ids = Input(shape=(512, ), dtype='int32', name='input_ids')
        attention_mask = Input(shape=(512, ), dtype='int32', name='attention_mask')
        transformer = transformer_model.roberta([input_ids, attention_mask])
        code_bert = transformer.last_hidden_state[:, 0, :]
        code_bert = Dropout(dropout_prob)(code_bert)

        if shared_layer:
            shared = Dense(768, 
                            kernel_initializer=kernel_initializer, 
                            activation='tanh')(code_bert)
            shared = Dropout(dropout_prob)(shared)
        else:
            shared = code_bert

        output = Dense(1, 
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=L2(l2_reg_lambda), 
                        bias_regularizer=L2(l2_reg_lambda),
                        activation='sigmoid', 
                        name=self.name)(shared)

        model = Model(inputs=[input_ids, attention_mask], outputs=[output])

        lr_scheduler = LinearDecayWithWarmup(initial_learning_rate=learning_rate, 
                                             warmup_steps=warmup_steps,
                                             decay_steps=decay_steps)


        loss = BinaryCrossentropy() if class_weight is None else WeightedBinaryCrossEntropy(zero_weight=class_weight[0], one_weight=class_weight[1])
        model.compile(loss=loss,
                      optimizer=AdamW(lr_scheduler, weight_decay, global_clipnorm=max_grad_norm, epsilon=adam_epsilon),
                      metrics=['accuracy', Precision(), Recall(), F1()])

        model.summary()

        return model
 