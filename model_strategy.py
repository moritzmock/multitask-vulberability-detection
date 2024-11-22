from data_encoders import MultiTaskDataEncoder, VulOnlyDataEncoder, SATDOnlyDataEncoder, VulSATDDataEncoder
from models import MultiTaskModelFactory, SingleTaskModelFactory
from scores_calculator import SingleTaskScoresCalculator, MultiTaskScoresCalculator

from tf_keras.initializers import TruncatedNormal
from ext.codebert_tokenizer import CodeBERTTokenizer
from ext.codebert_tokenizer_fast import CodeBERTTokenizerFast

class ModelStrategy():

    def __init__(self, model_name):
        self.model_name = model_name

    def create_encoder(self, separate_comments, mask_satd_keywords, truncation_side = 'right'):
        tokenizer = CodeBERTTokenizer.from_pretrained("microsoft/codebert-base", truncation_side=truncation_side)
        max_length = 512

        if self.model_name == 'multitask':
            encoder = MultiTaskDataEncoder(tokenizer, max_length, separate_comments, mask_satd_keywords)
        elif self.model_name == 'vulonly':
            encoder = VulOnlyDataEncoder(tokenizer, max_length, separate_comments, mask_satd_keywords)
        elif self.model_name == 'satdonly':
            encoder = SATDOnlyDataEncoder(tokenizer, max_length, separate_comments, mask_satd_keywords)
        elif self.model_name == 'vulsatd':
            encoder = VulSATDDataEncoder(tokenizer, max_length, separate_comments, mask_satd_keywords)
        else:
            print('Unknown algorithm. Valid options are default, hidden and multitask')
        return encoder


    def create_model(self, learning_rate, dropout_prob, l2_reg_lambda, shared_layer, warmup_steps, decay_steps, gamma, class_weight):
        
        print("-----------------------")

        if self.model_name == 'multitask':
            model_factory = MultiTaskModelFactory()
        elif self.model_name == 'vulonly':
            model_factory = SingleTaskModelFactory('vulnerable')
        elif self.model_name == 'satdonly':
            model_factory = SingleTaskModelFactory('satd')
        elif self.model_name == 'vulsatd':
            model_factory = SingleTaskModelFactory('vulsatd')
        else:
            print('Unknown algorithm. Valid options are default, hidden, multitask, vulonly, satdonly, and vulsatd.')

        return model_factory.build_model(learning_rate,
                                         dropout_prob,
                                         l2_reg_lambda,
                                         shared_layer,
                                         kernel_initializer=TruncatedNormal(stddev=0.02),
                                         weight_decay=0,
                                         max_grad_norm=1.0,
                                         adam_epsilon=1e-8,
                                         warmup_steps=warmup_steps,
                                         decay_steps=decay_steps,
                                         gamma=gamma,
                                         class_weight=class_weight)
    
    def create_scores_calculator(self):
        if self.model_name == 'multitask':
            return MultiTaskScoresCalculator()
        return SingleTaskScoresCalculator()
    
    def get_metric_to_monitor(self):
        return 'val_multitask_f1_distance' if self.model_name == 'multitask' else 'val_f1'

    def get_metric_direction(self):
        return 'min' if self.model_name == 'multitask' else 'max'