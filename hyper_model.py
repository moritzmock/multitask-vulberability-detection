from keras_tuner import HyperModel

class HyperModelFactoryAdapter(HyperModel):


    def __init__(self, 
                 factory, 
                 learning_rate_generator,
                 dropout_prob_generator,
                 l2_reg_lambda_generator,
                 shared_layer_generator,
                 batch_size_generator
                 ):
        self.factory = factory
        self.learning_rate_generator = learning_rate_generator
        self.dropout_prob_generator = dropout_prob_generator
        self.l2_reg_lambda_generator = l2_reg_lambda_generator
        self.shared_layer_generator = shared_layer_generator
        self.batch_size_generator = batch_size_generator

    def build(self, hp):
        
        return self.factory.create_model(self.learning_rate_generator(hp), 
                                  self.dropout_prob_generator(hp),
                                  self.l2_reg_lambda_generator(hp),
                                  self.shared_layer_generator(hp)
                                  )

    

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=self.batch_size_generator(hp),
            **kwargs
        )