from keras_tuner import GridSearch
from keras_tuner.engine import tuner_utils
import copy

class NoCheckGridSearch(GridSearch):

    def __init__(
        self,
        hypermodel=None,
        objective=None,
        max_trials=None,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        **kwargs,
    ):
        super().__init__(hypermodel, 
                         objective, 
                         max_trials, 
                         seed, 
                         hyperparameters, 
                         tune_new_entries,
                         allow_new_entries,
                         max_retries_per_trial,
                         max_consecutive_failed_trials,
                         **kwargs)
        
    def run_trial(self, trial, *args, **kwargs):


        callbacks = kwargs.pop("callbacks", [])

        # Run the training process multiple times.
        histories = []
        for execution in range(self.executions_per_trial):
            copied_kwargs = copy.copy(kwargs)
            #callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial, execution)
            callbacks.append(tuner_utils.TunerCallback(self, trial))
            
            copied_kwargs["callbacks"] = callbacks
            obj_value = self._build_and_fit_model(trial, *args, **copied_kwargs)

            histories.append(obj_value)
        return histories