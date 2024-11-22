import numpy as np
from utils import calculate_scores

class SingleTaskScoresCalculator():

    def calculate(self, labels, predictions):
        labels = np.array(labels)
        calculate_scores(predictions, labels)


class MultiTaskScoresCalculator():

    def calculate(self, labels, predictions):

        satd_predictions = predictions[0]
        vul_predictions = predictions[1]

        labels = np.array(labels).squeeze().transpose()

        print("Measures for multitask - SATD")
        calculate_scores(satd_predictions, labels[0])
        print('\n')

        print("Measures for multitask - Vul")
        calculate_scores(vul_predictions, labels[1])
        print('\n')

        print("Measures for multitask - VulSATD")
        satd_predictions_boolean = satd_predictions > 0.5
        vul_predictions_boolean = vul_predictions > 0.5

        vulSATD_prediction = np.logical_and(satd_predictions_boolean, vul_predictions_boolean)

        vulSATD_label = np.logical_and(labels[0], labels[1])

        calculate_scores(vulSATD_prediction, vulSATD_label)
        print('\n')

