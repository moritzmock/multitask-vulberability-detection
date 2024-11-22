import tf_keras
import tensorflow as tf
import pandas as pd
import numpy as np
from f1 import F1
from ext.linear_decay_with_warmup import LinearDecayWithWarmup
from ext.weighted_binary_crossentropy import WeightedBinaryCrossEntropy


def load_model(path):
    model = tf_keras.models.load_model(path, 
                                       custom_objects={'F1': F1, 
                                                       'LinearDecayWithWarmup': LinearDecayWithWarmup, 
                                                       'WeightedBinaryCrossEntropy': WeightedBinaryCrossEntropy})
    
    return model

def load_split(file, comment_column, code_column):
    entries = pd.read_csv(file)
    print('Invalid entries: {}'.format(entries[entries.isnull().any(axis=1)]))
    entries.fillna('', inplace=True)
    vul_column = 'W'
    if 'Devign' in entries.columns:
        vul_column = 'Devign'
    elif 'Big-Vul' in entries.columns:
        vul_column = 'Big-Vul'
    if 'Class' not in entries.columns:
        entries['Class'] = entries.apply(lambda row: row.W * 2 + row.MAT, axis=1)
    if 'LeadingComment' not in entries.columns:
        entries['LeadingComment'] = ''

    return tf.data.Dataset.from_tensor_slices((entries[comment_column], entries[code_column], entries['MAT'], entries[vul_column], entries['Class']))

def load_split_dataset(data_folder, comment_column, code_column):

    train_mod = load_split(data_folder + "/train.csv", comment_column, code_column)
    val_mod = load_split(data_folder + "/val.csv", comment_column, code_column)
    test_mod = load_split(data_folder + "/test.csv", comment_column, code_column)
        
    print("Size of the training set: ", len(train_mod))
    print("Size of the validation set: ", len(val_mod))
    print("Size of the test set: ", len(test_mod))

    return train_mod, val_mod, test_mod

def calculate_scores(predictions, label):

    if hasattr(label, "ndim") and label.ndim > 1:
        label = label.squeeze()

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for index in range(len(predictions)):
        prediction = predictions[index] if isinstance(predictions[index], bool) else predictions[index][0] > 0.5

        if(label[index] == True):
            if(prediction == True):
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if(prediction == False):
                tn = tn + 1
            else:
                fp = fp + 1

    print("tp -> ", tp)
    print("tn -> ", tn)
    print("fp -> ", fp)
    print("fn -> ", fn)

    precision = tp / (tp + fp) if tp + fp > 0 else -1
    recall = tp / (tp + fn) if tp + fn > 0 else -1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall > 0 else -1

    print("\nprecision -> ", precision)
    print("recall -> ", recall)
    print("accuracy -> ", accuracy)
    print("f1 -> ", f1)

def generate_path(output_dir, model_name, learning_rate, number_of_epochs, batch_size, dropout_prob, l2_reg_lambda):
    return "{}/weights_{}_lr_{}_ne_{}_bs_{}_dp_{}_l2_{}.tf".format(output_dir, model_name, learning_rate, number_of_epochs, batch_size, dropout_prob, l2_reg_lambda)

def get_positive_ratio_for_data(y_train):
        traininig_size = len(y_train)
        print('Training data size = {}'.format(traininig_size))
        positive_in_training = np.count_nonzero(y_train)
        ratio = positive_in_training / traininig_size
        print('Positive: {}'.format(positive_in_training))
        print('Positive Ratio: {}'.format(ratio))
        return ratio