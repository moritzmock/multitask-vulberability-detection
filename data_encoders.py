import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

import re

# Adapted from https://towardsdatascience.com/discover-the-sentiment-of-reddit-subgroup-using-roberta-model-10ab9a8271b8

class AbstractDataEncoder():

    def __init__(self, tokenizer, max_length, bi_modal, mask_satd_keywords):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bi_modal = bi_modal
        self.satd_keywords_regex = re.compile('FIXME|TODO|HACK|\sXXX\s', re.IGNORECASE)
        self.handle_satd_keywords = lambda c: c
        if mask_satd_keywords == 'mask':
            self.handle_satd_keywords = self.do_mask_satd_keywords
        elif mask_satd_keywords == 'remove':
            self.handle_satd_keywords = self.remove_satd_keywords

    def do_mask_satd_keywords(self, comment):
        return self.satd_keywords_regex.sub('<mask>', comment)
        
    def remove_satd_keywords(self, comment):
        return self.satd_keywords_regex.sub('', comment)

    def convert_example_to_feature(self, code, comments=None):

        if self.bi_modal:
            return self.tokenizer.encode_plus(comments, code,
                                        add_special_tokens=True,  
                                        max_length=self.max_length,
                                        pad_to_max_length=True, 
                                        return_attention_mask=True, 
                                        truncation=True
                                        )
        else:
            return self.tokenizer.encode_plus(code,
                                        add_special_tokens=True, 
                                        max_length=self.max_length,
                                        pad_to_max_length=True, 
                                        return_attention_mask=True, 
                                        truncation=True
                                        )

    # map to the expected input to TFRobertaForSequenceClassification
    def map_example_to_dict(self, input_ids, attention_masks, label):
        return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
            }, label

    def map_example_to_dict_not_label(self, input_ids, attention_masks):
        return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
            }

    def convert_label(self, satd, vulnerable, label):
        pass

    def encode_examples(self, ds, limit=-1):
        # Prepare Input list
        input_ids_list = []
        attention_mask_list = []
        label_list = []

        if (limit > 0):
            ds = ds.take(limit)

        for comments, code, satd, vulnerable, label in tfds.as_numpy(ds):
            comments_str = comments.decode()
            code_str = code.decode()
            bert_input = self.convert_example_to_feature(code_str, comments_str)
            input_ids_list.append(bert_input['input_ids'])
            attention_mask_list.append(bert_input['attention_mask'])
            label_list.append(self.convert_label(satd, vulnerable, label))

        return { 'input_ids':  tf.convert_to_tensor(input_ids_list), 
                 'attention_mask': tf.convert_to_tensor(attention_mask_list) }, label_list
    
    def encode_train_and_validation_data(self, training_data, validation_data, limit):
        ds_train_encoded, train_labels = self.encode_examples(training_data, limit)
        ds_validation_encoded, validation_labels = self.encode_examples(validation_data, limit)

        train_labels = np.array(train_labels).squeeze().transpose()
        validation_labels = np.array(validation_labels).squeeze().transpose()

        y_train = self.convert_labels_tensor(train_labels)
        y_validation = self.convert_labels_tensor(validation_labels)

        return ds_train_encoded, y_train, ds_validation_encoded, y_validation

class SingleTaskDataEncoder(AbstractDataEncoder):

    def __init__(self, tokenizer, max_length, separate_comments, mask_satd_keywords):
        super().__init__(tokenizer, max_length, separate_comments, mask_satd_keywords)

    def convert_labels_tensor(self, labels):
        return tf.convert_to_tensor(labels)

class BaseMultiTaskDataEncoder(AbstractDataEncoder):

    def __init__(self, tokenizer, max_length, separate_comments, mask_satd_keywords):
        super().__init__(tokenizer, max_length, separate_comments, mask_satd_keywords)

    def convert_labels_tensor(self, labels):
        return { 'satd': tf.convert_to_tensor(labels[0]), 'vulnerable': tf.convert_to_tensor(labels[1]) }

class DefaultDataEncoder(SingleTaskDataEncoder):

    def __init__(self, tokenizer, max_length, separate_comments, mask_satd_keywords):
        super().__init__(tokenizer, max_length, separate_comments, mask_satd_keywords)

    def convert_label(self, satd, vulnerable, label):
        return [label]

class OneHotDataEncoder(SingleTaskDataEncoder):

    def __init__(self, tokenizer, max_length, separate_comments, mask_satd_keywords):
        super().__init__(tokenizer, max_length, separate_comments, mask_satd_keywords)

    def convert_label(self, satd, vulnerable, label):
        labels = [ 0, 0, 0, 0 ]
        labels[label] = 1
        return labels

class MultiTaskDataEncoder(BaseMultiTaskDataEncoder):

    def __init__(self, tokenizer, max_length, separate_comments, mask_satd_keywords):
        super().__init__(tokenizer, max_length, separate_comments, mask_satd_keywords)

    def convert_label(self, satd, vulnerable, label):
        return ([satd], [vulnerable])

    def map_example_to_dict(self, input_ids, attention_masks, label):
        return {
                    "input_ids": input_ids,
                    "attention_mask": attention_masks
                }, {
                    "satd": label[0],
                    "vulnerable": label[1]
                }

class VulOnlyDataEncoder(SingleTaskDataEncoder):

    def __init__(self, tokenizer, max_length, separate_comments, mask_satd_keywords):
        super().__init__(tokenizer, max_length, separate_comments, mask_satd_keywords)

    def convert_label(self, satd, vulnerable, label):
        return [vulnerable]

class SATDOnlyDataEncoder(SingleTaskDataEncoder):

    def __init__(self, tokenizer, max_length, separate_comments, mask_satd_keywords):
        super().__init__(tokenizer, max_length, separate_comments, mask_satd_keywords)

    def convert_label(self, satd, vulnerable, label):
        return [satd]

class VulSATDDataEncoder(SingleTaskDataEncoder):

    def __init__(self, tokenizer, max_length, separate_comments, mask_satd_keywords):
        super().__init__(tokenizer, max_length, separate_comments, mask_satd_keywords)

    def convert_label(self, satd, vulnerable, label):
        return [satd and vulnerable]
        