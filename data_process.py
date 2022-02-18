from typing import Dict, List, Tuple

import numpy as np
from transformers import BertTokenizer

from preprocess import Preprocessor
import tensorflow
from tensorflow.keras.preprocessing import sequence
from tqdm import tqdm
from tensorflow.python.keras.utils.np_utils import to_categorical


class Data(Preprocessor):
    def __init__(self, data) -> None:
        super().__init__()
        self.task = 'classification'
        self.label = data['labels']
        self.data = data
        self.data_size = len(self.data)

    def prepare(self, test_index, train_index):
        valid_len = int(0.1 * len(train_index))
        y = to_categorical(self.label)
        self.y_train, self.y_test = y[self.train_index[valid_len:]], y[self.test_index]
        self.y_validation = y[self.train_index[:valid_len]]
        product_comment_validation = self.data['comment'][train_index[:valid_len]]
        product_comment_train = self.data['comment'][train_index[valid_len:]]
        product_comment_test = self.data['comment'][test_index]

        self.x_train, x_train_dict = tokenize_batch_using_bert(text_data=product_comment_train,
                                                               transformer_model_name='HooshvareLab/bert-base-parsbert-uncased',
                                                               max_length=100)
        self.train_dataset = tensorflow.data.Dataset.from_tensor_slices((x_train_dict, self.y_train))
        x_test, x_test_dict = tokenize_batch_using_bert(text_data=product_comment_test,
                                                        transformer_model_name='HooshvareLab/bert-base-parsbert-uncased',
                                                        max_length=100)
        self.test_dataset = tensorflow.data.Dataset.from_tensor_slices((x_test_dict, self.y_test))
        x_validation, x_validation_dict = tokenize_batch_using_bert(text_data=product_comment_validation,
                                                                    transformer_model_name='HooshvareLab/bert-base-parsbert-uncased',
                                                                    max_length=100)
        self.validation_dataset = tensorflow.data.Dataset.from_tensor_slices((x_validation_dict, self.y_validation))


def tokenize_batch_using_bert(text_data: np.asarray, transformer_model_name: str, max_length: int) -> Tuple[
    List[int], Dict[str, np.ndarray]]:
    """
    tokenizes a list of strings using BertTokenizer  then saves the result and returns it.
    """
    # tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
    tokenizer = BertTokenizer.from_pretrained(transformer_model_name)
    data = tokenizer(
        text_data.tolist(),
        verbose=True,
        add_special_tokens=True,  # add [CLS] and [SEP] tokens
        return_attention_mask=True,
        return_token_type_ids=True,  # not needed for this type of ML task
        padding='max_length',  # add 0 pad tokens to the sequences less than max_length
        truncation=True,
        max_length=max_length,  # truncates if len(s) > max_length
    )

    data_list = np.concatenate((np.expand_dims(np.asarray(data["input_ids"]), axis=1),
                                np.expand_dims(np.asarray(data["attention_mask"]), axis=1),
                                np.expand_dims(np.asarray(data['token_type_ids']), axis=1)), axis=1)
    data_list = [data_list[:, 0, :], data_list[:, 1, :], data_list[:, 2, :]]

    data_dict = {"input_ids": data["input_ids"],
                 "attention_mask": data["attention_mask"],
                 "token_type_ids": data['token_type_ids']}
    return data_list, data_dict
