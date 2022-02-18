from abc import ABC, abstractmethod

from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.utils.np_utils import to_categorical


class Preprocessor(ABC):

    def __init__(self) -> None:
        self.y_train, self.y_test, self.y_validation = [], [], []
        self.x_train, self.x_test = [], []
        self.test_index, self.test_text = [], []
        self.train_index, self.train_text = [], []
        self.validation_dataset = None
        self.train_dataset = None
        self.test_dataset = None

    def prepare_data_for_train(self, column_name='comment'):
        skf = StratifiedKFold(n_splits=int(0.1 ** -1), shuffle=False)
        self.train_index, self.test_index = next(skf.split(self.data[column_name], self.label))
        self.test_text = self.data[column_name][self.test_index]
        self.prepare(self.test_index, self.train_index)

    def prepare_data_for_test(self, the_data):
        y = to_categorical(the_data.label)
        self.test_text = the_data.data[[]]
        self.y_test, self.y_train = y, []
        self.prepare([], the_data, [x for x in range(len(the_data.data))])

    def prepare(self, test_index, train_index):
        pass
