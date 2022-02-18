import collections
import math
import numpy as np
import deep_model_design
import plotting


class ModelTrainer(deep_model_design.DeepModelDesign):
    def __init__(self, model_name, input1, embedding1):
        super().__init__(input1, embedding1)
        self.model = None
        self.model_name = model_name
        self.result_name = model_name + 'res'
        self.plot_path = model_name + 'plt'
        self.model_path = ''

    def build_model(self, embedding, max_sequence_length, dropout, class_num, num_filter, filter_size):
        if 'cnn' in self.model_name:
            self.model = self.model_type_dict(self.model_name)(max_sequence_length=max_sequence_length,
                                                               embedding_dim=embedding.embedding_dim,
                                                               dropout=dropout,
                                                               num_filter=num_filter,
                                                               filter_size=filter_size,
                                                               class_num=class_num)

        elif 'lstm' in self.model_name:
            self.model = self.model_type_dict(self.model_name)(dropout=dropout,
                                                               class_num=class_num,
                                                               )

    def create_class_weight(self, labels: np.ndarray, mu=0.15) -> dict:
        """
        computes the weights based on the distribution of the labels
        """
        labels = [np.argmax(x) for x in labels]
        labels_dict = dict(collections.Counter(labels))
        total = np.sum(list(labels_dict.values()))
        keys = labels_dict.keys()
        class_weight = dict()
        for key in keys:
            score = math.log(mu * total / float(labels_dict[key]))
            class_weight[key] = score if score > 1.0 else 1.0
        return class_weight

    def train(self, preprocessor, plot=True, tf_data_flag=True):
        class_weight = self.create_class_weight(preprocessor.y_train)
        if tf_data_flag:
            # for input_tmp in preprocessor.train_dataset.batch(batch_size).take(batch_size):
            #     print()

            history = self.model.fit(preprocessor.train_dataset.batch(32), epochs=10,
                                     validation_data=preprocessor.validation_dataset.batch(32),
                                     verbose=1,
                                     class_weight=class_weight
                                     )
        else:
            history = self.model.fit(x=preprocessor.x_train, y=preprocessor.y_train,
                                     epochs=10,
                                     validation_data=(preprocessor.x_test, preprocessor.y_test),
                                     verbose=1, batch_size=21,
                                     class_weight=class_weight
                                     )

        if plot:
            self._plot_train(history)
            self._test_analysis(preprocessor)

    def save_model(self, embedding_name):
        if 'transformer' in embedding_name:
            self.model.save_weights(self.model_path + self.result_name + 'weights_' + self.model_name + '.hd5')
        else:
            self.model.save(self.model_path + self.result_name + 'weights_' + self.model_name + '.hdf5')

    def _plot_train(self, history):
        plotting.plot_acc_loss(history=history, result_name=self.result_name, plot_path=self.plot_path)
        plotting.log_best_scores(history, self.result_name, self.plot_path)

    def _test_analysis(self, preprocessor):
        prediction = self.model.predict(preprocessor.validation_dataset.batch(32))
        prediction_max = np.empty_like(prediction)
        for i in range(len(prediction)):
            prediction_max[i, :] = np.int8(prediction[i, :] == prediction[i, :].max())
        comparing = np.all(prediction_max == preprocessor.y_test, axis=1)
        compare_dict = {}
        with open(self.model_path + self.result_name + '_WrongAnswer_' + self.model_name + '.csv', 'w') as f:
            f.write("جمله" + "," + "نوع تشخیص" + "," + "پاسخ درست" + "\n")
            for i in range(len(comparing)):
                if not comparing[i]:
                    f.write(preprocessor.test_text[i])
                    class_list_name = preprocessor.class_list_name
                    tmp = "," + class_list_name.get(np.argmax(prediction_max[i, :])) + \
                          "," + class_list_name.get(np.argmax(preprocessor.y_test[i, :]))
                    if compare_dict.get(tmp) is not None:
                        compare_dict[tmp] += 1
                    else:
                        compare_dict[tmp] = 1
                    f.write(tmp)
                    f.write("\n")
        plotting.plotting_conf_2(prediction_max, preprocessor.y_test, preprocessor.class_list_name,
                                 result_name=self.result_name, plot_path=self.plot_path)
        plotting.ploting_confusion_matrix(prediction_max, preprocessor.y_test, preprocessor.class_list_name,
                                          result_name=self.result_name, plot_path=self.plot_path)
