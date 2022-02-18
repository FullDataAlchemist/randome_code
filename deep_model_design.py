from tensorflow import keras, int32
from tensorflow.keras import layers
import tensorflow as tf

from tensorflow.python.keras.losses import CategoricalCrossentropy


class DeepModelDesign:
    def __init__(self, input1, embedding1):
        self.input1, self.embedding1 = input1, embedding1

    def model_type_dict(self, argument):
        model_type_dict = {
            'cnn': self._cnn_model,
            'cnn_2input': self._cnn_2input_model,
            'lstm_2input': self._lstm_2input_model,
            'bilstm': self._bidirectional_lstm_model,
        }
        return model_type_dict.get(argument, "Invalid model type")

    def _cnn_model(self, input1, embedding1,
                   max_sequence_length,
                   embedding_dim,
                   dropout,
                   num_filter,
                   filter_size,
                   class_num,
                   ):
        if embedding1 is not None:
            reshape1 = layers.Reshape((max_sequence_length, embedding_dim, 1))(embedding1)
        else:
            reshape1 = layers.Reshape((max_sequence_length, embedding_dim, 1))(input1)

        conv1 = layers.Conv2D(num_filter, kernel_size=(filter_size[0], embedding_dim), padding='valid',
                              kernel_initializer='normal', activation='relu')(reshape1)
        mxpool1 = layers.MaxPool2D(pool_size=(max_sequence_length - filter_size[0] + 1 - 2, 1), strides=(1, 1),
                                   padding='valid')(conv1)
        flatten1 = layers.Flatten()(mxpool1)
        dropout1 = layers.Dropout(dropout)(flatten1)
        dense1 = layers.Dense(units=class_num, activation='softmax')(dropout1)
        model = keras.Model(input1, dense1)
        adam = keras.optimizers.Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-3)
        model.compile(optimizer=adam, loss=CategoricalCrossentropy(),
                      metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()])
        return model

    def _cnn_2input_model(self, input1, embedding1,
                          max_sequence_length,
                          embedding_dim,
                          dropout,
                          num_filter,
                          filter_size,
                          class_num,
                          second_feature_len,
                          ):
        if embedding1 is not None:
            reshape = layers.Reshape((max_sequence_length, embedding_dim, 1))(embedding1)
        else:
            reshape = layers.Reshape((max_sequence_length, embedding_dim, 1))(input1)

        batchnorm1 = layers.BatchNormalization()(reshape)
        conv_0 = layers.Conv2D(num_filter, kernel_size=(filter_size[0], embedding_dim), padding='valid',
                               kernel_initializer='normal', activation='relu')(batchnorm1)
        conv_1 = layers.Conv2D(num_filter, kernel_size=(filter_size[1], embedding_dim), padding='valid',
                               kernel_initializer='normal', activation='relu')(batchnorm1)
        conv_2 = layers.Conv2D(num_filter, kernel_size=(filter_size[2], embedding_dim), padding='valid',
                               kernel_initializer='normal', activation='relu')(batchnorm1)

        maxpool_0 = layers.MaxPool2D(pool_size=(max_sequence_length - filter_size[0] + 1, 1), strides=(1, 1),
                                     padding='valid')(
            conv_0)
        maxpool_1 = layers.MaxPool2D(pool_size=(max_sequence_length - filter_size[1] + 1, 1), strides=(1, 1),
                                     padding='valid')(
            conv_1)
        maxpool_2 = layers.MaxPool2D(pool_size=(max_sequence_length - filter_size[2] + 1, 1), strides=(1, 1),
                                     padding='valid')(
            conv_2)

        concatenated_tensor1 = layers.Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten1 = layers.Flatten()(concatenated_tensor1)
        dropout1 = layers.Dropout(dropout)(flatten1)
        output1 = layers.Dense(units=class_num, activation='sigmoid')(dropout1)

        input2 = layers.Input(shape=(second_feature_len,))
        concatenatedFeatures = layers.Concatenate(axis=1)([output1, input2])
        dense2 = layers.Dense(units=class_num, activation='softmax')(concatenatedFeatures)
        dropout2 = layers.Dropout(dropout)(dense2)
        output2 = layers.Dense(units=class_num, activation='softmax')(dropout2)

        model = keras.Model(inputs=[input1, input2], outputs=output2)
        adam = keras.optimizers.Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-3)
        model.compile(optimizer=adam, loss=CategoricalCrossentropy(),
                      metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()])

        return model

    def _bidirectional_lstm_model(self,
                                  dropout,
                                  class_num,
                                  ):
        if self.embedding1 is not None:
            batchnorm1 = layers.BatchNormalization()(self.embedding1)
        else:
            batchnorm1 = layers.BatchNormalization()(self.input1)
        dropout2 = layers.Dropout(dropout)(batchnorm1)
        dense1 = layers.Dense(32, use_bias=False, activation='tanh',
                              kernel_regularizer=keras.regularizers.l2(0.001))(dropout2)
        bilstm1 = layers.Bidirectional(layers.LSTM(64, kernel_regularizer=keras.regularizers.l2(0.001)
                                                   , dropout=dropout, recurrent_dropout=dropout))(dense1)
        dense2 = layers.Dense(class_num, activation='softmax', name='output')(bilstm1)
        model = keras.Model(self.input1, dense2)
        model.compile(loss=keras.losses.CategoricalCrossentropy(),
                      optimizer=keras.optimizers.Adam(1e-3),
                      metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()])
        return model

    def _lstm_2input_model(self,
                           dropout,
                           class_num,
                           second_feature_len, ):
        lstm1 = layers.LSTM(64, activation='tanh',
                            kernel_regularizer=keras.regularizers.l2(0.001), )(self.embedding1)
        batchnorm1 = layers.BatchNormalization()(lstm1)
        dropout1 = layers.Dropout(dropout)(batchnorm1)
        dense2 = layers.Dense(class_num, activation='sigmoid')(dropout1)
        input2 = layers.Input(shape=(second_feature_len,), )
        concatenatedFeatures = layers.Concatenate(axis=1)([dense2, input2])
        output1 = layers.Dense(class_num, activation='softmax')(concatenatedFeatures)
        model = keras.Model([self.input1, input2], output1)
        model.compile(loss=CategoricalCrossentropy(), optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()])
        return model
