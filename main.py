import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Tuple
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers import BertTokenizer
from tensorflow import keras, int32
from transformers import TFBertModel, TFBertForSequenceClassification, BertConfig, XLNetModel
import model_handler
from data_process import Data
from tensorflow.keras import layers
import tensorflow as tf
from hazm import *


def get_embedding_layer(embedding_dim, embedding_name, max_num_words, max_sequence_length, w2v_weights,
                        transformer_model_name):
    if 'no_embedding' in embedding_name:
        input_layer = layers.Input(shape=(max_sequence_length,), dtype='int32')
        embedding_layer = layers.Embedding(max_num_words, embedding_dim, input_length=max_sequence_length)(
            input_layer)
    elif 'transformer' in embedding_name:
        embedding_layer, input_layer = get_transformer_embedding(TFBertModel, max_sequence_length,
                                                                 transformer_model_name)
    elif 'xl-net' in embedding_name:
        embedding_layer, input_layer = get_transformer_embedding(XLNetModel, max_sequence_length,
                                                                 transformer_model_name)
    else:
        input_layer = layers.Input(shape=(max_sequence_length,), dtype='int32')
        embedding_layer = layers.Embedding(input_dim=max_num_words, output_dim=embedding_dim,
                                           input_length=max_sequence_length,
                                           embeddings_initializer=keras.initializers.Constant(w2v_weights),
                                           # weights=[word2vec],
                                           name='embedding_layer', trainable=False)(input_layer)
    return input_layer, embedding_layer


def get_transformer_embedding(model, max_sequence_length, transformer_model_name):
    bert = model.from_pretrained(transformer_model_name, trainable=False)
    input_ids = tf.keras.layers.Input(shape=(max_sequence_length,), dtype=int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(max_sequence_length,), dtype=int32, name="attention_mask")
    token_type_ids = tf.keras.layers.Input(shape=(max_sequence_length,), dtype=int32, name="token_type_ids")
    input_layer = [input_ids, attention_mask, token_type_ids]
    bert = bert(input_layer)
    embedding = bert[0]
    return embedding, input_layer


def lemmatize_data(digi_data):
    tokenizer = WordTokenizer()
    lemmatizer = Lemmatizer()
    normalizer = Normalizer()
    res = list()
    for each in tqdm(digi_data.data['comment']):
        res.append(' '.join(lemmatizer.lemmatize(w) for w in tokenizer.tokenize(normalizer.normalize((str(each))))))
    digi_data.data['comment'] = pd.DataFrame(res)


def stemm_data(digi_data):
    tokenizer = WordTokenizer()
    stemmer = Stemmer()
    normalizer = Normalizer()

    res = list()
    for each in tqdm(digi_data.data['comment']):
        res.append(' '.join(stemmer.stem(w) for w in tokenizer.tokenize(normalizer.normalize((str(each))))))
    digi_data.data['comment'] = pd.DataFrame(res)


def remove_stopwords(digi_data):
    tokenizer = WordTokenizer()
    normalizer = Normalizer()
    res = list()
    for each in tqdm(digi_data.data['comment']):
        res.append(
            ' '.join(w for w in tokenizer.tokenize(normalizer.normalize((str(each)))) if w not in stopwords_list()))
    digi_data.data['comment'] = pd.DataFrame(res)


if __name__ == '__main__':
    data = pd.read_csv('comment-it.csv')
    labels = pd.DataFrame(pd.cut(data['میانگین نظرات'], bins=3, labels=np.arange(3), right=False))
    labels.columns = ["labels"]
    data = pd.concat([data, labels], axis=1)
    data.convert_dtypes()
    data.dropna()

    digi_data = Data(data)
    lemmatizer_flag = False
    stemmer_flag = False
    stopword_remove_flag = False
    if stopword_remove_flag:
        remove_stopwords(digi_data)
    if lemmatizer_flag:
        lemmatize_data(digi_data)
    if stemmer_flag:
        stemm_data(digi_data)

    digi_data.prepare_data_for_train()

    input1, embedding1 = get_embedding_layer(embedding_dim=768,
                                             embedding_name='transformer',
                                             max_num_words=400000,
                                             max_sequence_length=100,
                                             transformer_model_name='HooshvareLab/bert-base-parsbert-uncased',
                                             w2v_weights=None)
    new_model = model_handler.ModelTrainer('bilstm', input1, embedding1)
    new_model.build_model(embedding='HooshvareLab/bert-base-parsbert-uncased',
                          max_sequence_length=100,
                          dropout=0.2,
                          class_num=3,
                          num_filter=3,
                          filter_size=5)
    new_model.train(digi_data)
    new_model.save_model('digi_model')
