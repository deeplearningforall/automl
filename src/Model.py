from keras.models import Model,load_model
from keras.models import model_from_json
from keras.models import Model,load_model
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.layers import LSTM, GRU
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPool1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pandas as pd
import numpy as np
import json


class NNBuilder():

    def __init__(self):
        '''
        Set mappings from layer class to function to add the layers
        '''

        self.mapping = {'LSTM': self.add_lstm, 'GRU': self.add_gru, 'Dense': self.add_dense,
                        'Embedding': self.add_embedding,
                        'Conv1D': self.add_conv, 'MaxPool1D': self.add_max_pool, 'Flatten': self.add_flatten}
        self.merge_mode = {'concatenate': Concatenate}
        self.down_stream = {3: ['LSTM', 'Dense', 'Conv1D', 'MaxPool1D', 'Flatten'], 2: ['Dense']}
        self.up_stream = {3: ['LSTM', 'Dense', 'Conv1D', 'MaxPool1D', 'Embedding'], 2: ['Flatten', 'LSTM']}
        self.layer_configs = {
            'LSTM': {'cells': [64, 128, 256], 'return_sequences': [True, False], 'bidirectional': [True, False]},
            'GRU': {'cells': [64, 128, 256], 'return_sequences': ['True', 'False']},
            'Dense': {'units': [64, 128, 256], 'activation': ['relu', 'sigmoid']},
            'MaxPool1D': {'pool_size': [2]},
            'Conv1D': {'filters': [64, 128, 256], 'kernel_size': [2, 3, 4],
                       'activation': ['sigmoid', 'relu']},
            'Flatten': {}}

        self.embedding_layer_config = {'input_dim': 10000, 'output_dim': 100}
        self.dense_layer_config = {'activation': 'softmax'}
        self.model_config = {'loss': 'sparse_categorical_crossentropy', 'optimizer': 'rmsprop', 'metrics': 'accuracy'}

    def add_lstm(self, config, prev_layer):
        '''
        :param config: the parameters passed for lstm layer;
        :param prev_layer: the upstream layer of the current layer;
        :return: the added lstm layer.
        '''
        if config['bidirectional']:
            lstm_layer = Bidirectional(LSTM(config['cells'], return_sequences=config['return_sequences']))(prev_layer)
        else:
            lstm_layer = LSTM(config['cells'], return_sequences=config['return_sequences'])(prev_layer)
        return lstm_layer

    def add_gru(self, config, prev_layer):
        '''
        :param config: the parameters passed for gru layer;
        :param prev_layer: the upstream layer of the current layer;
        :return: the added gru layer.
        '''
        if config['bidirectional']:
            gru_layer = Bidirectional(GRU(config['cells'], return_sequences=config['return_sequences']))(prev_layer)
        else:
            gru_layer = GRU(config['cells'], return_sequences=config['return_sequences'])(prev_layer)
        return gru_layer

    def add_embedding(self, config, prev_layer):
        '''
        :param config: the parameters passed for embedding layer;
        :param prev_layer: the upstream layer of the current layer;
        :return: the added embedding layer.
        '''
        embedding_layer = Embedding(config['input_dim'], config['output_dim'], input_length=config['input_length'], \
                                    trainable=config['trainable'])(prev_layer)
        return embedding_layer

    def add_dense(self, config, prev_layer):
        '''
        :param config: the parameters passed for dense layer;
        :param prev_layer: the upstream layer of the current layer;
        :return: the added dense layer.
        '''
        dense_layer = Dense(config['units'], activation=config['activation'])(prev_layer)
        return dense_layer

    def add_conv(self, config, prev_layer):
        '''
        :param config: the parameters passed for convolutional layer;
        :param prev_layer: the upstream layer of the current layer;
        :return: the added convolutional layer.
        '''
        conv_layer = Conv1D(filters=config['filters'], kernel_size=config['kernel_size'],
                            activation=config['activation'])(prev_layer)
        return conv_layer

    def add_max_pool(self, config, prev_layer):
        '''
        :param config: the parameters passed for convolutional layer;
        :param prev_layer: the upstream layer of the current layer;
        :return: the added max pooling layer.
        '''
        pooling_layer = MaxPool1D(pool_size=config['pool_size'])(prev_layer)
        return pooling_layer

    def add_flatten(self, config, prev_layer):
        '''
        :param config: the parameters passed for convolutional layer;
        :param prev_layer: the upstream layer of the current layer;
        :return: the added flatten layer.
        '''
        flatten_layer = Flatten()(prev_layer)
        return flatten_layer

    def add_layers(self, layers, prev_layer):
        '''
        :param layers: the layers needed to be added to a neural net. It can be a layer, such as LSTM, or can be
                        a sub neural net.
        :param prev_layer: the upstream layer for the neural net.
        :return: the output layer of the neural net.
        '''
        for layer in layers:
            if type(layer) == type(list()):
                merge_mode = layer[-1]
                col = []
                for i in range(len(layer) - 1):
                    inner_layers = layer[i]
                    col.append(self.add_layers(inner_layers, prev_layer))
                prev_layer = self.merge_mode[merge_mode](axis=-1)(col)
            else:
                layer_class = layer['type']
                if layer_class == 'Flatten':
                    layer_config = {}
                else:
                    layer_config = layer['config']
                prev_layer = self.mapping[layer_class](layer_config, prev_layer)

        return prev_layer

    def gold_miner(self, embedding_layer_dim, output_layer_dim, num_intermediate_layers):
        '''

        :param embedding_layer_dim: the dimension of embedding layer;
        :param output_layer_dim: the dimension of dense layer;
        :param num_intermediate_layers: the number of layers in between embedding and dense layers;
        :return: intermediate layers in json format.
        '''
        layers = []
        prev_dim = embedding_layer_dim
        for index in range(num_intermediate_layers):
            layer = {}
            if index != num_intermediate_layers - 1:
                possible_new_layers = self.down_stream[prev_dim]

                new_layer = possible_new_layers[np.random.randint(low=0, high=max(1, len(possible_new_layers) - 1))]
                layer['type'] = new_layer
                for para in self.layer_configs[new_layer]:
                    options = self.layer_configs[new_layer][para]
                    value = options[np.random.randint(low=0, high=max(1, len(options) - 1))]
                    if 'config' not in layer:
                        layer['config'] = {}
                    layer['config'][para] = value

            else:
                possible_new_layers = list(
                    set(self.down_stream[prev_dim]).intersection(set(self.up_stream[output_layer_dim])))
                new_layer = possible_new_layers[np.random.randint(low=0, high=max(1, len(possible_new_layers) - 1))]
                layer['type'] = new_layer

                for para in self.layer_configs[new_layer]:
                    options = self.layer_configs[new_layer][para]
                    value = options[np.random.randint(low=0, high=max(1, len(options) - 1))]
                    if 'config' not in layer:
                        layer['config'] = {}
                    layer['config'][para] = value
            layers.append(layer)
        return layers

    def create_layer_config(self, sequences_matrix, Y_train, num_intermediate_layers):
        '''
        :param sequences_matrix: input of the neural net;
        :param Y_train: output of the neural net;
        :param num_intermediate_layers: the number of layers in between embedding and dense layers;
        :return: the layers of the entire neural net in json format.
        '''
        layers = []

        input_len = sequences_matrix.shape[1]
        input_layer = {'type': 'Input', 'config': {'shape': input_len}}
        layers.append(input_layer)

        embedding_layer = {'type': 'Embedding', 'config': {'input_dim': self.embedding_layer_config['input_dim'],
                                                           'output_dim': self.embedding_layer_config['output_dim'], \
                                                           'input_length': input_len, 'trainable': True}}
        layers.append(embedding_layer)

        intermediate_layers = self.gold_miner(3, 2, num_intermediate_layers)

        if intermediate_layers[len(intermediate_layers) - 1]['type'] in ['LSTM', 'GRU']:
            intermediate_layers[len(intermediate_layers) - 1]['config']['return_sequences'] = False
        layers += intermediate_layers

        output_layer = {'type': 'Dense',
                        'config': {'units': max(Y_train) + 1, 'activation': self.dense_layer_config['activation']}}
        layers.append(output_layer)

        return layers

    def build_model(self, sequences_matrix, Y_train, num_intermediate_layers):
        '''
        :param layers: the layers to build neural net;
        :param sequences_matrix: input of the neural net;
        :return: the compiled model.
        '''
        layers = self.create_layer_config(sequences_matrix, Y_train, num_intermediate_layers)
        sequenceInput = Input(shape=(layers[0]['config']['shape'],), dtype='int32')
        output_layer = self.add_layers(layers[1:], sequenceInput)
        mod = Model(sequenceInput, [output_layer])
        mod.summary()
        mod.compile(loss=self.model_config['loss'], optimizer=self.model_config['optimizer'],
                    metrics=[self.model_config['metrics']])
        return mod


if __name__ == '__main__':

    da = pd.read_csv('../data/amazon_reviews_us_Mobile_Electronics_v1_00.tsv', sep='\t', error_bad_lines=False)
    da = da[['review_body', 'star_rating']]
    da = da.dropna()
    X = da.review_body
    Y = da.star_rating
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)
    max_words = 1000
    max_len = 20
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_train)
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

    builder = NNBuilder()
    num_intermediate_layers = 10
    model = builder.build_model(sequences_matrix, Y_train, num_intermediate_layers)
    model.fit(sequences_matrix, Y_train, batch_size=128, epochs=3, validation_split=0.2)





