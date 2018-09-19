from keras.models import Model,load_model
from keras.models import model_from_json
from keras.models import Model,load_model
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.layers import LSTM
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
import json

class NNBuilder():

    def __init__(self):
        '''
        Set mappings from layer class to function to add the layers
        '''

        self.mapping = {'LSTM': self.add_lstm, 'Dense': self.add_dense, 'Embedding': self.add_embedding,
                        'Conv1D': self.add_conv, 'MaxPool1D': self.add_max_pool, 'Flatten':self.add_flatten}
        self.merge_mode = {'concatenate': Concatenate}

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

    def add_embedding(self, config, prev_layer):
        '''
        :param config: the parameters passed for embedding layer;
        :param prev_layer: the upstream layer of the current layer;
        :return: the added embedding layer.
        '''
        embedding_layer = Embedding(config['input_dim'], config['output_dim'],input_length=config['input_length'], \
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
        conv_layer = Conv1D(filters=config['filters'], kernel_size=config['kernel_size'], activation=config['activation'])(prev_layer)
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
                for i in range(len(layer) -1):
                    inner_layers = layer[i]
                    col.append(self.add_layers(inner_layers, prev_layer))
                prev_layer = self.merge_mode[merge_mode](axis=-1)(col)
            else:
                layer_class = layer['type']
                layer_config = layer['config']
                prev_layer = self.mapping[layer_class](layer_config, prev_layer)

        return prev_layer

    def build_model(self, layers, sequences_matrix):
        '''
        :param layers: the layers to build neural net;
        :param sequences_matrix: input of the neural net;
        :return: the compiled model.
        '''
        sequenceInput = Input(shape=(layers[0]['config']['shape'],), dtype='int32')
        output_layer = self.add_layers(layers[1:], sequenceInput)
        mod = Model(sequenceInput,[output_layer])
        mod.summary()
        mod.compile(loss=model_config['loss'], optimizer=model_config['optimizer'], metrics=[model_config['metrics']])
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

    layers = [{'type': 'Input', 'config': {'shape': 20}},\
              {'type': 'Embedding', 'config': {'input_dim': 1000, 'output_dim': 20, 'input_length': 20,  'trainable':False}}, \
              {'type': 'LSTM', 'config': {'cells': 128, 'return_sequences': True, 'bidirectional': True}}, \
              {'type': 'LSTM', 'config': {'cells':128, 'return_sequences': True, 'bidirectional': True}}, \
              [[{'type': 'Conv1D', 'config': {'filters': 100, 'kernel_size': 5, 'activation': 'relu'}}, \
              {'type': 'MaxPool1D', 'config': {'pool_size': 2}}, \
              {'type': 'Flatten', 'config': {}}],\
               [{'type': 'Conv1D', 'config': {'filters': 100, 'kernel_size': 5, 'activation': 'relu'}}, \
                {'type': 'MaxPool1D', 'config': {'pool_size': 2}}, \
                {'type': 'Flatten', 'config': {}}], 'concatenate'
               ], \
              {'type': 'Dense', 'config': {'units':5, 'activation': 'softmax'}}]

    model_config = {'loss':'sparse_categorical_crossentropy','optimizer': 'rmsprop', 'metrics': 'accuracy'}

    builder = NNBuilder()
    model = builder.build_model(layers, sequences_matrix)
    model.fit(sequences_matrix, Y_train, batch_size=128, epochs=1, validation_split=0.2)
