from keras.models import Sequential, clone_model
from keras.layers import Dense, Activation
from keras.layers import Embedding, LSTM
from keras import optimizers
from keras.preprocessing.sequence import pad_sequences
import numpy as np

"""
Implementation for Deep neural net.

model is the online network
old_model is the target network

"""
class DNN:

    def __init__(self, input_dim, hidden_layer_size, hidden_layers, output_class, learning_rate):
        self.learning_rate = learning_rate
        self.word_index = {'LSTM_RST': 1, 'LSTM_RSF': 2, 'LSTM_RST_BI': 3, \
                             'LSTM_RSF_BI': 4, 'GRU_RST': 5, 'GRU_RSF': 6, \
                             'GRU_RST_BI': 7, 'GRU_RSF_BI': 8, \
                             'Dense_Sigmoid': 9, 'Dense_Tanh': 10, \
                             'Dense_Relu': 11, 'MaxPool1D': 12, 'Conv1D': 13, \
                             'Flatten': 14, 'input_layer': 15, 'Embedding': 16, 'output_layer': 17}
        self.output_index = {'add:LSTM_RST': 0, 'add:LSTM_RSF': 1, 'add:LSTM_RST_BI': 2, \
                             'add:LSTM_RSF_BI': 3, 'add:GRU_RST': 4, 'add:GRU_RSF': 5, \
                             'add:GRU_RST_BI': 6, 'add:GRU_RSF_BI': 7, \
                             'add:Dense_Sigmoid': 8, 'add:Dense_Tanh': 9, \
                             'add:Dense_Relu': 10, 'add:MaxPool1D': 11, 'add:Conv1D': 12, \
                             'add:Flatten': 13}
        self.model = self.create_nn(input_dim, learning_rate)
        self.old_model = self.create_nn(input_dim, learning_rate)
        self.clone()

    """
    Returns a sequential network with 2 hidden layer, 1 input layer and 1 output layer. The final layer uses
    linear activation. The loss function is mean squared error and the optimizer used is Adam.
    """
    def create_nn(self, input_dim, learning_rate):
        model = Sequential()
        model.add(Embedding(len(self.word_index)+1, output_dim=128, input_shape=(input_dim,)))
        model.add(LSTM(128, return_sequences = False))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(len(self.output_index)))
        self.learning_rate = learning_rate
        model.compile(optimizer=optimizers.Adam(lr=self.learning_rate), loss='mse')
        return model

    """
    Calls fit method to train the neural network
    """
    def train(self, training_set, epoch=1, batch_size=None):
        x = []
        for train_x in training_set[0]:
            x.append([self.word_index[temp] for temp in train_x])
        x = pad_sequences(x, maxlen=20)
        print(x.shape)
        print(training_set[1].shape)
        self.model.fit(np.array(x), training_set[1], epochs=epoch, verbose=0, batch_size=batch_size)
        return self

    def predict(self, feature):
        feature = [self.word_index[temp] for temp in feature]
        feature = pad_sequences([feature], maxlen=20)
        a = self.model.predict(feature)[0]
        a = np.argmax(a)
        return {j:i for i, j in self.output_index.items()}[a]

    def batch_predict(self, feature, new_model = True):
        print(feature)
        x = []
        for train_x in feature:
            x.append([self.word_index[temp] for temp in train_x])
        x = pad_sequences(x, maxlen=20)
        if(new_model):
            a = self.model.predict(x)
        else:
            a = self.old_model.predict(x)
        return a

    def save(self, path):
        self.model.save_weights(path+".h5")

    def load(self, path):
        self.model.load_weights(path)

    """
    copies the weights from model to old_model.
    """
    def clone(self):
        self.old_model.set_weights(self.model.get_weights())

def main():
    model = DNN(20,64,2,4,0.01)

    train_x = [['input_layer', 'Embedding', 'output_layer']]*1000
    train_y = np.random.rand(1000, 14)
    training_set = (train_x, train_y)

    model.train(training_set)
    test = ['input_layer', 'Embedding', 'output_layer']
    pred = model.predict(test)
    print (pred)

if __name__ == '__main__':
    main()
