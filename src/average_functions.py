from keras.models import Sequential, clone_model
from keras.layers import Dense, Activation
from keras import optimizers
import numpy as np

"""
Implementation for Deep neural net.

model is the online network
old_model is the target network

"""
class DNN:

    def __init__(self, input_dim, hidden_layer_size, hidden_layers, output_class, learning_rate):
        self.learning_rate = learning_rate
        self.model = self.create_nn(input_dim, hidden_layer_size, hidden_layers, output_class, learning_rate)
        self.old_model = self.create_nn(input_dim, hidden_layer_size, hidden_layers, output_class, learning_rate)
        self.clone()

    """
    Returns a sequential network with 2 hidden layer, 1 input layer and 1 output layer. The final layer uses
    linear activation. The loss function is mean squared error and the optimizer used is Adam.
    """
    def create_nn(self, input_dim, hidden_layer_size, hidden_layers, output_class, learning_rate):
        model = Sequential()
        model.add(Dense(hidden_layer_size,input_shape=(input_dim,), activation='relu'))
        model.add(Dense(hidden_layer_size, activation='relu'))
        #self.model.add(Dense(hidden_layer_size, activation='relu'))
        #self.model.add(Dense(hidden_layer_size, activation='relu'))
        model.add(Dense(output_class))
        learning_rate = learning_rate
        model.compile(optimizer=optimizers.Adam(lr=self.learning_rate), loss='mse')
        return model

    """
    Calls fit method to train the neural network
    """
    def train(self, training_set, epoch=1, batch_size=None):
        self.model.fit(training_set[0], training_set[1], epochs=epoch, verbose=0, batch_size=batch_size)
        return self

    def predict(self, feature):
        
        return self.model.predict(feature)

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
    model = DNN(8,64,2,4,0.01)

    train_x = np.random.rand(1000,8)
    train_y = np.random.rand(1000,4)
    training_set = (train_x, train_y)

    model.train(training_set)
    test = np.random.rand(1,8)
    pred = model.predict(test)
    print (pred)

if __name__ == '__main__':
    main()
