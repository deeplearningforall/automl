from Model import *
from Model import NNBuilder

class AgentUtility():
    def  __init__(self, dataset_loc, input_col, output_col, sep=',', max_words=None, maxlen=20):
        '''
            :param Dataset_loc: The loction of  the dataset.
            :param input_col: The input column name in the dataset.
            :param output_col: The output column name in the dataset.
            :param sep: Seperator for reading the csv file.
            :param max_words: Max vocab size to use for reading the csv file(None for unlimited).
            :param maxlen: Max sequence length.
            :return: None.
        '''
        df = pd.read_csv(dataset_loc, sep, error_bad_lines = False)
        df = df[[input_col, output_col]]
        df = df.dropna()
        x = df[input_col]
        y = df[output_col]
        le = LabelEncoder()
        self.y = le.fit_transform(y)
        self.tok = Tokenizer(num_words=max_words)
        self.tok.fit_on_texts(x)
        sequences = self.tok.texts_to_sequences(x)
        self.x = sequence.pad_sequences(sequences, maxlen=maxlen)
        self.builder = NNBuilder(len(self.tok.word_index)+1, max(self.y)+1, maxlen)
        self.model_layers = ['input_layer', 'Embedding', 'output_layer']
        self.model = self.builder.agent_model(self.model_layers)
        #self.perform_operation('add:LSTM_RST')
        #self.perform_operation('update:Conv1D')
        #self.perform_operation('add:MaxPool1D')
        #print(self.model.summary())
        #a = self.model.fit(self.x, self.y, batch_size=512, epochs=3, validation_split=0.2)

    def perform_operation(self, name):
        '''
        :param name: The name of operation too do. eg.  add:LSTM_RST for adding a lstm with return sequence as true., update:LSTM_RSF, Delete
        :return: None
        '''
        if(name.split(':')[0] == 'add'):
            self.model_layers.insert(-1,name.split(':')[1])
        elif(name.split(':')[0] == 'update'):
            self.model_layers[-2] = name.split(':')[1]
        else:
            self.model_layers.pop(-2)
        self.model = self.builder.agent_model(self.model_layers)

AgentUtility('../data/amazon_reviews_us_Mobile_Electronics_v1_00.tsv', 'review_body', 'star_rating', '\t')
'''
da = pd.read_csv('../data/amazon_reviews_us_Mobile_Electronics_v1_00.tsv', sep='\t', error_bad_lines=False)
da = da[['review_body', 'star_rating']]
da = da.dropna()
X = da.review_body
Y = da.star_rating
le = LabelEncoder()
Y = le.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)
max_words = 1000
max_len = 20
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

builder = NNBuilder()
num_intermediate_layers = 1
model = builder.build_model(sequences_matrix, Y_train, num_intermediate_layers)
model.fit(sequences_matrix, Y_train, batch_size=128, epochs=3, validation_split=0.2)
'''