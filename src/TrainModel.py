from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras.layers import Embedding, Reshape, Dense, Input, Flatten, Conv1D, Dot, Add, Concatenate, MaxPool1D, merge, \
    Dropout, LSTM, GRU, Bidirectional
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


class TrainModel():
    def __init__(self):
        self.tokenizer = Tokenizer()

    def cleanText(self, text):
        try:
            text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
            text = re.sub(r'[\n\r]', '', text)
            text = re.sub(r'"', '', text)
        except:
            text = ''
        return text.strip().lower()

    def createSequnceList(self, text):
        if isinstance(text, str):
            return text.lower().split()
        else:
            return []

    def tokenize(self, text):
        self.tokenizer.fit_on_texts(text)
        # print(self.tokenizer.word_index)
        return self.tokenizer

    def createSequence(self, text, seqLen=40, padding="pre", truncating='pre', value=0.0):
        text = self.createSequnceList(text)
        seq = self.tokenizer.texts_to_sequences(text)
        return pad_sequences(seq, 20)

    def trainTestSplit(self, x, y, testSize=0.1):
        return train_test_split(x, y, test_size=testSize, stratify=y)

    def initializeEmbeddings(self, w2v, embeddingSize=100):
        wordIndex = self.tokenizer.word_index
        embeddingMatrix = np.zeros((len(wordIndex) + 1, embeddingSize))
        for word, i in wordIndex.items():
            try:
                embedddingVector = w2v.wv(word)
            except:
                embeddingVector = None
            if embeddingVector is not None:
                embeddingMatrix[int(i)] = embeddingVector
        return embeddingMatrix
