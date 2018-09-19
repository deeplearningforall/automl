import spacy
from gensim.models.phrases import Phraser
from gensim.models import Phrases
from gensim.models import Word2Vec
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py

nlp = spacy.load('en')


class PreProcessText():
    def createSequnceList(self, text):
        if isinstance(text, str):
            return text.lower().split()
        else:
            return []

    def getLen(self, column):
        if column.dtypes == 'object':
            column = column.apply(cleanText)
            length = column.str.len()
            return min(length), max(length), int(np.mean(length))

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

    def extractPhrasesTopics(self, column, minCount=5, threshold=100):
        sentences = column.apply(self.createSequnceList)
        common_terms = ["of", "with", "without", "and", "or", "the", "a", "do", "not", "you", "thank", "at", "all", ]
        bigram = Phrases(sentences, min_count=minCount, threshold=threshold, common_terms=common_terms)
        phrases = []
        scores = []
        nouns = []
        for phrase, score in bigram.export_phrases(sentences):
            doc = nlp(phrase.decode('utf-8'))
            pos = []
            for token in doc:
                pos.append(token.pos_)
                if set(pos) & set(['NOUN', 'NNP', 'NN']):
                    phrases.append((phrase.decode('utf-8')))
                    scores.append(score)
                    nouns.append(str(token))

        p = dict(zip(phrases, scores))
        t = dict(zip(nouns, scores))
        return sorted(t.items(), key=operator.itemgetter(1), reverse=True), sorted(p.items(),
                                                                                   key=operator.itemgetter(1),
                                                                                   reverse=True)

    def getUniqueTokens(self, column):
        if column.dtypes == 'object':
            return len(str(column).split())
        else:
            return 0

    def getUniqueNouns(self, column):
        if column.dtypes == 'object':
            doc = nlp(str(column.apply(self.cleanText)))
            nouns = []
            for token in doc:
                nouns.extend([token.text for token in doc if
                              token.is_stop != True and token.is_punct != True and token.pos_ == "NOUN"])
            return len(nouns)
        # self.getWordLenDistribution(nouns)
        else:
            return 0

    def getWordLenDistribution(self, column):
        if column.dtypes == 'object':
            distribution = [len(x) for x in self.cleanText(str(column)).split()]
            print(distribution)
            return distribution
        else:
            return 0

    def createWord2Vec(self, column, embeddingSize=100, window=5, minCount=10, workers=2, skipGram=0):
        column = column.apply(self.cleanText)
        sentences = column.apply(self.createSequnceList)
        model = Word2Vec(sentences, size=100, window=5, min_count=minCount, workers=workers, sg=skipGram)
        return model
