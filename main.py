import numpy as np
import pandas as pd

from Sentiment import SentimentAnalysis
from utils import helper, preprocess

"""
Yet to do:
    expand context
    using pretrained glove
"""

if __name__ == '__main__':

    data = pd.read_csv("F:/Internship/Omdena/Arabic-Chapter/data/IMDB Dataset.csv", nrows=2000)
    text, label = data['review'], data['sentiment']


    #LSTM
    sentiment = SentimentAnalysis(preprocess.tokenizer, vocab_size=13000, maxlen=150, embedding_vector=10, method="lstm")
    text = sentiment.tokenize(text, stop_words=['and', 'a', 'is', 'the', 'in', 'be', 'will'])
    text, label, unique_words, word_dict = sentiment.vectorize(text, label)
    text, label = np.array(text), np.array(label)
    model = sentiment.fit(text, label, epochs=5, test_size=0.2)

    #Validation
    text = ["Today is a bad day for me!", "I'm happy"]
    sentiment.predict(text, model)


    text, label = data['review'], data['sentiment']
    #Bidirectional RNN
    sentiment = SentimentAnalysis(preprocess.tokenizer, vocab_size=13000, maxlen=150, embedding_vector=10, method="bidRNN")
    text = sentiment.tokenize(text, stop_words=['and', 'a', 'is', 'the', 'in', 'be', 'will'])
    text, label, unique_words, word_dict = sentiment.vectorize(text, label)
    text, label = np.array(text), np.array(label)
    model = sentiment.fit(text, label, epochs=5, test_size=0.2)

    #Validation
    text = ["Today is a bad day for me!", "I'm happy"]
    sentiment.predict(text, model)


    text, label = data['review'], data['sentiment']
    #1D Convolution
    sentiment = SentimentAnalysis(preprocess.tokenizer, vocab_size=13000, maxlen=150, embedding_vector=10, method="1DConv")
    text = sentiment.tokenize(text, stop_words=['and', 'a', 'is', 'the', 'in', 'be', 'will'])
    text, label, unique_words, word_dict = sentiment.vectorize(text, label)
    text, label = np.array(text), np.array(label)
    model = sentiment.fit(text, label, epochs=5, test_size=0.2)

    #Validation
    text = ["Today is a bad day for me!", "I'm happy"]
    sentiment.predict(text, model)


    text, label = data['review'], data['sentiment']
    #Simple RNN
    sentiment = SentimentAnalysis(preprocess.tokenizer, vocab_size=13000, maxlen=150, embedding_vector=10, method="simpleRNN")
    text = sentiment.tokenize(text, stop_words=['and', 'a', 'is', 'the', 'in', 'be', 'will'])
    text, label, unique_words, word_dict = sentiment.vectorize(text, label)
    text, label = np.array(text), np.array(label)
    model = sentiment.fit(text, label, epochs=5, test_size=0.2)

    #Validation
    text = ["Today is a bad day for me!", "I'm happy"]
    sentiment.predict(text, model)



