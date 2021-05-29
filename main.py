import os

import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

from Sentiment import SentimentAnalysis
from utils import helper, preprocess
from utils.config import config

"""
Yet to do:
    # verify preprocessing
    # weights
    # architechture
"""

if __name__ == '__main__':

    data = pd.read_csv(config['data_path'])
    text, label = data['review'], data['sentiment']


    #Check config file to alter the hyperparameters
    sentiment = SentimentAnalysis(preprocess.tokenizer, vocab_size=config['vocab_size'], maxlen=config['maxlen'], embedding_vector=config['embedding_vector'], method=config['method'],)
    text = sentiment.tokenize(text, punctuations=config['punctuations'], stop_words=config['stop_words'])
    text, label, unique_words, word_dict = sentiment.vectorize(text, label)
    text, label = np.array(text), np.array(label)
    model = sentiment.fit(text, label, epochs=config['epochs'], test_size=config['test_size'])

    #Validation
    text = ["Today is a bad day for me!", 
            "I'm happy"]
    # model = tf.keras.models.load_model("./models/model.h5")
    sentiment.predict_(text, model)
