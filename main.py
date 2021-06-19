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
    # verify preprocessing, use uncleaned data
    # weights
    # architechture
"""

if __name__ == '__main__':

    train_data = pd.read_csv(config['train_data_path'])
    val_data = pd.read_csv(config['val_data_path'])
    test_data = pd.read_csv(config['test_data_path'])

    train_data = train_data.dropna().reset_index(drop=True)
    val_data = val_data.dropna().reset_index(drop=True)
    test_data = test_data.dropna().reset_index(drop=True)

    train_text, train_label = train_data['cleaned_text'].values, train_data['Class_camel'].values
    val_text, val_label = val_data['cleaned_text'].values, val_data['Class_camel'].values
    test_text, test_label = test_data['cleaned_text'].values.copy(), test_data['Class_camel'].values.copy()


    #Check config file to alter the hyperparameters
    sentiment = SentimentAnalysis(preprocess.tokenizer, vocab_size=config['vocab_size'], maxlen=config['maxlen'], embedding_vector=config['embedding_vector'], method=config['method'],)
    
    train_text = sentiment.tokenize(train_text, punctuations=config['punctuations'], stop_words=config['stop_words'])
    train_text, train_label, unique_words, word_dict = sentiment.vectorize(train_text, train_label)

    val_text = sentiment.tokenize(val_text, punctuations=config['punctuations'], stop_words=config['stop_words'])
    val_text, val_label, _, _ = sentiment.vectorize(val_text, val_label)

    test_text = sentiment.tokenize(test_text, punctuations=config['punctuations'], stop_words=config['stop_words'])
    test_text, test_label, _, _ = sentiment.vectorize(test_text, test_label)

    model = sentiment.fit(train_text, train_label, validation_data=(val_text, val_label), epochs=config['epochs'], method=config['method'])
    sentiment.evaluate(test_text, test_label, model, batch_size=32)

    #Validation
    text = test_data['cleaned_text']
    # model = tf.keras.models.load_model("models/lstm_model.h5")
    sentiment.predict_(text[:4], model, batch_size=32)
    
