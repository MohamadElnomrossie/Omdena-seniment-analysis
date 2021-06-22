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

    train_data = helper.get_folds(train_data, 'cleaned_text', 'Class_camel', split=2, getvalue=0)
    val_data = helper.get_folds(val_data, 'cleaned_text', 'Class_camel', split=2, getvalue=0)
    test_data = helper.get_folds(test_data, 'cleaned_text', 'Class_camel', split=2, getvalue=0)

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
    # text = test_data['cleaned_text']
    # model = tf.keras.models.load_model("models/lstm_model.h5")
    # sentiment.predict_(text[:4], model, batch_size=32)
    print(f"\nText : {test_data['cleaned_text'].iloc[45]} Label : {test_data['Class_camel'].iloc[45]}")
    print(sentiment.predict_([test_data['cleaned_text'].iloc[45]], model, batch_size=32))

    print(f"Text : {test_data['cleaned_text'].iloc[0]} Label : {test_data['Class_camel'].iloc[0]}")
    print(sentiment.predict_([test_data['cleaned_text'].iloc[0]], model, batch_size=32))

    print(f"Text : {test_data['cleaned_text'].iloc[35]} Label : {test_data['Class_camel'].iloc[35]}")
    print(sentiment.predict_([test_data['cleaned_text'].iloc[35]], model, batch_size=32))
    
