import numpy as np
import tensorflow as tf
from sklearn import model_selection, preprocessing
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

from utils import helper


class SentimentAnalysis:
    def __init__(self, tokenizer, vocab_size=30, maxlen=10, embedding_vector=5, method="simpleRNN"):
        self.method = method
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.embedding_vector = embedding_vector


    def tokenize(self, text, stop_words=['and', 'a', 'is', 'the', 'in', 'be', 'will']):
        tokens, maxlen = self.tokenizer(text, stop_words)
        if not self.maxlen:
            self.maxlen = maxlen

        return tokens


    def vectorize(self, text, label):
        vector, temp, all_ = [], [], []
        for d in text:
            for i in d:
                temp.extend(one_hot(i, self.vocab_size))
            vector.append(temp)
            temp=[]
        vector = pad_sequences(vector, maxlen=self.maxlen, padding="post")
        for x in text:
            all_.extend(x)
        word_dict = helper.word_dictionary(all_)

        label = preprocessing.LabelEncoder().fit_transform(label)

        return vector, label, list(word_dict.keys()), word_dict


    def fit(self, text, label, epochs=10, test_size=0.2):

        trainX, validX, trainY, validY = model_selection.train_test_split(text, label, random_state=42, shuffle=True)
        print(trainX.shape, validX.shape, trainY.shape, validY.shape)

        # trainX = tf.Tensor(tf.data.Dataset.from_tensors(tf.constant(trainX)).batch(16, drop_remainder=True), value_index=, dtype=tf.int32)
        # validY = tf.constant(validY).set_shape([16, validY.shape[0], validY.shape[1]])
        
        model = helper.get_model(trainX, trainY, self.vocab_size, self.embedding_vector, self.maxlen, self.method)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        print(model.summary())

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)  
        mc = ModelCheckpoint('models/model.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)
        model.fit(trainX, trainY, validation_data=(validX, validY), epochs=epochs, callbacks=[es, mc], verbose=1)
        model.save_weights("models/weights.h5", overwrite=True)
        loss, acc = model.evaluate(validX, validY, workers=-1)
        print("Validation loss: {}  Validatoin acc: {}".format(loss, acc))

        return model

    def predict(self, text, model):
        for i in text:
            pred = helper.get_prediction(self.tokenize, self.vectorize, i, model)
            print('-'*20)
            print("Positive" if pred[0][0] > 0.5 else "Negative")
            print('-'*20)
