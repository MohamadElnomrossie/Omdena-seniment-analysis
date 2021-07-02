import numpy as np
import tensorflow as tf
from sklearn import model_selection, preprocessing, metrics
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adamax, Adadelta
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import tensorflow_addons as tfa

from utils import helper
from utils.config import config


class SentimentAnalysis:
    def __init__(self, tokenizer, vocab_size=30000, maxlen=256, embedding_vector=50, method="simpleRNN"):
        self.method = method
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.embedding_vector = embedding_vector
        self.optim = {'adam':Adam,
                    'adamax':Adamax,
                    'adadelta':Adadelta,
                    'SGD':SGD,
                    'RMSprop':RMSprop}


    def tokenize(self, text, punctuations=[], stop_words=[]):
        tokens, maxlen, vocab = self.tokenizer(text, punctuations, stop_words)
        if self.maxlen == 'auto':
            self.maxlen = maxlen
        if self.vocab_size == 'auto':
            self.vocab_size = vocab
        return tokens


    def vectorize(self, text, label, return_label=True):
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
        if return_label:
            onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
            label = np.array(label).reshape(len(label), 1)
            label = onehot_encoder.fit_transform(label)
            return vector, label, list(word_dict.keys()), word_dict
        return vector, list(word_dict.keys()), word_dict


    def fit(self, trainX, trainY, validation_data=(), epochs=10, batch_size=32, method='simpleRNN'):

        validX, validY = validation_data
        model = helper.get_model(trainX, trainY, self.vocab_size, self.embedding_vector, self.maxlen, self.method)
        model.compile(optimizer=self.optim[config['optim']](learning_rate=config['learning_rate']), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy",tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.AUC()])

        tqdm_callback = tfa.callbacks.TQDMProgressBar()
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=5)  
        mc = ModelCheckpoint(config['save_model_path'] + method + "_model.h5", monitor='val_loss', mode='min', save_best_only=True,verbose=1)
        model.fit(trainX, trainY, batch_size=batch_size, validation_data=(validX, validY), epochs=epochs, callbacks=[mc, tqdm_callback], verbose=0)
        model.save_weights(config['save_weights_path'] + method + "_weights.h5", overwrite=True)

        return model

    def evaluate(self, text, label, model, batch_size=32):
        loss, acc, pre, rec, auc  = model.evaluate(text, label, workers=-1, batch_size=batch_size)
        print("\nValidation loss: {}  Validation acc: {} Precision: {} Recall: {} Auc Roc: {}".format(loss, acc, pre, rec, auc))

    def predict_(self, text, model, batch_size=32, print_=True):
        text = helper.predict(text, model, self.tokenizer, self.vocab_size, self.maxlen)
        pred = model.predict(text, batch_size=batch_size)
        if print_:
            for p in pred:
                print('-'*20)
                pp = np.argmax(p)
                if pp == 0:
                    print(f"Negative {p[pp]}")
                elif pp == 1:
                    print(f"Neutral {p[pp]}")
                else:
                    print(f"Positive {p[pp]}")
                print('-'*20)
        else:
            gtruth = ["Negative", "Neutral", "Positive"]
            return [gtruth[np.argmax(p)] for p in pred]
