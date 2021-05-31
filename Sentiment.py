import numpy as np
import tensorflow as tf
from sklearn import model_selection, preprocessing
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

from utils import helper
from utils.config import config


class SentimentAnalysis:
    def __init__(self, tokenizer, vocab_size=30, maxlen=10, embedding_vector=5, method="simpleRNN"):
        self.method = method
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.embedding_vector = embedding_vector


    def tokenize(self, text, punctuations=[], stop_words=[]):
        tokens, maxlen, vocab = self.tokenizer(text, punctuations, stop_words)
        if self.maxlen == 'auto':
            self.maxlen = maxlen
        if self.vocab_size == 'auto':
            self.vocab_size = vocab
        print(vocab, maxlen)
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

        #label = preprocessing.LabelEncoder().fit_transform(label)
        onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        label = np.array(label).reshape(len(label), 1)
        label = onehot_encoder.fit_transform(label)
        return vector, label, list(word_dict.keys()), word_dict


    def fit(self, text, label, epochs=10, test_size=0.2):
        trainX, validX, trainY, validY = model_selection.train_test_split(text, label, random_state=42, shuffle=True)

        # trainX = tf.Tensor(tf.data.Dataset.from_tensors(tf.constant(trainX)).batch(16, drop_remainder=True), value_index=, dtype=tf.int32)
        # validY = tf.constant(validY).set_shape([16, validY.shape[0], validY.shape[1]])
        print(trainX.shape, validX.shape, trainY.shape, validY.shape)
        model = helper.get_model(trainX, trainY, self.vocab_size, self.embedding_vector, self.maxlen, self.method)
        model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
        print(model.summary())

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)  
        mc = ModelCheckpoint(config['save_model_path'], monitor='val_loss', mode='min', save_best_only=True,verbose=1)
        model.fit(trainX, trainY, validation_data=(validX, validY), epochs=epochs, callbacks=[es, mc], verbose=1)
        model.save_weights(config['save_weights_path'], overwrite=True)
        loss, acc = model.evaluate(validX, validY, workers=-1)
        print("Validation loss: {}  Validation acc: {}".format(loss, acc))

        return model

    def predict_(self, text, model):
        text = helper.predict(text, model, self.tokenizer, self.vocab_size, self.maxlen)
        pred = model.predict(text)
        print(pred)
        for p in pred:
            print('-'*20)
            pp = np.argmax(p)
            if pp == 0:
                print(f"Neutral {p[pp]}")
            elif pp == 1:
                print(f"Negative {p[pp]}")
            else:
                print(f"Positive {p[pp]}")
            print('-'*20)
        
