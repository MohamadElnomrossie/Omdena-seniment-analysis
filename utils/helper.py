from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot


def word_dictionary(text):
    text = set(list(text))
    dictionary = {}
    for i, word in enumerate(text):
        dictionary[word] = i

    return dictionary

def predict(text, model, tokenize, vocab, maxlen):
    text, _, _ = tokenize(text)
    vector, temp = [], []
    for d in text:
        for i in d:
            temp.extend(one_hot(i, vocab))
        vector.append(temp)
        temp=[]
    vector = pad_sequences(vector, maxlen=maxlen, padding="post")
    return vector

def get_model(X, y, vocab_size, embedding_size, maxlen, method):
    if method == "simpleRNN":
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_size, input_length=maxlen ,name="embedding"))
        model.add(SimpleRNN(64))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))
    
    elif method == "bidRNN":
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_size, input_length=maxlen ,name="embedding"))
        model.add(Bidirectional(LSTM(64)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))

    elif method == "1DConv":
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_size, input_length=maxlen ,name="embedding"))
        model.add(Conv1D(64, 6, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))

    elif method == "lstm":
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_size, input_length=maxlen ,name="embedding"))
        model.add(LSTM(64))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))

    return model
