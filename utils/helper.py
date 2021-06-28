from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from sklearn.model_selection import StratifiedKFold

def get_folds(df, source, target, split=4, getvalue=0):
    df.loc[:, "kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    X = df[source].values
    y = df[target].values
    skf = StratifiedKFold(n_splits=split)

    for fold_, (train_, val_) in enumerate(skf.split(X=X, y=y)):
        df.loc[val_, "kfold"] = fold_
    return df.loc[df['kfold'] == getvalue]

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
        sequence = Input(shape=(maxlen,), dtype='int32')
        embedded = Embedding(vocab_size, embedding_size, input_length=maxlen)(sequence)
        forward_layer = LSTM(64, return_sequences=True)(embedded)
        backward_layer = LSTM(64, activation='relu', return_sequences=True,
                            go_backwards=True)(embedded)
        merged = concatenate([forward_layer, backward_layer])
        bid = Bidirectional(merged)(merged)
        after_dp = Dropout(0.5)(bid)
        output = Dense(3, activation='softmax')(after_dp)
        model = Model(inputs=sequence, outputs=output)

    elif method == "1DConv":
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_size, input_length=maxlen ,name="embedding"))
        model.add(Convolution1D(filters=64,kernel_size=7,activation='relu'))
        model.add(MaxPooling1D(pool_size=3))
        model.add(Convolution1D(filters=64,kernel_size=7,activation='relu'))
        model.add(MaxPooling1D(pool_size=3))
        model.add(Convolution1D(filters=32,kernel_size=3,activation='relu'))
        model.add(MaxPooling1D(pool_size=3))
        model.add(Flatten())
        model.add(Dense(activation='relu',units=64))
        model.add(Dropout(0.2))
        model.add(Dense(activation='relu',units=32))
        model.add(Dropout(0.1))
        model.add(Dense(units=3,activation='softmax'))

    elif method == "lstm":
        sequence = Input(shape=(maxlen,), dtype='int32')
        embedded = Embedding(vocab_size, embedding_size, input_length=maxlen)(sequence)
        forwards = LSTM(64)(embedded)
        backwards = LSTM(64, go_backwards=True)(embedded)
        merged = concatenate([forwards, backwards])
        after_dp = Dropout(0.2)(merged)
        output = Dense(3, activation='softmax')(after_dp)
        model = Model(inputs=sequence, outputs=output)

    return model
