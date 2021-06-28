import pickle
with open("Datasets/stopWords.pkl", 'rb') as f:
    stop_words = list(pickle.load(f))
    stop_words = list(set(stop_words + ['و','في','من','بواسطة','أ','هو','و','في','سيكون','إلى','كان','كن','هو','ال','و','ما','ء','ه','س']))

config = {
    'vocab_size':60000,
    'maxlen':256,
    'embedding_vector':100,

    'method':'1DConv', #other - simpleRNN, bidRNN, 1DConv, lstm
    'stop_words':stop_words,
    'punctuations':"""'!"-#$%&'()*+,«».؛،/:؟?@[\]^_`{|}~""",

    'epochs':50,
    'optim':'adam', # other - adamax, adadelta, SGD, Adam, RMSprop
    'learning_rate':1e-4,

    'save_model_path':'models/',
    'save_weights_path':"models/",
    'train_data_path':"Datasets/Final_Dataset/Dataset/train.csv",
    'val_data_path':"Datasets/Final_Dataset/Dataset/val.csv",
    'test_data_path':"Datasets/Final_Dataset/Dataset/test.csv",
}
