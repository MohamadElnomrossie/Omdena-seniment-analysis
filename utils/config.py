import pickle
with open("Datasets/stopWords.pkl", 'rb') as f:
    stop_words = list(pickle.load(f))
    stop_words = list(set(stop_words + ['و','في','من','بواسطة','أ','هو','و','في','سيكون','إلى','كان','كن','هو','ال','و','ما','ء','ه','س']))

config = {
    'vocab_size':60000,# 1200
    'maxlen':256,# 150
    'embedding_vector':50,

    'method':'1DConv',
    'stop_words':stop_words,
    'punctuations':"""'!"-#$%&'()*+,«».؛،/:؟?@[\]^_`{|}~""",

    'epochs':25,
    # 'test_size':0.2,

    'save_model_path':'models/',
    'save_weights_path':"models/",
    'train_data_path':"Datasets/Final_Dataset/Dataset/train.csv",
    'val_data_path':"Datasets/Final_Dataset/Dataset/val.csv",
    'test_data_path':"Datasets/Final_Dataset/Dataset/test.csv",
}
