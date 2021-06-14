
config = {
    'vocab_size':1200,# 1200
    'maxlen':150,# 150
    'embedding_vector':10,

    'method':'simpleRNN',
    'stop_words':['و','في','من','بواسطة','أ','هو','و','في','سيكون','إلى','كان','كن','هو','ال','و','ما','ء','ه'],
    'punctuations':"""'!"#$%&'()*+,«».؛،/:؟?@[\]^_`{|}~""",

    'epochs':8,
    # 'test_size':0.2,

    'save_model_path':'models/',
    'save_weights_path':"models/",
    'train_data_path':"Datasets/Final_Dataset/Dataset/train.csv",
    'val_data_path':"Datasets/Final_Dataset/Dataset/val.csv",
    'test_data_path':"Datasets/Final_Dataset/Dataset/test.csv",
}
