
config = {
    'vocab_size':'auto',# 1200
    'maxlen':'auto',# 1500
    'embedding_vector':10,

    'method':'simpleRNN',
    'stop_words':['and', 'a', 'is', 'the', 'in', 'be', 'will', 't', 'll', 'of', 'to', 'was', 'its', 'm', 've'],
    'punctuations':"""'!"#$%&'()*+,«».؛،/:؟?@[\]^_`{|}~""",

    'epochs':8,
    'test_size':0.2,

    'save_model_path':'models/model.h5',
    'save_weights_path':"models/weights.h5",
    'data_path':"Datasets/AJGT.xlsx"
}