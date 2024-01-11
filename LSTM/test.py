# from pyvi import ViTokenizer, ViPosTagger
# import string
# import os
# import pickle
# import keras

# # def clean_document(doc):
# #     doc = ViTokenizer.tokenize(doc) #Pyvi Vitokenizer library
# #     doc = doc.lower() #Lower
# #     tokens = doc.split() #Split in_to words
# #     # table = str.maketrans('', '', string.punctuation.replace("_", "")) #Remove all punctuation
# #     # tokens = [w.translate(table) for w in tokens]
# #     table = str.maketrans('', '', string.punctuation.replace("_", "")) #Remove all punctuation
# #     tokens = [word for word in tokens if word]
# #     return tokens

# # print(clean_document("Xin chào, mọi người."))

# path = './3.Full'
# with open(path + '/tokenizer.pkl', 'rb') as f:
#     tokenizer = pickle.load(f)

# print(len(tokenizer.word_index.keys()))

import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from pyvi import ViTokenizer, ViPosTagger
import string

# path = './1.VanHoa150'
# path = './3.Full'
# path = '0.Full'
path = '2.VanHoaFull'
length = 10

with open(path + '/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    
# with open(path + '/sequences_digit.pkl', 'rb') as f:
#     sequences_digit = pickle.load(f)

# with open(path + '/sequences.pkl', 'rb') as f:
#     sequences = pickle.load(f)

# model = load_model(path + '/51_acc_language_model.h5')
print(tokenizer.word_index)
print(len(tokenizer.word_index.keys()))