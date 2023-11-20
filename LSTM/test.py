from pyvi import ViTokenizer, ViPosTagger
import string
import os
import pickle
import keras

def clean_document(doc):
    doc = ViTokenizer.tokenize(doc) #Pyvi Vitokenizer library
    doc = doc.lower() #Lower
    tokens = doc.split() #Split in_to words
    # table = str.maketrans('', '', string.punctuation.replace("_", "")) #Remove all punctuation
    # tokens = [w.translate(table) for w in tokens]
    table = str.maketrans('', '', string.punctuation.replace("_", "")) #Remove all punctuation
    tokens = [word for word in tokens if word]
    return tokens

print(clean_document("Xin chào, mọi người."))