import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from pyvi import ViTokenizer, ViPosTagger
import string

# path = './1.VanHoa150'
path = './3.Full'
length = 10

with open(path + '/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    
with open(path + '/sequences_digit.pkl', 'rb') as f:
    sequences_digit = pickle.load(f)

with open(path + '/sequences.pkl', 'rb') as f:
    sequences = pickle.load(f)

model = load_model(path + '/51_acc_language_model.h5')

def clean_document(doc):
    doc = ViTokenizer.tokenize(doc) #Pyvi Vitokenizer library
    doc = doc.lower() #Lower
    tokens = doc.split() #Split in_to words
    table = str.maketrans('', '', string.punctuation.replace("_", "")) #Remove all punctuation
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word]
    return tokens


def preprocess_input(doc):
    tokens = clean_document(doc)
    tokens = tokenizer.texts_to_sequences(tokens)
    tokens = pad_sequences([tokens], maxlen=length, truncating='pre')
    return np.reshape(tokens, (1,length))

def generate_text(text_input, n_words):
    tokens = preprocess_input(text_input)
    result = np.reshape(tokens, (len(tokens[0])))
    result = [x for x in result if x]
    # print(result)

    for _ in range(n_words):
        # next_digit = model.predict_classes(tokens)
        # # Vì predict_classes đã bị loại bỏ ở các phiên bản keras gần đây nên thay thế bên dưới
        probabilities = model.predict(tokens)
        next_digit= np.argmax(probabilities, axis=1)
        result.append(next_digit)
        tokens = np.append(tokens, next_digit)
        tokens = np.delete(tokens, 0)
        tokens = np.reshape(tokens, (1, length))
    
    # Mapping to text  
    tokens = np.reshape(tokens, (length))
    out_word = []
    for token in result:
        for word, index in tokenizer.word_index.items():
            if index == token:
                out_word.append(word)
                break

    return ' '.join(out_word)

def get_raw_sentence(sentence):
    sentence = sentence.split('_')
    return ' '.join(sentence)

def main_generation():
    input_seq = input('Nhập văn bản: ')
    number_word = int(input('Nhập số lượng chữ muốn sinh ra: '))
    sentence = generate_text(text_input=input_seq, n_words=number_word)
    sentence = get_raw_sentence(sentence)
    print(sentence)

main_generation()