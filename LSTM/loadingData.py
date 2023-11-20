from pyvi import ViTokenizer, ViPosTagger
import string
import os
import pickle
import keras

def clean_document(doc):
    doc = ViTokenizer.tokenize(doc) #Pyvi Vitokenizer library
    doc = doc.lower() #Lower
    tokens = doc.split() #Split in_to words
    table = str.maketrans('', '', string.punctuation.replace("_", "")) #Remove all punctuation
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word]
    return tokens


def load_all_folder(folder_dir, INPUT_LENGTH):
    sequences = []
    count = 1

    for file_dir in os.listdir(folder_dir):
        for filename in os.listdir(folder_dir + '/' + file_dir):
            print(count, "Loading in file", filename)
            count+=1
            if count > 1000:
                break
            f1 = open(folder_dir + '/' + file_dir + "/" + filename, encoding='utf-16')
            doc = f1.read()
            tokens = clean_document(doc)

            for i in range(INPUT_LENGTH + 1, len(tokens)):
                seq = tokens[i-INPUT_LENGTH-1:i]
                line = ' '.join(seq)
                sequences.append(line)

    return sequences

def load_one_folder(folder_dir, file_dir, INPUT_LENGTH):
    count = 1
    sequences = []

    for filename in os.listdir(folder_dir + '/' + file_dir):
        print(count, "Loading in file", filename)
        count+=1
        # if count == 150:
        #     break
        f1 = open(folder_dir + '/' + file_dir + "/" + filename, encoding='utf-16')
        doc = f1.read()
        tokens = clean_document(doc)

        for i in range(INPUT_LENGTH + 1, len(tokens)):
            seq = tokens[i-INPUT_LENGTH-1:i]
            line = ' '.join(seq)
            sequences.append(line)
    
    return sequences


def loading_data():
    INPUT_LENGTH = 10
    # INPUT_LENGTH = 50
    folder_dir = "./Train_Full"
    path = '3.Full/'
    file_dir = "Vanhoa"

    sequences = load_all_folder(folder_dir, INPUT_LENGTH)
    # sequences = load_one_folder(folder_dir, file_dir, INPUT_LENGTH)
    

    with open(path + "sequences.pkl", "wb") as f:
        pickle.dump(sequences, f)

    tokenizer = keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~ ')
    tokenizer.fit_on_texts(sequences)

    with open(path + "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    sequences_digit = tokenizer.texts_to_sequences(sequences)

    with open(path + "sequences_digit.pkl", "wb") as f:
        pickle.dump(sequences_digit, f)

loading_data()