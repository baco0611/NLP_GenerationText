import pickle
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Embedding, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import Sequence

# path = '2.VanHoaFull'
path = '3.Full'

with open(path + "/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open(path + "/sequences_digit.pkl", "rb") as f:
    sequences_digit = pickle.load(f)

# Tạo một class generator
class DataGenerator(Sequence):
    def __init__(self, X, y, vocab_size, batch_size):
        self.X = X
        self.y = y
        self.vocab_size = vocab_size
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_X = self.X[start:end]
        batch_y = self.y[start:end]
        batch_y = keras.utils.to_categorical(batch_y, num_classes=self.vocab_size)
        return batch_X, batch_y

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)
print(len(sequences_digit))
sequences_digit = np.array(sequences_digit)
X, y = sequences_digit[:,:-1], sequences_digit[:,-1]
# y = keras.utils.to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]
batch_size = 64
epochs = 50

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=10))
model.add(BatchNormalization())
model.add(LSTM(batch_size, return_sequences=True))
model.add(LSTM(batch_size))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(vocab_size, activation='softmax'))
model.summary()

# Tạo một callback EarlyStopping
early_stopping = EarlyStopping(monitor='accuracy', min_delta=0, patience=3, verbose=1, mode='max', baseline=0.75)
stop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5, verbose = 1, mode = 'auto')
callbacks = [stop]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Tạo và sử dụng generator
generator = DataGenerator(X, y, vocab_size, batch_size)

# Huấn luyện model sử dụng fit_generator
model.fit_generator(generator, epochs=epochs, callbacks=[callbacks])

# Lưu model
model.save(path + '/acc_language_model.hdf5')
