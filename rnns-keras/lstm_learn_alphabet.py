import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

# fix random seed for reproducibility
np.random.seed(7)

# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# create mapping of characters to integers (0-25) and the reverse
char_to_int = {c: i for i, c in enumerate(alphabet)}
int_to_char = {i: c for i, c in enumerate(alphabet)}

# print(char_to_int, int_to_char)

# prepare the dataset of input to output pais encoded as integers
seq_lenght = 1
dataX = []
dataY = []

for i in range(0, len(alphabet)-seq_lenght, 1):
    seq_in = alphabet[i:i+seq_lenght]
    seq_out = alphabet[i + seq_lenght]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    print(f"{seq_in} -> {seq_out}")

X = pad_sequences(dataX, maxlen=seq_lenght, dtype='float32')

# reshape X to be [samples, time steps, features?]
X = np.reshape(dataX, (X.shape[0], seq_lenght, 1))
# print(f"X.shape: {X.shape}")
# print(f"X -> {X}")

# normalize
X = X / float(len(alphabet))
# print(f"X -> {X}")

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# create and fit the model
model = Sequential()
model.add(LSTM(16, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(X, y, epochs=5000, batch_size=len(dataX), verbose=2, shuffle=False)

# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print(f"Model Accuracy: {scores[1]*100}")

# demostrate some model predictions
for pattern in dataX:
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, "->", result)

# demonstrate predicting random patterns
print("Test a Random Pattern:")
for i in range(0,20):
    pattern_index = np.random.randint(len(dataX))
    pattern = dataX[pattern_index]
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, "->", result)