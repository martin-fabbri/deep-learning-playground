import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# fix random seed for reproducibility
np.random.seed(7)

# load the dataset but keep the top n words, zero rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

print(f"X_train: {X_train}")
print(f"X_train.shape: {X_train.shape}")
print(f"len(X_train[0]): {len(X_train[0])}")
print(f"len(X_train[1]): {len(X_train[1])}")

print(f"Y_train: {y_train}")
print(f"Y_train.shape: {y_train.shape}")
print(f"Y_train[0]: {y_train[0]}")

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
print(f"X_train: {X_train}")
print(f"X_train.shape: {X_train.shape}")
print(f"len(X_train[0]): {len(X_train[0])}")
print(f"len(X_train[1]): {len(X_train[1])}")

# create the model
embedding_vector_lenght = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_lenght, input_length=max_review_length))
# model.add(Dropout(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())
model.fit(X_train, y_train, epochs=5, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

