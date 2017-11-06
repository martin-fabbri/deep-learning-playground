import numpy as np
from pickle import dump
from pickle import load
from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences

def load_doc(filename):
    try:
        with open(filename, 'r') as file:
            text = file.read()
            return text
    except EnvironmentError:
        print('Cannot open file')

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

raw_text = load_doc('rhyme.txt')
print(raw_text)

# clean
tokens = raw_text.split()
raw_text = ' '.join(tokens)

print(raw_text)

# create sequences
length = 10
sequences = list()
for i in range(length, len(raw_text)):
    seq = raw_text[i-length: i+1]
    sequences.append(seq)
    print(seq)

print(f'Total sequences: {len(sequences)}')

seq_filename = 'char_sequence.txt'
save_doc(sequences, seq_filename)

raw_text = load_doc(seq_filename)
lines = raw_text.split('\n')

chars = sorted(set(raw_text))
mapping = dict((c, i) for i, c in enumerate(chars))

sequences = list()
for line in lines:
	# integer encode line
	encoded_seq = [mapping[char] for char in line]
	# store
	sequences.append(encoded_seq)

vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]

print(X, y)


sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
print(sequences)
X= np.array(sequences)
y = to_categorical(y, num_classes=vocab_size)

print(f"X shape: {X.shape}")
print(f"vocab_size: {vocab_size}")

model = Sequential()
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=2)
model.save('model.h5')
dump(mapping, open('mapping.pkl','wb'))

def generate_seq(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
	# generate a fixed number of characters
	for _ in range(n_chars):
		# encode the characters as integers
		encoded = [mapping[char] for char in in_text]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# one hot encode
		encoded = to_categorical(encoded, num_classes=len(mapping))
		encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])
		# predict character
		yhat = model.predict_classes(encoded, verbose=0)
		# reverse map integer to character
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		# append to input
		in_text += char
	return in_text

model = load_model('model.h5')
mapping = load(open('mapping.pkl', 'rb'))
print(generate_seq(model, mapping, 10, 'Sing a son', 20))