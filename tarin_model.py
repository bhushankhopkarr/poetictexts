import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

# Download and read the Shakespeare text file
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()[300000:800000]

# Prepare data
characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

SEQ_LENGTH, STEP_SIZE = 40, 3
sentences = [text[i: i + SEQ_LENGTH] for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE)]
next_characters = [text[i + SEQ_LENGTH] for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE)]

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

# Build and compile the model
model = Sequential([
    LSTM(128, input_shape=(SEQ_LENGTH, len(characters))),
    Dense(len(characters), activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

# Train the model
model.fit(x, y, batch_size=256, epochs=4)
model.save('textgenerator.model/textgenerator.h5')
print("Model saved to textgenerator.model/textgenerator.h5")
