import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

# Read the file
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

# Use a subset of the text
text = text[300000:800000]

# Create a sorted list of unique characters in the text
characters = sorted(set(text))

# Create dictionaries to map characters to indices and vice versa
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3

model = tf.keras.models.load_model('textgenerator.model/textgenerator.h5')

#text generation function
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Function to generate text
def generate_text(model, text, char_to_index, index_to_char, length=100, temperature=1.0):
    SEQ_LENGTH = 40
    start_index = np.random.randint(0, len(text) - SEQ_LENGTH - 1)
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated = sentence
    
    for _ in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(char_to_index)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character

    return generated

print('---------0.2---------')
print(generate_text(300, 0.2))
print('---------0.4---------')
print(generate_text(300, 0.4))
print('---------0.6---------')
print(generate_text(300, 0.6))
print('---------0.8---------')
print(generate_text(300, 0.8))
print('---------1---------')
print(generate_text(300, 1.0))