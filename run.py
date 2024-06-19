import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('textgenerator.model/textgenerator.h5')

# Read the text file
filepath = tf.keras.utils.get_file('shakespeare.txt',
                                   'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode('utf-8').lower()[300000:800000]

# Create character mappings
characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

SEQ_LENGTH = 40

# Sampling function
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.argmax(np.random.multinomial(1, preds, 1))

# Text generation function
def generate_text(model, text, char_to_index, index_to_char, length=100, temperature=1.0):
    start_index = np.random.randint(0, len(text) - SEQ_LENGTH - 1)
    sentence = text[start_index:start_index + SEQ_LENGTH]
    generated = sentence

    for _ in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(char_to_index)))
        for t, char in enumerate(sentence):
            x[0, t, char_to_index[char]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character

    return generated

# Generate and print text with different temperatures
temperatures = [0.2, 0.4, 0.6, 0.8, 1.0]
for temp in temperatures:
    print(f'--------- Temperature {temp} ---------')
    print(generate_text(model, text, char_to_index, index_to_char, length=300, temperature=temp))
    print('-' * 80)