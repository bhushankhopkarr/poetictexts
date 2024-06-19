import numpy as np
import tensorflow as tf

# Load the pre-trained model
def load_model(model_path='textgenerator.model/textgenerator.h5'):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to sample predictions
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