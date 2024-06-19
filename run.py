import numpy as np
import tensorflow as tf

# Function to load the pre-trained model
def load_model(model_path='textgenerator.model/textgenerator.h5'):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Function to preprocess text
def load_text(file_path, start=300000, end=800000):
    try:
        text = open(file_path, 'rb').read().decode('utf-8').lower()
        return text[start:end]
    except Exception as e:
        print(f"Error reading text file: {e}")
        raise

# Function to create character mappings
def create_char_mappings(text):
    characters = sorted(set(text))
    char_to_index = {c: i for i, c in enumerate(characters)}
    index_to_char = {i: c for i, c in enumerate(characters)}
    return char_to_index, index_to_char

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

def main():
    # Load the pre-trained model
    model = load_model()

    # Load and preprocess the text
    filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    text = load_text(filepath)

    # Create character mappings
    char_to_index, index_to_char = create_char_mappings(text)

    # Generate and print text with different temperatures
    temperatures = [0.2, 0.4, 0.6, 0.8, 1.0]
    for temp in temperatures:
        print(f'--------- Temperature {temp} ---------')
        generated_text = generate_text(model, text, char_to_index, index_to_char, length=300, temperature=temp)
        print(generated_text)
        print('-' * 80)

if __name__ == "__main__":
    main()