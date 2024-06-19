# Text Generation with Shakespeare's Works

This project uses a recurrent neural network (RNN) with Long Short-Term Memory (LSTM) layers to generate text based on Shakespeare's works. The model is trained on a subset of Shakespeare's text and can generate new text sequences based on the learned patterns.

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- NumPy

You can install the required packages using pip:

```
pip install tensorflow numpy
```
Training the Model 
To train the model, run the main.py script. This script will download the text data, preprocess it, and train the LSTM model.

Run main.py to Train the Model
```
python main.py
```
###
This script performs the following tasks:

Downloads and reads the Shakespeare text file.
Preprocesses the text data into training examples.
Builds and trains the LSTM model.
Saves the trained model to *textgenerator.model/textgenerator.h5*.

Generating Text <br>
Once the model is trained, you can use the run.py script to generate text based on the trained model.

Run run.py to Generate Text
```
python run.py
```
This script uses the trained model to generate text. You can specify different temperatures to control the randomness of the output. The higher the temperature, the more random the output.

Example Output<br>
The script prints generated text with different temperature settings:

Temperature 0.2<br>
![Screenshot 2024-06-19 113645](https://github.com/bhushankhopkarr/poetictexts/assets/121181515/12697b0c-0922-4212-b587-488bcb9d455b)

Temperature 0.4<br>
![Screenshot 2024-06-19 113654](https://github.com/bhushankhopkarr/poetictexts/assets/121181515/b38d6299-9962-45e0-9828-71cc5442309b)

Temperature 0.6<br>
![Screenshot 2024-06-19 113704](https://github.com/bhushankhopkarr/poetictexts/assets/121181515/af046316-51ec-419b-ab10-25512c93412b)

Temperature 0.8<br>
![Screenshot 2024-06-19 113713](https://github.com/bhushankhopkarr/poetictexts/assets/121181515/6986ad40-8b59-4837-8ba2-a51d8548c7aa)

Temperature 1.0<br>
![Screenshot 2024-06-19 113730](https://github.com/bhushankhopkarr/poetictexts/assets/121181515/924e43cb-aa57-4b23-9c3e-0e9061b0a158)

