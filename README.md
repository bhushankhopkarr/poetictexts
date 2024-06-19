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
Saves the trained model to textgenerator.h5.

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
Temperature 0.4<br>
Temperature 0.6<br>
Temperature 0.8<br>
Temperature 1.0<br>
