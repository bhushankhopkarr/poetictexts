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

Run train_model.py to Train the Model
```
python train_model.py
```
###
This script performs the following tasks:

Downloads and reads the Shakespeare text file.
Preprocesses the text data into training examples.
Builds and trains the LSTM model.
Saves the trained model to *textgenerator.model/textgenerator.h5*.

Generating Text <br>
Once the model is trained, you can use the run.py script to generate text based on the trained model.

Run main.py to Generate Text
```
python main.py
```
This script uses the trained model to generate text. You can specify different temperatures to control the randomness of the output. The higher the temperature, the more random the output.

Example Output<br>
The script prints generated text with different temperature settings:

Temperature 0.2<br>
![Screenshot 2024-06-19 130229](https://github.com/bhushankhopkarr/poetictexts/assets/121181515/95801374-de4d-4cdd-adeb-bef9f6d07f8b)

Temperature 0.4<br>
![Screenshot 2024-06-19 130236](https://github.com/bhushankhopkarr/poetictexts/assets/121181515/e94a8ea3-8027-485f-9286-cd222eb96a00)

Temperature 0.6<br>
![Screenshot 2024-06-19 130242](https://github.com/bhushankhopkarr/poetictexts/assets/121181515/4f4ac067-2be9-479d-a427-faed2010e617)

Temperature 0.8<br>
![Screenshot 2024-06-19 130256](https://github.com/bhushankhopkarr/poetictexts/assets/121181515/4517b060-336b-4595-bdad-03558a73ee0e)

Temperature 1.0<br>
![image](https://github.com/bhushankhopkarr/poetictexts/assets/121181515/f4a46a10-70f8-4dd9-a316-89ab6a77ba58)


