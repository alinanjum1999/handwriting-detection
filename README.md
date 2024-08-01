# Handwriting Recognition Project

## Overview
This project focuses on recognizing handwritten digits using a machine learning model. The project utilizes TensorFlow and Keras libraries to train, evaluate, and deploy a neural network for recognizing handwritten digits.

## Project Structure

## Files and Directories

- **digits/**: Contains the dataset of handwritten digits, organized into training and testing sets.
- **variables/**: Directory containing the model variables saved during training.
- **.gitattributes**: Git configuration file for managing repository attributes.
- **README.md**: This file. Provides an overview and instructions for the project.
- **fingerprint.pb**: Protocol buffer file, possibly containing model fingerprints or input data specifications.
- **handwritingrecognition.py**: The main Python script for training and testing the handwriting recognition model.
- **keras_metadata.pb**: Metadata for the Keras model, stored in protocol buffer format.
- **saved_model.pb**: The saved TensorFlow model in protocol buffer format.
- **test.model**: A pre-trained model saved in binary format.

## Getting Started

### Prerequisites
- Python 3.7 or later
- TensorFlow 2.0 or later
- Keras
- NumPy
- Matplotlib (for visualization)

Install the required libraries using pip:
```bash
pip install tensorflow keras numpy matplotlib
```
Dataset
Ensure the dataset of handwritten digits is placed in the digits/ directory, organized into train and test folders, each containing subfolders for each digit (0-9).

Training the Model
To train the model, run the handwritingrecognition.py script:
```bash
python handwritingrecognition.py --mode train
```
This will train the model using the training dataset and save the model to saved_model.pb.

Evaluating the Model
To evaluate the model on the test dataset, run:
```bash
python handwritingrecognition.py --mode evaluate
```
Script Details
handwritingrecognition.py
This script contains functions for:

Loading and preprocessing the dataset
Building the neural network model
Training the model
Evaluating the model
Saving and loading the model
Command-line Arguments
--mode: Specify the mode of operation. Options are train and evaluate.
Example usage:
```bash
python handwritingrecognition.py --mode train
```
Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
The MNIST dataset for handwritten digits.
TensorFlow and Keras libraries for building and training the model.<br/>
Contact
For any questions or issues, please contact alinanjum1999@gmail.com or open an issue here.

