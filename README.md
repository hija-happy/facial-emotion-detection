# Facial Emotion Detection 

Welcome to the Emotion Detection Project! This project helps recognize emotions from images using machine learning. Follow these simple steps to use it:

## What You Need

1. **Python**: Make sure you have Python installed on your computer.
2. **Required Files**:
   - `prepare_data.py`: Prepares the data for training.
   - `train_model.py`: Trains the model to recognize emotions.
   - `predict_model.py`: Uses the trained model to predict emotions from new images.
   - **Data Files**: 
     - `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy` (generated after running `prepare_data.py` and `train_model.py`)
   - **CSV File**:
     - Download the CSV file used for training from [this link](https://www.kaggle.com/datasets/deadskull7/fer2013). 

## How to Use

1. **Setup**:
   - Download all the files from this repository.
   - Download the CSV file from the provided link and save it in the same directory.
   - Make sure you have the required Python packages installed. You can install them using `pip`:
     ```bash
     pip install numpy tensorflow pandas scikit-learn
     ```

2. **Prepare Data**:
   - Run this command to prepare the data from the CSV file:
     ```bash
     python prepare_data.py
     ```
   - This will create files `X_train.npy`, `X_test.npy`, `y_train.npy`, and `y_test.npy` that will be used for training the model.

3. **Train the Model**:
   - Run this command to train the model using the prepared data:
     ```bash
     python train_model.py
     ```
   - This will create a model file named `emotion_model.h5` that is used for predicting emotions.

4. **Predict Emotions**:
   - Run this command to predict emotions from a new image:
     ```bash
     python predict_model.py
     ```
   - Make sure to replace the `img_path` in `predict_model.py` with the path to your image file.

## Files Explained

- **`prepare_data.py`**: This script loads image data from the CSV file, preprocesses it, and splits it into training and testing sets. The preprocessed data is saved as `.npy` files.

- **`train_model.py`**: This script loads the preprocessed data, builds a neural network model to recognize emotions, trains the model on the data, and saves the trained model to a file.

- **`predict_model.py`**: This script loads a trained model, preprocesses a new image, and predicts the emotion shown in that image.

## Notes

- Ensure that the paths to the CSV file and image files in `prepare_data.py` and `predict_model.py` are correctly set according to your file locations.
- For any questions or issues, feel free to reach out!
