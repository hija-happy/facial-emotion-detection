<<<<<<< HEAD
#prepare_data.py
# import pandas as pd
=======
import pandas as pd
>>>>>>> 84d27ca (initial commit)
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(csv_file):
    data = pd.read_csv(csv_file)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).values
    return faces, emotions

def preprocess_data(faces):
    faces = faces / 255.0
    return faces

if __name__ == "__main__":
    csv_file = r"D:\6TH SEM\CG_lab\fer13\fer2013.csv"
    faces = preprocess_data(faces) # type: ignore
    X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2, random_state=42) # type: ignore
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    
    

