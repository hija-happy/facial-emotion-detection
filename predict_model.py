import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

def predict_emotion(img_path, model_path='emotion_model.h5'):
    model = load_model(model_path)
    
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    # Predict the emotion
    predictions = model.predict(img_array)
    emotion_index = np.argmax(predictions[0])
    
    # Define emotion labels
    emotion_labels = {
        0: 'Anger',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happiness',
        4: 'Sadness',
        5: 'Surprise',
        6: 'Neutral'
    }
    
    # Get the emotion name
    emotion = emotion_labels.get(emotion_index, 'Unknown')
    return emotion

if __name__ == "__main__":
    img_path = r"D:\6TH SEM\CG_lab\sadbaby.jpg"
    emotion = predict_emotion(img_path)
    print(f'Predicted emotion: {emotion}')

