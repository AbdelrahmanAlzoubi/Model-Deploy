import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd

# Load the model
model = load_model('pneumothorax_classifier.h5')  # Your saved model

# Define class names
class_names = ['Simple Pneumothorax', 'Tension Pneumothorax']

# Preprocessing + Prediction Function
def predictor(img_path):
    img = load_img(img_path, target_size=(224, 224))  # Resize for your model
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    preds = model.predict(img)
    preds = pd.DataFrame(preds[0], columns=["confidence"])
    preds["class"] = class_names
    preds = preds.sort_values(by="confidence", ascending=False).reset_index(drop=True)
    return preds
