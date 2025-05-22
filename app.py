import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model('pneumothorax_classifier.h5')

# Class names
class_names = ['Simple Pneumothorax', 'Tension Pneumothorax']

# Page title
st.title("Pneumothorax Type Classifier")
st.write("Upload a chest X-ray image, and the model will predict whether it's a simple or tension pneumothorax.")

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

# Image preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

if uploaded_file is not None:
    # Display uploaded image
    image_data = Image.open(uploaded_file).convert("RGB")
    st.image(image_data, caption='Uploaded Chest X-ray', use_column_width=True)

    # Preprocess image
    img = preprocess_image(image_data)

    # Predict
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Display result
    st.markdown(f"### Prediction: **{class_names[predicted_class]}**")
    st.markdown(f"Confidence: {prediction[0][predicted_class]:.2%}")
