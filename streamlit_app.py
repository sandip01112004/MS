import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")  # Replace with the path to your model
    return model

model = load_model()

# Define class names (adjust based on your project)
CLASS_NAMES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match the model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("MediScan: AI-Powered Medical Image Diagnosis")
st.write("Upload a medical eye image to predict its condition (e.g., Cataract, Diabetic Retinopathy).")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Analyzing the image...")

    # Preprocess the image and make predictions
    input_image = preprocess_image(image)
    predictions = model.predict(input_image)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display the results
    st.write(f"### Predicted Condition: **{predicted_class}**")
    st.write(f"### Confidence: **{confidence:.2f}%**")
