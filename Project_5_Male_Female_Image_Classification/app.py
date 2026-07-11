import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# Load trained model
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gender_model.pkl")

model = joblib.load(MODEL_PATH)

IMG_SIZE = 64

st.set_page_config(
    page_title="Gender Classification",
    page_icon="🧑",
    layout="centered"
)

st.title("🧑 Male / Female Image Classifier")
st.write("Upload an image to predict whether it is Male or Female.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to OpenCV format
    img = np.array(image)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Flatten
    img = img.flatten().reshape(1, -1)

    # Prediction
    prediction = model.predict(img)[0]

    classes = ["Male", "Female"]

    st.subheader("Prediction")
    st.success(classes[prediction])

# About the developer
st.markdown("---")
st.markdown(
    """
<div style="text-align:center;">
    <h3>Connect with Me</h3>
    <a href="https://github.com/Anamikaa200" target="_blank">
        <img src="https://img.shields.io/badge/GitHub-Abhay--cody-black?style=for-the-badge&logo=github">
    </a>
    <br><br>
    <a href="https://www.linkedin.com/in/anamika-yadav-64b688340/" target="_blank">
        <img src="https://img.shields.io/badge/LinkedIn-Abhay%20Kumar%20Gupta-blue?style=for-the-badge&logo=linkedin">
    </a>
</div>
""",
    unsafe_allow_html=True,
)
