import os
import requests
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="COVID-19 Detection",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 COVID-19 Detection from Chest X-Ray")
st.write("Upload a Chest X-Ray image to predict whether it is **COVID** or **NORMAL**.")

# ---------------------------------
# Hugging Face Model Details
# ---------------------------------
MODEL_URL = "https://huggingface.co/a1bhay23/Detection_of_Covid_19_from_Chest_X_ray/resolve/main/model.keras"
MODEL_PATH = "model.keras"


# ---------------------------------
# Download Model (Only Once)
# ---------------------------------
@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Downloading AI Model..."):

            response = requests.get(MODEL_URL)

            response.raise_for_status()

            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)

    model = tf.keras.models.load_model(MODEL_PATH)

    return model


model = load_model()

# ---------------------------------
# Upload Image
# ---------------------------------
uploaded_file = st.file_uploader(
    "Choose a Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    img = image.resize((299, 299))

    img = np.array(img)

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    with st.spinner("Analyzing..."):

        prediction = model.predict(img)

    probability = float(prediction[0][0])

    if probability >= 0.5:

        st.error("🦠 Prediction: COVID")

        st.progress(probability)

        st.write(f"Confidence : **{probability*100:.2f}%**")

    else:

        st.success("✅ Prediction: NORMAL")

        st.progress(1 - probability)

        st.write(f"Confidence : **{(1-probability)*100:.2f}%**")
