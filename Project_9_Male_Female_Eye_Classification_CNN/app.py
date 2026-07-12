import os
import numpy as np
import streamlit as st
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Male vs Female Eye Classification",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.stApp{
background-color:#F4F8FB;
}

/* Header */

.main-header{

background:linear-gradient(90deg,#0F62FE,#2563EB,#4F46E5);

padding:25px;

border-radius:18px;

text-align:center;

box-shadow:0px 8px 18px rgba(0,0,0,0.25);

margin-bottom:30px;

}

.main-header h1{

color:white;

font-size:42px;

font-weight:800;

margin-bottom:8px;

}

.main-header p{

color:white;

font-size:18px;

}

/* Cards */

.card{

background:white;

padding:20px;

border-radius:20px;

box-shadow:0 10px 25px rgba(0,0,0,.08);

margin-bottom:20px;

}

/* Prediction Card */

.prediction{

background:linear-gradient(135deg,#16A34A,#059669);

color:white;

padding:25px;

border-radius:20px;

text-align:center;

font-size:24px;

font-weight:bold;

}

/* Footer */

.footer{

text-align:center;

padding:25px;

font-size:18px;

}

/* Uploader */

.stFileUploader{

border:2px dashed #2563EB;

padding:18px;

border-radius:15px;

}

/* Sidebar */

[data-testid="stSidebar"]{

background:#0F172A;

}

[data-testid="stSidebar"] *{

color:white;

}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
<h1>👁 Male vs Female Eye Classification</h1>
<p>Deep Learning based Eye Gender Prediction using CNN</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("👨‍💻 Project Information")

    st.markdown("---")

    st.markdown("""
### 📌 About Project

This project uses a **Convolutional Neural Network (CNN)** to classify eye images into:

- 👨 Male
- 👩 Female

The model is automatically trained if no saved model is found.
""")

    st.markdown("---")

    st.metric("Image Size", "64 × 64")
    st.metric("Model", "CNN")
    st.metric("Framework", "TensorFlow")

    st.markdown("---")

    st.info("💡 Upload a clear eye image for the best prediction.")

# ============================================================
# Constants
# ============================================================

IMG_SIZE = (64, 64)

MODEL_PATH = "model.keras"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "train")

# ============================================================
# Load or Train Model
# ============================================================

@st.cache_resource
def get_model():

    # --------------------------------------------------------
    # Load existing model
    # --------------------------------------------------------

    if os.path.exists(MODEL_PATH):

        model = load_model(MODEL_PATH)

        return model, None

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not os.path.exists(TRAIN_DIR):

        st.error("❌ train folder not found.")

        st.stop()

    # --------------------------------------------------------
    # Prepare Dataset
    # --------------------------------------------------------

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.20
    )

    train_gen = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode="binary",
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode="binary",
        subset="validation"
    )

    # --------------------------------------------------------
    # CNN Model
    # --------------------------------------------------------

    model = Sequential([

        Conv2D(
            32,
            (3,3),
            activation="relu",
            input_shape=(64,64,3)
        ),

        MaxPooling2D(),

        Conv2D(
            64,
            (3,3),
            activation="relu"
        ),

        MaxPooling2D(),

        Conv2D(
            128,
            (3,3),
            activation="relu"
        ),

        MaxPooling2D(),

        Flatten(),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(0.5),

        Dense(
            1,
            activation="sigmoid"
        )

    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # --------------------------------------------------------
    # Train Model
    # --------------------------------------------------------

    with st.spinner("🔄 Training CNN Model... Please wait."):

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=5,
            verbose=1
        )

    # --------------------------------------------------------
    # Save Model
    # --------------------------------------------------------

    model.save(MODEL_PATH)

    # --------------------------------------------------------
    # Display Accuracy
    # --------------------------------------------------------

    st.success("✅ Model trained successfully!")

    st.subheader("📈 Training Accuracy")

    fig, ax = plt.subplots(figsize=(7,4))

    ax.plot(
        history.history["accuracy"],
        linewidth=3,
        label="Training Accuracy"
    )

    ax.plot(
        history.history["val_accuracy"],
        linewidth=3,
        label="Validation Accuracy"
    )

    ax.set_xlabel("Epoch")

    ax.set_ylabel("Accuracy")

    ax.legend()

    st.pyplot(fig)

    return model, train_gen.class_indices

# ============================================================
# Load Model
# ============================================================

model, class_idx = get_model()

if class_idx is None:

    class_idx = {
        "female":0,
        "male":1
    }
    
# ============================================================
# Upload Section
# ============================================================

st.markdown("""
<div class="card">
    <h2 style="text-align:center;color:#2563EB;">
        📤 Upload Eye Image
    </h2>
    <p style="text-align:center;color:gray;">
        Upload a JPG, JPEG or PNG image of an eye for prediction.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# Prediction
# ============================================================

if uploaded is not None:

    col1, col2 = st.columns([1, 1])

    # --------------------------------------------------------
    # Left Column : Image
    # --------------------------------------------------------

    with col1:

        st.markdown("""
        <div class="card">
            <h3 style="text-align:center;">
                🖼 Uploaded Image
            </h3>
        """, unsafe_allow_html=True)

        img = image.load_img(
            uploaded,
            target_size=IMG_SIZE
        )

        arr = image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        st.image(
            uploaded,
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Right Column : Prediction
    # --------------------------------------------------------

    with col2:

        with st.spinner("🔍 Analyzing Image..."):

            prediction = float(
                model.predict(arr, verbose=0)[0][0]
            )

        if prediction >= 0.5:

            label = "👨 Male"
            confidence = prediction * 100

            result_color = "#059669"

        else:

            label = "👩 Female"
            confidence = (1 - prediction) * 100

            result_color = "#EC4899"

        st.markdown(f"""
        <div style="
            background:{result_color};
            color:white;
            padding:25px;
            border-radius:18px;
            text-align:center;
            box-shadow:0px 8px 18px rgba(0,0,0,0.2);
        ">
            <h2>Prediction Result</h2>
            <h1>{label}</h1>
            <h3>Confidence: {confidence:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.subheader("📊 Confidence Score")

        st.progress(min(int(confidence), 100))

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

        st.success(f"Prediction: {label}")

        st.info(
            "The confidence score indicates how certain the model is about its prediction."
        )

# ============================================================
# No Image Uploaded
# ============================================================

else:

    st.markdown("""
    <div class="card" style="text-align:center;">
        <h3>📁 No Image Uploaded</h3>
        <p>Please upload an eye image to begin the prediction.</p>
    </div>
    """, unsafe_allow_html=True)
    
# ============================================================
# Footer
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
<style>

.footer-card{
    background: linear-gradient(135deg,#0F172A,#1E293B);
    color:white;
    padding:30px;
    border-radius:20px;
    text-align:center;
    box-shadow:0px 10px 25px rgba(0,0,0,0.25);
}

.footer-card h2{
    color:#60A5FA;
    margin-bottom:10px;
}

.footer-card p{
    font-size:17px;
    margin:8px;
}

.tech-box{
    background:#EEF4FF;
    padding:20px;
    border-radius:18px;
    box-shadow:0px 5px 15px rgba(0,0,0,0.08);
    margin-top:15px;
}

</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# ============================================================
# Technologies Used
# ============================================================

with col1:

    st.markdown("""
    <div class="tech-box">

    <h2 style="color:#2563EB;">
    🛠 Technologies Used
    </h2>

    ✅ Python

    ✅ TensorFlow / Keras

    ✅ CNN (Convolutional Neural Network)

    ✅ NumPy

    ✅ Matplotlib

    ✅ Streamlit

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# Project Features
# ============================================================

with col2:

    st.markdown("""
    <div class="tech-box">

    <h2 style="color:#2563EB;">
    ⭐ Features
    </h2>

    ✔ Automatic CNN Model Training

    ✔ Image Upload

    ✔ Eye Classification

    ✔ Confidence Score

    ✔ Interactive Dashboard

    ✔ Responsive UI

    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# Developer Section
# ============================================================

st.markdown("""
<div class="footer-card">

<h2>👨‍💻 Connect with Me</h2>

<p>
🐱 <b>GitHub</b><br>
<a href="https://github.com/Anamikaa200" target="_blank" style="color:#60A5FA;text-decoration:none;">
https://github.com/Anamikaa200
</a>
</p>

<p>
💼 <b>LinkedIn</b><br>
<a href="https://www.linkedin.com/in/anamika-yadav-64b688340"
target="_blank"
style="color:#60A5FA;text-decoration:none;">
www.linkedin.com/in/anamika-yadav-64b688340
</a>
</p>

<p style="margin-top:20px;">
Made with ❤️ using Streamlit & TensorFlow
</p>

</div>
""", unsafe_allow_html=True)

# ============================================================
# Copyright
# ============================================================

st.markdown(
    """
    <div style="text-align:center;
                color:gray;
                font-size:14px;
                margin-top:20px;">
        © 2026 Male vs Female Eye Classification • All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True,
)
