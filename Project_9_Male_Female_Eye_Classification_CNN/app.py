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

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Male vs Female Eye Classification",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# PREMIUM GLASSMORPHISM CSS
# ==========================================================

st.markdown("""
<style>

/* Hide Streamlit Branding */

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

/* ===========================
BACKGROUND
=========================== */

.stApp{

background:
linear-gradient(
135deg,
#07111F 0%,
#0F172A 35%,
#16213E 70%,
#1D3557 100%
);

background-attachment:fixed;

color:white;

}

/* Main Container */

.block-container{

padding-top:2rem;

padding-bottom:2rem;

max-width:1400px;

}

/* ===========================
HERO CARD
=========================== */

.hero{

background:rgba(255,255,255,.08);

backdrop-filter:blur(20px);

-webkit-backdrop-filter:blur(20px);

border:1px solid rgba(255,255,255,.15);

border-radius:25px;

padding:35px;

box-shadow:0px 10px 40px rgba(0,0,0,.35);

margin-bottom:30px;

text-align:center;

}

.hero h1{

font-size:48px;

font-weight:800;

color:white;

margin-bottom:8px;

}

.hero h3{

font-size:20px;

font-weight:400;

color:#CBD5E1;

}

/* ===========================
GLASS CARD
=========================== */

.glass{

background:rgba(255,255,255,.08);

backdrop-filter:blur(18px);

-webkit-backdrop-filter:blur(18px);

border:1px solid rgba(255,255,255,.12);

border-radius:22px;

padding:25px;

margin-bottom:20px;

box-shadow:0 8px 35px rgba(0,0,0,.28);

color:white;

}

/* ===========================
RESULT CARD
=========================== */

.result{

background:linear-gradient(
135deg,
#2563EB,
#4F46E5
);

border-radius:22px;

padding:30px;

color:white;

text-align:center;

box-shadow:0 10px 35px rgba(37,99,235,.35);

}

/* ===========================
SIDEBAR
=========================== */

[data-testid="stSidebar"]{

background:#08111D;

}

[data-testid="stSidebar"] *{

color:white !important;

}

/* ===========================
FILE UPLOADER
=========================== */

.stFileUploader{

background:rgba(255,255,255,.06);

padding:18px;

border-radius:18px;

border:2px dashed #60A5FA;

}

/* ===========================
BUTTON
=========================== */

.stButton>button{

background:linear-gradient(
90deg,
#2563EB,
#4F46E5
);

color:white;

border:none;

border-radius:15px;

padding:.7rem 2rem;

font-size:16px;

font-weight:700;

transition:.3s;

}

.stButton>button:hover{

transform:translateY(-3px);

}

/* ===========================
METRIC
=========================== */

[data-testid="metric-container"]{

background:rgba(255,255,255,.08);

backdrop-filter:blur(15px);

border-radius:18px;

padding:18px;

border:1px solid rgba(255,255,255,.12);

}

/* ===========================
TEXT
=========================== */

h1,h2,h3,h4,h5,h6{

color:white;

}

p{

color:#E5E7EB;

}

label{

color:white !important;

}

small{

color:#CBD5E1;

}

/* ===========================
SUCCESS
=========================== */

.stSuccess{

border-radius:15px;

}

.stInfo{

border-radius:15px;

}

.stWarning{

border-radius:15px;

}

.stError{

border-radius:15px;

}

/* ===========================
PROGRESS BAR
=========================== */

.stProgress > div > div{

background:#3B82F6;

}

/* ===========================
IMAGE
=========================== */

img{

border-radius:18px;

}

/* ===========================
HORIZONTAL LINE
=========================== */

hr{

border:1px solid rgba(255,255,255,.08);

}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HERO SECTION
# ==========================================================

st.markdown("""
<div class="hero">

<h1>
👁 Male vs Female Eye Classification
</h1>

<h3>
Deep Learning based Gender Classification using CNN
</h3>

</div>
""", unsafe_allow_html=True)

# ==========================================================
# PROFESSIONAL SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("# 🤖 AI Dashboard")

    st.markdown("---")

    st.markdown("""
### 📌 Project

This application uses a **Convolutional Neural Network (CNN)** to classify eye images as:

- 👨 Male
- 👩 Female

Upload an eye image and the trained model will predict its class with a confidence score.
""")

    st.markdown("---")

    st.subheader("⚙ Model Details")

    st.metric("Architecture", "CNN")
    st.metric("Image Size", "64 × 64")
    st.metric("Framework", "TensorFlow")
    st.metric("Input Channels", "RGB")

    st.markdown("---")

    st.subheader("📂 Dataset")

    st.info(
        """
Dataset Structure

train/

├── male/

└── female/
"""
    )

    st.markdown("---")

    st.subheader("💡 Tips")

    st.success(
        """
✔ Upload a clear eye image.

✔ JPG / JPEG / PNG supported.

✔ Better lighting improves prediction.
"""
    )

# ==========================================================
# CONSTANTS
# ==========================================================

IMG_SIZE = (64, 64)

MODEL_PATH = "model.keras"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "train")

# ==========================================================
# LOAD OR TRAIN MODEL
# ==========================================================

@st.cache_resource
def get_model():

    # ------------------------------------------------------
    # Load Existing Model
    # ------------------------------------------------------

    if os.path.exists(MODEL_PATH):

        model = load_model(MODEL_PATH)

        return model, None

    # ------------------------------------------------------
    # Check Dataset
    # ------------------------------------------------------

    if not os.path.exists(TRAIN_DIR):

        st.error("❌ train/ folder not found.")

        st.stop()

    # ------------------------------------------------------
    # Dataset Generator
    # ------------------------------------------------------

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

    # ======================================================
    # MODEL CARD
    # ======================================================

    st.markdown("""
<div class="glass">

<h2 style="text-align:center;">
🧠 Building CNN Model
</h2>

</div>
""", unsafe_allow_html=True)

    # ------------------------------------------------------
    # CNN MODEL
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # TRAIN MODEL
    # ------------------------------------------------------

    progress = st.progress(0)

    status = st.empty()

    status.info("🚀 Training model...")

    history = model.fit(

        train_gen,

        validation_data=val_gen,

        epochs=5,

        verbose=1

    )

    progress.progress(100)

    status.success("✅ Training Completed")

    # ------------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------------

    model.save(MODEL_PATH)

    # ======================================================
    # TRAINING GRAPH
    # ======================================================

    st.markdown("""
<div class="glass">

<h2 style="text-align:center;">
📈 Training Accuracy
</h2>

</div>
""", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9,4))

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

    ax.set_title("CNN Training Performance")

    ax.grid(alpha=0.3)

    ax.legend()

    st.pyplot(fig)

    return model, train_gen.class_indices

# ==========================================================
# LOAD MODEL
# ==========================================================

model, class_idx = get_model()

if class_idx is None:

    class_idx = {

        "female": 0,

        "male": 1

    }

# ==========================================================
# GLASS UPLOAD SECTION
# ==========================================================

st.markdown("""
<div class="glass">

<h2 style="text-align:center;">
📤 Upload Eye Image
</h2>

<p style="text-align:center;color:#CBD5E1;">
Supported Formats : JPG • JPEG • PNG
</p>

</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"]
)

# ==========================================================
# IMAGE PREDICTION
# ==========================================================

if uploaded is not None:

    left, right = st.columns([1, 1])

    # ======================================================
    # LEFT PANEL
    # ======================================================

    with left:

        st.markdown("""
        <div class="glass">
        <h2 style="text-align:center;">
        🖼 Uploaded Image
        </h2>
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

    # ======================================================
    # RIGHT PANEL
    # ======================================================

    with right:

        with st.spinner("🤖 AI is analyzing the image..."):

            prediction = float(
                model.predict(arr, verbose=0)[0][0]
            )

        if prediction >= 0.5:

            label = "👨 Male"

            confidence = prediction * 100

            color = "#2563EB"

            emoji = "👨"

        else:

            label = "👩 Female"

            confidence = (1 - prediction) * 100

            color = "#EC4899"

            emoji = "👩"

        st.markdown(f"""
        <div style="
            background:linear-gradient(135deg,{color},#4F46E5);
            padding:35px;
            border-radius:25px;
            text-align:center;
            color:white;
            box-shadow:0px 15px 35px rgba(0,0,0,.35);
        ">

        <h3>Prediction Result</h3>

        <h1 style="font-size:55px;">
        {emoji}
        </h1>

        <h1>{label}</h1>

        <h2>{confidence:.2f}% Confidence</h2>

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.subheader("📊 Confidence Score")

        st.progress(int(confidence))

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.success(f"Prediction : {label}")

        if confidence >= 95:

            st.success(
                "🎯 Excellent confidence. The model is highly certain."
            )

        elif confidence >= 80:

            st.info(
                "👍 Good confidence prediction."
            )

        else:

            st.warning(
                "⚠ Confidence is relatively low. Try another clearer image."
            )

# ==========================================================
# NO IMAGE
# ==========================================================

else:

    st.markdown("""

<div class="glass">

<h2 style="text-align:center;">
📂 Waiting for Image...
</h2>

<p style="text-align:center;color:#CBD5E1;">

Upload an eye image to begin prediction.

</p>

</div>

""", unsafe_allow_html=True)

# ==========================================================
# PROJECT SUMMARY
# ==========================================================

st.write("")

st.markdown("""
<div class="glass">

<h2 style="text-align:center;">
📊 Project Highlights
</h2>

</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Model", "CNN")

with c2:
    st.metric("Classes", "2")

with c3:
    st.metric("Input Size", "64×64")

# ==========================================================
# TECHNOLOGIES & FEATURES
# ==========================================================

st.write("")
st.markdown("""
<div class="glass">
<h2 style="text-align:center;">
🚀 Technologies & Features
</h2>
</div>
""", unsafe_allow_html=True)

tech_col, feature_col = st.columns(2)

with tech_col:

    st.markdown("""
<div class="glass">

<h3 style="text-align:center;">
🛠 Technologies Used
</h3>

✅ Python

✅ TensorFlow

✅ Keras

✅ Convolutional Neural Network (CNN)

✅ NumPy

✅ Matplotlib

✅ Streamlit

</div>
""", unsafe_allow_html=True)

with feature_col:

    st.markdown("""
<div class="glass">

<h3 style="text-align:center;">
⭐ Features
</h3>

✔ Eye Image Classification

✔ Automatic Model Loading

✔ CNN Training

✔ Real-Time Prediction

✔ Confidence Score

✔ Responsive Dashboard

✔ Modern Glassmorphism UI

</div>
""", unsafe_allow_html=True)

# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.write("")

st.markdown("""
<div class="glass">

<h2 style="text-align:center;">
📖 About This Model
</h2>

<p style="text-align:center; color:#E2E8F0; font-size:17px;">

This application uses a Convolutional Neural Network (CNN) trained on
male and female eye images.

The uploaded image is resized to <b>64 × 64</b> pixels,
normalized, and passed through the CNN to predict the gender
based on the eye image.

</p>

</div>
""", unsafe_allow_html=True)

# ==========================================================
# DEVELOPER SECTION
# ==========================================================

st.write("")

st.markdown("""
<div style="
background:linear-gradient(135deg,#2563EB,#4F46E5);
padding:35px;
border-radius:25px;
text-align:center;
box-shadow:0px 15px 40px rgba(0,0,0,.35);
">

<h1 style="color:white;">
👨‍💻 Developer
</h1>

<h2 style="color:white;">
Anamika Yadav
</h2>

<p style="color:#E5E7EB;font-size:18px;">
AI • Machine Learning • Deep Learning
</p>

</div>
""", unsafe_allow_html=True)

# ==========================================================
# SOCIAL LINKS
# ==========================================================

st.write("")

social1, social2 = st.columns(2)

with social1:

    st.markdown("""
<div class="glass" style="text-align:center;">

<h3>
🐱 GitHub
</h3>

<a href="https://github.com/Anamikaa200"
target="_blank"
style="
color:#60A5FA;
font-size:18px;
text-decoration:none;
">
github.com/Anamikaa200
</a>

</div>
""", unsafe_allow_html=True)

with social2:

    st.markdown("""
<div class="glass" style="text-align:center;">

<h3>
💼 LinkedIn
</h3>

<a href="https://www.linkedin.com/in/anamika-yadav-64b688340"
target="_blank"
style="
color:#60A5FA;
font-size:18px;
text-decoration:none;
">
linkedin.com/in/anamika-yadav-64b688340
</a>

</div>
""", unsafe_allow_html=True)

# ==========================================================
# FOOTER
# ==========================================================

st.write("")
st.divider()

st.markdown(
    """
<div style="
text-align:center;
padding:20px;
color:#94A3B8;
font-size:15px;
">

Made with ❤️ using Streamlit & TensorFlow

<br><br>

© 2026 Male vs Female Eye Classification

</div>
""",
    unsafe_allow_html=True,
)
