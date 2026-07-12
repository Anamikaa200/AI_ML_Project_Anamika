
# app.py
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Male vs Female Eye Classification", layout="wide")
st.title("👤 Male vs Female Eye Classification")

IMG_SIZE=(64,64)
MODEL_PATH="model.keras"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "train")


@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH), None

    if not os.path.exists(TRAIN_DIR):
        st.error("train/ folder not found.")
        st.stop()

    datagen=ImageDataGenerator(rescale=1./255,validation_split=0.2)

    train_gen=datagen.flow_from_directory(
        TRAIN_DIR,target_size=IMG_SIZE,batch_size=32,
        class_mode="binary",subset="training"
    )
    val_gen=datagen.flow_from_directory(
        TRAIN_DIR,target_size=IMG_SIZE,batch_size=32,
        class_mode="binary",subset="validation"
    )

    model=Sequential([
        Conv2D(32,(3,3),activation="relu",input_shape=(64,64,3)),
        MaxPooling2D(),
        Conv2D(64,(3,3),activation="relu"),
        MaxPooling2D(),
        Conv2D(128,(3,3),activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(128,activation="relu"),
        Dropout(0.5),
        Dense(1,activation="sigmoid")
    ])

    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

    history=model.fit(train_gen,validation_data=val_gen,epochs=5)

    model.save(MODEL_PATH)

    fig,ax=plt.subplots()
    ax.plot(history.history["accuracy"],label="Train")
    ax.plot(history.history["val_accuracy"],label="Validation")
    ax.set_title("Training Accuracy")
    ax.legend()
    st.pyplot(fig)

    return model, train_gen.class_indices

model,class_idx=get_model()

if class_idx is None:
    class_idx={"female":0,"male":1}

uploaded=st.file_uploader("Upload an image",type=["jpg","jpeg","png"])

if uploaded:
    img=image.load_img(uploaded,target_size=IMG_SIZE)
    arr=image.img_to_array(img)/255.0
    arr=np.expand_dims(arr,0)

    pred=float(model.predict(arr,verbose=0)[0][0])
    if pred>=0.5:
        label="Male"
        conf=pred*100
    else:
        label="Female"
        conf=(1-pred)*100

    st.image(uploaded,width=250)
    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {conf:.2f}%")

st.markdown("---")
st.markdown("### Connect with Me")
st.markdown("- GitHub: https://github.com/Abhay-cody")
st.markdown("- LinkedIn: https://www.linkedin.com/in/abhay-kumar-gupta-104a18397")