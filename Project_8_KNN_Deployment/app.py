import streamlit as st
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🌸 Iris Flower Species Prediction",
    page_icon="🌸",
    layout="centered"
)

# -----------------------------
# Train Model
# -----------------------------
@st.cache_resource
def train_model():
    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=1
    )

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, iris.target_names, accuracy


model, species, accuracy = train_model()

# -----------------------------
# Title
# -----------------------------
st.title("🌸 Iris Flower Species Prediction")
st.write("Predict the species of an Iris flower using the K-Nearest Neighbors (KNN) algorithm.")

st.success(f"Model Accuracy: **{accuracy*100:.2f}%**")

st.divider()

# -----------------------------
# Input
# -----------------------------
st.subheader("Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=4.0,
        max_value=8.0,
        value=5.1,
        step=0.1,
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=2.0,
        max_value=5.0,
        value=3.5,
        step=0.1,
    )

with col2:
    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=1.0,
        max_value=7.0,
        value=1.4,
        step=0.1,
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.1,
        max_value=3.0,
        value=0.2,
        step=0.1,
    )

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Species", use_container_width=True):

    sample = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    st.success(f"### 🌼 Predicted Species: **{species[prediction].title()}**")

    st.subheader("Prediction Probability")

    for i, name in enumerate(species):
        st.progress(float(probability[i]))
        st.write(f"**{name.title()} : {probability[i]*100:.2f}%**")

# -----------------------------
# About
# -----------------------------
st.divider()

st.subheader("About this Project")

st.write("""
- **Algorithm:** K-Nearest Neighbors (KNN)
- **Dataset:** Iris Dataset (Scikit-learn)
- **Features Used:**
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
""")

st.divider()

st.markdown("### 👨‍💻 Developed by Anamika Yadav")

st.markdown(
"""
- 🔗 **GitHub:** https://github.com/Anamikaa200
- 💼 **LinkedIn:** www.linkedin.com/in/anamika-yadav-64b688340
"""
)
