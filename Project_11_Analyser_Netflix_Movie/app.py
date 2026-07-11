import streamlit as st
import pandas as pd
from transformers import pipeline

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Netflix Review Analyzer",
    page_icon="🎬",
    layout="wide"
)

# ----------------------------------
# Load Model
# ----------------------------------
@st.cache_resource
def load_model():
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    return classifier

classifier = load_model()

# ----------------------------------
# Load Dataset
# ----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("netflix movie dhurandhar 2.csv", sep=";")

df = load_data()

# ----------------------------------
# Title
# ----------------------------------
st.title("🎬 Netflix Movie Review Analyzer")
st.write(
    """
Analyze the sentiment of movie reviews using a Hugging Face Transformer model.
The app predicts whether a review is **Positive** or **Negative**.
"""
)

# ----------------------------------
# Sidebar
# ----------------------------------
st.sidebar.header("Dataset Information")

st.sidebar.write(f"Total Reviews : **{len(df)}**")

positive = len(df[df["Class"] == "POSITIVE"])
negative = len(df[df["Class"] == "NEGATIVE"])

st.sidebar.write(f"Positive Reviews : **{positive}**")
st.sidebar.write(f"Negative Reviews : **{negative}**")

# ----------------------------------
# Tabs
# ----------------------------------
tab1, tab2 = st.tabs(["Review Analyzer", "Dataset"])

# ----------------------------------
# Review Prediction
# ----------------------------------
with tab1:

    st.subheader("Enter Your Review")

    review = st.text_area(
        "Movie Review",
        height=180,
        placeholder="Type your movie review here..."
    )

    if st.button("Analyze Review"):

        if review.strip() == "":
            st.warning("Please enter a review.")
        else:

            prediction = classifier(review)[0]

            label = prediction["label"]
            score = prediction["score"]

            if label == "POSITIVE":
                st.success("😊 Positive Review")
            else:
                st.error("😞 Negative Review")

            st.metric("Confidence", f"{score*100:.2f}%")

# ----------------------------------
# Dataset Viewer
# ----------------------------------
with tab2:

    st.subheader("Dataset Preview")

    st.dataframe(df, use_container_width=True)

    st.subheader("Sentiment Distribution")

    chart = df["Class"].value_counts()

    st.bar_chart(chart)

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Hugging Face Transformers")
