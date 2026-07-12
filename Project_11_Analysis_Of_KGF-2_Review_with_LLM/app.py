import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch

# -------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="KGF Chapter 2 Review Analyzer",
    page_icon="🎬",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown("""
<style>
.main-title{
    font-size:40px;
    color:#FFD700;
    text-align:center;
    font-weight:bold;
}
.sub-title{
    text-align:center;
    color:white;
}
.result{
    padding:15px;
    border-radius:10px;
    font-size:20px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("🎬 Project Details")

st.sidebar.info("""
### NLP Project

**Movie:** KGF Chapter 2

### Features
✔ Sentiment Analysis

✔ Question Answering

### Models

Sentiment:
DistilBERT SST-2

Question Answering:
MiniLM SQuAD2
""")

# -------------------------------------------------
# Title
# -------------------------------------------------
st.markdown(
    "<h1 class='main-title'>KGF Chapter 2 Review Analysis using LLMs</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='sub-title'>Analyze Movie Reviews with Hugging Face Transformers</p>",
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# Load Models
# -------------------------------------------------
@st.cache_resource
def load_sentiment():

    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

@st.cache_resource
def load_qa():

    tokenizer = AutoTokenizer.from_pretrained(
        "deepset/minilm-uncased-squad2"
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        "deepset/minilm-uncased-squad2"
    )

    return tokenizer, model


classifier = load_sentiment()
tokenizer, qa_model = load_qa()

# -------------------------------------------------
# Tabs
# -------------------------------------------------
tab1, tab2 = st.tabs([
    "😊 Sentiment Analysis",
    "❓ Question Answering"
])

# =================================================
# SENTIMENT ANALYSIS
# =================================================
with tab1:

    st.header("Movie Review Sentiment")

    review = st.text_area(
        "Enter a KGF Chapter 2 Review",
        height=180,
        placeholder="Example: KGF Chapter 2 is an amazing movie..."
    )

    if st.button("Analyze Sentiment"):

        if review.strip() == "":
            st.warning("Please enter a review.")
        else:

            result = classifier(review)[0]

            label = result["label"]
            score = result["score"]

            if label == "POSITIVE":

                st.success("😊 POSITIVE REVIEW")

            else:

                st.error("😞 NEGATIVE REVIEW")

            st.metric(
                "Confidence Score",
                f"{score*100:.2f}%"
            )

            st.progress(float(score))

# =================================================
# QUESTION ANSWERING
# =================================================
with tab2:

    st.header("Ask Questions About a Review")

    context = st.text_area(
        "Movie Review",
        height=180,
        placeholder="Paste any movie review here..."
    )

    question = st.text_input(
        "Ask a Question",
        placeholder="Example: What did the reviewer like?"
    )

    if st.button("Get Answer"):

        if context == "" or question == "":

            st.warning("Please enter both review and question.")

        else:

            inputs = tokenizer(
                question,
                context,
                return_tensors="pt",
                truncation=True
            )

            with torch.no_grad():

                outputs = qa_model(**inputs)

            start = torch.argmax(outputs.start_logits)
            end = torch.argmax(outputs.end_logits) + 1

            answer = tokenizer.decode(
                inputs["input_ids"][0][start:end],
                skip_special_tokens=True
            )

            st.success("Answer")

            st.write(answer)

st.divider()

st.markdown(
"""
<center>

Made with ❤️ using Streamlit & Hugging Face Transformers

</center>
""",
unsafe_allow_html=True
)