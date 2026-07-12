import os
import streamlit as st
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)
import torch

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="KGF Chapter 2 Review Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

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

background:linear-gradient(
135deg,
#090909,
#161616,
#242424
);

color:white;

}

.block-container{

padding-top:2rem;
max-width:1400px;

}

.hero{

background:rgba(255,255,255,.05);

backdrop-filter:blur(18px);

padding:30px;

border-radius:20px;

border:1px solid rgba(255,255,255,.08);

margin-top:20px;

margin-bottom:25px;

text-align:center;

}

.hero h1{

font-size:48px;

font-weight:800;

color:#FFD700;

}

.hero p{

font-size:18px;

color:#DDDDDD;

}

.glass{

background:rgba(255,255,255,.05);

backdrop-filter:blur(16px);

padding:20px;

border-radius:18px;

border:1px solid rgba(255,255,255,.08);

margin-bottom:18px;

}

[data-testid="stSidebar"]{

background:#111111;

}

[data-testid="stSidebar"] *{

color:white !important;

}

.stButton>button{

width:100%;

background:linear-gradient(
90deg,
#FFD700,
#F59E0B
);

color:black;

font-weight:bold;

border:none;

border-radius:12px;

padding:12px;

}

</style>
""", unsafe_allow_html=True)

# ============================================================
# BANNER IMAGE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_PATH = os.path.join(BASE_DIR, "KGF_2.jpg")

if os.path.exists(IMAGE_PATH):

    st.image(
        IMAGE_PATH,
        use_container_width=True,
    )

else:

    st.warning(
        "Banner image 'KGF_2.jpg' not found."
    )

# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

<h1>
🎬 KGF Chapter 2 Review Analysis using LLMs
</h1>

<p>

Analyze movie reviews using
Hugging Face Transformers for
Sentiment Analysis and
Question Answering.

</p>

</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎬 Project Dashboard")

    st.markdown("---")

    st.subheader("📌 Project")

    st.info("""
Movie : **KGF Chapter 2**

This application demonstrates
Natural Language Processing using
Large Language Models.
""")

    st.markdown("---")

    st.subheader("✨ Features")

    st.success("😊 Sentiment Analysis")

    st.success("❓ Question Answering")

    st.success("🤖 Hugging Face Transformers")

    st.success("📊 Interactive Dashboard")

    st.markdown("---")

    st.subheader("🧠 Models")

    st.write("**Sentiment Model**")
    st.caption("DistilBERT SST-2")

    st.write("**QA Model**")
    st.caption("MiniLM SQuAD2")

    st.markdown("---")

    st.subheader("⚙ Technologies")

    st.write("• Python")
    st.write("• Streamlit")
    st.write("• Transformers")
    st.write("• PyTorch")

st.divider()

# ============================================================
# LOAD HUGGING FACE MODELS
# ============================================================

@st.cache_resource
def load_sentiment_model():

    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    return classifier


@st.cache_resource
def load_qa_model():

    tokenizer = AutoTokenizer.from_pretrained(
        "deepset/minilm-uncased-squad2"
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        "deepset/minilm-uncased-squad2"
    )

    return tokenizer, model


# ============================================================
# LOAD MODELS
# ============================================================

with st.spinner("🔄 Loading AI Models..."):

    classifier = load_sentiment_model()

    tokenizer, qa_model = load_qa_model()

st.success("✅ Models Loaded Successfully")

# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2 = st.tabs([
    "😊 Sentiment Analysis",
    "❓ Question Answering"
])

# ============================================================
# SENTIMENT ANALYSIS PAGE
# ============================================================

with tab1:

    st.markdown("""
<div class="glass">

<h2 style="text-align:center;">
😊 Movie Review Sentiment Analysis
</h2>

<p style="text-align:center;color:#CCCCCC;">
Analyze whether a KGF Chapter 2 review is
Positive or Negative.
</p>

</div>
""", unsafe_allow_html=True)

    review = st.text_area(
        "Movie Review",
        height=220,
        placeholder="""
Example:

KGF Chapter 2 is one of the best action movies ever made.
Yash delivered an outstanding performance and the background
music was phenomenal.
"""
    )

    analyze_btn = st.button(
        "🚀 Analyze Sentiment",
        use_container_width=True
    )

# ============================================================
# QUESTION ANSWERING PAGE
# ============================================================

with tab2:

    st.markdown("""
<div class="glass">

<h2 style="text-align:center;">
❓ Question Answering
</h2>

<p style="text-align:center;color:#CCCCCC;">
Paste any movie review and ask questions about it.
</p>

</div>
""", unsafe_allow_html=True)

    context = st.text_area(
        "Movie Review",
        height=220,
        placeholder="""
Paste a complete KGF Chapter 2 review here...
"""
    )

    question = st.text_input(
        "Question",
        placeholder="Example: What did the reviewer like?"
    )

    answer_btn = st.button(
        "🤖 Get Answer",
        use_container_width=True
    )

st.divider()

# ============================================================
# SENTIMENT ANALYSIS LOGIC
# ============================================================

if analyze_btn:

    if review.strip() == "":

        st.warning("⚠ Please enter a movie review.")

    else:

        with st.spinner("🤖 AI is analyzing the review..."):

            result = classifier(review)[0]

        label = result["label"]
        score = result["score"]

        if label == "POSITIVE":

            sentiment = "😊 POSITIVE"
            color = "#16A34A"
            emoji = "😄"

        else:

            sentiment = "😞 NEGATIVE"
            color = "#DC2626"
            emoji = "😔"

        st.markdown(f"""
<div style="
background:linear-gradient(135deg,{color},#111827);
padding:30px;
border-radius:20px;
text-align:center;
color:white;
box-shadow:0px 10px 30px rgba(0,0,0,.35);
">

<h2>Prediction Result</h2>

<h1 style="font-size:60px;">
{emoji}
</h1>

<h2>{sentiment}</h2>

<h1>{score*100:.2f}%</h1>

</div>
""", unsafe_allow_html=True)

        st.write("")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Sentiment",
                sentiment
            )

        with col2:

            st.metric(
                "Confidence",
                f"{score*100:.2f}%"
            )

        st.write("")

        st.subheader("📊 Confidence Score")

        st.progress(float(score))

        st.write("")

        if score >= 0.95:

            st.success(
                "🎯 The model is highly confident about this prediction."
            )

        elif score >= 0.80:

            st.info(
                "👍 Good confidence prediction."
            )

        else:

            st.warning(
                "⚠ The confidence is moderate. A longer review may improve prediction."
            )

        st.markdown("---")

        st.subheader("📝 Review")

        st.info(review)

st.divider()

# ============================================================
# QUESTION ANSWERING LOGIC
# ============================================================

if answer_btn:

    if context.strip() == "" or question.strip() == "":

        st.warning("⚠ Please enter both the review and your question.")

    else:

        with st.spinner("🤖 AI is finding the answer..."):

            inputs = tokenizer(
                question,
                context,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            with torch.no_grad():

                outputs = qa_model(**inputs)

            start = torch.argmax(outputs.start_logits)
            end = torch.argmax(outputs.end_logits) + 1

            answer = tokenizer.decode(
                inputs["input_ids"][0][start:end],
                skip_special_tokens=True
            )

        st.markdown("""
<div class="glass">

<h2 style="text-align:center;">
💡 Answer
</h2>

</div>
""", unsafe_allow_html=True)

        if answer.strip():

            st.success(answer)

        else:

            st.warning("The model could not find a suitable answer.")

        st.write("")

        with st.expander("📄 Review Used"):

            st.write(context)

# ============================================================
# PROJECT DASHBOARD
# ============================================================

st.markdown("""
<div class="glass">

<h2 style="text-align:center;">
📊 Project Dashboard
</h2>

</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Sentiment Model", "DistilBERT")

with c2:
    st.metric("QA Model", "MiniLM")

with c3:
    st.metric("Framework", "Transformers")

st.write("")

# ============================================================
# TECHNOLOGIES & FEATURES
# ============================================================

left, right = st.columns(2)

with left:

    st.markdown("""
<div class="glass">

### 🛠 Technologies Used

- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- DistilBERT
- MiniLM
- NLP

</div>
""", unsafe_allow_html=True)

with right:

    st.markdown("""
<div class="glass">

### ⭐ Features

- Movie Review Sentiment Analysis
- Question Answering
- Confidence Score
- Interactive Dashboard
- Modern UI
- Fast Inference
- Hugging Face Models

</div>
""", unsafe_allow_html=True)

# ============================================================
# ABOUT PROJECT
# ============================================================

with st.expander("ℹ About This Project"):

    st.markdown("""

### 🎬 KGF Chapter 2 Review Analysis using LLMs

This project demonstrates the use of **Natural Language Processing (NLP)** and **Large Language Models (LLMs)** to analyze movie reviews.

### Workflow

1. User enters a movie review.
2. DistilBERT predicts whether the review is Positive or Negative.
3. MiniLM answers questions based on the review.
4. Results are displayed with confidence scores.

### Models Used

- DistilBERT SST-2
- MiniLM SQuAD2

""")

# ============================================================
# DEVELOPER
# ============================================================

st.write("")

st.markdown("""
<div class="glass" style="text-align:center;">

<h2>👩‍💻 Developer</h2>

<h3>Anamika Yadav</h3>

<p>
Artificial Intelligence • Machine Learning • NLP
</p>

</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    st.link_button(
        "📂 GitHub Profile",
        "https://github.com/Anamikaa200",
        use_container_width=True
    )

with col2:

    st.link_button(
        "💼 LinkedIn Profile",
        "https://www.linkedin.com/in/anamika-yadav-64b688340",
        use_container_width=True
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
<div style="
text-align:center;
padding:20px;
color:#CCCCCC;
">

🎬 <b>KGF Chapter 2 Review Analysis using LLMs</b>

<br><br>

Made with ❤️ using

<b>Streamlit • Hugging Face Transformers • PyTorch</b>

<br><br>

© 2026 Anamika Yadav

</div>
""",
    unsafe_allow_html=True,
)
