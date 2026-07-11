import os
import requests
import streamlit as st
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Health Insurance RAG Chatbot",
    page_icon="🏥",
    layout="wide",
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("👨‍💻 Developer")

st.sidebar.success("Abhay Kumar Gupta")

st.sidebar.markdown("---")

st.sidebar.markdown("### 🔗 Connect with Me")

st.sidebar.markdown(
    """
📂 **GitHub Repository**

https://github.com/Abhay-cody/AI_ML_PROJECTS/tree/main/Project_10_RAG_Chatbots_for_Health_Insurance
"""
)

st.sidebar.markdown(
    """
💼 **LinkedIn Profile**

https://www.linkedin.com/in/abhay-kumar-gupta-104a18397
"""
)

st.sidebar.markdown("---")

try:
    st.sidebar.link_button(
        "📂 Open GitHub Repository",
        "https://github.com/Abhay-cody/AI_ML_PROJECTS/tree/main/Project_10_RAG_Chatbots_for_Health_Insurance",
    )

    st.sidebar.link_button(
        "💼 Visit LinkedIn",
        "https://www.linkedin.com/in/abhay-kumar-gupta-104a18397",
    )

except Exception:
    st.sidebar.markdown(
        "[📂 Open GitHub Repository](https://github.com/Abhay-cody/AI_ML_PROJECTS/tree/main/Project_10_RAG_Chatbots_for_Health_Insurance)"
    )

    st.sidebar.markdown(
        "[💼 Visit LinkedIn](https://www.linkedin.com/in/abhay-kumar-gupta-104a18397)"
    )

st.sidebar.markdown("---")

st.sidebar.info(
    "This chatbot answers questions related to Health Insurance Policies using Google's Gemini model."
)

# =====================================================
# MAIN PAGE
# =====================================================

st.title("🏥 Health Insurance RAG Chatbot")

st.markdown(
    """
Ask questions about **Health Insurance Policies**.

The chatbot retrieves information from the Health Insurance webpage and answers using **Google Gemini**.
"""
)

st.markdown("---")

# =====================================================
# API KEY
# =====================================================

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input(
        "🔑 Enter Gemini API Key",
        type="password"
    )

if not api_key:
    st.warning("Please enter your Gemini API Key from the sidebar.")
    st.stop()

# =====================================================
# WEBSITE
# =====================================================

URL = "https://www.starhealth.in/health-insurance/types-of-health-insurance/"

# =====================================================
# LOAD WEBSITE
# =====================================================

@st.cache_data(show_spinner=True)
def load_website():

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        URL,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    text = soup.get_text(separator="\n")

    return text


# =====================================================
# VECTOR DATABASE
# =====================================================

@st.cache_resource(show_spinner=True)
def create_vector_db(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    documents = splitter.create_documents([text])

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )

    db = FAISS.from_documents(
        documents,
        embeddings,
    )

    return db


# =====================================================
# BUILD DATABASE
# =====================================================

try:

    website_text = load_website()

    vector_db = create_vector_db(
        website_text
    )

except Exception as e:

    st.error("Unable to load website.")

    st.exception(e)

    st.stop()

# =====================================================
# QUESTION
# =====================================================

question = st.text_input(
    "💬 Ask your question"
)

if question:

    with st.spinner("Searching..."):

        docs = vector_db.similarity_search(
            question,
            k=4,
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are an AI assistant specializing in Health Insurance.

Answer ONLY using the context below.

If the answer is unavailable in the context, reply:

"I couldn't find that information in the provided document."

Context:

{context}

Question:

{question}
"""

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.2,
        )

        response = llm.invoke(prompt)

        st.markdown("## ✅ Answer")

        st.write(response.content)

st.markdown("---")

st.caption(
    "Developed by Abhay Kumar Gupta | Health Insurance RAG Chatbot"
)
