import os
import requests
import streamlit as st
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Health Insurance RAG Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title{
    font-size:40px;
    font-weight:bold;
    color:#2E8B57;
}

.subtitle{
    font-size:18px;
    color:gray;
}

.stButton>button{
    width:100%;
    border-radius:8px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("👨‍💻 Developer")

    st.success("Anamika Yadav")

    st.markdown("---")

    st.markdown("### 📂 Project Links")

    github_url = "https://github.com/Anamikaa200"

    linkedin_url = "www.linkedin.com/in/anamika-yadav-64b688340"

    try:
        st.link_button("📂 GitHub Profile", github_url)
        st.link_button("💼 LinkedIn Profile", linkedin_url)
    except:
        st.markdown(f"[GitHub Profile]({github_url})")
        st.markdown(f"[LinkedIn Profile]({linkedin_url})")

    st.markdown("---")

    st.info(
        """
This chatbot retrieves information from a Health Insurance website
and answers your questions using OpenAI GPT.
"""
    )

# ============================================================
# MAIN PAGE
# ============================================================

st.title("Health Insurance RAG ChatBot")  

st.markdown(
    '<p class="subtitle">Ask anything related to Health Insurance Policies.</p>',
    unsafe_allow_html=True,
)

st.markdown("---")

# ============================================================
# OPENAI API KEY
# ============================================================

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input(
        "🔑 Enter OpenAI API Key",
        type="password",
    )

if not api_key:
    st.warning("Please enter your OpenAI API Key.")
    st.stop()
# ============================================================
# WEBSITE URL
# ============================================================

URL = "https://www.starhealth.in/health-insurance/types-of-health-insurance/"

# ============================================================
# LOAD WEBSITE
# ============================================================

@st.cache_data(show_spinner="Loading Health Insurance website...")
def load_website():

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/138.0 Safari/537.36"
        )
    }

    response = requests.get(
        URL,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    # Remove unwanted tags
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    text = soup.get_text(separator="\n")

    # Clean text
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    cleaned_text = "\n".join(lines)

    return cleaned_text


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

@st.cache_resource(show_spinner="Creating Vector Database...")
def create_vector_db(text, api_key):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    documents = splitter.create_documents([text])

    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
)

    db = FAISS.from_documents(
        documents,
        embeddings,
    )

    return db


# ============================================================
# BUILD VECTOR DATABASE
# ============================================================

try:

    website_text = load_website()

    vector_db = create_vector_db(
        website_text,
        api_key,
    )

except Exception as e:

    st.error("❌ Unable to load the Health Insurance website.")

    st.exception(e)

    st.stop()


# ============================================================
# DISPLAY WEBSITE STATUS
# ============================================================

st.success("✅ Knowledge Base Loaded Successfully")

with st.expander("📄 View Website Information"):

    st.write(
        """
The chatbot is using information retrieved from the
Health Insurance webpage to answer your questions.
        """
    )

    st.caption(
        f"Loaded approximately **{len(website_text):,} characters** of text."
    )

st.markdown("---")

# ============================================================
# QUESTION ANSWERING
# ============================================================

st.subheader("💬 Ask Your Health Insurance Question")

question = st.text_input(
    "Enter your question here...",
    placeholder="Example: What are the different types of health insurance?"
)

if question:

    with st.spinner("🔍 Searching relevant information..."):

        try:

            docs = vector_db.similarity_search(
                question,
                k=4,
            )

            if not docs:
                st.warning("No relevant information was found.")
                st.stop()

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

        except Exception as e:

            st.error("Error while searching the knowledge base.")

            st.exception(e)

            st.stop()

    prompt = f"""
You are an expert AI assistant specializing in Health Insurance.

Your task is to answer ONLY from the provided context.

Rules:

1. Do NOT make up information.
2. If the answer is unavailable, reply exactly:

"I couldn't find that information in the provided document."

3. Keep answers clear and concise.
4. Use bullet points whenever appropriate.
5. Explain in simple English.

==========================
CONTEXT
==========================

{context}

==========================
QUESTION
==========================

{question}

==========================
ANSWER
==========================
"""

    with st.spinner("🤖 GPT is generating an answer..."):

        try:

            llm = ChatOpenAI(

                model="gpt-4.1-mini",

                api_key=api_key,

                temperature=0.2,

            )

            response = llm.invoke(prompt)

            answer = response.content.strip()

            st.markdown("## ✅ Answer")

            if answer:

                st.success(answer)

            else:

                st.warning("Gemini returned an empty response.")

        except Exception as e:

            st.error("Failed to generate an answer.")

            st.exception(e)

# ============================================================
# OPTIONAL CONTEXT VIEWER
# ============================================================

if question:

    with st.expander("📚 Retrieved Context"):

        st.write(context)

st.markdown("---")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
<div style='text-align:center;'>

### 🏥 Health Insurance RAG Chatbot

Developed with ❤️ using

**Streamlit • LangChain • FAISS • OpenAI GPT • BeautifulSoup**

</div>
""",
unsafe_allow_html=True,
)

st.markdown("---")

with st.expander("ℹ️ About this Project"):

    st.markdown(
        """
### Project Description

This application is a Retrieval-Augmented Generation (RAG) chatbot for answering
questions related to Health Insurance.

### Workflow

1. Scrape Health Insurance webpage
2. Extract webpage text
3. Split into chunks
4. Generate embeddings using Google Embedding Model
5. Store embeddings in FAISS
6. Retrieve relevant chunks
7. Generate answers using Gemini 2.5 Flash

### Technologies Used

- Python
- Streamlit
- LangChain
- FAISS
- Google Gemini
- BeautifulSoup
- Requests
"""
    )

st.markdown("---")

st.caption(
    "© 2026 Anamika Yadav | Health Insurance RAG Chatbot"
)
