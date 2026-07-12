import os
import requests
import streamlit as st

from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI,
)

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
#07111F,
#132238,
#1E3A5F
);

color:white;

}

.block-container{

max-width:1400px;

padding-top:2rem;

}

/* ========================= */

.hero{

background:rgba(255,255,255,.08);

backdrop-filter:blur(20px);

border-radius:25px;

padding:35px;

border:1px solid rgba(255,255,255,.12);

text-align:center;

margin-bottom:25px;

box-shadow:0px 8px 30px rgba(0,0,0,.25);

}

.hero h1{

font-size:46px;

font-weight:800;

color:white;

}

.hero p{

font-size:18px;

color:#CBD5E1;

}

/* ========================= */

.glass{

background:rgba(255,255,255,.08);

backdrop-filter:blur(18px);

border-radius:20px;

padding:20px;

border:1px solid rgba(255,255,255,.12);

margin-bottom:18px;

box-shadow:0px 8px 25px rgba(0,0,0,.25);

}

/* ========================= */

[data-testid="stSidebar"]{

background:#09121E;

}

[data-testid="stSidebar"] *{

color:white !important;

}

/* ========================= */

.stButton>button{

width:100%;

background:linear-gradient(
90deg,
#2563EB,
#4F46E5
);

color:white;

border:none;

border-radius:12px;

font-weight:bold;

padding:12px;

}

/* ========================= */

.stTextInput>div>div>input{

border-radius:12px;

}

/* ========================= */

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

</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

<h1>
🏥 Health Insurance RAG Chatbot
</h1>

<p>

Ask questions about Health Insurance using
Retrieval-Augmented Generation (RAG)
powered by OpenAI GPT-4.1 Mini.

</p>

</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 AI Dashboard")

    st.markdown("---")

    st.subheader("👨‍💻 Developer")

    st.success("Anamika Yadav")

    st.markdown("---")

    st.subheader("🔗 Connect")

    st.link_button(
        "📂 GitHub",
        "https://github.com/Anamikaa200",
    )

    st.link_button(
        "💼 LinkedIn",
        "https://www.linkedin.com/in/anamika-yadav-64b688340",
    )

    st.markdown("---")

    st.subheader("📌 About")

    st.info(
        """
This chatbot retrieves information
from a Health Insurance website.

It uses:

• OpenAI Embeddings

• FAISS

• GPT-4.1 Mini

to answer your questions.
"""
    )

# ============================================================
# API KEY
# ============================================================

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:

    api_key = st.sidebar.text_input(
        "🔑 OpenAI API Key",
        type="password",
    )

if not api_key:

    st.warning(
        "Please enter your OpenAI API Key."
    )

    st.stop()

# ============================================================
# SOURCE WEBSITE
# ============================================================

URL = "https://www.starhealth.in/health-insurance/types-of-health-insurance/"

# ============================================================
# LOAD WEBSITE
# ============================================================

@st.cache_data(show_spinner="🌐 Loading Health Insurance website...")
def load_website():

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/138.0 Safari/537.36"
        )
    }

    try:

        response = requests.get(
            URL,
            headers=headers,
            timeout=30,
        )

        response.raise_for_status()

    except requests.exceptions.RequestException as e:

        raise RuntimeError(
            f"Unable to access the source website.\n\n{e}"
        )

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    # Remove unnecessary tags
    for tag in soup(
        ["script", "style", "noscript", "svg", "iframe"]
    ):
        tag.extract()

    text = soup.get_text(separator="\n")

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    cleaned_text = "\n".join(lines)

    return cleaned_text

# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

@st.cache_resource(show_spinner="🧠 Creating Vector Database...")
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

    vector_db = FAISS.from_documents(
        documents,
        embeddings,
    )

    return vector_db

# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================

try:

    website_text = load_website()

    vector_db = create_vector_db(
        website_text,
        api_key,
    )

except Exception as e:

    st.error("❌ Failed to build the knowledge base.")

    st.exception(e)

    st.stop()

# ============================================================
# KNOWLEDGE BASE STATUS
# ============================================================

st.success("✅ Knowledge Base Loaded Successfully")

with st.expander("📄 Knowledge Base Information"):

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Characters",
            f"{len(website_text):,}"
        )

    with col2:

        chunks = len(
            RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            ).create_documents([website_text])
        )

        st.metric(
            "Document Chunks",
            chunks
        )

    st.markdown("---")

    st.markdown(
        f"""
**Source Website**

{URL}

The chatbot answers questions using information retrieved
from this webpage and indexed into a FAISS vector database.
"""
    )

st.markdown("---")

# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================================
# QUESTION SECTION
# ============================================================

st.markdown("""
<div class="glass">
<h2>💬 Ask Your Health Insurance Question</h2>
<p style="color:#CBD5E1;">
Ask anything related to health insurance policies.
</p>
</div>
""", unsafe_allow_html=True)

question = st.text_input(
    "Question",
    placeholder="Example: What are the different types of health insurance?",
    label_visibility="collapsed",
)

# ============================================================
# QUESTION ANSWERING
# ============================================================

if question:

    # --------------------------------------------------------
    # Retrieve Similar Documents
    # --------------------------------------------------------

    with st.spinner("🔍 Searching the knowledge base..."):

        try:

            results = vector_db.similarity_search_with_score(
                question,
                k=4,
            )

            if not results:

                st.warning(
                    "No relevant information was found."
                )

                st.stop()

            docs = [doc for doc, score in results]

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

        except Exception as e:

            st.error(
                "Error while searching the knowledge base."
            )

            st.exception(e)

            st.stop()

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are an expert AI assistant specializing in Health Insurance.

Answer ONLY from the supplied context.

Rules:

- Never fabricate information.
- If the answer is unavailable, reply exactly:

"I couldn't find that information in the provided document."

- Use simple English.
- Prefer bullet points whenever appropriate.
- Be concise but informative.

=========================
CONTEXT
=========================

{context}

=========================
QUESTION
=========================

{question}

=========================
ANSWER
=========================
"""

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    with st.spinner("🤖 GPT-4.1 Mini is generating an answer..."):

        try:

            llm = ChatOpenAI(
                model="gpt-4.1-mini",
                api_key=api_key,
                temperature=0.2,
            )

            response = llm.invoke(prompt)

            answer = response.content.strip()

        except Exception as e:

            st.error(
                "Failed to generate an answer."
            )

            st.exception(e)

            st.stop()

    # --------------------------------------------------------
    # Save History
    # --------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": answer,
        }
    )

    # --------------------------------------------------------
    # Display Answer
    # --------------------------------------------------------

    st.markdown("""
<div class="glass">
<h2>✅ Answer</h2>
</div>
""", unsafe_allow_html=True)

    st.success(answer)

    # --------------------------------------------------------
    # Retrieved Context
    # --------------------------------------------------------

    with st.expander("📚 Retrieved Context"):

        st.write(context)

# ============================================================
# CHAT HISTORY
# ============================================================

if st.session_state.chat_history:

    st.markdown("""
<div class="glass">
<h2>🕘 Conversation History</h2>
</div>
""", unsafe_allow_html=True)

    for i, chat in enumerate(
        reversed(st.session_state.chat_history),
        start=1,
    ):

        with st.expander(
            f"Question {i}: {chat['question']}"
        ):

            st.markdown("**Answer**")

            st.write(chat["answer"])

st.markdown("---")

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

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "LLM",
        "GPT-4.1 Mini"
    )

with col2:
    st.metric(
        "Embedding Model",
        "text-embedding-3-small"
    )

with col3:
    st.metric(
        "Vector Store",
        "FAISS"
    )

st.write("")

# ============================================================
# TECHNOLOGIES
# ============================================================

left, right = st.columns(2)

with left:

    st.markdown("""
<div class="glass">

## 🛠 Technologies Used

- Python
- Streamlit
- LangChain
- FAISS
- BeautifulSoup
- Requests
- OpenAI GPT-4.1 Mini
- OpenAI Embeddings

</div>
""", unsafe_allow_html=True)

with right:

    st.markdown("""
<div class="glass">

## ⭐ Features

- Website Scraping
- Retrieval-Augmented Generation (RAG)
- Semantic Search
- FAISS Vector Database
- OpenAI Embeddings
- GPT-4.1 Mini
- Conversation History
- Context Viewer
- Modern Dashboard UI

</div>
""", unsafe_allow_html=True)

st.write("")

# ============================================================
# WORKFLOW
# ============================================================

st.markdown("""
<div class="glass">

## 🔄 Project Workflow

1️⃣ Scrape the Health Insurance webpage

⬇

2️⃣ Extract and clean webpage text

⬇

3️⃣ Split text into chunks

⬇

4️⃣ Generate embeddings using
OpenAI **text-embedding-3-small**

⬇

5️⃣ Store embeddings in **FAISS**

⬇

6️⃣ Retrieve relevant document chunks

⬇

7️⃣ Generate the final answer using
**GPT-4.1 Mini**

</div>
""", unsafe_allow_html=True)

st.write("")

# ============================================================
# ABOUT PROJECT
# ============================================================

with st.expander("ℹ About this Project", expanded=False):

    st.markdown("""

### Health Insurance RAG Chatbot

This application demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline.

Instead of relying only on the language model's general knowledge, it:

- Scrapes information from a Health Insurance website.
- Cleans and preprocesses the webpage content.
- Splits the content into manageable chunks.
- Converts the chunks into vector embeddings using OpenAI.
- Stores the embeddings in a FAISS vector database.
- Retrieves the most relevant information for each question.
- Uses GPT-4.1 Mini to generate answers based only on the retrieved context.

This approach improves answer relevance and reduces hallucinations compared to using an LLM alone.

""")

# ============================================================
# DEVELOPER
# ============================================================

st.write("")

st.markdown("""
<div class="glass" style="text-align:center;">

<h2>👨‍💻 Developer</h2>

<h3>Anamika Yadav</h3>

<p>
AI • Machine Learning • Retrieval-Augmented Generation
</p>

</div>
""", unsafe_allow_html=True)

developer_col1, developer_col2 = st.columns(2)

with developer_col1:

    st.link_button(
        "📂 GitHub Profile",
        "https://github.com/Anamikaa200",
        use_container_width=True,
    )

with developer_col2:

    st.link_button(
        "💼 LinkedIn Profile",
        "https://www.linkedin.com/in/anamika-yadav-64b688340",
        use_container_width=True,
    )

st.write("")

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
<div style="text-align:center; color:#CBD5E1; padding:20px;">

🏥 <b>Health Insurance RAG Chatbot</b>

<br><br>

Built with ❤️ using

<b>Streamlit • LangChain • FAISS • OpenAI GPT-4.1 Mini</b>

<br><br>

© 2026 Anamika Yadav

</div>
""",
    unsafe_allow_html=True,
)
