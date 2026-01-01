import os 
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# 1. Page Config
st.set_page_config(page_title="Aral-AI", page_icon="🇵🇭")

# 2. Load Environment Variables
load_dotenv()

# 3. Define Functions
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

def process_pdf(uploaded_file):
    """Save upload to temp file, load it, chunk it, and index it."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = get_embeddings()
    vector_db = FAISS.from_documents(texts, embeddings)

    # Clean up temp file
    os.remove(tmp_path)
    return vector_db

# 4. The Sidebar (Settings)
with st.sidebar:
    st.title("🇵🇭 Aral-AI Settings")

    # A. File Uploader
    uploaded_file = st.file_uploader("Upload your Module (PDF)", type="pdf")

    # B. Persona Selector (Dynamic Prompting)
    persona = st.selectbox(
        "Choose Tutor Style:",
        ("Taglish (Default)", "Cebuano (Bisaya)", "English (Formal)")
    )

    st.info("Powered by Groq (Llama 3) & LangChain")

# 5. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Kamusta! Upload your module and ask me anything."}
    ]

# 6. Main App Logic
st.title("🇵🇭 Aral-AI: Your Smart Filipino Tutor")

# Display previous message
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User Input
if user_input := st.chat_input("Ask a question about your module..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Check if PDF is processed
    if uploaded_file is None:
        response_text = "Please upload your module first sa sidebar, lods!"
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.chat_message("assistant").write(response_text)
    else:
        # If NEW file, process it
        if "vector_db" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
            with st.spinner("Wait lang, reading your module..."):
                st.session_state.vector_db = process_pdf(uploaded_file)
                st.session_state.current_file = uploaded_file.name
                st.success("Module loaded!")

        # DEFINE PROMPTS BASED ON SELECTION
        if persona == "Taglish (Default)":
            custom_template = """You are a Filipino tutor. Answer in Taglish (Tagalog-Engish mix).
            Be friendly and easy to understand. If the answer is not in the module, say "Sorry lods, wala sa module yan."
            Context: {context}
            Question: {question}"""
        elif persona == "Cebuano (Bisaya)":
            custom_template = """You are a Filipino tutor. Answer in Cebuano (Bisaya).
            Be friendly. If the answer is not in the module, say "Wala na sa module bai."
            Context: {context}
            Question: {question}"""
        elif persona == "English (Formal)":
            custom_template = """You are a formal academic tutor. Answer in clear, concise English.
            Context: {context}
            Question: {question}"""

        # Run the Chain
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=os.getenv("GROQ_API_KEY")
        )

        prompt = PromptTemplate(template=custom_template, input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_db.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain.invoke(user_input)
                st.write(response["result"])
                st.session_state.messages.append({"role": "assistant", "content": response['result']})