import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Setup: Define where the PDF is and where to save the database
PDF_PATH = "data/sample_module.pdf"
DB_PATH = "vectorstore/db_faiss"

def create_vector_db():
    print("Loading PDF...")
    if not os.path.exists(PDF_PATH):
        print(f"Error: File {PDF_PATH} does not exist.")
        return
    
    # A. Load the Document
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    # B. Split the Document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # C. Embed the Chunks
    print("Generating embeddings for chunks...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # D. Create the Vector Database
    vector_db = FAISS.from_documents(texts, embeddings)

    # E. Save locally the vector database
    vector_db.save_local(DB_PATH)
    print(f"Success!Vector database saved to {DB_PATH}")

if __name__ == "__main__":
    create_vector_db()