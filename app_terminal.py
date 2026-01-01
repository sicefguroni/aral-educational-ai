import os 
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# 1. Setup
load_dotenv()
DB_PATH = "vectorstore/db_faiss"

def start_app():
    # 2. Load the Vector Database
    print("Loading Vector Database...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Load the local vector database
    try:
        vector_db = FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        print(f"Error loading vector database: {e}")
        print("Did you run create_db.py first?")
        return

    # 3. Setup the Groq LLM
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # 4. The "Taglish" Prompt Template
    custom_prompt_template = """
    You are a friendly Filipino tutot helping a student review.
    Use the following pieces of context to answer the question at the end.

    Rules:
    1. Answer in a mix of English and Tagalog (Taglish).
    2. Make it easy to understand (Explain Like I'm 5).
    3. If you don't know the answer based on the context, say "Sorry lods, wala sa module yan."
    4. Keep it encouraging and motivating!

    Context: {context}

    Question: {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

    # 5. Create the Retrieval QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )

    # 6. Start the Loop w/ Chat Interface
    print("\nAral-AI is Ready! (Type 'exit' to quit)")
    print("-----------------------------------------")

    while True:
        query = input("\nStudent: ")
        if query.lower() == "exit":
            break
    
        print("Thinking...")
        try:
            response = qa_chain.invoke(query)
            print(f"\nTutor: {response['result']}")
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    start_app()