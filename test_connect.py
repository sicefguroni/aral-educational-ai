import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# 1. Initialize the "Chat Model"
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

print("🔌 Connecting to Groq (Llama 3)...")

# 2. Test Question
question = "Translate 'Good morning, student' into Taglish."
print(f"📝 Asking: {question}")

try:
    response = llm.invoke(question)
    print("\n🚀 SUCCESS! AI Response:")
    print(response.content)
except Exception as e:
    print(f"\n❌ Error: {e}")