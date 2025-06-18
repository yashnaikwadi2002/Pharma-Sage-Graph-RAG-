# test_ai.py
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print("🔑 Gemini Key loaded:", api_key[:10], "...")

embedder = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

chunks = ["This is pharma chunk A", "This is pharma chunk B"]
embeddings = embedder.embed_documents(chunks)

print("✅ Embeddings retrieved:", len(embeddings), "chunks")
print("🔢 Example vector:", embeddings[0][:5], "...")
