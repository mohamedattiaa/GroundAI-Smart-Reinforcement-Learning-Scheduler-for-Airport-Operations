# === Interactive Chatbot for Airport Ground Operations (RAG + LLaMA 3) ===
# Works locally with Ollama and ChromaDB
import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# --- STEP 1: Load your dataset ---
data_folder = "data"
files = ["aircraft.csv", "vehicles.csv", "tasks.csv", "assignments.csv"]
docs = []

for file in files:
    path = os.path.join(data_folder, file)
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {file}")
        continue
    df = pd.read_csv(path)
    text = df.to_string(index=False)
    docs.append(Document(page_content=text, metadata={"source": file}))

# --- STEP 2: Split documents ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# --- STEP 3: Create / Load Chroma Vector Database ---
persist_dir = "chroma_db"
if os.path.exists(persist_dir):
    print("✅ Loading existing Chroma database...")
    db = Chroma(persist_directory=persist_dir, embedding_function=OllamaEmbeddings(model="llama3:8b"))
else:
    print("🚀 Creating new Chroma database...")
    embeddings = OllamaEmbeddings(model="llama3:8b")
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    db.persist()

# --- STEP 4: Initialize Local LLM ---
llm = Ollama(model="llama3:8b")

# --- STEP 5: Interactive chat loop ---
print("\n🛫 Airport AI Assistant ready! (type 'exit' to quit)\n")

while True:
    query = input("❓ Ask me about the airport data: ").strip()
    if query.lower() in ["exit", "quit", "q"]:
        print("👋 Goodbye!")
        break

    # Retrieve most relevant data chunks
    results = db.similarity_search(query, k=3)
    context = "\n\n".join([r.page_content for r in results])

    # Build prompt
    prompt = f"Use the following airport operations data to answer:\n{context}\n\nQuestion: {query}"
    answer = llm.invoke(prompt)
    print("\n🧠 AI Answer:\n", answer)
    print("-" * 80)
