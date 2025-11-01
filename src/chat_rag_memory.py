# === Airport Chatbot with Memory + RAG + LLaMA 3 ===
import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# --- Load dataset ---
data_folder = "data"
files = ["aircraft.csv", "vehicles.csv", "tasks.csv", "assignments.csv"]
docs = []
for file in files:
    path = os.path.join(data_folder, file)
    df = pd.read_csv(path)
    docs.append(Document(page_content=df.to_string(index=False), metadata={"source": file}))

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# --- Vector Store ---
persist_dir = "chroma_db"
embeddings = OllamaEmbeddings(model="llama3:8b")
db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
retriever = db.as_retriever(search_kwargs={"k": 3})

# --- Memory ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- LLM + Chain ---
llm = Ollama(model="llama3:8b")
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

# --- Chat Loop ---
print("\n🧠 Airport Assistant with Memory (type 'exit' to quit)\n")

while True:
    question = input("❓ You: ").strip()
    if question.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    answer = qa_chain.run(question)
    print(f"🤖 LLaMA: {answer}\n")
