from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pandas as pd
import os

# --- Load datasets ---
data_folder = "data"
files = ["aircraft.csv", "vehicles.csv", "tasks.csv", "assignments.csv"]
docs = []

for file in files:
    path = os.path.join(data_folder, file)
    df = pd.read_csv(path)
    text = df.to_string(index=False)
    docs.append(Document(page_content=text, metadata={"source": file}))

# --- Split text into chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# --- Create embeddings & local vector store ---
embeddings = OllamaEmbeddings(model="llama3:8b")
db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="chroma_db")

# --- Initialize LLM ---
llm = Ollama(model="llama3:8b")

# --- Query function ---
def ask_ai(query):
    # Search the most relevant data chunks
    results = db.similarity_search(query, k=3)
    context = "\n\n".join([r.page_content for r in results])
    prompt = f"Use the airport data below to answer:\n\n{context}\n\nQuestion: {query}"
    answer = llm.invoke(prompt)
    print("\n🧠 AI Answer:\n", answer)

# --- Example queries ---
ask_ai("Which vehicle handled aircraft A1?")
ask_ai("List all tasks related to aircraft A2.")
ask_ai("What is the duration of the refueling task?")
