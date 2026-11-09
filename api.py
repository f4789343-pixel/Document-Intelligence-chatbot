from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
import shutil
from langchain_community.document_loaders import TextLoader
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
app = FastAPI()

def retreiver(chunks, vector_store):
  bm25_retriever = BM25Retriever.from_documents(chunks, k=3)

  vector_retriever = vector_store.as_retriever(search_kwargs={"k":3})

  hybrid_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5,0.5])
  return hybrid_retriever

UPLOAD_DIR = Path("documents/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
@app.post("/upload")
def predict(file: UploadFile = File(...)):
  file_path = UPLOAD_DIR / file.filename
  with open(file_path, "wb") as buffer:
    shutil.copyfileobj(file.file, buffer)
  loader = TextLoader(str(file_path), encoding="utf-8")
  documents = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
  chunks = text_splitter.split_documents(documents)
  embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
  vector_store = Chroma(collection_name="busines_information", embedding_function=embedding_model, persist_directory="./chroma_db")
  vector_store.add_documents(chunks)
  hybrid_retriever = retreiver(chunks, vector_store)
  app.state.hybrid_retriever = hybrid_retriever
  return {"message": "Document uploaded and indexed successfully"}

reranking = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieval_context(retrieved_chunks,query, top_k=3):
  pairs = [[query, doc.page_content] for doc in retrieved_chunks]
  scores = reranking.predict(pairs)
  ranked_results = sorted(zip(retrieved_chunks, scores), key=lambda item: item[1], reverse=True)
  context = "\n\n".join(doc.page_content for doc, _ in ranked_results[:top_k])
  return context

class Question(BaseModel):
  question: str
@app.post("/chat")

def chat(data: Question):
  if not hasattr(app.state, "hybrid_retriever"):
    raise HTTPException(status_code=409, detail="Upload a document before asking a question")

  hybrid_retriever = app.state.hybrid_retriever
  retrieved_chunks = hybrid_retriever.invoke(data.question)
  context = retrieval_context(retrieved_chunks, data.question)
  prompt = ChatPromptTemplate.from_template("""
       Answer the question using only the provided context.
       If the answer is not available in the context, say "I don't know based on the provided documents."
       Context:
       {context}
       Question:
       {question}
       Answer:
       """)
  llm = ChatOllama(model="qwen3:4b-instruct", temperature=0, reasoning=False, model_kwargs={"think": False})
  rag_chain = prompt | llm
  response = rag_chain.invoke({"context": context, "question": data.question})
  return {"response": response.content}




  

  











  
  
