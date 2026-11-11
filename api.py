from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import shutil
from langchain_community.document_loaders import TextLoader
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
import time

load_dotenv()
app = FastAPI()


def retriever(chunks, vector_store):
  bm25_retriever = BM25Retriever.from_documents(chunks, k=3)

  vector_retriever = vector_store.as_retriever(search_type="mmr",search_kwargs={"k":3,"fetch_k":15,"lambda_mult":0.5})

  hybrid_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5,0.5])
  return hybrid_retriever

embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
original_embed_documents = embedding_model.embed_documents

def timed_embed_documents(texts):
    start = time.perf_counter()
    embeddings = original_embed_documents(texts)
    end = time.perf_counter()

    print(f"Embedding generation: {end-start:.2f}s")
    return embeddings

embedding_model.embed_documents = timed_embed_documents
UPLOAD_DIR = Path("documents/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
@app.post("/upload")
def upload(files: list[UploadFile] = File(...)):
  start = time.perf_counter()
  
  for file in files:
   t1 = time.perf_counter()
    
   file_path = UPLOAD_DIR / file.filename
   with open(file_path, "wb") as buffer:
    shutil.copyfileobj(file.file, buffer)

   t2 = time.perf_counter()
   loader = TextLoader(str(file_path), encoding="utf-8")
   documents = loader.load()
   t3 = time.perf_counter()
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
   chunks = text_splitter.split_documents(documents)
   t4 = time.perf_counter()
   vector_store = Chroma(collection_name="busines_information", embedding_function=embedding_model, persist_directory="./chroma_db")
   vector_store.add_documents(chunks)
   t5 = time.perf_counter()
   hybrid_retriever = retriever(chunks, vector_store)
   app.state.hybrid_retriever = hybrid_retriever
   t6 = time.perf_counter()
   end = time.perf_counter()
  print(f"File saving: {t2-t1:.2f}s")
  print(f"Document loading: {t3-t2:.2f}s")
  print(f"Text splitting: {t4-t3:.2f}s")
  print(f"Embedding + storage: {t5-t4:.2f}s")
  print(f"Retriever creation: {t6-t5:.2f}s")
  print(f"upload time: {end - start:.2f} seconds")
  return {"message": "Document uploaded and indexed successfully"}


print("OpenRouter API key loaded:", bool(os.getenv("OPENROUTER_API_KEY")))
class Question(BaseModel):
  question: str
@app.post("/chat")

def chat(data: Question):
  start = time.perf_counter()
  if not hasattr(app.state, "hybrid_retriever"):
    raise HTTPException(status_code=409, detail="Upload a document before asking a question")

  hybrid_retriever = app.state.hybrid_retriever
  t1 = time.perf_counter()
  retrieved_chunks = hybrid_retriever.invoke(data.question)
  t2 = time.perf_counter()
  context = "\n\n".join(doc.page_content for doc in retrieved_chunks[:3])
  t3 = time.perf_counter()
  prompt = ChatPromptTemplate.from_template("""
    Answer the question using only the provided context.

    Keep the answer concise and direct.
    Do not include lengthy reasoning or explanations.
    Provide the answer in 3-5 sentences when possible.

    If the answer is unavailable in the context, say:
    "I don't know based on the provided documents."

    Context:
    {context}

    Question:
    {question}

    Answer:""")
  try:
   llm = ChatOpenRouter(model="liquid/lfm-3-1.7b:free", temperature=0, max_tokens=2048)
   rag_chain = prompt | llm
   response = rag_chain.invoke({"context": context, "question": data.question})
  except Exception as e:
    raise HTTPException(
        status_code=503,
        detail=f"LLM service temporarily unavailable: {str(e)}")
  print("RAW RESPONSE:", repr(response.content))
  print("METADATA:", response.response_metadata)
  print("USAGE:", response.usage_metadata)
  t4 = time.perf_counter()
  end = time.perf_counter()
  print(f"Retrieval: {t2-t1:.2f}s")
  print(f"Context preparation: {t3-t2:.2f}s")
  print(f"LLM generation: {t4-t3:.2f}s")
  print(f"Total: {t4-start:.2f}s")
  print(f"response time: {end - start:.2f} seconds")
  return {"response": response.content}





  

  











  
  
