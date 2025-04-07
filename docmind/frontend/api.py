# app/api.py
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import shutil
import uuid

from app.ingestion import DocumentProcessor
from app.indexing import DocumentIndexer
from app.retrieval import LightRAGRetriever
from app.agents import DocMindAgent
from app.config import DATA_DIR

# Initialize components
processor = DocumentProcessor()
indexer = DocumentIndexer()
retriever = LightRAGRetriever(indexer)
agent = DocMindAgent(indexer, retriever)

# Define API models
class ChatRequest(BaseModel):
    message: str
    retrieval_type: Optional[str] = "hybrid"

class ChatResponse(BaseModel):
    response: str
    
app = FastAPI(title="DocMind API", 
              description="API for the DocMind document management system",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system with default documents
@app.on_event("startup")
async def startup_event():
    default_docs = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) 
                   if os.path.isfile(os.path.join(DATA_DIR, f))]
    
    if default_docs:
        nodes = processor.ingest_documents(file_paths=default_docs)
        indexer.create_vector_index(nodes)
        indexer.create_knowledge_graph(nodes)
        retriever.setup_bm25_retriever(nodes)
        print(f"Initialized with {len(default_docs)} documents")

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    response = agent.chat(request.message)
    return {"response": response}

# Document search endpoint
@app.post("/search")
async def search(query: str, retrieval_type: str = "hybrid"):
    results = agent.search_documents(query, retrieval_type)
    return results

# Upload documents endpoint
@app.post("/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    saved_files = []
    
    # Save uploaded files
    for file in files:
        temp_file_path = os.path.join(DATA_DIR, f"{uuid.uuid4()}_{file.filename}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(temp_file_path)
    
    # Process files in background
    background_tasks.add_task(process_new_documents, saved_files)
    
    return {"message": f"Uploaded {len(saved_files)} files. Processing in background."}

async def process_new_documents(file_paths):
    """Process new documents in the background."""
    nodes = processor.ingest_documents(file_paths=file_paths)
    
    # Update indexes
    indexer.create_vector_index(nodes, "vector_index_updated")
    indexer.create_knowledge_graph(nodes, "kg_index_updated")
    
    # Update retrievers
    retriever.setup_bm25_retriever(nodes)
    print(f"Processed {len(file_paths)} new documents")

# Knowledge graph endpoint
@app.get("/knowledge-graph")
async def query_knowledge_graph(query: str, max_results: int = 5):
    results = agent.search_knowledge_graph(query, max_results)
    return results

def start_api():
    """Start the FastAPI server."""
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start_api()