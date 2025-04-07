# app/main.py
import os
from app.ingestion import DocumentProcessor
from app.indexing import DocumentIndexer
from app.retrieval import LightRAGRetriever
from app.agents import DocMindAgent
from app.interface import launch_interface
from app.config import DATA_DIR

def main():
    """Main entry point for the DocMind application."""
    print("Starting DocMind system...")
    
    # Step 1: Process documents from data directory
    print("Processing documents...")
    processor = DocumentProcessor()
    default_docs = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) 
                   if os.path.isfile(os.path.join(DATA_DIR, f))]
    nodes = processor.ingest_documents(file_paths=default_docs)
    print(f"Processed {len(nodes)} nodes from documents.")
    
    # Step 2: Create indexes
    print("Creating indexes...")
    indexer = DocumentIndexer()
    indexer.create_vector_index(nodes)
    indexer.create_knowledge_graph(nodes)
    print("Indexes created successfully.")
    
    # Step 3: Set up retriever
    print("Setting up retrieval system...")
    retriever = LightRAGRetriever(indexer)
    retriever.setup_bm25_retriever(nodes)
    print("Retrieval system initialized.")
    
    # Step 4: Create agent
    print("Initializing DocMind agent...")
    agent = DocMindAgent(indexer, retriever)
    print("Agent initialized successfully.")
    
    # Step 5: Launch the interface
    print("Launching user interface...")
    launch_interface(agent)
    print("Interface launched. DocMind is ready!")

if __name__ == "__main__":
    main()