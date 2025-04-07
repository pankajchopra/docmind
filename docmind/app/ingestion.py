# app/ingestion.py
import os
from typing import List, Dict, Any
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from app.config import DATA_DIR

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 20):
        """Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
    def load_documents(self, directory: str = None, file_paths: List[str] = None) -> List[Document]:
        """Load documents from directory or specific file paths."""
        if directory is None:
            directory = DATA_DIR
            
        reader = SimpleDirectoryReader(
            input_dir=directory,
            file_paths=file_paths,
            filename_as_id=True
        )
        
        documents = reader.load_data()
        
        print(f"Loaded {len(documents)} document(s)")
        for doc in documents:
            print(f"- {doc.metadata.get('file_name', 'Unknown')}: {len(doc.text)} characters")
            
        return documents
    
    def process_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Process documents into nodes with metadata."""
        nodes = self.node_parser.get_nodes_from_documents(documents)
        
        processed_nodes = []
        for node in nodes:
            # Extract and enhance metadata
            metadata = node.metadata.copy()
            # Add additional metadata (e.g., creation date, word count)
            metadata["word_count"] = len(node.text.split())
            metadata["char_count"] = len(node.text)
            
            processed_nodes.append({
                "text": node.text,
                "metadata": metadata,
                "id": node.id_
            })
            
        print(f"Processed into {len(processed_nodes)} text chunks")
        return processed_nodes
    
    def ingest_documents(self, directory: str = None, file_paths: List[str] = None) -> List[Dict[str, Any]]:
        """Complete ingestion pipeline: load and process documents."""
        documents = self.load_documents(directory, file_paths)
        return self.process_documents(documents)


# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    nodes = processor.ingest_documents()
    print(f"Ingestion complete. Created {len(nodes)} nodes.")