# app/utils.py
import os
import hashlib
from typing import Dict, Any, List
from datetime import datetime
from llama_index.core import Document

def create_document_id(text: str) -> str:
    """Create a unique document ID based on content."""
    return hashlib.md5(text.encode()).hexdigest()

def extract_document_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from a document file."""
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1].lower()
    file_size = os.path.getsize(file_path)
    modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
    
    metadata = {
        "file_name": file_name,
        "file_extension": file_ext,
        "file_size_bytes": file_size,
        "last_modified": modified_time.isoformat(),
    }
    
    return metadata

def nodes_to_documents(nodes: List[Dict[str, Any]]) -> List[Document]:
    """Convert processed nodes back to Document objects."""
    documents = []
    for node in nodes:
        doc = Document(
            text=node["text"],
            metadata=node["metadata"],
            doc_id=node.get("id", create_document_id(node["text"]))
        )
        documents.append(doc)
    
    return documents