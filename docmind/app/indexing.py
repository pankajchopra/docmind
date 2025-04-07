# app/indexing.py
import os
from typing import List, Dict, Any, Optional
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from app.config import INDEX_STORE_DIR, DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL, OPENAI_API_KEY
from app.utils import nodes_to_documents

class DocumentIndexer:
    def __init__(self, 
                 llm_model: str = DEFAULT_LLM_MODEL,
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        """Initialize the document indexer with specified models."""
        # Configure default models
        self.llm = OpenAI(model=llm_model, api_key=OPENAI_API_KEY)
        self.embed_model = OpenAIEmbedding(model=embedding_model, api_key=OPENAI_API_KEY)
        
        # Set as default in Settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        # Initialize indexes
        self.vector_index = None
        self.kg_index = None
    
    def create_vector_index(self, 
                          nodes: List[Dict[str, Any]], 
                          index_name: str = "vector_index") -> VectorStoreIndex:
        """Create a vector store index from processed nodes."""
        # Convert nodes to documents
        documents = nodes_to_documents(nodes)
        
        # Create vector index
        self.vector_index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True
        )
        
        # Save the index
        index_path = os.path.join(INDEX_STORE_DIR, index_name)
        self.vector_index.storage_context.persist(persist_dir=index_path)
        
        print(f"Vector index created and saved to {index_path}")
        return self.vector_index
    
    def create_knowledge_graph(self, 
                             nodes: List[Dict[str, Any]], 
                             index_name: str = "kg_index") -> KnowledgeGraphIndex:
        """Create a knowledge graph index from processed nodes."""
        # Convert nodes to documents
        documents = nodes_to_documents(nodes)
        
        # Create knowledge graph index
        self.kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            max_triplets_per_chunk=10,
            include_embeddings=True,
            show_progress=True
        )
        
        # Save the index
        index_path = os.path.join(INDEX_STORE_DIR, index_name)
        self.kg_index.storage_context.persist(persist_dir=index_path)
        
        print(f"Knowledge graph index created and saved to {index_path}")
        return self.kg_index
    
    def load_index(self, index_name: str, index_type: str = "vector") -> Optional[Any]:
        """Load a previously saved index."""
        index_path = os.path.join(INDEX_STORE_DIR, index_name)
        
        if not os.path.exists(index_path):
            print(f"Index not found at {index_path}")
            return None
        
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        
        if index_type == "vector":
            self.vector_index = load_index_from_storage(storage_context)
            return self.vector_index
        elif index_type == "kg":
            self.kg_index = load_index_from_storage(storage_context)
            return self.kg_index
        else:
            raise ValueError(f"Unknown index type: {index_type}")
    
    def create_query_engine(self, index_type: str = "vector", **kwargs):
        """Create a query engine from the specified index type."""
        if index_type == "vector" and self.vector_index:
            return self.vector_index.as_query_engine(**kwargs)
        elif index_type == "kg" and self.kg_index:
            return self.kg_index.as_query_engine(**kwargs)
        else:
            raise ValueError("Index not available or invalid type specified")


# Example usage
if __name__ == "__main__":
    from app.ingestion import DocumentProcessor
    
    # Process documents
    processor = DocumentProcessor()
    nodes = processor.ingest_documents()
    
    # Create indexes
    indexer = DocumentIndexer()
    vector_index = indexer.create_vector_index(nodes)
    kg_index = indexer.create_knowledge_graph(nodes)
    
    # Test query
    query_engine = indexer.create_query_engine("vector")
    response = query_engine.query("What are the main topics in these documents?")
    print(response)