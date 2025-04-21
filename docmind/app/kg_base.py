# kg_base.py
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LlamaIndex imports for KnowledgeGraphIndex functionality
from llama_index.core import Document as LlamaIndexDocument
from llama_index.core import StorageContext
from llama_index.core import KnowledgeGraphIndex
from llama_index.llms.openai import OpenAI as LlamaOpenAI

import os
import hashlib
import json
import logging
from typing import Literal
from config import DATA_DIR, CHROMA_STORE_DIR, DOCUMENT_HASHES_FILE, CHUNK_SIZE, CHUNK_OVERLAP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration constants
GRAPH_STORAGE_TYPES = Literal["neo4j", "sqlite", "memory"]


class Config:
    """Configuration class to hold all settings."""
    @classmethod
    def initialize(cls):
        """Initialize the configuration by ensuring directories exist."""
        # Ensure all directories exist



class BaseGraphManager:
    """Base manager class with common functionality for different graph storage types."""

    def __init__(self):
        """Initialize shared components."""
        Config.initialize()

        # Set up LLM and embedding models
        self.langchain_embed_model = OpenAIEmbeddings()
        self.langchain_llm = ChatOpenAI(temperature=0)
        self.llama_llm = LlamaOpenAI()

        # Set up the vector store
        self.vector_store = self._setup_vector_store()

        # Create text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        # Load existing document hashes
        self.document_hashes = self._load_document_hashes()

    def _setup_vector_store(self):
        """Initialize and return the LangChain Chroma vector store."""
        try:
            # Try to load existing DB
            vector_store = Chroma(
                persist_directory=CHROMA_STORE_DIR,
                embedding_function=self.langchain_embed_model
            )
            logger.info("Loaded existing Chroma vector store")
        except Exception as e:
            # Create new if it doesn't exist or has issues
            logger.info(f"Creating new Chroma vector store: {e}")
            vector_store = Chroma(
                embedding_function=self.langchain_embed_model,
                persist_directory=CHROMA_STORE_DIR
            )
        return vector_store

    def _load_document_hashes(self):
        """Load previously processed document hashes."""
        try:
            with open(DOCUMENT_HASHES_FILE, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_document_hashes(self):
        """Save document hashes to track changes."""
        with open(DOCUMENT_HASHES_FILE, "w") as f:
            json.dump(self.document_hashes, f)

    def get_document_hash(self, langchain_document: LangchainDocument):
        """Generate a hash for a document's content."""
        return hashlib.sha256(langchain_document.page_content.encode()).hexdigest()

    def convert_to_llama_document(self, langchain_doc: LangchainDocument) -> LlamaIndexDocument:
        """Convert a LangChain document to a LlamaIndex document."""
        return LlamaIndexDocument(
            text=langchain_doc.page_content,
            metadata=langchain_doc.metadata
        )

    def process_document_for_vector_store(self, langchain_document: LangchainDocument):
        """Process a document for vector storage, updating if changed."""
        doc_id = langchain_document.metadata.get("source")
        if not doc_id:
            logger.warning("LangChain Document missing 'source' metadata. Skipping.")
            return False

        current_hash = self.get_document_hash(langchain_document)

        if doc_id not in self.document_hashes or self.document_hashes[doc_id] != current_hash:
            logger.info(f"Document '{doc_id}' has changed or is new. Updating vector store...")

            # Split document into chunks
            chunks = self.text_splitter.split_documents([langchain_document])

            # Add document_id to all chunks
            for chunk in chunks:
                chunk.metadata["document_id"] = doc_id

            # Check if document already exists in Chroma
            if doc_id in self.document_hashes:
                # Delete existing document by document_id
                try:
                    self.vector_store.delete(
                        filter={"document_id": doc_id}
                    )
                    logger.info(f"Deleted existing vectors for document '{doc_id}'")
                except Exception as e:
                    logger.warning(f"Error deleting existing vectors: {e}")

            # Add new document chunks to Chroma
            self.vector_store.add_documents(chunks)

            # Update hash
            self.document_hashes[doc_id] = current_hash
            logger.info(f"Successfully updated vector store for document '{doc_id}'")
            return True
        else:
            logger.info(f"Document '{doc_id}' has not changed. Skipping vector update.")
            return False

    def process_document_for_graph(self, langchain_document: LangchainDocument):
        """Process a document for graph storage - override in subclasses."""
        raise NotImplementedError("Subclasses must implement process_document_for_graph")

    def delete_graph_elements_for_doc(self, doc_id: str):
        """Delete existing graph elements for a document - override in subclasses."""
        raise NotImplementedError("Subclasses must implement delete_graph_elements_for_doc")

    def visualize_graph(self, doc_id=None):
        """Create a visualization of the graph - override in subclasses."""
        raise NotImplementedError("Subclasses must implement visualize_graph")

    def process_documents(self, doc_paths=None):
        """Process documents and update indices."""
        # Load documents using LangChain's TextLoader
        if doc_paths:
            # Process specific documents
            documents = []
            for path in doc_paths:
                loader = TextLoader(path)
                documents.extend(loader.load())
        else:
            # Process all documents in the data directory
            from langchain.document_loaders import DirectoryLoader
            loader = DirectoryLoader(Config.DATA_DIR, glob="**/*.txt, *.pdf", loader_cls=TextLoader)
            documents = loader.load()

        logger.info(f"Loaded {len(documents)} documents")

        # Process each document
        updated_docs = 0
        for doc in documents:
            # Process for vector store
            vector_updated = self.process_document_for_vector_store(doc)

            # Process for graph store
            graph_updated = self.process_document_for_graph(doc)

            if vector_updated or graph_updated:
                updated_docs += 1

                # Create visualization if document was updated
                doc_id = doc.metadata.get("source")
                if doc_id:
                    try:
                        self.visualize_graph(doc_id)
                    except Exception as e:
                        logger.error(f"Failed to create graph visualization: {e}")

        # Save updated document hashes
        self.save_document_hashes()

        # Persist the vector store
        self.vector_store.persist()

        logger.info(f"Processing complete. Updated {updated_docs} documents.")

        return updated_docs