# chromadb_ingestion.py
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import hashlib
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChromaDBIngestion:
    """Utility class for ingesting documents into ChromaDB."""

    def __init__(self,
                 openai_api_key=None,
                 chroma_persist_dir="./chroma_db",
                 document_hashes_file="document_hashes.json",
                 data_dir="./data",
                 chunk_size=1024,
                 chunk_overlap=20):
        """Initialize the ChromaDB ingestion utility."""
        # Set API key from environment variable if not provided
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        # Set file paths
        self.chroma_persist_dir = chroma_persist_dir
        self.document_hashes_file = document_hashes_file
        self.data_dir = data_dir

        # Ensure directories exist
        Path(self.chroma_persist_dir).mkdir(exist_ok=True, parents=True)

        # Set up embedding model
        self.embed_model = OpenAIEmbeddings()

        # Set up the vector store
        self.vector_store = self._setup_vector_store()

        # Create text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Load existing document hashes
        self.document_hashes = self._load_document_hashes()

    def _setup_vector_store(self):
        """Initialize and return the LangChain Chroma vector store."""
        try:
            # Try to load existing DB
            vector_store = Chroma(
                persist_directory=self.chroma_persist_dir,
                embedding_function=self.embed_model
            )
            logger.info("Loaded existing Chroma vector store")
        except Exception as e:
            # Create new if it doesn't exist or has issues
            logger.info(f"Creating new Chroma vector store: {e}")
            vector_store = Chroma(
                embedding_function=self.embed_model,
                persist_directory=self.chroma_persist_dir
            )
        return vector_store

    def _load_document_hashes(self):
        """Load previously processed document hashes."""
        try:
            with open(self.document_hashes_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_document_hashes(self):
        """Save document hashes to track changes."""
        with open(self.document_hashes_file, "w") as f:
            json.dump(self.document_hashes, f)

    def get_document_hash(self, document: Document):
        """Generate a hash for a document's content."""
        return hashlib.sha256(document.page_content.encode()).hexdigest()

    def process_document(self, document: Document):
        """Process a document for vector storage, updating if changed."""
        doc_id = document.metadata.get("source")
        if not doc_id:
            logger.warning("Document missing 'source' metadata. Skipping.")
            return False

        current_hash = self.get_document_hash(document)

        if doc_id not in self.document_hashes or self.document_hashes[doc_id] != current_hash:
            logger.info(f"Document '{doc_id}' has changed or is new. Updating vector store...")

            # Split document into chunks
            chunks = self.text_splitter.split_documents([document])

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

    def process_documents(self, doc_paths=None):
        """Process documents and update the vector store."""
        # Load documents using LangChain's DirectoryLoader
        if doc_paths:
            # Process specific documents
            documents = []
            for path in doc_paths:
                loader = TextLoader(path)
                documents.extend(loader.load())
        else:
            # Process all documents in the data directory
            loader = DirectoryLoader(self.data_dir, glob="**/*.txt", loader_cls=TextLoader)
            documents = loader.load()

        logger.info(f"Loaded {len(documents)} documents")

        # Process each document
        updated_docs = 0
        for doc in documents:
            if self.process_document(doc):
                updated_docs += 1

        # Save updated document hashes
        self.save_document_hashes()

        # Persist the vector store
        self.vector_store.persist()

        logger.info(f"Processing complete. Updated {updated_docs} documents.")

        return updated_docs


def main(doc_paths=None):
    """Main function to process documents and update ChromaDB."""
    logger.info("Starting ChromaDB document ingestion")

    # Initialize the ChromaDB ingestion utility
    chroma_ingestion = ChromaDBIngestion()

    # Process documents
    updated_docs = chroma_ingestion.process_documents(doc_paths)

    return updated_docs


if __name__ == "__main__":
    import sys

    doc_paths = sys.argv[1:] if len(sys.argv) > 1 else None
    main(doc_paths)