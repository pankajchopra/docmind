# app/indexing.py
import os
import logging
from typing import List
from langchain.schema.document import Document
from langchain_openai import ChatOpenAI  # Changed from OpenAI to ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.chains import RetrievalQA

from app.ai_models import AVAILABLE_EMBEDDINGS
from app.config import CHROMA_STORE_DIR, DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL, OPENAI_API_KEY
from app.customeExceptions import ChromaException, VectorStoreException

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class DocumentIndexer:
    def __init__(self,
                 llm_model: str = DEFAULT_LLM_MODEL,
                 embedding_model: str = AVAILABLE_EMBEDDINGS["openai-small"]["model"],
                 base_url: str = "https://api.openai.com/v1",
                 collection_name: str = "MyStockCollection"):
        """Initialize the document indexer with specified models."""
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not self.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Configure default models
        # Use ChatOpenAI instead of OpenAI for modern models
        self.llm = ChatOpenAI(
            model=llm_model,
            api_key=OPENAI_API_KEY,
            temperature=0
        )
        self.embedding_model = OpenAIEmbeddings(
            model=embedding_model,
            base_url=base_url,
            api_key=OPENAI_API_KEY
        )
        self.collection_name = collection_name
        self.vector_store = None

    def create_vector_index(self,
                            documents: List[Document],
                            persist: bool = True) -> Chroma:
        """Create a vector store index from processed documents."""
        if not documents:
            error_msg = "Cannot create index with empty document list"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            from app.ingestion_vector import DocumentProcessor
            processor = DocumentProcessor()
            logger.info(f"Creating vector index with {len(documents)} documents")
            vector_store = processor.save_to_chroma(documents)
            if vector_store is None:
                raise VectorStoreException("Failed to create vector store")
            self.vector_store = vector_store
            return vector_store
        except Exception as e:
            error_msg = f"Error creating vector index: {e}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg) from e

    def load_index(self) -> Chroma:
        """Load a previously saved index, i.e. load chroma vector store."""
        if not os.path.exists(CHROMA_STORE_DIR):
            error_msg = f"Index not found at {CHROMA_STORE_DIR}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.info(f"Loading index from {CHROMA_STORE_DIR}")
            self.vector_store = Chroma(persist_directory=CHROMA_STORE_DIR,
                                       embedding_function=self.embedding_model,
                                       collection_name=self.collection_name
                                       )
            return self.vector_store
        except Exception as e:
            error_msg = f"Error loading index: {e}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg) from e

    def update_index(self, documents: List[Document]) -> Chroma:
        """Update the index with new documents, avoiding duplicates."""
        if not documents:
            error_msg = "Cannot update index with empty document list"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Load existing index if available
        if self.vector_store is None:
            try:
                logger.info("No active vector store, attempting to load existing one")
                self.load_index()
                logger.info("Successfully loaded existing index")
            except FileNotFoundError:
                logger.info(f"Index not found. Creating a new one with {len(documents)} documents")
                return self.create_vector_index(documents)
            except VectorStoreException as e:
                logger.info(f"Failed to load index: {e}. Creating a new one")
                return self.create_vector_index(documents)

        logger.info(f"Updating index with {len(documents)} documents")
        return self.create_vector_index(documents)

    def delete_collection(self):
        """Delete the entire collection."""
        try:
            if self.vector_store is None:
                self.load_index()

            if self.vector_store:
                logger.info(f"Deleting collection {self.collection_name}")
                self.vector_store.delete_collection()
                self.vector_store = None
            else:
                logger.warning("No vector store to delete")
        except Exception as e:
            error_msg = f"Error deleting collection: {e}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg) from e

    def create_retriever(self, search_type="similarity", k=4, **kwargs):
        """Create a retriever from the vector store.
        Performs straightforward vector similarity search
        Returns the k most similar documents to the query
        Simple and fast, good for basic retrieval tasks
        Supports different search types
        like "similarity" or "mmr" (Maximum Marginal Relevance)
        """
        try:
            if self.vector_store is None:
                logger.info("No active vector store, attempting to load existing one")
                self.load_index()

            logger.info(f"Creating {search_type} retriever with k={k}")
            # Create base retriever
            retriever = self.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs={"k": k, **kwargs}
            )
            return retriever
        except Exception as e:
            error_msg = f"Error creating retriever: {e}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg) from e

    def create_contextual_retriever(self, threshold=0.7, k=4):
        """Create a contextual compression retriever with embeddings filter.
        This is a more advanced retrieval mechanism that adds a filtering layer
        on top of the base retriever. Helps reduce irrelevant results by filtering out documents that aren't similar enough
        More selective and precise, potentially improving relevance
        """
        try:
            logger.info(f"Creating contextual retriever with threshold={threshold}, k={k}")
            base_retriever = self.create_retriever(k=k)

            # Create embeddings filter
            filter_compressor = EmbeddingsFilter(
                embeddings=self.embedding_model,
                similarity_threshold=threshold
            )

            # Create contextual compression retriever
            contextual_retriever = ContextualCompressionRetriever(
                base_compressor=filter_compressor,
                base_retriever=base_retriever
            )
            return contextual_retriever
        except Exception as e:
            error_msg = f"Error creating contextual retriever: {e}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg) from e

    def create_qa_chain(self, retriever=None):
        """Create a QA chain using the specified retriever or default retriever.
         create_qa_chain method creates a question-answering pipeline that ties together
         your retrieval system with language model capabilities
         Document retrieval: Finding the most relevant chunks
        Context processing: Preparing those chunks for the LLM
        Answer generation: Using the LLM to create a coherent answer from those chunks
        Source tracking: Keeping track of which documents contributed to the answer
        """
        try:
            if retriever is None:
                logger.info("No retriever provided, creating default retriever")
                retriever = self.create_retriever()

            logger.info("Creating QA chain")
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            return qa_chain
        except Exception as e:
            error_msg = f"Error creating QA chain: {e}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg) from e


# Example usage
if __name__ == "__main__":
    from app.ingestion_vector import DocumentProcessor

    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)

        logger.info("Starting document processing and indexing")

        # Process documents
        processor = DocumentProcessor()

        # Combine all documents
        all_documents = processor.ingest_all()
        logger.info(f"Processed {len(all_documents)} total documents")

        # Create indexes
        indexer = DocumentIndexer()
        vector_store = indexer.create_vector_index(all_documents)
        logger.info("Vector index created successfully")

        # Test query using different retrieval methods
        retriever = indexer.create_retriever()
        results = retriever.get_relevant_documents("What is Azure Kubernetes Service?")
        logger.info(f"Found {len(results)} relevant documents for 'Azure Kubernetes Service'")

        if len(results) == 0:
            results = retriever.get_relevant_documents("Federal Deductions?")
            logger.info(f"Found {len(results)} relevant documents for 'Federal Deductions?'")

        # Test QA chain
        qa_chain = indexer.create_qa_chain()
        response = qa_chain({"query": "What are the main topics in these documents?"})
        logger.info("QA chain response received")
        logger.info(response["result"])
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
