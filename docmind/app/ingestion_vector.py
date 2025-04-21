# app/ingestion.py
import json
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import bs4
from langchain.schema.document import Document
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion_from_networkx import NetworkXGraphManager
from app.ai_models import AVAILABLE_EMBEDDINGS
from app.customeExceptions import ChromaException, DocumentProcessingException
from app.config import CHROMA_STORE_DIR, DATA_DIR, METADATA_FILE, WEB_META_FILE, PROJECT_ROOT

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 20):
        """Initialize the document processor."""
        self.metadata_map = {}
        self.project_root = Path(__file__).parent.parent
        self.default_chunk_size = chunk_size
        self.default_chunk_overlap = chunk_overlap

        try:
            # Load metadata for documents
            with open(METADATA_FILE, "r") as f:
                self.metadata_map = json.load(f)
                logger.info(f"Loaded metadata for {len(self.metadata_map)} documents")

            # Load website data from JSON
            with open(WEB_META_FILE, "r") as f:
                self.website_data = json.load(f)
                logger.info(f"Loaded website data with {len(self.website_data.get('MoneyMarket', []))} entries")
        except FileNotFoundError as e:
            logger.error(f"Metadata file not found: {e}")
            raise DocumentProcessingException(f"Failed to load metadata: {e}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in metadata file: {e}")
            raise DocumentProcessingException(f"Failed to parse metadata: {e}") from e

    def _create_text_splitter(self, chunk_size: Optional[int] = None,
                              chunk_overlap: Optional[int] = None,
                              separators: List[str] = None) -> RecursiveCharacterTextSplitter:
        """Create a text splitter with specified parameters."""
        if separators is None:
            separators = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size if chunk_size else self.default_chunk_size,
            chunk_overlap=chunk_overlap if chunk_overlap else self.default_chunk_overlap,
            separators=separators,
            is_separator_regex=False,
            keep_separator=True,
            length_function=len,
            add_start_index=True
        )

    def _process_chunks(self, chunks: List[Document], source_path: str = None) -> List[Document]:
        """Process document chunks by adding IDs and metadata."""
        if not chunks:
            logger.warning("No chunks to process")
            return []

        chunks = self.calculate_document_id(chunks)

        # If source path provided, add appropriate metadata
        if source_path:
            file_metadata = self.get_metadata(source_path)
            for chunk in chunks:
                chunk.metadata.update(file_metadata)
                _source = Path(source_path).resolve().relative_to(PROJECT_ROOT).as_posix()
                chunk.metadata["source"] = _source
                chunk.metadata["file_path"] = source_path

        return chunks

    def ingest_web_pages(self, chunk_size: Optional[int] = None,
                         chunk_overlap: Optional[int] = None) -> List[Document]:
        """Ingest web pages from predefined configuration."""
        logger.info("Starting web page ingestion")
        total_chunks: List[Document] = []

        try:
            for item in self.website_data.get("MoneyMarket", []):
                url = item.get("Website URL")
                if not url:
                    logger.warning("Skipping web page with missing URL")
                    continue

                logger.info(f"Processing web page: {url}")

                metadata = {
                    "source": url,
                    "title": item.get("Page Title", ""),
                    "description": item.get("Description", ""),
                    "category": item.get("Category", ""),
                    "keywords": item.get("Keywords", ""),
                    "language": item.get("Language", "en")
                }

                # SoupStrainer for filtering content
                try:
                    content_strainer = bs4.SoupStrainer("div", attrs={"class": "content"})
                    loader_content = WebBaseLoader(url, bs_kwargs={"parse_only": content_strainer})
                    _documents = loader_content.load()

                    if _documents:
                        text_splitter = self._create_text_splitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        chunks = text_splitter.split_documents(_documents)
                        chunks = self.calculate_document_id(chunks)

                        for chunk in chunks:
                            chunk.metadata.update(metadata)
                        total_chunks.extend(chunks)
                        logger.info(f"Added {len(chunks)} chunks from {url}")
                    else:
                        logger.warning(f"No content extracted from {url}")
                except Exception as e:
                    logger.error(f"Error processing web page {url}: {e}")
                    continue

            logger.info(f"Completed web page ingestion. Total chunks: {len(total_chunks)}")
            return total_chunks
        except Exception as e:
            error_msg = f"Failed to ingest web pages: {e}"
            logger.error(error_msg)
            raise DocumentProcessingException(error_msg) from e

    def _load_documents(self, file_path: str, loader_class) -> List[Document]:
        """Load documents using the specified loader."""
        try:
            logger.info(f"Loading file: {file_path}")
            loader = loader_class(file_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {file_path}")

            for i, doc in enumerate(documents):
                logger.debug(f"Document {i} content length: {len(doc.page_content)}")

            return documents
        except Exception as e:
            error_msg = f"Error loading document {file_path}: {e}"
            logger.error(error_msg)
            raise DocumentProcessingException(error_msg) from e

    def _split_and_process_documents(self, documents: List[Document],
                                     file_path: str,
                                     chunk_size: Optional[int] = None,
                                     chunk_overlap: Optional[int] = None) -> List[Document]:
        """Split documents into chunks and process them."""
        if not documents:
            logger.warning(f"No documents to process from {file_path}")
            return []

        try:
            text_splitter = self._create_text_splitter(chunk_size, chunk_overlap)
            chunks = text_splitter.split_documents(documents)

            if not chunks:
                logger.warning(f"No chunks created from {file_path}")
                return []

            return self._process_chunks(chunks, file_path)
        except Exception as e:
            error_msg = f"Error splitting/processing documents from {file_path}: {e}"
            logger.error(error_msg)
            raise DocumentProcessingException(error_msg) from e

    def ingest_text_documents(self, directory: str = None,
                              chunk_size: Optional[int] = None,
                              chunk_overlap: Optional[int] = None) -> List[Document]:
        """Load and process text documents from a directory."""
        logger.info("Starting text document ingestion")
        all_chunks = []

        if directory is None:
            directory = DATA_DIR

        try:
            for filename in os.listdir(directory):
                if filename.lower().endswith(".txt"):
                    file_path = os.path.join(directory, filename)
                    try:
                        documents = self._load_documents(file_path, TextLoader)
                        chunks = self._split_and_process_documents(
                            documents,
                            file_path,
                            chunk_size,
                            chunk_overlap
                        )
                        all_chunks.extend(chunks)
                        logger.info(f"Processed {len(chunks)} chunks from {filename}")
                    except DocumentProcessingException as e:
                        logger.error(f"Skipping file {filename} due to error: {e}")
                        continue

            if not all_chunks:
                logger.warning("No text documents were successfully processed")

            subset = all_chunks[round(len(all_chunks) / 6) * 5:] if all_chunks else []
            logger.info(f"Completed text document ingestion. Total chunks: {len(subset)}")
            return subset
        except Exception as e:
            error_msg = f"Failed to ingest text documents: {e}"
            logger.error(error_msg)
            raise DocumentProcessingException(error_msg) from e

    def ingest_documents(self, directory: str = None,
                         chunk_size: Optional[int] = None,
                         chunk_overlap: Optional[int] = None) -> List[Document]:
        """Load and process PDF documents from a directory."""
        logger.info("Starting PDF document ingestion")
        all_chunks = []

        if directory is None:
            directory = DATA_DIR

        try:
            for filename in os.listdir(directory):
                if filename.lower().endswith(".pdf"):
                    file_path = os.path.join(directory, filename)
                    try:
                        documents = self._load_documents(file_path, PDFPlumberLoader)
                        chunks = self._split_and_process_documents(
                            documents,
                            file_path,
                            chunk_size,
                            chunk_overlap
                        )
                        all_chunks.extend(chunks)
                        logger.info(f"Processed {len(chunks)} chunks from {filename}")
                    except DocumentProcessingException as e:
                        logger.error(f"Skipping file {filename} due to error: {e}")
                        continue

            if not all_chunks:
                logger.warning("No PDF documents were successfully processed")

            subset = all_chunks[round(len(all_chunks) / 6) * 5:] if all_chunks else []
            logger.info(f"Completed PDF document ingestion. Total chunks: {len(subset)}")
            return subset
        except Exception as e:
            error_msg = f"Failed to ingest PDF documents: {e}"
            logger.error(error_msg)
            raise DocumentProcessingException(error_msg) from e

    def save_to_chroma(self, documents: List[Document]) -> Chroma:
        """Save documents to Chroma vector store."""
        if not documents:
            error_msg = "No documents provided to save to Chroma"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Saving {len(documents)} documents to Chroma")

        try:
            embedding_function = OpenAIEmbeddings(
                model=AVAILABLE_EMBEDDINGS["openai-small"]["model"],
                base_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY")
            )

            chroma_db = Chroma(
                persist_directory=CHROMA_STORE_DIR,
                embedding_function=embedding_function
            )

            # Get document IDs directly from metadata
            document_ids = [doc.metadata.get("id") for doc in documents]
            if None in document_ids:
                raise ValueError("Some documents are missing ID in metadata")

            # Get existing IDs from ChromaDB
            collection = chroma_db._collection
            existing_ids = set()

            if collection is None:
                logger.info("Collection not found. Creating a new one...")
                chroma_db = Chroma.from_documents(
                    documents=documents,
                    ids=document_ids,
                    persist_directory=CHROMA_STORE_DIR,
                    embedding=embedding_function,
                    collection_name="MyStockCollection"
                )
                logger.info(f"Created new collection with {len(documents)} documents")
            else:
                logger.info("Collection found. Checking for existing documents...")
                # Get existing IDs
                existing_results = collection.get(include=[])
                existing_ids = set(existing_results["ids"]) if existing_results["ids"] else set()

                # Filter out documents that already exist
                new_docs = []
                new_docs_ids = []
                for doc, doc_id in zip(documents, document_ids):
                    if doc_id not in existing_ids:
                        new_docs.append(doc)
                        new_docs_ids.append(doc_id)

                # Only add documents if there are new ones
                if new_docs:
                    chroma_db.add_documents(
                        documents=new_docs,
                        ids=new_docs_ids
                    )
                    logger.info(f"Added {len(new_docs)} new documents to ChromaDB. Skipped {len(documents) - len(new_docs)} existing documents.")
                else:
                    logger.info("No new documents to add.")

            return chroma_db
        except Exception as e:
            error_msg = f"Error saving documents to Chroma: {e}"
            logger.error(error_msg)
            raise ChromaException(error_msg) from e

    @staticmethod
    def read_text_file(file_path: str) -> str:
        """Reads the content of a text file."""
        logger.info(f"Reading text file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        except FileNotFoundError as e:
            error_msg = f"File not found at {file_path}"
            logger.error(error_msg)
            raise DocumentProcessingException(error_msg) from e
        except Exception as e:
            error_msg = f"Error reading file {file_path}: {e}"
            logger.error(error_msg)
            raise DocumentProcessingException(error_msg) from e

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get metadata for a file from the metadata map."""
        try:
            # Get relative path from project root
            relative_path = Path(file_path).resolve().relative_to(PROJECT_ROOT).as_posix()
            metadata = self.metadata_map.get(relative_path, {})
            logger.debug(f"Retrieved metadata for {relative_path}: {metadata}")
            return metadata
        except Exception as e:
            logger.warning(f"Error retrieving metadata for {file_path}: {e}")
            return {}

    def calculate_document_id(self, chunks: List[Document]) -> List[Document]:
        """Calculate unique IDs for document chunks."""
        if not chunks:
            return []

        logger.info(f"Calculating IDs for {len(chunks)} chunks")
        last_page_id = None
        current_page_index = 0
        current_page = 0

        for i, chunk in enumerate(chunks):
            # Safety check if metadata exists
            if not hasattr(chunk, 'metadata'):
                logger.warning(f"Chunk {i} has no metadata attribute")
                chunk.metadata = {}

            source = chunk.metadata.get("source", "")
            page = chunk.metadata.get("page", "")

            if page == "":
                current_page += 1
                page = current_page
                logger.debug(f"Empty page tag in metadata in chunk {i} incremented to {current_page}")

            # Ensure we have non-empty values
            if source == "":
                logger.debug(f"Empty source tag in metadata in chunk {i}")
                source = "unknown"

            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_page_index += 1
            else:
                current_page_index = 0

            chunk_id = f"{current_page_id}:{current_page_index}"
            chunk.metadata["id"] = chunk_id
            chunk.id = chunk_id
            logger.debug(f"Set ID for chunk {i}: {chunk_id}")

            last_page_id = current_page_id

        return chunks

    def ingest_all(self):
        # Process PDF documents
        try:
            documents = self.ingest_documents()
            logger.info(f"PDF Document Ingestion complete. Created {len(documents)} documents.")
        except DocumentProcessingException as e:
            logger.error(f"Failed to process PDF documents: {e}")
            documents = []
        # Process text documents
        try:
            txt_documents = self.ingest_text_documents()
            logger.info(f"Text Document Ingestion complete. Created {len(txt_documents)} documents.")
        except DocumentProcessingException as e:
            logger.error(f"Failed to process text documents: {e}")
            txt_documents = []
        # Process web documents
        try:
            web_documents = self.ingest_web_pages()
            logger.info(f"Web Document Ingestion complete. Created {len(web_documents)} documents.")
        except DocumentProcessingException as e:
            logger.error(f"Failed to process web documents: {e}")
            web_documents = []
        # Combine all documents
        return documents + txt_documents + web_documents


# Example usage
if __name__ == "__main__":
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        logger.info("Starting document processing")
        processor = DocumentProcessor()
        all_documents = processor.ingest_all()

        logger.info(f"Total documents: {len(all_documents)}")

        if all_documents:
            try:
                chroma_db = processor.save_to_chroma(all_documents)

                # Test query
                query = "Maximizing contributions to 401k"
                logger.info(f"Testing query: '{query}'")

                results = chroma_db.similarity_search_with_relevance_scores(
                    query,
                    k=5
                )

                for i, (doc, score) in enumerate(results):
                    logger.info(f"Result {i + 1}: Score: {score}")
                    logger.info(f"Content: {doc.page_content[:100]}...")
            except ChromaException as e:
                logger.error(f"Failed to save to Chroma: {e}")

                # For graph creation - we need to process each document type appropriately
                try:
                    networkx = NetworkXGraphManager()

                    # Process documents for the graph
                    # Note: This should work with your existing implementation if modified to handle lists
                    if networkx.process_documents(all_documents):
                        print("Graph created successfully")

                        # Get graph retriever for use with LightRAG
                        graph_retriever = networkx.get_llama_graph_retriever()
                        print("Graph retriever created successfully")

                        # This retriever can now be passed to LightRAGRetriever
                    else:
                        print("Graph creation failed")
                except Exception as e:
                    logger.error(f"Error creating knowledge graph: {e}", exc_info=True)
        else:
            logger.warning("No documents to save to Chroma")
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {e}", exc_info=True)
