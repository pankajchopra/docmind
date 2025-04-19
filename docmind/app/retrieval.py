# app/retrieval.py
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema.retriever import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from flask import Flask, request, jsonify


class LightRAGRetriever:
    """Enhanced retrieval system using LightRAG techniques with LangChain."""

    def __init__(self, vector_store=None, llm=None, documents=None):
        """Initialize with vector store and LLM."""
        self.llm = llm
        self.vector_store = vector_store
        self.documents = documents
        self.vector_retriever = None
        self.bm25_retriever = None
        self.kg_retriever = None  # Knowledge graph retriever

        # Initialize vector retriever if vector store is provided
        if vector_store:
            self.vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        # Initialize BM25 if documents provided
        if documents:
            self.setup_bm25_retriever(documents)

    def setup_bm25_retriever(self, documents):
        """Set up a BM25 retriever from documents."""
        if not documents:
            return None

        self.bm25_retriever = BM25Retriever.from_documents(
            documents,
            k=5
        )
        return self.bm25_retriever

    def setup_knowledge_graph_retriever(self, kg_index):
        """Set up knowledge graph retriever."""
        if kg_index:
            try:
                self.kg_retriever = kg_index.as_retriever(retriever_mode="keyword")
                return self.kg_retriever
            except Exception as e:
                print(f"Error setting up knowledge graph retriever: {e}")
        return None

    def hybrid_retrieval(self, query_str: str) -> List[Document]:
        """Perform hybrid retrieval combining vector and keyword search."""
        if not self.vector_retriever:
            raise ValueError("Vector retriever must be initialized")

        try:
            # Get vector results
            vector_docs = self.vector_retriever.get_relevant_documents(query_str)
            all_docs = vector_docs

            # Add BM25 results if available
            if self.bm25_retriever:
                bm25_docs = self.bm25_retriever.get_relevant_documents(query_str)
                # Combine and deduplicate
                doc_contents = set(doc.page_content for doc in all_docs)

                for doc in bm25_docs:
                    if doc.page_content not in doc_contents:
                        all_docs.append(doc)
                        doc_contents.add(doc.page_content)

            # Add knowledge graph results if available
            if self.kg_retriever:
                try:
                    kg_docs = self.kg_retriever.get_relevant_documents(query_str)
                    doc_contents = set(doc.page_content for doc in all_docs)

                    for doc in kg_docs:
                        if doc.page_content not in doc_contents:
                            all_docs.append(doc)
                            doc_contents.add(doc.page_content)
                except Exception as e:
                    print(f"Error retrieving from knowledge graph: {e}")

            return all_docs
        except Exception as e:
            print(f"Error in hybrid retrieval: {e}")
            # Return whatever we have or an empty list
            return vector_docs if 'vector_docs' in locals() else []

    def graph_augmented_retrieval(self, query_str: str) -> List[Document]:
        """Perform graph-augmented retrieval prioritizing knowledge graph results."""
        if not self.kg_retriever:
            if self.vector_retriever:
                # Fall back to vector retrieval if KG not available
                return self.vector_retriever.get_relevant_documents(query_str)
            raise ValueError("Neither knowledge graph nor vector retriever is initialized")

        try:
            # Get results from knowledge graph
            kg_docs = self.kg_retriever.get_relevant_documents(query_str)

            # Get vector results if available
            vector_docs = []
            if self.vector_retriever:
                vector_docs = self.vector_retriever.get_relevant_documents(query_str)

            # Combine and deduplicate, prioritizing knowledge graph results
            all_docs = kg_docs
            doc_contents = set(doc.page_content for doc in all_docs)

            for doc in vector_docs:
                if doc.page_content not in doc_contents:
                    all_docs.append(doc)
                    doc_contents.add(doc.page_content)

            return all_docs
        except Exception as e:
            print(f"Error in graph augmented retrieval: {e}")
            # Fall back to vector retrieval
            if self.vector_retriever:
                return self.vector_retriever.get_relevant_documents(query_str)
            return []

    def query_transformation_retrieval(self, query_str: str) -> Tuple[str, List[Document]]:
        """Use LLM to transform the query for improved retrieval results."""
        if not self.vector_retriever:
            raise ValueError("Vector retriever must be initialized")

        if not self.llm:
            # Fall back to regular retrieval if no LLM
            docs = self.hybrid_retrieval(query_str)
            return query_str, docs

        try:
            # Use LLM to transform the query
            transformed_query = self.llm.invoke(
                f"Please reformulate the following question into a more detailed and specific question that would help with document retrieval: {query_str}"
            )

            # Check if the response is a string or an object with content attribute
            if hasattr(transformed_query, 'content'):
                transformed_query = transformed_query.content

            print(f"Original query: {query_str}")
            print(f"Transformed query: {transformed_query}")

            # Use transformed query for retrieval
            docs = self.hybrid_retrieval(transformed_query)

            return transformed_query, docs
        except Exception as e:
            print(f"Error in query transformation: {e}")
            # Fall back to regular retrieval
            docs = self.hybrid_retrieval(query_str)
            return query_str, docs

    def reranking_retrieval(self, query_str: str, initial_results: List[Document]) -> List[Document]:
        """Rerank initial retrieval results using embedding similarity."""
        if not initial_results:
            return []

        if not self.vector_store or not hasattr(self.vector_store, "embeddings"):
            return initial_results  # Return original results if embeddings not available

        try:
            # Create embedding filter for reranking
            embeddings = self.vector_store.embeddings
            reranker = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7)

            # Apply reranking
            reranked_docs = reranker.compress_documents(initial_results, query=query_str)
            return reranked_docs
        except Exception as e:
            print(f"Error in reranking: {e}")
            return initial_results

    def create_advanced_retrieval_pipeline(self, query_str: str, retrieval_type: str = "hybrid"):
        """Create an advanced retrieval pipeline based on specified retrieval type."""
        if not query_str:
            raise ValueError("Query string cannot be empty")

        try:
            # Get documents based on retrieval type
            if retrieval_type == "hybrid":
                docs = self.hybrid_retrieval(query_str)
            elif retrieval_type == "graph":
                docs = self.graph_augmented_retrieval(query_str)
            elif retrieval_type == "transformed":
                _, docs = self.query_transformation_retrieval(query_str)
            else:
                # Default to hybrid if unknown type
                print(f"Unknown retrieval type: {retrieval_type}, using hybrid instead")
                docs = self.hybrid_retrieval(query_str)

            # Apply reranking if we have more than one document
            if len(docs) > 1 and self.vector_store and hasattr(self.vector_store, "embeddings"):
                docs = self.reranking_retrieval(query_str, docs)

            # Create a custom retriever that returns our reranked docs
            class CustomRetriever(BaseRetriever):
                def __init__(self, docs):
                    self.docs = docs
                    super().__init__()

                def _get_relevant_documents(self, query):
                    return self.docs

                async def _aget_relevant_documents(self, query):
                    return self.docs

            # Create custom retriever with our reranked docs
            custom_retriever = CustomRetriever(docs)

            # Create QA chain with our custom retriever
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=custom_retriever,
                return_source_documents=True
            )

            return qa_chain
        except Exception as e:
            print(f"Error creating retrieval pipeline: {e}")

            # Create a fallback QA chain with empty retriever
            class EmptyRetriever(BaseRetriever):
                def _get_relevant_documents(self, query):
                    return []

                async def _aget_relevant_documents(self, query):
                    return []

            empty_retriever = EmptyRetriever()

            return RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=empty_retriever,
                return_source_documents=True
            )


# RESTful API implementation for LightRAG retriever
def create_retrieval_api(app, indexer, retriever):
    """Create Flask endpoints for retrieval operations."""

    @app.route('/api/search', methods=['POST'])
    def search():
        """Search documents with specified retrieval type."""
        data = request.json
        query = data.get('query', '')
        retrieval_type = data.get('retrieval_type', 'hybrid')

        if not query:
            return jsonify({"error": "Query is required"}), 400

        try:
            # Create retrieval pipeline
            qa_chain = retriever.create_advanced_retrieval_pipeline(query, retrieval_type)

            # Execute query
            response = qa_chain({"query": query})

            # Format source documents for response
            source_docs = []
            if "source_documents" in response:
                for doc in response["source_documents"]:
                    source_docs.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })

            return jsonify({
                "answer": response["result"],
                "source_documents": source_docs
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/retrieval-types', methods=['GET'])
    def get_retrieval_types():
        """Get available retrieval types."""
        return jsonify({
            "retrieval_types": [
                {
                    "id": "hybrid",
                    "name": "Hybrid Search",
                    "description": "Combines vector similarity and keyword search"
                },
                {
                    "id": "graph",
                    "name": "Knowledge Graph",
                    "description": "Prioritizes knowledge graph connections"
                },
                {
                    "id": "transformed",
                    "name": "Query Transformation",
                    "description": "Uses LLM to improve the query before search"
                }
            ]
        })

    return app


# Example usage
if __name__ == "__main__":
    from app.indexing import DocumentIndexer
    from app.ingestion import DocumentProcessor
    from flask import Flask
    import os

    # Configure Flask
    app = Flask(__name__)

    # Process documents
    processor = DocumentProcessor()
    pdf_documents = processor.ingest_documents()
    txt_documents = processor.ingest_text_documents()
    web_documents = processor.ingest_web_pages()

    documents = pdf_documents + txt_documents + web_documents

    # Create indexer
    indexer = DocumentIndexer()
    vector_store = indexer.create_vector_index(documents)

    # Create LLM
    llm = ChatOpenAI(temperature=0)

    # Create retriever
    retriever = LightRAGRetriever(vector_store, llm, documents)

    # Create API
    app = create_retrieval_api(app, indexer, retriever)

    # Run server
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))