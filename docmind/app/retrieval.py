# app/retrieval.py
from typing import List, Dict, Any, Optional, Tuple
from llama_index.core.retrievers import VectorIndexRetriever, KGTableRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BM25Retriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_pipeline import QueryPipeline
from app.config import RETRIEVAL_TOP_K

class LightRAGRetriever:
    """Enhanced retrieval using LightRAG techniques."""
    
    def __init__(self, indexer):
        """Initialize with document indexer."""
        self.indexer = indexer
        self.vector_retriever = None
        self.kg_retriever = None
        self.bm25_retriever = None
        
        # Initialize retrievers if indexes exist
        if self.indexer.vector_index:
            self.vector_retriever = VectorIndexRetriever(
                index=self.indexer.vector_index,
                similarity_top_k=RETRIEVAL_TOP_K
            )
            
        if self.indexer.kg_index:
            self.kg_retriever = KGTableRetriever(
                index=self.indexer.kg_index,
                similarity_top_k=RETRIEVAL_TOP_K
            )
    
    def setup_bm25_retriever(self, nodes):
        """Set up a BM25 retriever from nodes."""
        documents = nodes_to_documents(nodes)
        self.bm25_retriever = BM25Retriever.from_defaults(
            documents=documents,
            similarity_top_k=RETRIEVAL_TOP_K
        )
        return self.bm25_retriever
    
    def hybrid_retrieval(self, query_str: str) -> List[Any]:
        """Perform hybrid retrieval combining vector and keyword search."""
        if not self.vector_retriever or not self.bm25_retriever:
            raise ValueError("Both vector and BM25 retrievers must be initialized")
        
        # Get results from both retrievers
        vector_nodes = self.vector_retriever.retrieve(query_str)
        bm25_nodes = self.bm25_retriever.retrieve(query_str)
        
        # Combine and deduplicate nodes
        all_nodes = []
        node_ids = set()
        
        for node in vector_nodes + bm25_nodes:
            if node.node.node_id not in node_ids:
                all_nodes.append(node)
                node_ids.add(node.node.node_id)
        
        return all_nodes
    
    def graph_augmented_retrieval(self, query_str: str) -> List[Any]:
        """Perform graph-augmented retrieval."""
        if not self.vector_retriever or not self.kg_retriever:
            raise ValueError("Both vector and knowledge graph retrievers must be initialized")
        
        # Get direct matches from vector index
        vector_nodes = self.vector_retriever.retrieve(query_str)
        
        # Get related nodes from knowledge graph
        kg_nodes = self.kg_retriever.retrieve(query_str)
        
        # Combine and deduplicate
        all_nodes = []
        node_ids = set()
        
        # Prioritize knowledge graph results but include both
        for node in kg_nodes + vector_nodes:
            if node.node.node_id not in node_ids:
                all_nodes.append(node)
                node_ids.add(node.node.node_id)
        
        return all_nodes

    def query_transformation_retrieval(self, query_str: str) -> Tuple[str, List[Any]]:
        """Use query transformation to improve retrieval."""
        if not self.vector_retriever:
            raise ValueError("Vector retriever must be initialized")
        
        # Use LLM to transform the query
        llm = self.indexer.llm
        response = llm.complete(
            f"Please reformulate the following question into a more detailed and specific question that would help with document retrieval: {query_str}"
        )
        transformed_query = response.text
        
        print(f"Original query: {query_str}")
        print(f"Transformed query: {transformed_query}")
        
        # Use transformed query for retrieval
        nodes = self.vector_retriever.retrieve(transformed_query)
        
        return transformed_query, nodes
    
    def reranking_retrieval(self, query_str: str, initial_results: List[Any]) -> List[Any]:
        """Rerank initial retrieval results."""
        # Create a reranker using a similarity postprocessor
        reranker = SimilarityPostprocessor(similarity_cutoff=0.7)
        
        # Rerank the nodes based on relevance
        reranked_nodes = reranker.postprocess_nodes(
            initial_results,
            query_str=query_str
        )
        
        return reranked_nodes
    
    def create_advanced_retrieval_pipeline(self, query_str: str, retrieval_type: str = "hybrid"):
        """Create an advanced retrieval pipeline."""
        # Get nodes based on retrieval type
        if retrieval_type == "hybrid":
            nodes = self.hybrid_retrieval(query_str)
        elif retrieval_type == "graph":
            nodes = self.graph_augmented_retrieval(query_str)
        elif retrieval_type == "transformed":
            _, nodes = self.query_transformation_retrieval(query_str)
        else:
            raise ValueError(f"Unknown retrieval type: {retrieval_type}")
        
        # Apply reranking
        reranked_nodes = self.reranking_retrieval(query_str, nodes)
        
        # Create query engine
        query_engine = RetrieverQueryEngine.from_args(
            retriever=None,  # We're providing nodes directly
            llm=self.indexer.llm,
            node_postprocessors=[],
        )
        
        # Manually set the retrieved nodes
        query_engine._retriever = lambda _: reranked_nodes
        
        return query_engine


# Example usage
if __name__ == "__main__":
    from app.ingestion import DocumentProcessor
    from app.indexing import DocumentIndexer
    from app.utils import nodes_to_documents
    
    # Process documents
    processor = DocumentProcessor()
    nodes = processor.ingest_documents()
    
    # Create indexes
    indexer = DocumentIndexer()
    indexer.create_vector_index(nodes)
    indexer.create_knowledge_graph(nodes)
    
    # Create enhanced retriever
    retriever = LightRAGRetriever(indexer)
    retriever.setup_bm25_retriever(nodes)
    
    # Test hybrid retrieval
    query = "What are the key techniques for document retrieval?"
    hybrid_engine = retriever.create_advanced_retrieval_pipeline(query, "hybrid")
    response = hybrid_engine.query(query)
    print("Hybrid Retrieval Response:", response)
    
    # Test graph-augmented retrieval
    graph_engine = retriever.create_advanced_retrieval_pipeline(query, "graph")
    response = graph_engine.query(query)
    print("Graph-Augmented Retrieval Response:", response)