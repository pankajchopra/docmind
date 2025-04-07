# app/agents.py
from typing import List, Dict, Any, Optional
from llama_index.core.tools import FunctionTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from functools import wraps
import datetime
import wikipedia

# Tool decorator for simplifying tool creation
def tool(func):
    """Decorator to create a FunctionTool from a Python function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    # Create function tool
    function_tool = FunctionTool.from_defaults(fn=func)
    wrapper.tool = function_tool
    return wrapper

class DocMindAgent:
    """Main agent for the DocMind system."""
    
    def __init__(self, indexer, retriever):
        """Initialize with indexer and retriever components."""
        self.indexer = indexer
        self.retriever = retriever
        self.llm = self.indexer.llm
        
        # All available tools
        self.tools = [
            self.search_documents.tool,
            self.search_knowledge_graph.tool,
            self.search_wikipedia.tool,
            self.get_current_time.tool,
            self.summarize_document.tool
        ]
        
        # Create the agent
        self.agent = OpenAIAgent.from_tools(
            self.tools,
            llm=self.llm,
            verbose=True,
            system_prompt="""You are DocMind, an intelligent document assistant that helps users find and understand information.
            When given a task, carefully plan your approach using the available tools.
            Break down complex queries into smaller steps, and execute them sequentially.
            First search for relevant information, then analyze it, and finally synthesize your findings.
            Always show your reasoning and cite your sources."""
        )
    
    @tool
    def search_documents(self, query: str, retrieval_type: str = "hybrid") -> Dict[str, Any]:
        """Search documents using the specified retrieval type.
        
        Args:
            query: The search query
            retrieval_type: The type of retrieval to use ('hybrid', 'graph', or 'transformed')
            
        Returns:
            The search results
        """
        query_engine = self.retriever.create_advanced_retrieval_pipeline(query, retrieval_type)
        response = query_engine.query(query)
        
        return {
            "response": str(response),
            "source_nodes": [
                {
                    "text": node.node.text,
                    "metadata": node.node.metadata,
                    "score": node.score if hasattr(node, "score") else None
                }
                for node in getattr(response, "source_nodes", [])
            ]
        }
    
    @tool
    def search_knowledge_graph(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search the knowledge graph for entities and relationships.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            Knowledge graph query results
        """
        if not self.indexer.kg_index:
            return {"error": "Knowledge graph not initialized"}
        
        query_engine = self.indexer.kg_index.as_query_engine(
            response_mode="tree",
            verbose=True
        )
        response = query_engine.query(query)
        
        return {
            "response": str(response),
            "entities": [node.node.metadata.get("entity", "Unknown") 
                        for node in getattr(response, "source_nodes", [])[:max_results]]
        }
    
    @tool
    def search_wikipedia(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        """Search Wikipedia for the given query.
        
        Args:
            query: The search query
            num_results: The number of results to return
            
        Returns:
            Wikipedia search results
        """
        try:
            search_results = wikipedia.search(query, results=num_results)
            results = []
            
            for title in search_results:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    summary = wikipedia.summary(title, sentences=3, auto_suggest=False)
                    results.append({
                        "title": title,
                        "url": page.url,
                        "summary": summary
                    })
                except Exception as e:
                    results.append({
                        "title": title,
                        "error": str(e)
                    })
                    
            return {"results": results}
        except Exception as e:
            return {"error": str(e)}
    
    @tool
    def get_current_time(self) -> Dict[str, str]:
        """Get the current date and time."""
        current_time = datetime.datetime.now()
        return {
            "date": current_time.strftime("%Y-%m-%d"),
            "time": current_time.strftime("%H:%M:%S"),
            "timezone": datetime.datetime.now().astimezone().tzname(),
            "iso_format": current_time.isoformat()
        }
    
    @tool
    def summarize_document(self, document_id: str, max_length: int = 200) -> Dict[str, Any]:
        """Summarize a specific document by ID.
        
        Args:
            document_id: The ID of the document to summarize
            max_length: Maximum length of the summary in words
            
        Returns:
            A summary of the document
        """
        # This would normally fetch the document by ID from a database
        # For this example, we'll use a placeholder
        if not self.indexer.vector_index:
            return {"error": "Vector index not initialized"}
        
        # In real implementation, you'd retrieve the document by ID
        # For now, we'll use a query to find relevant content
        query_engine = self.indexer.vector_index.as_query_engine()
        response = query_engine.query(
            f"Generate a concise summary (max {max_length} words) of document {document_id}"
        )
        
        return {
            "document_id": document_id,
            "summary": str(response),
            "max_length": max_length
        }
    
    def chat(self, message: str) -> str:
        """Process a user message using the agent."""
        response = self.agent.chat(message)
        return str(response)


# Example usage
if __name__ == "__main__":
    from app.ingestion import DocumentProcessor
    from app.indexing import DocumentIndexer
    
    # Process documents
    processor = DocumentProcessor()
    nodes = processor.ingest_documents()
    
    # Create indexes
    indexer = DocumentIndexer()
    indexer.create_vector_index(nodes)
    indexer.create_knowledge_graph(nodes)
    
    # Create retriever
    from app.retrieval import LightRAGRetriever
    retriever = LightRAGRetriever(indexer)
    retriever.setup_bm25_retriever(nodes)
    
    # Create agent
    agent = DocMindAgent(indexer, retriever)
    
    # Test the agent
    response = agent.chat(
        "I need information about advanced RAG techniques. Can you search our documents and summarize what you find?"
    )
    print(response)