# app/agents.py
from typing import List, Dict, Any, Optional
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from functools import wraps
import datetime
import wikipedia
from flask import Flask, request, jsonify


# Tool decorator for simplifying tool creation
def tool(name, description):
    """Decorator to create a LangChain Tool."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Create LangChain tool
        wrapper.tool = Tool(
            name=name,
            description=description,
            func=func
        )
        return wrapper

    return decorator


class DocMindAgent:
    """Main agent for the DocMind system using LightRAG retrieval."""

    def __init__(self, indexer, retriever):
        """Initialize with indexer and retriever components."""
        self.indexer = indexer
        self.retriever = retriever
        self.llm = self.indexer.llm if hasattr(self.indexer, 'llm') else ChatOpenAI(temperature=0)

        # Initialize tools
        self.tools = self._initialize_tools()

        # Create the agent
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    def _initialize_tools(self):
        """Initialize all tools for the agent."""
        tools = []

        # Search documents tool
        @tool("search_documents", "Search documents using the specified retrieval type")
        def search_documents(query: str, retrieval_type: str = "hybrid") -> str:
            """Search documents with specified retrieval method.

            Args:
                query: The search query
                retrieval_type: The type of retrieval ('hybrid', 'graph', or 'transformed')

            Returns:
                Search results
            """
            qa_chain = self.retriever.create_advanced_retrieval_pipeline(query, retrieval_type)
            response = qa_chain({"query": query})

            # Format source documents for easier readability
            sources = []
            if "source_documents" in response:
                for i, doc in enumerate(response["source_documents"]):
                    sources.append(f"Source {i + 1}: {doc.metadata.get('source', 'Unknown')}")

            return f"{response['result']}\n\nSources: {', '.join(sources)}"

        tools.append(search_documents.tool)

        # Knowledge graph search tool
        @tool("search_knowledge_graph", "Search the knowledge graph for entities and relationships")
        def search_knowledge_graph(query: str) -> str:
            """Search knowledge graph for entities related to query.

            Args:
                query: The search query

            Returns:
                Knowledge graph query results
            """
            if not hasattr(self.retriever, 'kg_retriever') or not self.retriever.kg_retriever:
                return "Knowledge graph not initialized"

            try:
                docs = self.retriever.graph_augmented_retrieval(query)

                # Format results
                results = []
                for doc in docs:
                    entity = doc.metadata.get("entity", "Unknown")
                    results.append(f"Entity: {entity} - {doc.page_content[:100]}...")

                return "\n".join(results) if results else "No results found in knowledge graph"
            except Exception as e:
                return f"Error searching knowledge graph: {str(e)}"

        tools.append(search_knowledge_graph.tool)

        # Wikipedia search tool
        @tool("search_wikipedia", "Search Wikipedia for information")
        def search_wikipedia(query: str, num_results: int = 3) -> str:
            """Search Wikipedia for the given query.

            Args:
                query: The search query
                num_results: Number of results to return (default: 3)

            Returns:
                Wikipedia search results
            """
            try:
                search_results = wikipedia.search(query, results=num_results)
                results = []

                for title in search_results:
                    try:
                        summary = wikipedia.summary(title, sentences=3, auto_suggest=False)
                        results.append(f"### {title}\n{summary}")
                    except Exception as e:
                        results.append(f"### {title}\nError: {str(e)}")

                return "\n\n".join(results) if results else "No Wikipedia results found"
            except Exception as e:
                return f"Wikipedia search error: {str(e)}"

        tools.append(search_wikipedia.tool)

        # Current time tool
        @tool("get_current_time", "Get the current date and time")
        def get_current_time() -> str:
            """Get the current date and time."""
            current_time = datetime.datetime.now()
            return (
                f"Current date: {current_time.strftime('%Y-%m-%d')}\n"
                f"Current time: {current_time.strftime('%H:%M:%S')}\n"
                f"Timezone: {datetime.datetime.now().astimezone().tzname()}"
            )

        tools.append(get_current_time.tool)

        # Summarize document tool
        @tool("summarize_document", "Summarize a specific document")
        def summarize_document(document_id: str, max_length: int = 200) -> str:
            """Summarize a specific document by ID.This would normally fetch the document by ID from a database


            Args:
                document_id: The document identifier
                max_length: Maximum summary length in words (default: 200)

            Returns:
                Document summary
            """
            if not hasattr(self.retriever, 'vector_retriever') or not self.retriever.vector_retriever:
                return "Vector index not initialized"

            query = f"Generate a concise summary (max {max_length} words) of document {document_id}"

            # First try to find the document by ID
            if hasattr(self.indexer, 'vector_store') and self.indexer.vector_store:
                # Create a query chain
                try:
                    qa_chain = self.retriever.create_advanced_retrieval_pipeline(
                        f"document_id:{document_id}", "hybrid"
                    )
                    response = qa_chain({"query": query})

                    return f"Summary of document {document_id}:\n{response['result']}"
                except Exception as e:
                    return f"Error summarizing document: {str(e)}"
            else:
                return f"Could not find document with ID {document_id}"

        tools.append(summarize_document.tool)

        return tools

    def chat(self, message: str) -> str:
        """Process a user message using the agent."""
        try:
            response = self.agent.run(message)
            return response
        except Exception as e:
            return f"Error processing your request: {str(e)}\n\nPlease try rephrasing your question."


# Create RESTful API for agent
def create_agent_api(app, indexer, retriever):
    """Create Flask endpoints for agent interactions."""

    @app.route('/api/agent/chat', methods=['POST'])
    def agent_chat():
        """Process chat messages using the agent."""
        data = request.json
        message = data.get('message', '')

        if not message:
            return jsonify({"error": "Message is required"}), 400

        try:
            # Create agent if not already created
            agent = DocMindAgent(indexer, retriever)

            # Get response from agent
            response = agent.chat(message)

            return jsonify({
                "response": response
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/agent/tools', methods=['GET'])
    def get_tools():
        """Get available agent tools."""
        # Create temporary agent to get tools
        agent = DocMindAgent(indexer, retriever)

        tools = []
        for tool in agent.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description
            })

        return jsonify({"tools": tools})

    return app


# Example usage
if __name__ == "__main__":
    from app.indexing import DocumentIndexer
    from app.retrieval import LightRAGRetriever
    from app.ingestion import DocumentProcessor
    from flask import Flask

    # Process documents
    processor = DocumentProcessor()
    documents = processor.ingest_documents()
    web_documents = processor.ingest_web_pages()

    # Combine all documents
    all_documents = documents + web_documents

    # Create indexer
    indexer = DocumentIndexer()
    vector_store = indexer.create_vector_index(documents)

    # Create retriever
    retriever = LightRAGRetriever(vector_store, indexer.llm, documents)

    # Create agent
    agent = DocMindAgent(indexer, retriever)

    # Test the agent
    response = agent.chat("I need information about advanced RAG techniques.")
    print(response)

    # Create Flask app
    app = Flask(__name__)
    app = create_agent_api(app, indexer, retriever)

    # Run server
    app.run(debug=True, host='0.0.0.0', port=5000)