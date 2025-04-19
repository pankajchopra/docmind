from langchain.graphs import NetworkxEntityGraph
from langchain_experimental.graph_retrievers.entity import EntityGraphRetriever
from langchain.schema import Document


class KnowledgeGraphRetriever:
    def __init__(self, graph, llm, retrieval_query_template=None):
        """Initialize with NetworkX graph and LLM."""
        self.graph = graph
        self.llm = llm

        # Create entity graph retriever
        self.entity_retriever = EntityGraphRetriever(
            graph=graph,
            llm=llm,
            retrieval_query_template=retrieval_query_template or "Given the query: {query}, what entities might be relevant?",
            max_entities_per_query=5
        )

    def get_relevant_documents(self, query_str):
        """Retrieve documents based on graph relationships."""
        return self.entity_retriever.get_relevant_documents(query_str)