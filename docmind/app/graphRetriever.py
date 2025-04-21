from langchain_core.documents import Document


class GraphRetriever:
    """A retriever that uses a NetworkX graph to find relevant information."""

    def __init__(self, nx_graph):
        """Initialize with a NetworkX graph."""
        self.nx_graph = nx_graph

    def retrieve(self, query, k=5):
        """Retrieve information from the graph based on the query.

        Args:
            query: The query string
            k: Number of results to return

        Returns:
            List of document objects with relevant information
        """
        # Extract potential entities from the query
        entities = self._extract_entities(query)

        results = []
        for entity in entities:
            if entity in self.nx_graph:
                # Get neighboring nodes and their relationships
                neighbors = self.nx_graph.neighbors(entity)
                for neighbor in neighbors:
                    # Get edge data (relationships)
                    edges = self.nx_graph.get_edge_data(entity, neighbor)

                    # For each relationship type
                    for key, edge_data in edges.items():
                        rel_type = edge_data.get('type', 'related_to')
                        doc_id = edge_data.get('document_id', 'unknown')

                        # Create a document-like object
                        doc = Document(
                            page_content=f"{entity} {rel_type} {neighbor}",
                            metadata={
                                "entity": entity,
                                "relation": rel_type,
                                "target": neighbor,
                                "document_id": doc_id
                            }
                        )
                        results.append(doc)

                        if len(results) >= k:
                            return results

        return results

    def _extract_entities(self, query):
        """Extract potential entities from the query.

        This is a simple implementation that should be enhanced with NLP techniques.
        """
        # This is a placeholder - in a real implementation, you'd use NER
        # For now, just return all nodes that partially match the query
        potential_entities = []
        for node in self.nx_graph.nodes():
            if isinstance(node, str) and any(term.lower() in node.lower() for term in query.split()):
                potential_entities.append(node)

        return potential_entities