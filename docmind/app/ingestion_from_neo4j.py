# kg_neo4j.py
from kg_base import BaseGraphManager, logger, LangchainDocument

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, MAX_TRIPLETS_PER_CHUNK, GRAPH_VISUALIZATIONS_DIR
from langchain_community.graphs import Neo4jGraph
from llama_index.core import StorageContext
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.graph_stores import Neo4jGraphStore

import matplotlib.pyplot as plt
import networkx as nx
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import datetime
import os


class Neo4jGraphManager(BaseGraphManager):
    """Manager class for Neo4j graph storage."""

    def __init__(self):
        """Initialize Neo4j graph storage."""
        super().__init__()

        # Set up the Neo4j graph connections
        self.langchain_graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD
        )

        self.llama_graph_store = Neo4jGraphStore(
            uri=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD
        )

        logger.info("Initialized Neo4j graph storage")

    def delete_graph_elements_for_doc(self, doc_id: str):
        """Delete existing graph elements for a document before updating."""
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        try:
            with driver.session() as session:
                query = """
                MATCH (n)
                WHERE n.document_id = $doc_id
                DETACH DELETE n
                """
                session.run(query, doc_id=doc_id)
                logger.info(f"Deleted existing Neo4j graph elements for document '{doc_id}'")
        except ServiceUnavailable as e:
            logger.error(f"Error connecting to Neo4j: {e}")
        finally:
            driver.close()

    def process_document_for_graph(self, langchain_document: LangchainDocument):
        """Process a document for graph storage using LlamaIndex KnowledgeGraphIndex."""
        doc_id = langchain_document.metadata.get("source")
        if not doc_id:
            logger.warning("LangChain Document missing 'source' metadata. Skipping graph processing.")
            return False

        current_hash = self.get_document_hash(langchain_document)

        # Process graph only if document has changed
        if doc_id not in self.document_hashes or self.document_hashes[doc_id] != current_hash:
            logger.info(f"Updating knowledge graph for document '{doc_id}'...")

            try:
                # Delete existing graph data for this document
                self.delete_graph_elements_for_doc(doc_id)

                # Convert to LlamaIndex document
                llama_doc = self.convert_to_llama_document(langchain_document)

                # Create storage context with graph store
                storage_context = StorageContext.from_defaults(graph_store=self.llama_graph_store)

                # Use LlamaIndex's KnowledgeGraphIndex for triple extraction
                kg_index = KnowledgeGraphIndex.from_documents(
                    documents=[llama_doc],
                    storage_context=storage_context,
                    max_triplets_per_chunk=MAX_TRIPLETS_PER_CHUNK,
                    include_embeddings=True,
                    llm=self.llama_llm
                )

                logger.info(f"Successfully updated knowledge graph for document '{doc_id}'")
                return True
            except Exception as e:
                logger.error(f"Error updating graph for document '{doc_id}': {e}")
                return False
        else:
            logger.info(f"Document '{doc_id}' has not changed. Skipping graph update.")
            return False

    def visualize_graph(self, doc_id=None):
        """Create a visualization of the Neo4j graph."""
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

        try:
            with driver.session() as session:
                # Query for the graph data
                if doc_id:
                    query = """
                    MATCH (n)-[r]->(m)
                    WHERE n.document_id = $doc_id
                    RETURN n, r, m
                    """
                    result = session.run(query, doc_id=doc_id)
                else:
                    query = """
                    MATCH (n)-[r]->(m)
                    RETURN n, r, m
                    """
                    result = session.run(query)

                # Build a NetworkX graph
                graph = nx.MultiDiGraph()

                for record in result:
                    source_node = dict(record["n"])
                    source_id = source_node.get("name", source_node.get("id", str(id(source_node))))

                    target_node = dict(record["m"])
                    target_id = target_node.get("name", target_node.get("id", str(id(target_node))))

                    relationship = dict(record["r"])
                    rel_type = type(record["r"]).__name__

                    # Add nodes and edge
                    graph.add_node(source_id, **source_node)
                    graph.add_node(target_id, **target_node)
                    graph.add_edge(source_id, target_id, type=rel_type, **relationship)

                # Visualize the graph
                doc_label = doc_id if doc_id else "all_documents"
                return self._visualize_networkx_graph(graph, doc_label)

        except Exception as e:
            logger.error(f"Error visualizing Neo4j graph: {e}")
        finally:
            driver.close()

    def _visualize_networkx_graph(self, graph, doc_label):
        """Create a visualization of a NetworkX graph."""
        if len(graph) == 0:
            logger.warning(f"No nodes to visualize for {doc_label}")
            return None

        # Create the visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph)

        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=500, alpha=0.8)

        # Draw edges
        edge_labels = {(u, v): d.get("type", "") for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

        # Draw labels
        labels = {n: d.get("label", n) for n, d in graph.nodes(data=True)}
        nx.draw_networkx_labels(graph, pos, labels=labels)

        # Save to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{doc_label}_{timestamp}.png"
        output_path = os.path.join(GRAPH_VISUALIZATIONS_DIR, filename)
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Created graph visualization at {output_path}")
        return output_path


def main(doc_paths=None):
    """Main function to process documents with Neo4j storage."""
    logger.info("Starting document processing with Neo4j storage")

    # Initialize the Neo4j graph manager
    graph_manager = Neo4jGraphManager()

    # Process documents
    updated_docs = graph_manager.process_documents(doc_paths)

    return updated_docs


if __name__ == "__main__":
    import sys

    doc_paths = sys.argv[1:] if len(sys.argv) > 1 else None
    main(doc_paths)