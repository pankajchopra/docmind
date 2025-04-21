# kg_networkx.py
from kg_base import BaseGraphManager, logger, LangchainDocument
from config import MAX_TRIPLETS_PER_CHUNK, GRAPH_VISUALIZATIONS_DIR, GRAPH_JSON_FILE, DATA_DIR
from llama_index.core import StorageContext
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from lightrag.core.retriever import GraphRetriever
import matplotlib.pyplot as plt
import networkx as nx
import json
import datetime
import os


class NetworkXGraphManager(BaseGraphManager):
    """Manager class for NetworkX in-memory graph storage."""

    def __init__(self, all_docs:LangchainDocument = None):
        """Initialize NetworkX graph storage."""
        super().__init__()
        self.doc_path = DATA_DIR
        # For in-memory storage, we use SimpleGraphStore for LlamaIndex integration
        self.llama_graph_store = SimpleGraphStore()

        # For in-memory storage, we'll need a NetworkX graph for storage and visualization
        self.nx_graph = nx.MultiDiGraph()
        self.kg_index = None
        # Load any existing graph from JSON file
        if not all_docs:
            if not self.load_memory_graph_from_json():
                print("No existing graph file found at", GRAPH_JSON_FILE)
            else:
                print("Loaded existing graph from", GRAPH_JSON_FILE)
        else:

            self.process_documents(doc_paths)
        logger.info("Initialized NetworkX in-memory graph storage")

    def load_memory_graph_from_json(self)-> bool:
        """Load the in-memory graph from a JSON file."""
        try:
            with open(GRAPH_JSON_FILE, "r") as f:
                graph_data = json.load(f)

            # Create a new graph
            self.nx_graph = nx.MultiDiGraph()

            # Add nodes
            for node in graph_data["nodes"]:
                self.nx_graph.add_node(
                    node["id"],
                    label=node.get("label", node["id"]),
                    document_id=node.get("document_id", "unknown")
                )

            # Add edges
            for edge in graph_data["edges"]:
                self.nx_graph.add_edge(
                    edge["source"],
                    edge["target"],
                    type=edge.get("type", "related_to"),
                    document_id=edge.get("document_id", "unknown")
                )

            logger.info(f"Loaded in-memory graph from {GRAPH_JSON_FILE}")
            return True
        except FileNotFoundError:
            logger.info(f"No existing graph file found at {GRAPH_JSON_FILE}")
            return False

    def _save_memory_graph_to_json(self):
        """Save the in-memory graph to a JSON file."""
        # Convert NetworkX graph to a JSON-serializable format
        graph_data = {
            "nodes": [],
            "edges": []
        }

        # Add nodes
        for node, attrs in self.nx_graph.nodes(data=True):
            node_data = {
                "id": node,
                "label": attrs.get("label", node),
                "document_id": attrs.get("document_id", "unknown")
            }
            graph_data["nodes"].append(node_data)

        # Add edges
        for source, target, attrs in self.nx_graph.edges(data=True):
            edge_data = {
                "source": source,
                "target": target,
                "type": attrs.get("type", "related_to"),
                "document_id": attrs.get("document_id", "unknown")
            }
            graph_data["edges"].append(edge_data)

        # Save to file
        with open(GRAPH_JSON_FILE, "w") as f:
            json.dump(graph_data, f, indent=2)

        logger.info(f"Saved in-memory graph to {GRAPH_JSON_FILE}")

    def delete_graph_elements_for_doc(self, doc_id: str):
        """Delete existing graph elements for a document."""
        # Filter out nodes with matching document_id
        nodes_to_remove = [
            node for node, attrs in self.nx_graph.nodes(data=True)
            if attrs.get("document_id") == doc_id
        ]
        self.nx_graph.remove_nodes_from(nodes_to_remove)
        logger.info(f"Deleted existing in-memory graph elements for document '{doc_id}'")

    def process_document_for_graph(self, langchain_document: LangchainDocument):
        """Process a document for graph storage using LlamaIndex KnowledgeGraphIndex."""
        doc_id = langchain_document.metadata.get("id")
        source = langchain_document.metadata.get("source")
        if not doc_id or not source:
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
                if(self.kg_index):
                    self.kg_index = self.kg_index
                else:
                    self.kg_index = KnowledgeGraphIndex.from_documents(
                        documents=[llama_doc],
                        storage_context=storage_context,
                        max_triplets_per_chunk=MAX_TRIPLETS_PER_CHUNK,
                        include_embeddings=True,
                        llm=self.llama_llm
                    )

                # Store the triples in memory using NetworkX
                self._store_triples_from_llama_index(self.kg_index, doc_id)

                # Save the updated graph to JSON
                self._save_memory_graph_to_json()

                logger.info(f"Successfully updated knowledge graph for document '{doc_id}'")
                return True
            except Exception as e:
                logger.error(f"Error updating graph for document '{doc_id}': {e}")
                return False
        else:
            logger.info(f"Document '{doc_id}' has not changed. Skipping graph update.")
            return False

    def _store_triples_from_llama_index(self, kg_index, doc_id):
        """Store the extracted triples in memory using NetworkX."""
        # Extract triples from the SimpleGraphStore in self.kg_index
        triples = []

        # Access the graph store from the index
        graph_store = self.kg_index._storage_context.graph_store

        # For SimpleGraphStore, we need to access its internal representation
        if hasattr(graph_store, "_rel_map"):
            for rel_type, rels in graph_store._rel_map.items():
                for subj, objs in rels.items():
                    for obj in objs:
                        triples.append((subj, rel_type, obj))

        # Store the triples in memory using NetworkX
        self._store_triples_in_memory(triples, doc_id)

    def _store_triples_in_memory(self, triples, doc_id):
        """Store the extracted triples in memory using NetworkX."""
        # Process and store each triple
        for subject, relation, object_ in triples:
            # Add nodes if they don't exist
            for node in [subject, object_]:
                if not self.nx_graph.has_node(node):
                    self.nx_graph.add_node(node, label=node, document_id=doc_id)

            # Add the relationship
            self.nx_graph.add_edge(
                subject,
                object_,
                type=relation,
                document_id=doc_id
            )

        logger.info(f"Stored {len(triples)} triples in memory for document '{doc_id}'")

    def visualize_graph(self, doc_id=None):
        """Create a visualization of the in-memory graph."""
        if doc_id:
            # Filter to only include nodes from this document
            subgraph_nodes = [
                node for node, attrs in self.nx_graph.nodes(data=True)
                if attrs.get("document_id") == doc_id
            ]
            subgraph = self.nx_graph.subgraph(subgraph_nodes)
            return self._visualize_networkx_graph(subgraph, doc_id)
        else:
            return self._visualize_networkx_graph(self.nx_graph, "all_documents")

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

    def process_documents(self, doc_paths=None):
        """Override process_documents to handle memory-specific operations."""
        updated_docs = super().process_documents(doc_paths)

        # For memory storage, ensure the graph is saved to JSON
        self._save_memory_graph_to_json()

        return updated_docs

    def process_documents(self, documents):
        """Process multiple documents for graph creation.

        Args:
            documents: List of Document objects to process

        Returns:
            Boolean indicating success
        """
        success = True
        for doc in documents:
            if not self.process_document_for_graph(doc):
                success = False
        return success

    def get_llama_graph_retriever(self):
        """Return a LlamaIndex graph retriever."""
        if not self.kg_index:
            raise ValueError("Knowledge graph index has not been initialized")

        # Get the graph query engine from the knowledge graph index
        return self.kg_index.as_query_engine()
def main(doc_paths=None):
    """Main function to process documents with NetworkX in-memory storage."""
    logger.info("Starting document processing with NetworkX in-memory storage")

    # Initialize the NetworkX graph manager
    graph_manager = NetworkXGraphManager()

    # Process documents
    updated_docs = graph_manager.process_documents(doc_paths)

    return updated_docs


if __name__ == "__main__":
    import sys

    doc_paths = sys.argv[1:] if len(sys.argv) > 1 else None
    main(doc_paths)