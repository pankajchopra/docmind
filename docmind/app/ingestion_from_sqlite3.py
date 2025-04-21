# kg_sqlite.py
from kg_base import BaseGraphManager, logger, LangchainDocument
from config import SQLITE_DB_PATH, GRAPH_VISUALIZATIONS_DIR, MAX_TRIPLETS_PER_CHUNK
from llama_index.core import StorageContext
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore

import matplotlib.pyplot as plt
import networkx as nx
import sqlite3
import json
import datetime
import os


class SQLiteGraphManager(BaseGraphManager):
    """Manager class for SQLite graph storage."""

    def __init__(self):
        """Initialize SQLite graph storage."""
        super().__init__()

        # For SQLite, we use SimpleGraphStore for LlamaIndex integration
        self.llama_graph_store = SimpleGraphStore()

        # Initialize SQLite database
        self._init_sqlite_db()

        logger.info("Initialized SQLite graph storage")

    def _init_sqlite_db(self):
        """Initialize the SQLite database schema."""
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()

        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            label TEXT,
            document_id TEXT,
            properties TEXT
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS relationships (
            id TEXT PRIMARY KEY,
            source_id TEXT,
            target_id TEXT,
            type TEXT,
            document_id TEXT,
            properties TEXT,
            FOREIGN KEY (source_id) REFERENCES nodes (id),
            FOREIGN KEY (target_id) REFERENCES nodes (id)
        )
        ''')

        conn.commit()
        conn.close()

    def delete_graph_elements_for_doc(self, doc_id: str):
        """Delete existing graph elements for a document before updating."""
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()

        # Delete relationships first (foreign key constraints)
        cursor.execute(
            "DELETE FROM relationships WHERE document_id = ?",
            (doc_id,)
        )

        # Then delete nodes
        cursor.execute(
            "DELETE FROM nodes WHERE document_id = ?",
            (doc_id,)
        )

        conn.commit()
        conn.close()
        logger.info(f"Deleted existing SQLite graph elements for document '{doc_id}'")

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

                # Store triples in SQLite
                self._store_triples_from_llama_index(kg_index, doc_id)

                logger.info(f"Successfully updated knowledge graph for document '{doc_id}'")
                return True
            except Exception as e:
                logger.error(f"Error updating graph for document '{doc_id}': {e}")
                return False
        else:
            logger.info(f"Document '{doc_id}' has not changed. Skipping graph update.")
            return False

    def _store_triples_from_llama_index(self, kg_index, doc_id):
        """Store the extracted triples in SQLite from LlamaIndex."""
        # Extract triples from the SimpleGraphStore in kg_index
        triples = []

        # Access the graph store from the index
        graph_store = kg_index._storage_context.graph_store

        # For SimpleGraphStore, we need to access its internal representation
        if hasattr(graph_store, "_rel_map"):
            for rel_type, rels in graph_store._rel_map.items():
                for subj, objs in rels.items():
                    for obj in objs:
                        triples.append((subj, rel_type, obj))

        # Store the triples in SQLite
        self._store_triples_in_sqlite(triples, doc_id)

    def _store_triples_in_sqlite(self, triples, doc_id):
        """Store the extracted triples in SQLite."""
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()

        # Process and store each triple
        for subject, relation, object_ in triples:
            # Add nodes if they don't exist
            for node in [subject, object_]:
                cursor.execute(
                    "INSERT OR IGNORE INTO nodes (id, label, document_id, properties) VALUES (?, ?, ?, ?)",
                    (
                        node,
                        node,  # Using the node value as its label for simplicity
                        doc_id,
                        json.dumps({})  # Empty properties for now
                    )
                )

            # Add the relationship
            rel_id = f"{subject}_{relation}_{object_}_{doc_id}"
            cursor.execute(
                """
                INSERT INTO relationships 
                (id, source_id, target_id, type, document_id, properties) 
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    rel_id,
                    subject,
                    object_,
                    relation,
                    doc_id,
                    json.dumps({})  # Empty properties for now
                )
            )

        conn.commit()
        conn.close()
        logger.info(f"Stored {len(triples)} triples in SQLite for document '{doc_id}'")

    def visualize_graph(self, doc_id=None):
        """Create a visualization of the graph from SQLite data."""
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()

        # Create a new graph
        graph = nx.MultiDiGraph()

        # Query for nodes
        if doc_id:
            cursor.execute("SELECT id, label FROM nodes WHERE document_id = ?", (doc_id,))
        else:
            cursor.execute("SELECT id, label FROM nodes")

        for row in cursor.fetchall():
            node_id, label = row
            graph.add_node(node_id, label=label)

        # Query for relationships
        if doc_id:
            cursor.execute(
                "SELECT source_id, target_id, type FROM relationships WHERE document_id = ?",
                (doc_id,)
            )
        else:
            cursor.execute("SELECT source_id, target_id, type FROM relationships")

        for row in cursor.fetchall():
            source, target, rel_type = row
            graph.add_edge(source, target, type=rel_type)

        conn.close()

        # Visualize the graph
        doc_label = doc_id if doc_id else "all_documents"
        return self._visualize_networkx_graph(graph, doc_label)

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
    """Main function to process documents with SQLite storage."""
    logger.info("Starting document processing with SQLite storage")

    # Initialize the SQLite graph manager
    graph_manager = SQLiteGraphManager()

    # Process documents
    updated_docs = graph_manager.process_documents(doc_paths)

    return updated_docs


if __name__ == "__main__":
    import sys

    doc_paths = sys.argv[1:] if len(sys.argv) > 1 else None
    main(doc_paths)