import networkx as nx
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.graph_stores import Neo4jGraphStore
import json
import sqlite3
from typing import Optional, Dict, List, Tuple, Any


def load_llama_simple_graph(persist_path: str) -> nx.DiGraph:
    """
    Load a LlamaIndex SimpleGraphStore and convert it to NetworkX DiGraph

    Args:
        persist_path: Path to the persisted SimpleGraphStore JSON file

    Returns:
        NetworkX DiGraph object
    """
    # Load the SimpleGraphStore from disk
    graph_store = SimpleGraphStore.from_persist_path(persist_path)

    # Create a new directed graph
    G = nx.DiGraph()

    # Add all nodes and relationships from the graph store
    for rel_triplet in graph_store.get_all_relationships():
        # Extract the triplet components
        subj = rel_triplet.source
        pred = rel_triplet.rel_type
        obj = rel_triplet.target

        # Add nodes with their metadata if available
        if not G.has_node(subj):
            G.add_node(subj)

        if not G.has_node(obj):
            G.add_node(obj)

        # Add the edge with the relationship type as an attribute
        G.add_edge(subj, obj, relationship=pred)

    return G


def extract_graph_from_neo4j(uri: str, username: str, password: str, document_id: Optional[str] = None) -> nx.DiGraph:
    """
    Extract graph data from Neo4j and convert to NetworkX

    Args:
        uri: Neo4j connection URI (e.g., bolt://localhost:7687)
        username: Neo4j username
        password: Neo4j password
        document_id: Optional document ID to filter data

    Returns:
        NetworkX DiGraph object
    """
    from neo4j import GraphDatabase

    G = nx.DiGraph()

    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        with driver.session() as session:
            # Query to get all nodes and their properties
            node_query = """
            MATCH (n)
            WHERE $document_id IS NULL OR n.document_id = $document_id
            RETURN id(n) as id, labels(n) as labels, properties(n) as properties
            """

            nodes = session.run(node_query, document_id=document_id).data()

            # Add all nodes to the graph
            for node in nodes:
                node_id = str(node['id'])
                G.add_node(
                    node_id,
                    labels=node['labels'],
                    **node['properties']
                )

                # Add original text as a node attribute for better visualization
                if 'text' in node['properties']:
                    G.nodes[node_id]['label'] = node['properties']['text']

            # Query to get all relationships
            rel_query = """
            MATCH (s)-[r]->(t)
            WHERE $document_id IS NULL OR s.document_id = $document_id
            RETURN id(s) as source, id(t) as target, type(r) as type, properties(r) as properties
            """

            relationships = session.run(rel_query, document_id=document_id).data()

            # Add all relationships to the graph
            for rel in relationships:
                source = str(rel['source'])
                target = str(rel['target'])
                rel_type = rel['type']

                G.add_edge(
                    source,
                    target,
                    relationship=rel_type,
                    **rel['properties']
                )

                # Add relationship type as edge label for visualization
                G.edges[source, target]['label'] = rel_type

    return G


def extract_graph_from_sqlite(db_path: str, document_id: Optional[str] = None) -> nx.DiGraph:
    """
    Extract graph data from SQLite and convert to NetworkX

    Args:
        db_path: Path to SQLite database file
        document_id: Optional document ID to filter data

    Returns:
        NetworkX DiGraph object
    """
    G = nx.DiGraph()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Query for nodes (adjust table/column names to match your schema)
        node_query = """
        SELECT id, type, properties FROM nodes
        WHERE document_id = ? OR ? IS NULL
        """

        cursor.execute(node_query, (document_id, document_id))
        nodes = cursor.fetchall()

        # Add nodes to graph
        for node_id, node_type, props_json in nodes:
            props = json.loads(props_json)
            G.add_node(
                node_id,
                node_type=node_type,
                **props
            )

            # Add label for visualization
            if 'text' in props:
                G.nodes[node_id]['label'] = props['text']

        # Query for relationships (adjust table/column names to match your schema)
        rel_query = """
        SELECT source_id, target_id, type, properties FROM relationships
        WHERE document_id = ? OR ? IS NULL
        """

        cursor.execute(rel_query, (document_id, document_id))
        relationships = cursor.fetchall()

        # Add relationships to graph
        for source, target, rel_type, props_json in relationships:
            props = json.loads(props_json)
            G.add_edge(
                source,
                target,
                relationship=rel_type,
                **props
            )

            # Add relationship type as edge label
            G.edges[source, target]['label'] = rel_type

    return G


def save_graph_as_graphml(G: nx.Graph, output_path: str) -> None:
    """
    Save a NetworkX graph to GraphML format with proper attribute handling

    Args:
        G: NetworkX graph object
        output_path: Path to save the GraphML file
    """
    # Convert all attributes to strings for GraphML compatibility
    for node, attrs in G.nodes(data=True):
        for key, value in list(attrs.items()):
            if isinstance(value, (dict, list)):
                G.nodes[node][key] = json.dumps(value)

    for u, v, attrs in G.edges(data=True):
        for key, value in list(attrs.items()):
            if isinstance(value, (dict, list)):
                G.edges[u, v][key] = json.dumps(value)

    # Write the graph to GraphML
    nx.write_graphml(G, output_path)
    print(f"Graph saved to {output_path}")


def convert_and_visualize(
        source_type: str,
        source_path: str,
        output_path: str,
        document_id: Optional[str] = None,
        neo4j_credentials: Optional[Dict] = None
) -> nx.DiGraph:
    """
    Convert graph data from various sources to NetworkX and save as GraphML

    Args:
        source_type: Type of source ('simple_store', 'neo4j', or 'sqlite')
        source_path: Path to the source data or connection string
        output_path: Path to save the GraphML file
        document_id: Optional document ID to filter by
        neo4j_credentials: Neo4j credentials if using Neo4j

    Returns:
        The created NetworkX DiGraph
    """
    if source_type == 'simple_store':
        G = load_llama_simple_graph(source_path)

    elif source_type == 'neo4j':
        if not neo4j_credentials:
            raise ValueError("Neo4j credentials required for Neo4j source type")

        G = extract_graph_from_neo4j(
            source_path,
            neo4j_credentials['username'],
            neo4j_credentials['password'],
            document_id
        )

    elif source_type == 'sqlite':
        G = extract_graph_from_sqlite(source_path, document_id)

    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    # Save the graph as GraphML
    save_graph_as_graphml(G, output_path)

    # Print some graph statistics
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    return G


if __name__ == "__main__":
    # Example usage:
    # 1. Convert from LlamaIndex SimpleGraphStore
    # convert_and_visualize(
    #     source_type='simple_store',
    #     source_path='./graph_store.json',
    #     output_path='./knowledge_graph.graphml'
    # )

    # 2. Extract from Neo4j
    # convert_and_visualize(
    #     source_type='neo4j',
    #     source_path='bolt://localhost:7687',
    #     output_path='./neo4j_graph.graphml',
    #     document_id='data/sample.txt',  # Optional: filter by document_id
    #     neo4j_credentials={'username': 'neo4j', 'password': 'password'}
    # )

    # 3. Extract from SQLite
    # convert_and_visualize(
    #     source_type='sqlite',
    #     source_path='./knowledge_graph.db',
    #     output_path='./sqlite_graph.graphml'
    # )

    print("Done! The graph can now be visualized in tools like Gephi, Cytoscape, or yEd.")