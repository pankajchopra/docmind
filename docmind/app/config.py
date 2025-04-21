# app/config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM Settings
DEFAULT_LLM_MODEL = "gpt-4o"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Index Settings
INDEX_STORE_DIR = os.path.join(DATA_DIR, "indexes")
os.makedirs(INDEX_STORE_DIR, exist_ok=True)

CHROMA_STORE_DIR = os.path.join(DATA_DIR, "chroma")
os.makedirs(CHROMA_STORE_DIR, exist_ok=True)

GRAPH_VISUALIZATIONS_DIR = os.path.join(DATA_DIR, "graph_visualizations")
os.makedirs(GRAPH_VISUALIZATIONS_DIR).mkdir(exist_ok=True)

# File paths
DOCUMENT_HASHES_FILE = os.path.join(DATA_DIR, "document_hashes.json")
GRAPH_JSON_FILE = os.path.join(DATA_DIR, "memory_graph.json")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")
WEB_META_FILE = os.path.join(DATA_DIR, "website_metadata.json")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "knowledge_graph.db")
# Retrieval Settings
RETRIEVAL_TOP_K = 3

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")

# SQLite details
SQLITE_DB_PATH = os.setenv("SQLITE_DB_PATH", SQLITE_DB_PATH)

# File paths
DOCUMENT_HASHES_FILE = os.setenv("DOCUMENT_HASHES_FILE", DOCUMENT_HASHES_FILE)
CHROMA_PERSIST_DIR = os.setenv("CHROMA_STORE_DIR",  CHROMA_STORE_DIR)
CHROMA_STORE_DIR = os.setenv("CHROMA_STORE_DIR",  CHROMA_STORE_DIR)
DATA_DIR = os.setenv("DATA_DIR", DATA_DIR)
GRAPH_JSON_FILE = os.setenv("GRAPH_JSON_FILE", GRAPH_JSON_FILE)
GRAPH_VISUALIZATIONS_DIR = os.setenv("GRAPH_VISUALIZATIONS_DIR", GRAPH_VISUALIZATIONS_DIR)

# Model settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))
MAX_TRIPLETS_PER_CHUNK = int(os.getenv("MAX_TRIPLETS_PER_CHUNK", "10"))
