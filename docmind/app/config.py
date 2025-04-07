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

# Index Settings
INDEX_STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "indexes")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Ensure directories exist
os.makedirs(INDEX_STORE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Retrieval Settings
RETRIEVAL_TOP_K = 3