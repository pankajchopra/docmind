from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from app.ai_models import AVAILABLE_EMBEDDINGS
from app.config import CHROMA_STORE_DIR


def get_embedding():
    return OpenAIEmbeddings(
        model=AVAILABLE_EMBEDDINGS["openai-small"]["model"],
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY"))


# Define the database path
persist_directory = CHROMA_STORE_DIR
# Connect to your existing persistent Chroma DB
db = Chroma(persist_directory=CHROMA_STORE_DIR, embedding_function=get_embedding)

# Get all collections
collections = db.get()
print(f"Collections: {collections}")

# Get all documents
all_docs = db.get()
print(f"Number of documents: {len(all_docs['ids'])}")

# Print documents and metadata
for i, (doc_id, document, metadata) in enumerate(zip(all_docs["ids"], all_docs["documents"], all_docs["metadatas"])):
    print(f"Document {i}:")
    print(f"  ID: {doc_id}")
    print(f"  Content: {document}")
    print(f"  Metadata: {metadata}")
    print("-" * 50)

# You can also query specific documents by ID
specific_docs = db.get(ids=["some_id_1", "some_id_2"])