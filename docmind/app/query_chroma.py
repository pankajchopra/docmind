import os
import chromadb
import shutil
from app.config import CHROMA_STORE_DIR

# Define the database path
persist_directory = CHROMA_STORE_DIR

# Check ChromaDB version
print(f"Using ChromaDB version: {chromadb.__version__}")

# First, try to back up the existing database
backup_dir = f"{persist_directory}_backup"
if os.path.exists(persist_directory) and not os.path.exists(backup_dir):
    print(f"Creating backup of existing database at {backup_dir}")
    shutil.copytree(persist_directory, backup_dir)
    print("Backup created successfully")

# Option 1: Try to use the existing database
try:
    print("Attempting to use existing database...")
    client = chromadb.PersistentClient(path=persist_directory)

    # Try to access the collection directly
    collection_name = "MyStockCollection"
    collection = client.get_collection(name=collection_name)
    print(f"Successfully loaded collection: {collection_name}")

    # Try to add a sample document if the collection is empty
    try:
        count = collection.count()
        print(f"Collection has {count} documents")

        if count == 0:
            print("Adding a sample document to the collection...")
            collection.add(
                documents=["This is a sample document about stocks"],
                metadatas=[{"source": "example"}],
                ids=["sample1"]
            )
            print("Sample document added successfully")
    except Exception as e:
        print(f"Error checking/adding documents: {e}")

    # Query the collection
    results = collection.query(
        query_texts=["stock"],
        n_results=5,
        include=["documents", "metadatas"]
    )
    print("Query Results:", results)

except Exception as e:
    print(f"Error using existing database: {e}")

    # Option 2: Create a new database if needed
    print("\nAttempting to create a new database...")

    # Rename or delete the old database directory
    if os.path.exists(persist_directory):
        print(f"Renaming existing directory to {persist_directory}_old")
        if os.path.exists(f"{persist_directory}_old"):
            shutil.rmtree(f"{persist_directory}_old")
        os.rename(persist_directory, f"{persist_directory}_old")

    # Create new client with fresh database
    client = chromadb.PersistentClient(path=persist_directory)
    print("New database created successfully")

    # Create a new collection
    collection_name = "MyStockCollection"
    collection = client.create_collection(name=collection_name)
    print(f"Created new collection: {collection_name}")

    # Add sample data
    collection.add(
        documents=["This is a document about stocks", "This is another document about investments"],
        metadatas=[{"source": "example1"}, {"source": "example2"}],
        ids=["doc1", "doc2"]
    )
    print("Added sample data to the new collection")

    # Query the new collection
    results = collection.query(
        query_texts=["stock"],
        n_results=5,
        include=["documents", "metadatas"]
    )
    print("Query Results:", results)