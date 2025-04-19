
class ChromaException(Exception):
    """Exception raised for errors in the Chroma database operations."""

    def __init__(self, message="Exception raised for errors in the Chroma database operations"):
        self.message = message
        super().__init__(self.message)


class VectorStoreException(Exception):
    """Exception raised for errors in vector store operations."""

    def __init__(self, message="Exception raised for errors in vector store operations."):
        self.message = message
        super().__init__(self.message)


class DocumentProcessingException(Exception):
    """Exception raised for errors in document processing."""

    def __init__(self, message="Exception raised for errors in document processing"):
        self.message = message
        super().__init__(self.message)
