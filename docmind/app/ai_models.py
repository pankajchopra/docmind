# Enhanced configuration with additional LLM and embedding options

# Available LLM options
AVAILABLE_LLMS = {
    # Anthropic models
    "claude-3-5-haiku": {
        "provider": "anthropic",
        "model": "claude-3-5-haiku-20240307",
        "description": "Fast, efficient model for high-throughput tasks"
    },
    "claude-3-5-sonnet": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20240620",
        "description": "Balanced performance for most general applications"
    },
    "claude-3-7-sonnet": {
        "provider": "anthropic",
        "model": "claude-3-7-sonnet-20250219",
        "description": "Advanced reasoning model with extended thinking capabilities"
    },
    "claude-3-opus": {
        "provider": "anthropic",
        "model": "claude-3-opus-20240229",
        "description": "Most powerful Claude model for complex tasks"
    },

    # OpenAI models
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "description": "OpenAI's flagship multimodal model"
    },
    "gpt-4-turbo": {
        "provider": "openai",
        "model": "gpt-4-turbo-2024-04-09",
        "description": "Optimized version of GPT-4 with improved performance"
    },
    "gpt-4-vision": {
        "provider": "openai",
        "model": "gpt-4-vision-preview",
        "description": "GPT-4 with image understanding capabilities"
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "description": "Fast and cost-effective model for many applications"
    },

    # Google models
    "gemini-pro": {
        "provider": "google",
        "model": "gemini-pro",
        "description": "Google's advanced text model with strong reasoning"
    },
    "gemini-ultra": {
        "provider": "google",
        "model": "gemini-ultra-vision",
        "description": "Google's most capable multimodal model"
    },

    # Microsoft/Azure models
    "azure-gpt4": {
        "provider": "azure",
        "model": "gpt-4",
        "deployment_name": "gpt4-deployment",
        "description": "Azure-hosted GPT-4"
    },

    # Hugging Face models
    "llama-3-70b": {
        "provider": "huggingface",
        "model": "meta-llama/Llama-3-70b-chat-hf",
        "description": "Meta's Llama 3 (70B) open model"
    },
    "llama-3-8b": {
        "provider": "huggingface",
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "description": "Smaller, faster Llama 3 (8B) model"
    },
    "mistral-large": {
        "provider": "huggingface",
        "model": "mistralai/Mistral-Large-2",
        "description": "Mistral's largest instruction-tuned model"
    },
    "mistral-small": {
        "provider": "huggingface",
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "description": "Efficient 7B parameter model from Mistral"
    },
    "falcon-40b": {
        "provider": "huggingface",
        "model": "tiiuae/falcon-40b-instruct",
        "description": "40B parameter model from TII UAE"
    },

    # Smaller Language Models (SLMs)
    "phi-3-mini": {
        "provider": "huggingface",
        "model": "microsoft/phi-3-mini-4k-instruct",
        "description": "Microsoft's compact but capable Phi-3 mini model"
    },
    "gemma-7b": {
        "provider": "huggingface",
        "model": "google/gemma-7b-it",
        "description": "Google's efficient 7B instruction-tuned model"
    },
    "tinyllama": {
        "provider": "huggingface",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "Extremely compact 1.1B parameter model"
    },

    # Local inference (for models that can run on consumer hardware)
    "local-llama": {
        "provider": "local",
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "description": "Locally hosted Llama 3 model using llama.cpp"
    },
    "local-mistral": {
        "provider": "local",
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "description": "Locally hosted Mistral model"
    }
}

# Available embedding options
AVAILABLE_EMBEDDINGS = {
    # OpenAI embeddings
    "openai-small": {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "description": "Efficient embedding model with good performance"
    },
    "openai-large": {
        "provider": "openai",
        "model": "text-embedding-3-large",
        "dimensions": 3072,
        "description": "Higher-quality embedding model with more dimensions"
    },

    # Google embeddings
    "google-gecko": {
        "provider": "google",
        "model": "text-embedding-gecko",
        "dimensions": 768,
        "description": "Google's embedding model for text"
    },

    # Hugging Face embeddings
    "sentence-transformers": {
        "provider": "huggingface",
        "model": "sentence-transformers/all-mpnet-base-v2",
        "dimensions": 768,
        "description": "High quality general purpose embeddings"
    },
    "e5-large": {
        "provider": "huggingface",
        "model": "intfloat/e5-large-v2",
        "dimensions": 1024,
        "description": "Improved embeddings for retrieval tasks"
    },
    "bge-large": {
        "provider": "huggingface",
        "model": "BAAI/bge-large-en-v1.5",
        "dimensions": 1024,
        "description": "State-of-the-art embedding model for RAG"
    },
    "jina-embeddings": {
        "provider": "huggingface",
        "model": "jinaai/jina-embeddings-v2-base-en",
        "dimensions": 768,
        "description": "Efficient embeddings optimized for retrieval"
    },

    # Cohere embeddings
    "cohere-embed": {
        "provider": "cohere",
        "model": "embed-english-v3.0",
        "dimensions": 1024,
        "description": "Cohere's English text embedding model"
    },
    "cohere-multilingual": {
        "provider": "cohere",
        "model": "embed-multilingual-v3.0",
        "dimensions": 1024,
        "description": "Multilingual embedding model from Cohere"
    },

    # Local embeddings
    "local-instructor": {
        "provider": "local",
        "model": "hkunlp/instructor-large",
        "dimensions": 768,
        "description": "Locally hosted instructor embeddings"
    },
    "local-fasttext": {
        "provider": "local",
        "model": "fasttext",
        "dimensions": 300,
        "description": "Lightweight local embeddings, good for limited resources"
    }
}

# Default selections
DEFAULT_LLM_PROVIDER = AVAILABLE_LLMS["claude-3-7-sonnet"]["provider"]
DEFAULT_LLM_MODEL = AVAILABLE_LLMS["claude-3-7-sonnet"]["model"]
DEFAULT_EMBEDDING_PROVIDER = AVAILABLE_EMBEDDINGS["openai-small"]["provider"]
DEFAULT_EMBEDDING_MODEL = AVAILABLE_EMBEDDINGS["openai-small"]["model"]