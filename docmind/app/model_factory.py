# app/model_factory.py
from typing import Dict, Any, Optional
import os


def get_llm_client(provider: str, model: str, **kwargs) -> Any:
    """Factory function to create appropriate LLM client based on provider.

    Args:
        provider: The model provider (anthropic, openai, google, etc.)
        model: The specific model to use
        **kwargs: Additional provider-specific parameters

    Returns:
        Initialized LLM client
    """
    if provider == "openai":
        from llama_index.llms.openai import OpenAI
        return OpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"))
    # elif provider == "anthropic":
    #     from llama_index.llms.anthropic import Anthropic
    #     return Anthropic(model=model, api_key=os.getenv("ANTHROPIC_API_KEY"))

    elif provider == "openai":
        from llama_index.llms.openai import OpenAI
        return OpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"))

    # elif provider == "google":
    #     from llama_index.llms.gemini import Gemini
    #     return Gemini(model=model, api_key=os.getenv("GOOGLE_API_KEY"))

    # elif provider == "azure":
    #     from llama_index.llms.azure_openai import AzureOpenAI
    #     return AzureOpenAI(
    #         model=model,
    #         engine=kwargs.get("deployment_name"),
    #         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    #     )
    #
    # elif provider == "huggingface":
    #     from llama_index.llms.huggingface import HuggingFaceLLM
    #     return HuggingFaceLLM(model_name=model)
    #
    # elif provider == "local":
    #     # For local models using llama.cpp
    #     from llama_index.llms.llama_cpp import LlamaCPP
    #     return LlamaCPP(model_path=f"models/{model.split('/')[-1]}.gguf")

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_embedding_client(provider: str, model: str, **kwargs) -> Any:
    """Factory function to create appropriate embedding client based on provider.

    Args:
        provider: The embedding provider (openai, google, huggingface, etc.)
        model: The specific model to use
        **kwargs: Additional provider-specific parameters

    Returns:
        Initialized embedding client
    """
    if provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        return OpenAIEmbedding(model_name=model, api_key=os.getenv("OPENAI_API_KEY"))

    # elif provider == "google":
    #     from llama_index.embeddings.google import GoogleGenerativeAIEmbedding
    #     return GoogleGenerativeAIEmbedding(model_name=model, api_key=os.getenv("GOOGLE_API_KEY"))
    #
    # elif provider == "huggingface":
    #     from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    #     return HuggingFaceEmbedding(model_name=model)
    #
    # elif provider == "cohere":
    #     from llama_index.embeddings.cohere import CohereEmbedding
    #     return CohereEmbedding(model=model, api_key=os.getenv("COHERE_API_KEY"))
    #
    # elif provider == "local":
    #     if model == "fasttext":
    #         from llama_index.embeddings.fastembed import FastEmbedEmbedding
    #         return FastEmbedEmbedding(model_name="all-MiniLM-L6-v2")
    #     else:
    #         from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    #         return HuggingFaceEmbedding(model_name=model, device="cuda" if kwargs.get("use_gpu", False) else "cpu")

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")