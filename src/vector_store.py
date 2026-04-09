"""
Vector Store Configuration
RUBRIC: Vector Store & RAG Setup (12 marks total)
- Azure AI Search vector store initialized correctly (4 marks)
- Azure OpenAI embeddings configured properly (4 marks)
- LangChain AzureSearch integration is correct (4 marks)

TASK: Initialize Azure AI Search vector store with embeddings
"""
import os
from src.config import Config
from langchain_community.vectorstores import AzureSearch

def get_vector_store(embedding_function):
    """
    Returns Azure AI Search vector store (no ChromaDB option)
    
    HINT: This function should:
    1. Get Azure Search credentials from Config
    2. Validate that endpoint and key are present
    3. Initialize AzureSearch with correct parameters
    4. Return the vector store instance
    
    Args:
        embedding_function: The LangChain embedding function to use
    """
    
    # HINT: Get configuration values from Config class
    endpoint = Config.AZURE_SEARCH_ENDPOINT
    key = Config.AZURE_SEARCH_KEY
    index_name = Config.AZURE_SEARCH_INDEX_NAME

    # HINT: Validate that required credentials are present
    if not endpoint or not key:
        raise ValueError("Azure Search endpoint and key must be set.") 
    # HINT: Initialize AzureSearch vector store
    # Required parameters: azure_search_endpoint, azure_search_key, 
    # index_name, embedding_function (use .embed_query method)
    vector_store = AzureSearch(
        azure_search_endpoint=endpoint, 
        azure_search_key=key,  
        index_name=index_name, 
        embedding_function=embedding_function.embed_query 
    )
    
    print(f"Initialized Azure AI Search (LangChain) for index '{index_name}'")
    return vector_store