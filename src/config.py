"""
Configuration Management
RUBRIC: Environment Setup & Configuration (8 marks total)
- Azure OpenAI credentials configured correctly (1 mark)
- Azure AI Search credentials set up properly (1 mark)
- config.py implemented with validation (3 marks)
- All required packages installed and imported without errors (3 marks)

TASK: Load all configuration from environment variables
"""
import os
from dotenv import load_dotenv

# HINT: Load environment variables from .env file
load_dotenv()  # HINT: load_dotenv()

class Config:
    """Configuration for Wanderlust Travel Chatbot"""
    
    # ====================
    # Azure OpenAI Configuration
    # ====================
    # HINT: Load Azure OpenAI credentials from environment
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  # HINT: "AZURE_OPENAI_API_KEY"
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")  # HINT: "AZURE_OPENAI_API_VERSION", "2023-05-15"
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")  # HINT: "AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")  # HINT: "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"

    # ====================
    # Azure AI Search Configuration (Only vector store - no ChromaDB)
    # ====================
    # HINT: Load Azure AI Search credentials
    AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT") 
    AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY") 
    AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "travel-kb-index")  # HINT: "AZURE_SEARCH_INDEX_NAME", "travel-kb-index"

    # ====================
    # Azure Storage (Optional)
    # ====================
    AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")  
    AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "travel-documents")  # HINT: "AZURE_STORAGE_CONTAINER_NAME", "travel-documents"

    # ====================
    # Azure Content Safety (Optional)
    # ====================
    AZURE_CONTENT_SAFETY_ENDPOINT = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")  
    AZURE_CONTENT_SAFETY_KEY = os.getenv("AZURE_CONTENT_SAFETY_KEY") 
    
    # ====================
    # Azure Monitor (Optional)
    # ====================
    APPLICATIONINSIGHTS_CONNECTION_STRING = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING") 
    
    # ====================
    # MLflow Configuration
    # ====================
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")  # HINT: "MLFLOW_TRACKING_URI"
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "wanderlust-travel-chatbot")  # HINT: "MLFLOW_EXPERIMENT_NAME", "wanderlust-travel-chatbot"
    
    # ====================
    # Ingestion Settings
    # ====================
    # HINT: Convert to integer, 0 means no limit
    INGESTION_LIMIT = int(os.getenv("INGESTION_LIMIT", "0"))  # HINT: "INGESTION_LIMIT", "0"