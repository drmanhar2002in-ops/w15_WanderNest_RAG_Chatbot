"""
Travel Search Engine with RAG
RUBRIC: Search Engine Implementation (13 marks total)
- TravelSearchEngine initialized correctly (4 marks)
- search_by_text performs similarity search (3 marks)
- synthesize_response generates grounded answers (4 marks)
- Governance checks integrated (2 marks)

TASK: Implement RAG search engine with governance integration
"""
import mlflow
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from src.config import Config
from governance.governance_gate import GovernanceGate
from src.vector_store import get_vector_store

class TravelSearchEngine:
    """RAG-powered search engine for travel queries"""
    
    def __init__(self):
        """
        Initialize search engine components
        
        HINT: Initialize:
        1. Governance gate
        2. Azure Chat OpenAI (LLM)
        3. Azure OpenAI Embeddings
        4. Vector store using get_vector_store
        """
        # HINT: Initialize governance gate
        self.governance_gate = GovernanceGate() 
        
        # HINT: Initialize Azure Chat OpenAI LLM
        # Required params: api_key, azure_endpoint, api_version, deployment_name, temperature
        self.llm = AzureChatOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            deployment_name=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
            temperature=Config.LLM_TEMPERATURE
        )
        
        # HINT: Initialize Azure OpenAI Embeddings
        # Required params: api_key, azure_endpoint, model, api_version
        self.embeddings = AzureOpenAIEmbeddings(
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            model=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
        ) 

        # HINT: Initialize Vector Store using get_vector_store function
        self.vector_store = get_vector_store(self.embeddings) 
    
    def search_by_text(self, query_text: str, k: int = 5):
        """
        Search for travel information using a text query
        
        HINT: This method should:
        1. Set MLflow experiment
        2. Start MLflow run
        3. Validate input with governance gate
        4. Perform similarity search on vector store
        5. Log metrics to MLflow
        6. Return results and query
        """
        
        # HINT: Set MLflow experiment name from Config
        mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)  
        
        with mlflow.start_run(run_name="search_travel_info"):
            print(f"DEBUG: Text Query: {query_text}")
            
            # HINT: Validate input using governance gate
            gov_check = self.governance_gate.validate_input(query_text)
            
            if not gov_check['passed']:  
                # HINT: Log governance failure event
                mlflow.log_event("GovernanceCheckFailed", {"violations": gov_check['violations']}) 
                return [], "Query blocked by security checks."

            # HINT: Log parameters to MLflow
            mlflow.log_param("k", k)  
            mlflow.log_param("query_text", query_text)  
            
            # HINT: Perform similarity search on vector store
            docs = self.vector_store.similarity_search(query_text, k=k) 
            
            # HINT: Log metric for number of results
            mlflow.log_metric("results_count", len(docs))
            
            return docs, query_text

    def synthesize_response(self, docs, user_query):
        """
        Generate a conversational response based on retrieved documents
        
        HINT: This method should:
        1. Start MLflow run
        2. Build context from retrieved documents
        3. Create a prompt for the LLM
        4. Generate response using LLM
        5. Validate output with governance gate
        6. Log response to MLflow
        7. Return final response
        """
        
        mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
        
        with mlflow.start_run(run_name="synthesize_response"):
            # HINT: Handle case when no documents found
            if not docs:
                return "I couldn't find any relevant information to answer your question." 
            
            # HINT: Build context from documents
            # Format: "- {content} (Source: {source})"
            context = "\n".join([
                f"- {doc.page_content} (Source: {doc.metadata.get('source', 'Unknown')})" 
                for doc in docs
            ])
            
            # HINT: Create prompt for LLM
            prompt = f"""
            You are a helpful travel assistant for Wanderlust Travels, an online travel agency.
            Use the following information from our knowledge base to answer the customer's question.
            
            Knowledge Base Information:
            {context}
            
            Customer Question: "{user_query}"
            
            Please provide a clear, helpful, and accurate answer based on the information above.
            If the information is not sufficient, let the customer know and provide general guidance.
            """  # HINT: context, user_query
            
            # HINT: Generate response using LLM
            response = self.llm.invoke(prompt).strip()  
            
            # HINT: Validate output using governance gate
            gov_check = self.governance_gate.validate_output(response) 
            
            if not gov_check['is_valid']:  
                return "I generated a response but it didn't pass safety checks. Please rephrase your question."  # HINT: "I generated a response but it didn't pass safety checks. Please rephrase your question."
            
            # HINT: Log response to MLflow as text file
            mlflow.log_text(response, "final_response.txt")
            
            return response