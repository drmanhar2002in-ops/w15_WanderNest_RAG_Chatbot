"""
Ragas Evaluation for Travel Chatbot
RUBRIC: Evaluation Framework (RAGAS) (8 marks total)
- RAGAS evaluation implemented (3 marks)
- Golden dataset created (2 marks)
- All four metrics computed (2 marks)
- Results saved with pass/fail logic (1 mark)

TASK: Implement Ragas evaluation with 4 metrics
"""
import os
import json
import logging
import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_openai import AzureChatOpenAI
#from langchain_openai import ChatOpenAI
from src.config import Config


from src.search_engine import TravelSearchEngine
from src.config import Config

# HINT: Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluation")


# Initialize evaluator
os.environ["OPENAI_API_KEY"] = Config.AZURE_OPENAI_API_KEY


class TravelChatbotEvaluator:
    """Evaluates Travel Chatbot using Ragas metrics"""
    
    def __init__(self):
        # HINT: Initialize search engine and golden dataset path
        self.engine = TravelSearchEngine() 
        self.golden_dataset_path = Path("data") / "golden_dataset.json"  # HINT: "data", "golden_dataset.json"

        self.llm = AzureChatOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            deployment_name="gpt-4.1",
            temperature=0.0
        )
    
    def load_golden_dataset(self) -> List[Dict]:
        """
        Load golden dataset for evaluation
        
        HINT: Check if file exists, if not create sample dataset
        """
        if not self.golden_dataset_path.exists(): 
            logger.warning(f"Golden dataset not found at {self.golden_dataset_path}")
            logger.info("Creating sample golden dataset...")
            return self._create_sample_dataset()  
    
        with open(self.golden_dataset_path, 'r') as f: 
            return json.load(f)  
    
    def _create_sample_dataset(self) -> List[Dict]:
        """
        Create sample golden dataset if not exists
        
        HINT: Create list of dicts with 'question' and 'ground_truth' keys
        Save to golden_dataset_path
        """
        sample_data = [
            {
                "question": "What are the baggage allowance rules for international flights?",  # HINT: "What are the baggage allowance rules for international flights?"
                "ground_truth": "The baggage allowance for international flights varies by cabin class and destination. Please check with Air India for specific details."  # HINT: Appropriate ground truth answer
            },
            {
                "question": "What is Air India's cancellation policy?",  # HINT: "What is Air India's cancellation policy?"
                "ground_truth": "Air India's cancellation policy depends on the fare type and booking class. Please refer to the terms and conditions of your ticket for detailed information."  # HINT: Appropriate ground truth answer
            },
            {
                "question": "Do I need a visa to travel from India to UK?",  # HINT: "Do I need a visa to travel from India to UK?"
                "ground_truth": "Yes, Indian citizens need a valid visa to travel to the UK."  # HINT: Appropriate ground truth answer
            },
            {
                "question": "What are the refund policies for flight cancellations?",  # HINT: "What are the refund policies for flight cancellations?"
                "ground_truth": "Refund policies for flight cancellations depend on the fare type and booking class. Please refer to the terms and conditions of your ticket for detailed information."  # HINT: Appropriate ground truth answer
            },
            {
                "question": "What documents do I need for international travel?",  # HINT: "What documents do I need for international travel?"
                "ground_truth": "For international travel, you typically need a valid passport, visa (if required), and sometimes additional documents like travel insurance or health certificates."  # HINT: Appropriate ground truth answer
            }
        ]
        
        # HINT: Save sample dataset
        self.golden_dataset_path.parent.mkdir(exist_ok=True)
        with open(self.golden_dataset_path, 'w') as f:  
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"Sample golden dataset saved to {self.golden_dataset_path}")
        return sample_data
    
    def generate_responses(self, questions: List[str]) -> tuple:
        """
        Generate responses for questions
        
        HINT: For each question:
        1. Search for documents
        2. Synthesize response
        3. Collect contexts
        Return (answers, contexts)
        """
        answers = []
        contexts = []
        
        for question in questions:
            logger.info(f"Generating answer for: {question}")
            
            try:
                # HINT: Search for relevant documents
                docs, _ = self.engine.search_by_text(question, k=3) 
                
                # HINT: Generate answer
                answer = self.engine.synthesize_response(docs, question)
                
                # HINT: Collect contexts (retrieved documents)
                #context_texts = [doc.page_content for doc in docs]
                context_texts = [doc.__dict__.get('page_content', '') for doc in docs]
                
                
                answers.append(answer)
                contexts.append(context_texts) 
                
            except Exception as e:
                logger.error(f"Error generating answer for '{question}': {e}")
                answers.append("Error generating answer.") 
                contexts.append([])
        
        return answers, contexts
    
    async def run_ragas_evaluation(self):
        """
        Run Ragas evaluation
        
        HINT: This method should:
        1. Load golden dataset
        2. Generate responses
        3. Prepare dataset dict
        4. Run Ragas evaluation with 4 metrics
        5. Save results
        """
        logger.info("=" * 70)
        logger.info("Starting Ragas Evaluation...")
        logger.info("=" * 70)
        
        # HINT: Load golden dataset
        golden_data = self.load_golden_dataset()  
        
        if not golden_data:
            logger.error("No evaluation data available")
            return None
        
        logger.info(f"Loaded {len(golden_data)} test cases")
        
        # HINT: Extract questions and ground truths
        questions = [item["question"] for item in golden_data] 
        ground_truths = [item["ground_truth"] for item in golden_data]
        categories = [item["category"] for item in golden_data] 
        
        # HINT: Generate answers and contexts
        logger.info("\nGenerating responses...")
        answers, contexts = self.generate_responses(questions) 
        
        # HINT: Prepare dataset for Ragas
        dataset_dict = {
            "question": questions,  
            "answer": answers,   
            "contexts": contexts,  
            "category": categories,
            "ground_truth": ground_truths  
        }
        
        # HINT: Create HuggingFace Dataset
        hf_dataset = Dataset.from_dict(dataset_dict) 
        
        logger.info("\nRunning Ragas metrics...")
        logger.info("Metrics: faithfulness, answer_relevancy, context_precision, context_recall")
        
        # HINT: Run evaluation
        try:
            results = evaluate(
                hf_dataset,
                metrics=[
                    faithfulness,  # HINT: faithfulness
                    answer_relevancy,  # HINT: answer_relevancy
                    context_precision,  # HINT: context_precision
                    context_recall,   # HINT: context_recall
                ],
                llm=self.llm
            )
            
            logger.info("\n" + "=" * 70)
            logger.info("EVALUATION RESULTS")
            logger.info("=" * 70)
            logger.info(f"\nRagas Scores:")
            faithfulness_val = self._normalize_metric(results['faithfulness'])
            answer_relevancy_val = self._normalize_metric(results['answer_relevancy'])
            context_precision_val = self._normalize_metric(results['context_precision'])
            context_recall_val = self._normalize_metric(results['context_recall'])
            logger.info(f"  Faithfulness:       {faithfulness_val:.4f}")  
            logger.info(f"  Answer Relevancy:   {answer_relevancy_val:.4f}")   
            logger.info(f"  Context Precision:  {context_precision_val:.4f}") 
            logger.info(f"  Context Recall:     {context_recall_val:.4f}")  
            logger.info("=" * 70)
            
            # HINT: Save detailed results
            self._save_results(results, dataset_dict)  # HINT: _save_results
            
            return results
            
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            logger.error("Make sure you have OPENAI_API_KEY set for Ragas to work")
            return None

    def _normalize_metric(self, value):
        try:
            if isinstance(value, list):
                return float(pd.Series(value).mean()) if value else 0.0
            if pd.isna(value):
                return 0.0
            return float(value)
        except Exception:
            return 0.0

    def _save_results(self, results: dict, dataset_dict: dict):
        """
        Save evaluation results to file
        
        HINT: Save summary JSON and detailed CSV
        """
        output_dir = Path("reports")  # HINT: "reports"
        output_dir.mkdir(exist_ok=True)
        
        # HINT: Save summary
        summary = {
            "faithfulness": self._normalize_metric(results['faithfulness']),  # HINT: 'faithfulness'
            "answer_relevancy": self._normalize_metric(results['answer_relevancy']),  # HINT: 'answer_relevancy'
            "context_precision": self._normalize_metric(results['context_precision']),  # HINT: 'context_precision'
            "context_recall": self._normalize_metric(results['context_recall']),  # HINT: 'context_recall'
            "total_test_cases": len(dataset_dict["question"])
        }
        
        summary_path = output_dir / "evaluation_summary.json"  # HINT: "evaluation_summary.json"
        with open(summary_path, 'w') as f:  # HINT: 'w'
            json.dump(summary, f, indent=2)  # HINT: dump
        
        logger.info(f"\n✅ Evaluation summary saved to {summary_path}")
        
        # HINT: Save detailed results
        detailed_df = pd.DataFrame(dataset_dict)  # HINT: dataset_dict
        detailed_path = output_dir / "evaluation_detailed.csv"  # HINT: "evaluation_detailed.csv"
        detailed_df.to_csv(detailed_path, index=False)
        
        logger.info(f"✅ Detailed results saved to {detailed_path}")
    
    def run(self):
        """Run evaluation (sync wrapper)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.run_ragas_evaluation())


def run_evaluation():
    """
    Main evaluation function
    
    HINT: Run evaluator and check if results pass thresholds
    """
    evaluator = TravelChatbotEvaluator()  
    results = evaluator.run() 
    
    if results:
        # HINT: Check if evaluation passes minimum thresholds
        min_faithfulness = 0.7  
        min_relevancy = 0.7 
        
        faithfulness_val = evaluator._normalize_metric(results['faithfulness'])
        answer_relevancy_val = evaluator._normalize_metric(results['answer_relevancy'])
        
        passed = (
            faithfulness_val >= min_faithfulness and  
            answer_relevancy_val >= min_relevancy
        )
        
        if passed:
            logger.info("\n✅ EVALUATION PASSED")
            return 0
        else:
            logger.warning("\n⚠️  EVALUATION BELOW THRESHOLDS")
            return 1
    else:
        logger.error("\n❌ EVALUATION FAILED")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = run_evaluation()
    sys.exit(exit_code)