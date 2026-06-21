import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from src.rag_chain import ask_question, get_llm
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL_NAME

def run_evaluation():
    """
    Runs the RAGAS evaluation framework on a small sample dataset.
    This is the key differentiator: it quantitatively measures hallucination 
    and answer quality, showing enterprise-level AI engineering skills.
    """
    
    # 1. Define synthetic Evaluation Dataset.
    # In a real project, you generate these from the PDFs or have clinicians write them.
    questions = [
        "What are the main risk factors for cardiovascular disease?",
        "How does metformin help in managing type 2 diabetes?",
        "What are the common side effects of chemotherapy?"
    ]
    
    # Expected answers (Ground Truths)
    ground_truths = [
        ["Hypertension, smoking, hyperlipidemia, and diabetes are major risk factors."],
        ["Metformin reduces hepatic glucose production and improves insulin sensitivity."],
        ["Common side effects include nausea, fatigue, hair loss, and increased risk of infection."]
    ]
    
    print("Collecting answers and contexts from the RAG chain for evaluation...")
    answers = []
    contexts = []
    
    for q in questions:
        try:
            res = ask_question(q)
            answers.append(res["answer"])
            # Extract content from sources
            doc_contents = [source["content"] for source in res["sources"]]
            contexts.append(doc_contents)
        except Exception as e:
            print(f"Skipping question due to error (Did you ingest PDFs?): {e}")
            answers.append("Error retrieving answer.")
            contexts.append(["Error retrieving context."])
    
    # 2. Prepare the HuggingFace dataset for RAGAS
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths # ragas expects 'ground_truth' or 'ground_truths' depending on version
    }
    
    dataset = Dataset.from_dict(data)
    
    # 3. Setup evaluator LLM and Embeddings
    # RAGAS needs an LLM to evaluate the RAG pipeline's outputs. We use Gemini.
    evaluator_llm = get_llm()
    evaluator_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print("Running RAGAS evaluation (measuring Faithfulness, Relevancy, etc.)...")
    
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )
        
        print("\n=== Evaluation Results ===")
        print(result)
        
        # Save results to a file for the resume/portfolio
        with open("evaluation_report.json", "w") as f:
            json.dump(dict(result), f, indent=4)
        print("Detailed results saved to evaluation_report.json")
        print("\nResume Bullet Metrics (Example):")
        print(f"Faithfulness: {result.get('faithfulness', 'N/A'):.2f}")
        print(f"Answer Relevancy: {result.get('answer_relevancy', 'N/A'):.2f}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("Note: Ensure your GEMINI_API_KEY is active and PDFs are ingested.")

if __name__ == "__main__":
    run_evaluation()
