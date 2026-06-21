import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from src.config import LLM_MODEL_NAME, SYSTEM_PROMPT
from src.vector_store import load_vector_store

# Load environment variables (like GEMINI_API_KEY)
load_dotenv()

def get_llm():
    """
    Initializes the Google Gemini Flash model.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY is not set correctly. Please update your .env file.")
        
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=0.0, # Temperature 0 for maximum factuality
        google_api_key=api_key,
    )

def _extract_text(content) -> str:
    """Helper to safely extract string text from LLM response content, which may be a list of dicts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
    return str(content)

def needs_decomposition(query: str) -> bool:
    """Heuristic check for comparison/multi-part questions."""
    comparison_signals = [
        r"\bcompare\b", r"\bcomparison\b", r"\bvs\.?\b", r"\bversus\b",
        r"\bdifference[s]?\b", r"\bboth (documents|papers|studies)\b",
        r"\bone paper.*other paper\b", r"\beach (document|paper|study)\b"
    ]
    return any(re.search(p, query, re.IGNORECASE) for p in comparison_signals)

def decompose_query(query: str, llm) -> list[str]:
    """Ask the LLM to split a comparison question into independent sub-queries."""
    prompt = f"""Split the following question into 2-3 independent, self-contained 
sub-questions, each one focused on a single topic or document, suitable for 
separate vector-database retrieval. Return ONLY the sub-questions, one per line, 
no numbering or extra text.

Question: {query}"""
    response = llm.invoke(prompt)
    content_text = _extract_text(response.content)
    sub_queries = [q.strip() for q in content_text.split("\n") if q.strip()]
    return sub_queries

def ask_question(query: str):
    """
    Runs a query through the RAG pipeline with Query Decomposition for multi-hop queries.
    """
    # 1. Load Vector Store
    vector_store = load_vector_store()
    
    # 2. Get LLM
    llm = get_llm()
    
    # 3. Retrieve relevant documents (with decomposition for comparative queries)
    docs_and_scores = []
    
    if needs_decomposition(query):
        print(f"Decomposing comparative query: {query}")
        sub_queries = decompose_query(query, llm)
        print(f"Generated sub-queries: {sub_queries}")
        
        seen = set()
        for sq in sub_queries:
            # Get top 3 chunks per sub-query
            results = vector_store.similarity_search_with_score(sq, k=3)
            for doc, score in results:
                # Deduplicate chunks based on source, page, and first 50 chars
                key = (doc.metadata.get("filename", ""), doc.metadata.get("page", ""), doc.page_content[:50])
                if key not in seen:
                    seen.add(key)
                    docs_and_scores.append((doc, score))
                    
        # Sort by L2 distance (lower is better) and cap at 6 chunks total to avoid context bloat
        docs_and_scores = sorted(docs_and_scores, key=lambda x: float(x[1]))[:6]
    else:
        # Standard single-hop retrieval
        docs_and_scores = vector_store.similarity_search_with_score(query, k=8)
    
    # 4. Format context for the LLM
    context_text = "\n\n---\n\n".join(doc.page_content for doc, _ in docs_and_scores)
    
    # 5. Format the prompt
    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
    formatted_prompt = prompt.format(context=context_text, question=query)
    
    # 6. Generate Answer
    response = llm.invoke(formatted_prompt)
    answer = _extract_text(response.content)
    
    # 7. Prepare source citations
    sources = []
    for doc, score in docs_and_scores:
        sources.append({
            "content": doc.page_content,
            "filename": doc.metadata.get("filename", "Unknown File"),
            "page": doc.metadata.get("page", "Unknown Page"),
            "score": round(float(score), 4) # Lower L2 distance means higher confidence
        })
        
    return {
        "answer": answer,
        "sources": sources
    }

if __name__ == "__main__":
    try:
        res = ask_question("Compare the cardiac risks discussed in both papers.")
        print("Answer:\n", res['answer'])
    except Exception as e:
        print(f"Failed to test RAG chain: {e}")
