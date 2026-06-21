import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Chunking Configuration
# Interview tip: Intelligently chosen chunk size ensures complete context without dilution.
# 512 tokens with 150-token overlap captures medical paragraph boundaries well.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 150

# Embedding Configuration
# all-MiniLM-L6-v2 is an excellent, efficient local model suitable for free tier / fast inference
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Configuration
# Gemini 1.5 Flash offers 1M token context window, fast processing, and is available in free tier
LLM_MODEL_NAME = "gemini-flash-latest"

# System Prompt for Hallucination Prevention
SYSTEM_PROMPT = """You are an expert Medical Research Assistant. Your task is to answer user queries using ONLY the provided medical research documents.
You must adhere strictly to the following rules:
1. Answer ONLY using the facts from the provided context.
2. If the answer is not contained within the context, you must explicitly state: "I cannot answer this question based on the provided documents."
3. Do not invent, hallucinate, or assume any medical information outside of the provided text.
4. If the user asks you to compare studies or papers, you must explicitly contrast their structural differences, such as:
   - Causal direction (e.g., treatments for prevention vs. side effects/harm caused by treatments)
   - Study design (e.g., randomized controlled trials vs. retrospective/observational cohorts)
   - Timescale (e.g., short-term follow-up vs. 15+ years latency)
   - Population (e.g., cardiac patients vs. cancer patients)
5. When you provide an answer, try to be concise and reference the context when appropriate.

Context:
{context}

Question:
{question}
"""
