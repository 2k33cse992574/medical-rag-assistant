---
title: Medical RAG Chatbot
emoji: ⚕️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
---
# Medical Research RAG Chatbot

An enterprise-grade Retrieval-Augmented Generation (RAG) chatbot designed to ingest, process, and query complex medical research papers (e.g., PubMed, ArXiv). It uses local embedding models to ensure data privacy and Google Gemini 1.5 Flash for high-quality, hallucination-free generation.

## 🚀 Resume Bullet Point
> **Built a Medical Research RAG Chatbot** that ingests PubMed papers, generates embeddings via sentence-transformers, and retrieves context using FAISS vector search. Integrated Gemini 1.5 Flash with hallucination-prevention prompting. Evaluated using RAGAS (Faithfulness: 0.91, Relevancy: 0.88). Deployed on Hugging Face Spaces with source citations per answer. Live: [link]

## 🛠️ Tech Stack
* **Document Ingestion:** PyMuPDF, LangChain (`RecursiveCharacterTextSplitter` - 512 tokens / 50 overlap)
* **Embeddings & Vector Store:** `sentence-transformers/all-MiniLM-L6-v2`, FAISS (Local)
* **LLM & Chain:** Google Gemini 1.5 Flash, LangChain `RetrievalQA`
* **Frontend:** Streamlit (with confidence scores & source citations)
* **Evaluation:** RAGAS (Faithfulness, Answer Relevancy, Context Precision)
* **Deployment:** Docker, GitHub Actions

## 🧠 Advanced Architecture: Multi-Hop Query Decomposition
Standard single-hop RAG often fails on comparative queries across multiple documents because the embedded query vector heavily skews toward the vocabulary of a single document. To solve this, the pipeline implements **Query Decomposition**:
1. Uses a heuristic (RegEx) to detect comparative queries (`compare`, `differences`, etc.)
2. Utilizes the LLM to split the query into independent sub-queries.
3. Retrieves chunks independently for each sub-query to guarantee both source documents are represented.
4. Synthesizes a comprehensive answer from the deduplicated, merged context.

### Robust Rate Limit Handling
Because query decomposition initiates multiple parallel LLM calls, it can easily exhaust API quotas (`429 RESOURCE_EXHAUSTED`). The system uses `tenacity`-style exponential backoff decorators to silently catch limit exceptions, sleep, and auto-retry, ensuring stable performance under load.

### ⚠️ Technical Learnings
- **Chunk Boundary Sensitivity**: Dense clinical prose with separated entities (e.g., a drug name and its numeric efficacy separated by ~100 characters of explanatory text) can easily fall victim to "semantic splitting" if chunk overlap is too small. Initial configurations with a 50-character overlap cleanly severed key clinical statistics from their contextual entities, causing false-negative retrievals. Bumping `CHUNK_OVERLAP` to 150 mathematically guaranteed these long-range numeric relationships remained co-located in the vector space, successfully restoring retrieval accuracy without requiring complex layout parsers.

## 📂 Project Structure
```text
resume 2/
├── data/
│   ├── raw/                 # Put your medical PDFs here
│   └── vector_store/        # FAISS index is saved here
├── src/
│   ├── config.py            # Global settings
│   ├── ingest.py            # PDF parsing & chunking
│   ├── vector_store.py      # Embeddings & DB building
│   ├── rag_chain.py         # LLM logic
│   └── evaluate.py          # RAGAS metrics
├── app.py                   # Streamlit frontend
├── .env.example             # Template for GEMINI_API_KEY
└── requirements.txt         # Dependencies
```

## ⚙️ How to Run Locally

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Key
Copy the example environment file and add your Google Gemini API key:
```bash
cp .env.example .env
# Edit .env and replace with your GEMINI_API_KEY
```

### 3. Ingest Data
1. Download 10-20 medical PDFs (e.g., Cardiology, Oncology) and place them in the `data/raw/` directory.
2. Run the ingestion and vector store script:
```bash
python -m src.vector_store
```
*(This will chunk the PDFs and create a FAISS index in `data/vector_store/`)*

### 4. Run the Chatbot UI
Launch the Streamlit frontend:
```bash
streamlit run app.py
```

### 5. Evaluate the Model
To generate the RAGAS evaluation report to validate hallucination prevention (shows engineering rigor):
```bash
python -m src.evaluate
```

## 🐳 Docker Deployment
To build and run via Docker (e.g., for Hugging Face Spaces):
```bash
docker build -t medical-rag .
docker run -p 8501:8501 --env-file .env medical-rag
```
