import os
import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import RAW_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def load_and_split_documents():
    """
    Loads all PDF documents from the raw data directory and splits them into chunks.
    Uses PyMuPDF for fast, reliable PDF parsing.
    """
    pdf_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {RAW_DATA_DIR}. Please add some medical PDFs.")
        return []

    print(f"Found {len(pdf_files)} PDFs. Starting ingestion...")
    documents = []
    
    for pdf_path in pdf_files:
        print(f"Loading {os.path.basename(pdf_path)}...")
        # PyMuPDF is fast and extracts metadata (like page numbers) efficiently
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        documents.extend(docs)
        
    print(f"Total pages loaded: {len(documents)}")
    
    # Intelligently chunking at specified size and overlap.
    # Using RecursiveCharacterTextSplitter for semantic awareness of paragraphs and boundaries.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")
    
    # Standardize metadata for easy access in the UI (citations)
    for chunk in chunks:
        # PyMuPDFLoader adds 'source' (full path) and 'page' (0-indexed)
        source = chunk.metadata.get('source', '')
        filename = os.path.basename(source)
        chunk.metadata['filename'] = filename
        
        # Ensure page number is human readable (1-indexed) if needed, 
        # but storing it as is fine, we'll format in UI.
        
    return chunks

if __name__ == "__main__":
    chunks = load_and_split_documents()
