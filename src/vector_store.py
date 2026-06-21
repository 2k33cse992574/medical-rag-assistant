import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import VECTOR_STORE_DIR, EMBEDDING_MODEL_NAME
from src.ingest import load_and_split_documents

def get_embeddings():
    """
    Returns the HuggingFace embeddings model.
    Sentence-transformers (all-MiniLM-L6-v2) runs locally and is free.
    """
    # model_kwargs={'device': 'cpu'} can be added if we want to ensure it runs on CPU, 
    # but HuggingFaceEmbeddings handles this automatically.
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def build_vector_store():
    """
    Loads chunks, generates embeddings, and saves the FAISS index locally.
    """
    chunks = load_and_split_documents()
    if not chunks:
        print("No documents to embed. Exiting.")
        return

    print(f"Generating embeddings using {EMBEDDING_MODEL_NAME} and building FAISS vector store...")
    embeddings = get_embeddings()
    
    # FAISS is used for fast, local similarity search
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save the index locally
    index_path = os.path.join(VECTOR_STORE_DIR, "faiss_index")
    vector_store.save_local(index_path)
    print(f"Vector store successfully saved to {index_path}")

def load_vector_store():
    """
    Loads the saved FAISS vector store from disk.
    """
    index_path = os.path.join(VECTOR_STORE_DIR, "faiss_index")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}. Please place PDFs in data/raw and run `python -m src.vector_store` first.")
        
    embeddings = get_embeddings()
    # allow_dangerous_deserialization=True is required in newer Langchain versions 
    # for loading local FAISS indices (since pickle is used under the hood).
    # Since we generate this file locally, it is safe.
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

if __name__ == "__main__":
    build_vector_store()
