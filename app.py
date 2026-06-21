import streamlit as st
from src.rag_chain import ask_question
import os

st.set_page_config(
    page_title="Medical Research Assistant",
    page_icon="⚕️",
    layout="wide"
)

# Custom CSS for professional UI look
st.markdown("""
<style>
    .source-card {
        background-color: #f8f9fa;
        border-left: 4px solid #198754;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 4px;
        font-size: 0.9em;
    }
    .confidence-high { color: #198754; font-weight: bold; }
    .confidence-med { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("⚕️ Medical Research Assistant")
st.markdown("""
This chatbot uses **Retrieval-Augmented Generation (RAG)** to answer medical queries based on ingested research papers. 
Powered by **Gemini 1.5 Flash**, **FAISS**, and **sentence-transformers**. 
_Answers include source citations and confidence scores to prevent hallucination._
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources & Confidence Scores"):
                for source in message["sources"]:
                    score = source['score']
                    # Note: FAISS uses L2 distance. Lower is better. Thresholds are illustrative.
                    if score < 1.0:
                        conf_class = "confidence-high"
                        conf_text = "High"
                    elif score < 1.5:
                        conf_class = "confidence-med"
                        conf_text = "Medium"
                    else:
                        conf_class = "confidence-low"
                        conf_text = "Low"
                        
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>Source:</strong> {source['filename']} (Page {source['page']})<br>
                        <strong>Confidence:</strong> <span class="{conf_class}">{conf_text} (L2 Distance: {score})</span><br>
                        <p style="margin-top: 5px; color: #555;"><i>"{source['content'][:250]}..."</i></p>
                    </div>
                    """, unsafe_allow_html=True)

# React to user input
if prompt := st.chat_input("Ask a medical question based on the ingested documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Searching medical literature..."):
            try:
                # Call our RAG chain
                response = ask_question(prompt)
                answer = response["answer"]
                sources = response["sources"]
                
                # Display the answer
                st.markdown(answer)
                
                # Display the sources in an expander
                with st.expander("View Sources & Confidence Scores"):
                    for source in sources:
                        score = source['score']
                        if score < 1.0:
                            conf_class = "confidence-high"
                            conf_text = "High"
                        elif score < 1.5:
                            conf_class = "confidence-med"
                            conf_text = "Medium"
                        else:
                            conf_class = "confidence-low"
                            conf_text = "Low"
                            
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source:</strong> {source['filename']} (Page {source['page']})<br>
                            <strong>Confidence:</strong> <span class="{conf_class}">{conf_text} (L2 Distance: {score})</span><br>
                            <p style="margin-top: 5px; color: #555;"><i>"{source['content'][:250]}..."</i></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"An error occurred: {e}. Ensure you have ingested PDFs, run `python -m src.vector_store`, and set your `GEMINI_API_KEY` in `.env`.")
