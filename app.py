import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import streamlit as st
import ollama

st.title("Personal AI Chatbot for PDFs")

uploaded_file = st.file_uploader("Upload Your PDF",type="pdf")

if uploaded_file is not None:
    st.write("File Uploaded Successfully!!")

    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    #clean Text
    text = text.replace("\n"," ")
    text = " ".join(text.split())

    st.write("PDF text extracted")

    #Chunk the text
    def chunk_text(text, max_length=300):
        words = text.split()
        chunks=[]
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    chunks = chunk_text(text)
    st.write(f"total Chunks created: {len(chunks)}")

    #load embedding model and create embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    st.write("Embeddings Created!")

    #create FAISS
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    #ask Questions
    raw_query = st.text_input("ask a question from your pdf:")
   
    query = f"Explain the {raw_query} described in the documents"

    query_embedding = model.encode([query], convert_to_numpy= True)
    faiss.normalize_L2(query_embedding)

    #search Faiss index
    distances, indices = index.search(query_embedding, 3)

    #compile top chunks
    context = "\n\n".join([chunks[i] for i in indices[0]])

    prompt = f"""
use the context below to answer the question clearly and concisely.

Context:
{context}

Questions:
{query}

Answer:
"""
    #Ask LLM
    response= ollama.chat(
        model = "llama3",
        messages = [{"role":"user","content":prompt}]
    )
    st.subheader("\nAnswer:")
    st.write(response["message"],["content"])



  





