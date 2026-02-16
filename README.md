#   RAG – Construction Assistant 

This project implements a Retrieval-Augmented Generation (RAG) pipeline for a construction marketplace assistant.  
The goal is to answer user questions strictly using internal company documents (Markdown files: `doc1.md`, `doc2.md`, `doc3.md`) instead of hallucinated model knowledge.

The system uses semantic retrieval, vector search, and grounded LLM generation to produce trustworthy and context-based answers.

---



## 1. Project Overview

This Mini-RAG system is built with:

- LangChain – for document loading, text splitting, embeddings, vector search, and retrieval.
- FAISS – for fast and local vector similarity search.
- Google Gemini 2.5 Flash – for grounded and accurate answer generation.
- Streamlit – for building a clean and interactive UI.
- Docker - containerized and run using Docker (Ensures consistent environment setup)

## RAG Pipeline Flow

1. Load company documents 
2. Chunk documents into meaningful segments
3. Convert chunks into vector embeddings
4. Store embeddings inside a local FAISS index
5. User asks a question
6. Query is embedded → FAISS retrieves top relevant chunks
7. Retrieved chunks are passed to the Gemini LLM
8. LLM generates an answer *only* using retrieved document text

---

## 2. Embedding Model & LLM Used 

### Embedding Model: `all-MiniLM-L6-v2`

#### Why this model?

- Lightweight and fast (CPU-friendly)
- High semantic similarity accuracy
- Works extremely well with FAISS
- Ideal for small-to-medium RAG systems

It provides high-quality embeddings without requiring GPU hardware.

---

### LLM: Gemini 2.5 Flash

#### Why this model?

- Fast inference suitable for interactive applications
- High reasoning and summarization capabilities
- Large context window
- Strong compliance with grounding instructions
- Cost-efficient for production use

---

## 3. Document Chunking & Retrieval Implementation

### Document Chunking

We use the `RecursiveCharacterTextSplitter` from LangChain with:

- `chunk_size = 500`
- `chunk_overlap = 50`

#### Why chunking?

- Embeddings work best on small, coherent text segments  
- Improves semantic search accuracy  
- Avoids mixing unrelated topics  
- Ensures the LLM receives relevant context only  

---

### Semantic Retrieval with FAISS

How retrieval works:

1. Each chunk is converted into an embedding vector  
2. Vectors are indexed inside FAISS  
3. User query → converted into an embedding  
4. FAISS performs similarity search  
5. Top-k most relevant chunks are returned  

#### Why FAISS?

- Very fast vector similarity search  
- Works entirely offline  
- Scales well with large document sets  
- Ideal for enterprise RAG systems  

---

#### 4. How Grounding to Retrieved Context Is Enforced

To prevent hallucinations, we enforce **strict grounding** in the LLM prompt:

#### Why this works

- The LLM only receives the retrieved chunks  
- Explicit instructions forbid external knowledge  
- If documents lack information → model must decline  
- Ensures reliable and fully explainable outputs  

---

## Example End-to-End Flow

1. User question:  
   “What factors cause construction delays?”

2. Query embedding is generated  
3. FAISS retrieves chunk:  
   “Delays occur due to material shortages, labor unavailability, and weather conditions.”

4. Gemini responds using only this chunk:  
   “Construction delays can occur due to material shortages, weather issues, and lack of labor.”

The UI displays:

- Retrieved chunks  
- Final grounded answer  

---

# Summary

This Mini RAG system provides:

- Semantic search over internal `.md` documents  
- Accurate and grounded LLM answers  
- Transparent context display  
- Streamlit-based chatbot UI
- Docker ensures consistent environment setup
- Lightweight CPU-friendly architecture  

Ideal for:

- Internal assistants  
- Enterprise policy/FAQ bots  
- Knowledge-base question answering  
- Document-grounded enterprise AI systems




# 📸 Screenshots – Working Demo

<img width="1354" height="799" alt="Screenshot 2026-02-07 210755" src="https://github.com/user-attachments/assets/8849652d-5df7-43c8-ad5c-3076faf889c4" />  
<img width="1625" height="798" alt="Screenshot 2026-02-07 210955" src="https://github.com/user-attachments/assets/8f45d328-c0cb-4529-a568-d5b42c915875" /> 
<img width="1694" height="863" alt="Screenshot 2026-02-07 211945" src="https://github.com/user-attachments/assets/9e06b149-f43b-49d3-82a1-5a70de4594f7" /> 
<img width="800" height="338" alt="Screenshot 2026-02-07 212006" src="https://github.com/user-attachments/assets/a1bb10d8-6591-4192-9ce7-f7ee0088754f" /> 
<img width="1352" height="821" alt="Screenshot 2026-02-07 212144" src="https://github.com/user-attachments/assets/4ecff129-e88c-4a00-9e9c-7eb6c1aff402" /> 
<img width="1053" height="720" alt="Screenshot 2026-02-07 212157" src="https://github.com/user-attachments/assets/0a1317ef-dd77-41f8-8e5a-d01ffbcaaa71" /> 
<img width="953" height="780" alt="Screenshot 2026-02-07 212409" src="https://github.com/user-attachments/assets/4276cca5-c0e4-4bbb-a3b2-faf2abeb285f" /> 
<img width="750" height="533" alt="Screenshot 2026-02-07 212427" src="https://github.com/user-attachments/assets/f5965793-d7df-4864-ab2e-6663a024eb60" />










