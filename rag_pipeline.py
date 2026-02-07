import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 1. Load and Chunk Documents 
def load_documents(folder_path="data"):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs


def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    for doc in docs:
        chunks.extend(splitter.split_text(doc))
    return chunks


# 2. Build FAISS Vector Store 
def build_or_load_faiss_index(chunks, index_path="vectorstore/faiss.index"):
    embedding_model = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )    
    
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_texts(chunks, embedding_model)
        vectorstore.save_local(index_path)

    return vectorstore


# 3. Retrieve Relevant Chunks 
def retrieve_chunks(query, vectorstore, k=3):
    results = vectorstore.similarity_search(query, k=k)
    return results


#  4. Gemini LLM (grounded generation) 
def generate_answer(query, retrieved_chunks):
    context = "\n\n".join([c.page_content for c in retrieved_chunks])

    prompt = f"""
You are an assistant that must ONLY answer based on the provided context.
If the answer cannot be found in the context, say "The documents do not contain this information."

CONTEXT:
{context}

USER QUESTION:
{query}

STRICT REQUIREMENT:
- Use only information from CONTEXT.
- Do not hallucinate.
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    return response.text, context
