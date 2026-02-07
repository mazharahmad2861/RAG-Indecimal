import streamlit as st
from rag_pipeline import load_documents, chunk_documents, build_or_load_faiss_index, retrieve_chunks, generate_answer

st.set_page_config(page_title="Mini RAG - Construction Assistant", layout="wide")

st.title(" Mini RAG: Construction Assistant")

# Step 1: Load and prepare documents
docs = load_documents()
chunks = chunk_documents(docs)
vectorstore = build_or_load_faiss_index(chunks)

query = st.text_input("Ask a question based on company documents:")

if st.button("Search"):
    if not query.strip():
        st.error("Please enter a query.")
    else:
        retrieved = retrieve_chunks(query, vectorstore, k=3)
        answer, context = generate_answer(query, retrieved)

        st.subheader(" Retrieved Context")
        for i, c in enumerate(retrieved, 1):
            st.markdown(f"**Chunk {i}:**")
            st.write(c.page_content)
            st.markdown("---")

        st.subheader(" Final Answer")
        st.write(answer)
