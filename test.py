from rag_pipeline import load_documents, chunk_documents, build_or_load_faiss_index, retrieve_chunks, generate_answer

def test_pipeline():
    docs = load_documents()
    chunks = chunk_documents(docs)
    vectorstore = build_or_load_faiss_index(chunks)

    query = "What factors affect construction project delays?"
    
    retrieved = retrieve_chunks(query, vectorstore, k=3)
    answer, context = generate_answer(query, retrieved)

    print("\n Retrieved Context ")
    print(context)

    print("\nGenerated Answer ")
    print(answer)

    assert len(retrieved) > 0, "No chunks retrieved"
    assert "do not contain" not in answer.lower(), "Unexpected grounding failure"

    print("\nTest Passed!")

if __name__ == "__main__":
    test_pipeline()
