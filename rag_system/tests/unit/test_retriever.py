from rag.app.retriever import retrieve_data
from langchain.schema import Document

def test_retrieve_data(monkeypatch):
    dummy_docs = [
        Document(page_content="Doc content", metadata={"source": "test"})
    ]

    monkeypatch.setattr(
        "retriever.retriever.get_relevant_documents",
        lambda q: dummy_docs
    )

    state = {"query": "test query"}
    result = retrieve_data(state)

    assert "context" in result
    assert "documents" in result
    assert len(result["documents"]) == 1
    assert "Doc content" in result["context"]
