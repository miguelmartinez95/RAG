import pytest
from rag.app.graph import generate_graph
from rag.app.state import RAGState

@pytest.mark.integration
def test_full_rag_pipeline(monkeypatch):
    # Mock retriever only
    from langchain.schema import Document

    monkeypatch.setattr(
        "retriever.retriever.get_relevant_documents",
        lambda q: [
            Document(page_content="Integration doc", metadata={"source": "int"})
        ]
    )

    graph = generate_graph()

    state = RAGState(query="What is RAG?")
    result = graph.invoke(state)

    assert result.answer is not None
    assert isinstance(result.score, float)
