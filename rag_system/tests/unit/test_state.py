from rag.app.state import RAGState


def test_rag_state_defaults():
    state = RAGState(query="What is RAG?")

    assert state.query == "What is RAG?"
    assert state.documents == []
    assert state.context == ""
    assert state.answer is None or state.answer == []
    assert state.score == 0.0
