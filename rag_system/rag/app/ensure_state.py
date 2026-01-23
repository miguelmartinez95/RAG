from typing import Any
from .state import RAGState

def ensure_rag_state(state: Any) -> RAGState:
    if isinstance(state, RAGState):
        return state
    if isinstance(state, dict):
        return RAGState(**state)
    raise TypeError(f"Invalid state type: {type(state)}")
