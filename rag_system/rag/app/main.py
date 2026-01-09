from fastapi import FastAPI
from .graph import generate_graph
from .request_model import QueryRequest
from .state import RAGState
from .generator import GeneratorModel
from fastapi.responses import StreamingResponse
import threading, queue, json


try:
    rag_graph = generate_graph()
    GRAPH_READY = True
except Exception as e:
    print(f"Graph failed to initialized: {e}")
    rag_graph = None
    GRAPH_READY = False


app = FastAPI(title="RAG Inference API")

@app.on_event("startup")
def load_model():
    import threading
    import logging

    def _preload():
        try:
            GeneratorModel.get_models()
            logging.info("Generator model preloaded")
        except Exception as e:
            logging.error(f"Generator preload failed: {e}")

    threading.Thread(target=_preload, daemon=True).start()

@app.post("/rag")
def run_rag(request: QueryRequest):
    state = RAGState(query=request.query)
    result = rag_graph.invoke(state)

    # Ensure final_state is RAGState, not dict
    if isinstance(result, dict):
        final_state = RAGState(**result)
    else:
        final_state = result

    return {
        "answer": final_state.answer,
        "context": final_state.context,
        "documents": [
            {
                "source": doc.metadata.get("source", ""),
                "content": doc.page_content
            }
            for doc in final_state.documents
        ],
        "score": final_state.score,
    }

@app.get("/health/live", tags=["health"])
def liveness():
    # Liveness just checks if the app process is alive
    return {"status": "alive"}

@app.get("/health/ready", tags=["health"])
def readiness():
    reasons = []

    if not GRAPH_READY:
        reasons.append("Graph not ready")

    try:
        from .retriever import db, reranker
        db.similarity_search("health", k=1)
        reranker.score([("health", "test")])
    except Exception as e:
        reasons.append(f"Retrieval not ready: {e}")

    # ✅ Correct readiness check
    if not GeneratorModel.is_ready():
        reasons.append("Generator model not loaded")

    if reasons:
        return {"status": "not ready", "reasons": reasons}, 503

    return {"status": "ready"}