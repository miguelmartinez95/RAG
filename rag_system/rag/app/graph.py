from langgraph.graph import StateGraph
from .retriever import retrieve_data
from .generator import generate_answer
from .evaluator import evaluate
from .state import RAGState

def generate_graph():

    graph = StateGraph(RAGState)

    graph.add_node("retriever", retrieve_data)
    graph.add_node("generator", generate_answer)
    graph.add_node("evaluator", evaluate)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "evaluator")

    graph.set_finish_point("evaluator")

    rag_graph = graph.compile()

    return rag_graph