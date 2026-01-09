import numpy as np
import time
import json
import yaml
import statistics
from .graph import generate_graph
from .retriever import retrieve_data, db, reranker, TOP_K, TOP_N
from langchain.embeddings import HuggingFaceEmbeddings

with open("config/rag_config.yaml", "r") as f:
    config = yaml.safe_load(f)

EVAL_DATA_PATH = "/app/rag/data/rag_eval.jsonl"
EMBEDDINGS_OUTPUT_DIR = f"/models/embeddings/sentence-transformers/{MODEL_NAME}"


def compute_metrics():
    graph = generate_graph()

    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_OUTPUT_DIR,
        encode_kwargs={"normalize_embeddings": True}
    )

    hits = []
    recalls = []
    reranker_gains = []
    context_precisions = []
    eval_scores = []
    groundedness_scores = []
    latencies = []

    with open(EVAL_DATA_PATH) as f:
        eval_data = [json.loads(l) for l in f]

    for sample in eval_data:
        query = sample["query"]
        gold_docs = set(sample["gold_docs"])

        # ---- Base retrieval only ----
        base_docs = db.similarity_search(query, k=TOP_K)
        base_ids = [d.metadata.get("source") for d in base_docs]

        hit = int(any(doc in gold_docs for doc in base_ids))
        recall = len(set(base_ids) & gold_docs) / max(len(gold_docs), 1)

        # ---- Reranking ----
        pairs = [(query, doc.page_content) for doc in base_docs]
        scores = reranker.score(pairs)
        ranked = sorted(zip(base_docs, scores), key=lambda x: x[1], reverse=True)[:TOP_N]
        reranked_docs = [doc for doc, _ in ranked]
        reranked_ids = [d.metadata.get("source") for d in reranked_docs]

        rank_before = np.mean([base_ids.index(d) if d in base_ids else TOP_K for d in gold_docs])
        rank_after = np.mean([reranked_ids.index(d) if d in reranked_ids else TOP_N for d in gold_docs])

        reranker_gain = rank_before - rank_after
        context_precision = sum(d in gold_docs for d in reranked_ids) / len(reranked_ids)

        # ---- Full RAG pipeline ----
        start = time.time()
        state = graph.run({"query": query})
        latency = (time.time() - start) * 1000

        score = state.score
        answer = state.answer
        context = state.context

        # ---- Groundedness ----
        a_emb = embedder.embed_query(answer)
        c_emb = embedder.embed_query(context)
        groundedness = np.dot(a_emb, c_emb)

        # ---- Collect metrics ----
        hits.append(hit)
        recalls.append(recall)
        reranker_gains.append(reranker_gain)
        context_precisions.append(context_precision)
        eval_scores.append(score)
        groundedness_scores.append(groundedness)
        latencies.append(latency)

    return {
        "retrieval_hit_rate@5": np.mean(hits),
        "retrieval_recall@5": np.mean(recalls),
        "reranker_gain": np.mean(reranker_gains),
        "context_precision@3": np.mean(context_precisions),
        "avg_eval_score": np.mean(eval_scores),
        "min_eval_score": min(eval_scores),
        "eval_score_std": statistics.stdev(eval_scores),
        "answer_groundedness": np.mean(groundedness_scores),
        "avg_latency_ms": np.mean(latencies),
    }