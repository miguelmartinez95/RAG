from pathlib import Path
import yaml
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


TOP_K = 10
TOP_N = 3


# ---- Config ----
with open("config/rag_config.yaml", "r") as f:
    config = yaml.safe_load(f)

RERANKER_OUTPUT_DIR = "/models/reranker_model/cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTOR_DB_PATH = "/vector_db"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_OUTPUT_DIR = f"/models/embeddings/sentence-transformers/{MODEL_NAME}"

# ---- Ensure vector DB exists locally ----
if not Path(VECTOR_DB_PATH).exists() or not any(Path(VECTOR_DB_PATH).glob("*")):
    raise RuntimeError(f"Vector DB not found at {VECTOR_DB_PATH}. Run the local build_index script first.")

# ---- Embeddings and Chroma retriever ----
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDINGS_OUTPUT_DIR,
    encode_kwargs={"normalize_embeddings": True}
)

db = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings,
    collection_name="default"
)

#base_retriever = db.as_retriever(
#    search_type="similarity",
#    search_kwargs={"k": 5}
#)

# ---- Reranker ----
if not Path(RERANKER_OUTPUT_DIR).exists():
    raise RuntimeError(f"Reranker model not found at {RERANKER_OUTPUT_DIR}")

reranker = HuggingFaceCrossEncoder(
    model_name=RERANKER_OUTPUT_DIR
)

#compressor = CrossEncoderReranker(
#    model=reranker,
#    top_n=3
#)
#
#retriever = ContextualCompressionRetriever(
#    base_retriever=base_retriever,
#    base_compressor=compressor
#)

# ---- Retrieval function ----
def retrieve_data(state):
    query = state.query

    # 1️⃣ Vector search
    docs = db.similarity_search(query, k=TOP_K)

    # 2️⃣ Prepare pairs for reranker
    pairs = [(query, doc.page_content) for doc in docs]

    # 3️⃣ Score with cross-encoder
    scores = reranker.score(pairs)

    # 4️⃣ Sort by relevance
    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )[:TOP_N]

    top_docs = [doc for doc, _ in ranked]

    # 5️⃣ Format context
    formatted_chunks = []
    for i, doc in enumerate(top_docs, start=1):
        formatted_chunks.append(
            f"""[Document {i}]
    Source: {doc.metadata.get("source", "unknown")}
    Content: {doc.page_content.strip()}
    """
        )

    state.documents = top_docs
    state.context = "\n------\n".join(formatted_chunks)

    print(f"RETRIEVER: {state.context}")

    return state