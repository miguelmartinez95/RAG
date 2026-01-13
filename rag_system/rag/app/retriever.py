from pathlib import Path
import yaml
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

TOP_K = 10
TOP_N = 3

# ---- Detect CI environment ----
IS_CI = os.getenv("CI", "false") == "true"

MODEL_NAME = "all-MiniLM-L6-v2"
# ---- Embeddings and Chroma retriever ----


if IS_CI:
    RERANKER_OUTPUT_DIR = os.path.join("C:", "kind-data", "actions-runner", "actions-runner", "_work", "RAG", "RAG",
                                       "rag_system", "tests", "ci_models", "reranker",
                                       "cross-encoder", "ms-marco-MiniLM-L-6-v2")
    VECTOR_DB_PATH = Path(
        "C:/kind-data/actions-runner/actions-runner/_work/RAG/RAG/rag_system/tests/tmp_vector_db"
    )
    EMBEDDINGS_OUTPUT_DIR = os.getenv(
        "EMBEDDINGS_MODEL_PATH",
        f"C:/kind-data/actions-runner/actions-runner/_work/RAG/RAG/rag_system/tests/ci_models/embeddings/sentence-transformers/{MODEL_NAME}",
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_OUTPUT_DIR,
        encode_kwargs={"normalize_embeddings": True}
    )
    # Build mini CI DB if missing
    if not VECTOR_DB_PATH.exists() or not any(VECTOR_DB_PATH.glob("*")):
        VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        print("Building CI mini vector DB...")
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document

        txt_files = list(VECTOR_DB_PATH.glob("*.txt"))
        if not txt_files:
            raise RuntimeError("No test documents found in tmp_vector_db for CI")

        documents = [
            Document(page_content=f.read_text(encoding="utf-8"), metadata={"source": f.name})
            for f in txt_files
        ]

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=20,
            chunk_overlap=2,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(VECTOR_DB_PATH),
            collection_name="default",
        )
    else:
        # Use persisted mini DB
        db = Chroma(
            persist_directory=str(VECTOR_DB_PATH),
            embedding_function=embeddings,
            collection_name="default",
        )
else:
    RERANKER_OUTPUT_DIR = "/models/reranker_model/cross-encoder/ms-marco-MiniLM-L-6-v2"
    VECTOR_DB_PATH=os.getenv("VECTOR_DB_PATH", "/vector_db")
    EMBEDDINGS_OUTPUT_DIR = f"/models/embeddings/sentence-transformers/{MODEL_NAME}"
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_OUTPUT_DIR,
        encode_kwargs={"normalize_embeddings": True}
    )
# ---- Ensure vector DB exists for non-CI runs ----
    if not Path(VECTOR_DB_PATH).exists() or not any(Path(VECTOR_DB_PATH).glob("*")):
        raise RuntimeError(f"Vector DB not found at {VECTOR_DB_PATH}. Run the local build_index script first.")


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