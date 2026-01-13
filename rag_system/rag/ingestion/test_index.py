
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

VECTOR_DB_PATH = Path(
    "C:/kind-data/actions-runner/actions-runner/_work/RAG/RAG/rag_system/tests/tmp_vector_db"
)
DOCS_PATH = VECTOR_DB_PATH  # in CI, txt files live here

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_MODEL_PATH = os.getenv(
    "EMBEDDINGS_MODEL_PATH",
    f"C:/kind-data/actions-runner/actions-runner/_work/RAG/RAG/rag_system/tests/ci_models/embeddings/sentence-transformers/{MODEL_NAME}",
)

def main():
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

    # If already populated, skip
    if any(p.suffix in {".sqlite3", ".parquet"} for p in VECTOR_DB_PATH.iterdir()):
        print("CI vector DB already exists – skipping build")
        return

    print("Building CI vector DB...")

    if not Path(EMBEDDINGS_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Embeddings model not found at {EMBEDDINGS_MODEL_PATH}"
        )

    txt_files = list(DOCS_PATH.glob("*.txt"))
    if not txt_files:
        raise RuntimeError("No .txt files found in tests/tmp_vector_db")

    documents = []
    for f in txt_files:
        documents.append(
            Document(
                page_content=f.read_text(encoding="utf-8"),
                metadata={"source": f.name},
            )
        )

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=20,
        chunk_overlap=2,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL_PATH,
        encode_kwargs={"normalize_embeddings": True},
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_PATH),
        collection_name="default",
    )

    print(f"CI vector DB built with {len(chunks)} chunks")


if __name__ == "__main__":
    main()
