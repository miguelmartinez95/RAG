from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from pathlib import Path
import yaml
import os
import glob

# ---- Config ----
with open("config/embeddings_config.yaml", "r") as f:
    config = yaml.safe_load(f)

VECTOR_DB_PATH = "/vector_db"               # PVC or local folder
DOCUMENT_PATH = "/data/documents"           # Where raw docs are stored
LOCAL_EMBEDDINGS_PATH = "/models/embeddings"  # Local embeddings model path


MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_MODEL_PATH = f"/models/embeddings/sentence-transformers/{MODEL_NAME}"

if not Path(EMBEDDINGS_MODEL_PATH).exists():
    raise FileNotFoundError(
        f"SentenceTransformer model not found at {EMBEDDINGS_MODEL_PATH}"
    )

print(f"Loading SentenceTransformer from {EMBEDDINGS_MODEL_PATH}")
st_model = SentenceTransformer(EMBEDDINGS_MODEL_PATH)

assert Path(EMBEDDINGS_MODEL_PATH, "modules.json").exists()

# ---- Load documents locally ----
def load_documents():
    if not Path(DOCUMENT_PATH).exists():
        raise RuntimeError(f"{DOCUMENT_PATH} does not exist")

    files = list(Path(DOCUMENT_PATH).glob("*.txt"))
    print(f"Found {len(files)} document files")
    if not files:
        raise RuntimeError("No documents found – ingestion cannot continue")

    documents = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append(Document(page_content=text, metadata={"source": file.name}))
    return documents

# ---- Build Chroma vector DB ----
def build_index():
    raw_documents = load_documents()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=20,
        chunk_overlap=2,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(raw_documents)

    # Pass the ST model to LangChain embeddings
    #embeddings = HuggingFaceEmbeddings(model_cls=lambda: st_model, encode_kwargs={"normalize_embeddings": True})

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL_PATH,
        encode_kwargs={"normalize_embeddings": True},
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
        collection_name="default",
    )
    print(f"Indexed {len(chunks)} chunks")
    return db

if __name__ == "__main__":
    # Ensure local directories exist
    Path(DOCUMENT_PATH).mkdir(parents=True, exist_ok=True)
    Path(VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
    Path(LOCAL_EMBEDDINGS_PATH).mkdir(parents=True, exist_ok=True)

    # Build vector DB directly on local disk
    build_index()

    print("✅ Vector DB ready locally — runtime containers can use it directly")
