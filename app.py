"""
Ask My Docs — Flask backend

Wraps the RAG pipeline (PDF/TXT ingestion -> chunking -> hybrid retrieval
(vector + BM25) -> cross-encoder reranking -> grounded generation) that was
prototyped in the notebook, behind four HTTP endpoints:

    GET  /        serves the chat frontend
    POST /upload  accepts a .pdf or .txt file, chunks + indexes it
    POST /ask     accepts {"question": "..."}, returns {"answer", "sources"}
    POST /reset   clears the in-memory knowledge base

Run with:
    export OPENAI_API_KEY="sk-..."
    python app.py
"""

import os
import tempfile

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from langchain_openai import ChatOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COLLECTION_NAME = "ask_my_docs"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
VECTOR_K = 10
BM25_K = 10
RERANK_TOP_N = 3
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Models — loaded once at startup (these are the slow parts)
# ---------------------------------------------------------------------------

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = len(embeddings.embed_query("dimension probe"))

print("Loading cross-encoder reranker...")
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

if not os.environ.get("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY is not set. /ask will fail until it is.")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))

prompt = ChatPromptTemplate.from_template(
    "You are answering a question using only the context provided below, "
    "which comes from documents the user uploaded. "
    "If the answer isn't contained in the context, say you don't know instead of guessing. "
    "Do not use outside knowledge.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ---------------------------------------------------------------------------
# In-memory vector store + knowledge base state
#
# Qdrant is run with location=":memory:" — no external database process and
# nothing written to disk, matching the "in-memory, no external vector DB"
# design goal, while still getting Qdrant's similarity search API. The
# knowledge base resets whenever the server restarts, and also via /reset.
# ---------------------------------------------------------------------------

qdrant_client = QdrantClient(location=":memory:")
qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
)
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# All indexed chunks, kept in memory too so BM25 (which has no incremental
# API) can be rebuilt whenever the knowledge base changes.
all_chunks: list[Document] = []
uploaded_files: list[str] = []

# Lazily (re)built retriever pipeline; invalidated on upload/reset.
_retriever = None


def invalidate_retriever():
    global _retriever
    _retriever = None


def get_retriever():
    """Build (or reuse) the hybrid retrieval + reranking pipeline."""
    global _retriever
    if _retriever is not None:
        return _retriever
    if not all_chunks:
        return None

    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": VECTOR_K},
    )

    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = BM25_K

    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    reranker = CrossEncoderReranker(model=cross_encoder, top_n=RERANK_TOP_N)

    _retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble,
    )
    return _retriever


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------


def load_documents(filepath: str, filename: str) -> list[Document]:
    """Extract raw text from an uploaded .pdf or .txt file."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        docs = PyPDFLoader(filepath).load()
    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        docs = [Document(page_content=text, metadata={})]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    for doc in docs:
        doc.metadata["source"] = filename

    return docs


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    if not filename:
        return jsonify({"error": "No file selected."}), 400
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Only .pdf and .txt files are supported."}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        docs = load_documents(tmp_path, filename)
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400
    finally:
        os.unlink(tmp_path)

    if not any(d.page_content.strip() for d in docs):
        return jsonify({"error": "No extractable text found in that file."}), 400

    chunks = text_splitter.split_documents(docs)

    vector_store.add_documents(chunks)
    all_chunks.extend(chunks)
    if filename not in uploaded_files:
        uploaded_files.append(filename)
    invalidate_retriever()

    return jsonify(
        {
            "filename": filename,
            "chunks_added": len(chunks),
            "total_chunks": len(all_chunks),
            "files": uploaded_files,
        }
    )


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Question is required."}), 400

    retriever = get_retriever()
    if retriever is None:
        return jsonify(
            {
                "answer": "There's nothing in the knowledge base yet — upload a "
                "PDF or text file first.",
                "sources": [],
            }
        )

    if not os.environ.get("OPENAI_API_KEY"):
        return jsonify({"error": "OPENAI_API_KEY is not set on the server."}), 500

    try:
        retrieved_docs = retriever.invoke(question)
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {e}"}), 500

    if not retrieved_docs:
        return jsonify(
            {
                "answer": "I couldn't find anything relevant to that question in "
                "the uploaded documents.",
                "sources": [],
            }
        )

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    sources = sorted({doc.metadata.get("source", "unknown") for doc in retrieved_docs})

    try:
        response = (prompt | llm).invoke({"context": context, "question": question})
    except Exception as e:
        return jsonify({"error": f"Generation failed: {e}"}), 500

    return jsonify({"answer": response.content, "sources": sources})


@app.route("/reset", methods=["POST"])
def reset():
    global all_chunks, uploaded_files

    qdrant_client.delete_collection(COLLECTION_NAME)
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    all_chunks = []
    uploaded_files = []
    invalidate_retriever()

    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
