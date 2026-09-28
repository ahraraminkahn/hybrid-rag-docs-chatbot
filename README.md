Under the hood it's a small Flask app wrapping a retrieval-augmented generation (RAG) pipeline: hybrid retrieval (vector + BM25), cross-encoder reranking, and grounded generation with an LLM.

Features
PDF and TXT ingestion – drag and drop or click to upload
Hybrid retrieval – semantic search (embeddings) combined with keyword search (BM25) for better recall
Cross-encoder reranking – the top candidates are re-scored so only the most relevant chunks reach the LLM
Grounded answers – the prompt instructs the model to answer only from the retrieved context, and to say "I don't know" otherwise
Source citations – each answer lists the files its context came from
No external database – Qdrant runs fully in memory, so there is nothing to install or configure besides Python packages
Clean single-page chat UI – no build step, no frontend framework
How it works
Upload (.pdf / .txt)
   └─► extract text ─► split into chunks (700 chars, 100 overlap)
                          ├─► embed ─► Qdrant (in-memory)
                          └─► kept in memory for BM25

Question
   └─► Ensemble retriever (vector top-10 + BM25 top-10, 50/50 weights)
          └─► Cross-encoder reranker (keep top 3)
                 └─► Prompt + context ─► GPT-4o-mini ─► answer + sources
Stage	Component
Loading	PyPDFLoader / plain text
Chunking	RecursiveCharacterTextSplitter
Embeddings	sentence-transformers/all-MiniLM-L6-v2
Vector store	Qdrant (:memory: mode)
Keyword search	BM25 (rank_bm25)
Reranker	cross-encoder/ms-marco-MiniLM-L-6-v2
LLM	OpenAI gpt-4o-mini
Web layer	Flask
Project structure
ask-my-docs/
├── app.py              # Flask backend + RAG pipeline
├── templates/
│   └── index.html      # Chat frontend
├── requirements.txt
├── .env                # Your OpenAI key (not committed)
└── README.md

Note: Flask's render_template looks in a templates/ folder, so index.html must live there.

Getting started
Prerequisites
Python 3.10+
An OpenAI API key
Installation
bash
git clone https://github.com/<your-username>/ask-my-docs.git
cd ask-my-docs

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
Configuration

Create a .env file in the project root:

OPENAI_API_KEY=sk-...

Or export it in your shell:

bash
export OPENAI_API_KEY="sk-..."
Run
bash
python app.py

Then open http://localhost:5000.

The first launch downloads the embedding and reranker models from Hugging Face, so startup can take a minute. Later launches use the local cache.

Usage
Upload a .pdf or .txt file using the sidebar.
Wait for the "Added N chunks" confirmation.
Ask a question in the chat box (Enter to send, Shift+Enter for a new line).
Read the answer and check the source chips beneath it.
Use Clear knowledge base to wipe everything and start fresh.
API
Method	Endpoint	Body	Response
GET	/	–	Chat UI
POST	/upload	multipart/form-data with file	{ filename, chunks_added, total_chunks, files }
POST	/ask	{ "question": "..." }	{ answer, sources }
POST	/reset	–	{ status: "cleared" }

Example:

bash
curl -F "file=@report.pdf" http://localhost:5000/upload

curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What were the key findings?"}'
Tuning

Constants at the top of app.py:

Setting	Default	Effect
CHUNK_SIZE	700	Characters per chunk
CHUNK_OVERLAP	100	Overlap between adjacent chunks
VECTOR_K	10	Candidates from vector search
BM25_K	10	Candidates from keyword search
RERANK_TOP_N	3	Chunks passed to the LLM after reranking

To change the LLM, edit the ChatOpenAI(model=...) line. To change the ensemble balance, edit weights=[0.5, 0.5] in get_retriever().

Limitations
In-memory only – the knowledge base is lost when the server restarts.
Single shared knowledge base – state is global, so all users of one server instance see the same documents. Not designed for multi-user use.
No conversation memory – each question is answered independently.
File-level citations – sources show filenames, not page numbers or passages.
Text-based files only – scanned PDFs without a text layer won't yield content (no OCR).
Development server – app.py runs Flask with debug=True. Use a production WSGI server (e.g. Gunicorn) and disable debug mode before deploying.
Re-uploading a file with the same name adds its chunks again rather than replacing them.
Ideas for next steps
Page-level citations and highlighted passages
Per-session knowledge bases
Persistent Qdrant storage
Streaming responses
Support for .docx and .md
Conversation history and follow-up questions
