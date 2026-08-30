# Hybrid RAG --- Project Brain

A Streamlit-based **Hybrid Retrieval-Augmented Generation (RAG)**
application that combines semantic vector search, BM25 keyword search,
and a CrossEncoder reranker before sending the retrieved context to a
local Ollama LLM.

## Overview

The application provides a chat interface where users can:

-   Ask questions about the documents stored in the RAG knowledge base.
-   Upload additional PDFs through the Streamlit sidebar.
-   Automatically extract text from normal PDFs.
-   Use OCR for scanned/image-based PDFs.
-   Split uploaded documents into overlapping chunks.
-   Generate embeddings for new chunks.
-   Add new chunks to the existing FAISS vector index and BM25 corpus.
-   Retrieve relevant information using a hybrid retrieval pipeline.
-   Rerank retrieved candidates using a CrossEncoder.
-   Generate an answer using a locally running Ollama model.

## Architecture

``` text
                         ┌──────────────────────┐
                         │   Streamlit Chat UI  │
                         └──────────┬───────────┘
                                    │
                              User Question
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │ Query Embedding Model  │
                       │ all-MiniLM-L6-v2       │
                       └───────────┬────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
                 ▼                                   ▼
        ┌─────────────────┐                 ┌─────────────────┐
        │   FAISS Search  │                 │   BM25 Search   │
        │ Semantic Search │                 │ Keyword Search  │
        └────────┬────────┘                 └────────┬────────┘
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   │
                            Hybrid Candidates
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │    CrossEncoder        │
                       │      Reranker          │
                       └───────────┬────────────┘
                                   │
                              Top 5 Chunks
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │    Prompt + Context    │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │       Ollama LLM       │
                       │         llama3          │
                       └───────────┬────────────┘
                                   │
                                   ▼
                              Final Answer
```

## Main Components

### 1. Streamlit

The frontend and application framework.

It provides:

-   Chat interface
-   Conversation history
-   PDF upload functionality
-   Loading/status messages
-   Sidebar settings

The application title is **Project Brain** and the main interface is
called **Hybrid RAG**.

### 2. Sentence Embeddings

The application uses:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

The Hugging Face `AutoTokenizer` and `AutoModel` are used to generate
embeddings.

Mean pooling is applied to token embeddings using the attention mask,
followed by L2 normalization.

This produces dense vector representations suitable for semantic
similarity search.

### 3. FAISS

FAISS is used as the vector database/index.

The precomputed embeddings are loaded from:

``` text
rag_foundation.npz
```

The application creates:

``` python
faiss.IndexFlatL2(dim)
```

and adds all stored embeddings to the index.

For every user question, the application retrieves the top 100
vector-search candidates.

### 4. BM25

BM25 provides traditional keyword-based retrieval.

The stored chunks are tokenized and indexed using:

``` python
BM25Okapi
```

For every question, the BM25 scores are calculated and the top 100
keyword-based candidates are selected.

### 5. Hybrid Retrieval

The application combines the candidates from:

-   FAISS semantic search
-   BM25 keyword search

The candidate sets are merged using a set union.

This allows the system to benefit from both:

-   **Semantic similarity** --- useful when the question uses different
    wording from the source.
-   **Keyword matching** --- useful for exact names, terms, numbers,
    forms, and phrases.

### 6. CrossEncoder Reranking

The combined candidates are reranked using:

``` text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Each candidate is evaluated as a:

``` text
(question, document_chunk)
```

pair.

The candidates are sorted by the CrossEncoder score, and the top 5
chunks are passed to the LLM.

This creates a three-stage retrieval pipeline:

``` text
FAISS + BM25
      ↓
Candidate Pool
      ↓
CrossEncoder
      ↓
Top 5 Context Chunks
```

## RAG Knowledge Base

The precomputed RAG foundation is stored in:

``` text
rag_foundation.npz
```

The file contains two important arrays:

``` text
embeddings
metadata
```

### embeddings

Contains the precomputed vector embeddings used by FAISS.

### metadata

Contains the document chunks and their associated information, including
the text used for embedding and the actual chunk text.

The application loads this data at startup and stores it in Streamlit
session state.

## PDF Ingestion

Uploaded PDFs are processed dynamically.

### Step 1 --- Read PDF

PyMuPDF (`fitz`) opens the uploaded PDF directly from memory.

### Step 2 --- Detect Scanned PDFs

The application checks how many pages contain meaningful extracted text.

If fewer than 20% of the pages contain more than 50 characters of
extracted text, the document is treated as a scanned PDF.

### Step 3 --- OCR

For scanned PDFs:

1.  Each page is rendered as an image at 150 DPI.
2.  Tesseract OCR extracts the text.
3.  Page orientation is detected and corrected when possible.
4.  OCR text is combined into a single document.

For normal PDFs, PyMuPDF text extraction is used instead.

### Step 4 --- Chunking

The extracted document is split using:

``` text
RecursiveCharacterTextSplitter
chunk_size = 800
chunk_overlap = 80
```

This creates overlapping chunks so that information near chunk
boundaries is less likely to be lost.

### Step 5 --- Embedding

Every new chunk is converted into an embedding using the same MiniLM
embedding model used for query retrieval.

### Step 6 --- Update FAISS

The new embeddings are added directly to the existing FAISS index.

### Step 7 --- Update BM25

The new chunks are added to the BM25 corpus and the BM25 index is
rebuilt.

The newly uploaded PDF therefore becomes searchable immediately without
rebuilding the original RAG foundation.

## LLM Generation

The final retrieved chunks are inserted into a prompt containing strict
instructions.

The LLM is accessed through the local Ollama API:

``` text
http://localhost:11434/api/generate
```

The configured model is:

``` text
llama3
```

The generation temperature is:

``` text
0.2
```

The prompt instructs the model to:

1.  Use only the supplied context.
2.  Say it does not know when the answer is not present.
3.  Be concise.
4.  Prioritize timelines, legal requirements, and form names.
5.  Avoid adding unsupported general advice.

This is intended to reduce hallucination and keep answers grounded in
retrieved documents.

## Runtime Flow

When a user submits a question:

``` text
User Question
      │
      ▼
Generate Query Embedding
      │
      ├───────────────► FAISS Top 100
      │
      └───────────────► BM25 Top 100
                              │
                              ▼
                       Merge Candidates
                              │
                              ▼
                      CrossEncoder Ranking
                              │
                              ▼
                         Top 5 Chunks
                              │
                              ▼
                    Construct RAG Prompt
                              │
                              ▼
                         Ollama llama3
                              │
                              ▼
                         Final Answer
```

## Project Files

The core files intended for the repository are:

``` text
RAG_project/
│
├── app.py
├── app2.py
├── rag_foundation.npz
├── requirements.txt
└── README.md
```

Large source/reference PDFs, ZIP archives, notebooks used for
experimentation, and checkpoint directories can be excluded from the Git
repository when they are not required to run the application.

## Requirements

The Python dependencies are listed in:

``` text
requirements.txt
```

The project uses libraries including:

-   Streamlit
-   NumPy
-   FAISS
-   PyTorch
-   Transformers
-   Sentence Transformers
-   Rank BM25
-   LangChain text splitters
-   PyMuPDF
-   PyTesseract
-   Pillow
-   Requests

## Local Setup

### 1. Create and activate a virtual environment

Example:

``` bash
python -m venv venv
```

Windows:

``` bash
venv\Scripts\activate
```

### 2. Install dependencies

``` bash
pip install -r requirements.txt
```

### 3. Install and configure Tesseract OCR

The current application expects Tesseract at:

``` text
C:\Program Files\TesseractOCR\tesseract.exe
```

If Tesseract is installed somewhere else, update the
`pytesseract.pytesseract.tesseract_cmd` path in the application.

### 4. Install Ollama

Ollama must be running locally because the application sends generation
requests to:

``` text
http://localhost:11434/api/generate
```

Make sure the configured model is available:

``` text
llama3
```

### 5. Run Streamlit

Both apps are a bit different. Run either of them.

``` bash
streamlit run app.py
```

Or

``` bash
streamlit run app2.py
```

## Important Notes

### GPU Support

The application automatically checks whether CUDA is available:

``` python
torch.cuda.is_available()
```

If CUDA is available, the embedding and reranking models use the GPU;
otherwise they run on the CPU.

### Streamlit Session State

The FAISS index, knowledge chunks, BM25 corpus, BM25 index, and chat
messages are stored in Streamlit session state.

This allows uploaded documents to remain available during the current
application session.

### Caching

The embedding and reranking models are loaded through Streamlit resource
caching so that they do not need to be repeatedly initialized on every
Streamlit rerun.

## Retrieval Parameters

The default retrieval configuration is:

  Component                    Setting
  -------------------------- ---------
  Initial FAISS candidates         100
  Initial BM25 candidates          100
  Final reranked chunks              5
  Chunk size                       800
  Chunk overlap                     80
  Query max token length           256
  LLM temperature                  0.2

## Design Summary

The project follows a **retrieve → rerank → generate** architecture.

The key idea is to avoid relying on a single retrieval method:

``` text
Dense Retrieval
     +
Sparse Retrieval
     ↓
Hybrid Candidate Set
     ↓
Neural Reranking
     ↓
Small High-Quality Context
     ↓
Grounded LLM Generation
```

This design combines the strengths of semantic search, exact keyword
matching, and neural relevance scoring before generation.
