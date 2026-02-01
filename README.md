# RAG Basics using LangChain

A Retrieval-Augmented Generation (RAG) project built with LangChain (v1.x compatible).  
This repo demonstrates the core RAG pipeline end-to-end:

1. Load a webpage as documents  
2. Split text into chunks  
3. Create embeddings and index them in a vector store (Chroma)  
4. Retrieve the most relevant chunks for a user question  
5. Inject retrieved context into a prompt  
6. Generate a grounded answer with an LLM

## Tech Stack
- LangChain (`langchain_core`, `langchain_community`, `langchain_openai`, `langchain_text_splitters`)
- ChromaDB (vector store)
- OpenAI Embeddings + Chat model
- BeautifulSoup (HTML parsing)

## Project Structure
    rag-basics-langchain/
    ├─ src/
    │ ├─ rag_basics.py
    │ └─ init.py
    ├─ requirements.txt
    ├─ .env.example
    ├─ .gitignore
    └─ README.md

## Setup

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Add your API key
Create a .env file (or export env vars) and set:
```bash
OPENAI_API_KEY="..."
```

### 4) Run
```bash
python -m src.rag_basics --question "What is Task Decomposition?"
```
Optional parameters:

--k (retrieval depth): number of chunks to retrieve (default: 4)

--chunk_size (default: 1000)

--chunk_overlap (default: 200)

--model (default: gpt-4o-mini)

--url (default: Lilian Weng's Agents blog post)

--persist_dir (e.g., ./chroma_db) to persist Chroma locally

Example:
```bash
python -m src.rag_basics \
  --question "Explain task decomposition in agents" \
  --k 5 \
  --chunk_size 800 \
  --chunk_overlap 150 \
  --persist_dir ./chroma_db
  ```