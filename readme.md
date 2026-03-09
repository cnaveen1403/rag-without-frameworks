# RAG Without Frameworks

This project demonstrates how Retrieval Augmented Generation (RAG) works internally without using frameworks like LangChain.

The goal is to understand the core mechanics of RAG systems used in modern AI applications.

## Architecture

User Question
      ↓
Embedding (Ollama)
      ↓
Vector Similarity Search
      ↓
Retrieve Relevant Chunks
      ↓
Prompt + Context
      ↓
LLM Generation

## Components

- FastAPI API server
- Local LLM using Ollama
- Embedding model: nomic-embed-text
- Custom vector store using numpy
- Cosine similarity search
- Document chunking

## Run Locally

Install dependencies:

pip install -r requirements.txt

Start server:

uvicorn app.main:app --reload

Open API docs:

http://127.0.0.1:8000/docs

## Future Improvements

- FAISS vector database
- Chunk overlap
- Reranking
- Evaluation metrics