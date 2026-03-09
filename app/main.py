from fastapi import FastAPI
from pydantic import BaseModel

from app.rag.rag_utils import chunk_text
from app.rag.vector_store import SimpleVectorStore
from app.services.llm_service import ask_local_llm

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


vector_store = SimpleVectorStore()

with open("documents.txt") as f:
    text = f.read()

chunks = chunk_text(text)

vector_store.add_documents(chunks)


@app.post("/ask")
async def ask_question(req: QueryRequest):

    relevant_chunks = vector_store.search(req.question)

    context = "\n".join(relevant_chunks)

    prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{req.question}
"""

    answer = ask_local_llm(prompt)

    return {"answer": answer}