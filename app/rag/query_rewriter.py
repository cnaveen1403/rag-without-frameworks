from app.services.llm_service import ask_local_llm


def rewrite_query(query: str):

    prompt = f"""
Rewrite the user query to make it more clear for document retrieval.

User query:
{query}

Rewritten query:
"""

    rewritten = ask_local_llm(prompt)

    return rewritten.strip()