from app.services.llm_service import ask_local_llm

def rerank_chunks(query, chunks):
    ranked = []
    for chunk in chunks:
        prompt = f"""You are ranking document relevance.

        Question: {query}
        Document: {chunk}

        score relevance from 1 to 10.
        Return only number
        """

        score = ask_local_llm(prompt)
        try:
            score = int(score.strip())
        except:
            score = 0

        ranked.append((chunk, score))
    
    ranked.sort(key=lambda x:x[1], reverse=True)

    return [x[0] for x in ranked]