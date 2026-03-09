import numpy as np

from app.rag.rag_utils import get_embedding


class SimpleVectorStore:

    def __init__(self):

        self.vectors = []
        self.texts = []

    def add_documents(self, docs):

        for doc in docs:

            embedding = get_embedding(doc)

            self.vectors.append(np.array(embedding))
            self.texts.append(doc)

    def search(self, query: str, top_k: int = 2):

        query_embedding = np.array(get_embedding(query))

        similarities = []

        for vec in self.vectors:

            sim = np.dot(query_embedding, vec) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(vec)
            )

            similarities.append(sim)

        top_indices = np.argsort(similarities)[-top_k:]

        return [self.texts[i] for i in top_indices]