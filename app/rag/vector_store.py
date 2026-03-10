import faiss
import numpy as np
import os
import pickle

from app.rag.rag_utils import get_embedding

class SimpleVectorStore:

    def __init__(self):
        self.texts = []
        self.index = None

        self.index_path = "faiss_index.bin"
        self.texts_path = "texts.pkl"

        if os.path.exists(self.index_path):

            self.index = faiss.read_index(self.index_path)

            with open(self.texts_path, "rb") as f:
                self.texts = pickle.load(f)

    def add_documents(self, docs):
        embeddings = []
        for doc in docs:
            embedding = get_embedding(doc)
            embeddings.append(embedding)
            self.texts.append(doc)

        embeddings = np.array(embeddings).astype("float32")
        dimension = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)

        faiss.write_index(self.index, self.index_path)

        with open(self.texts_path, "wb") as f:
            pickle.dump(self.texts, f)

    def search(self, query: str, top_k: int = 3):
        query_embedding = get_embedding(query)
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []

        for i in indices[0]:
            results.append(self.texts[i])

        return results

    def keyword_search(self, query, top_k=3):
        scores = []
        query_words = query.lower().split()

        for text in self.texts:
            score = 0
            for word in query_words:
                if word in text.lower():
                    score += 1
            scores.append(score)
            
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [self.texts[i] for i in top_indices]