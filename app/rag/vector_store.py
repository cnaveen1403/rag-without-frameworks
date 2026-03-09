import faiss
import numpy as np 

from app.rag.rag_utils import get_embedding

class SimpleVectorStore:

    def __init__(self):
        self.texts = []
        self.index = None

    def add_documents(self, docs):
        embeddings = []
        for doc in docs:
            embedding = get_embedding(doc)
            embeddings.append(embedding)
            self.texts.append(doc)

        embeddings = np.array(embeddings).astype("float32")
        dimensions = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimensions)
        self.index.add(embeddings)

    def search(self, query:str, top_k: int = 3):
        query_embedding = get_embedding(query)
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []

        for i in indices[0]:
            results.append(self.texts[i])

        return results