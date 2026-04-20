from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class RAGRetriever:
    def __init__(self):
        print("[INFO] Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("[INFO] Creating knowledge base...")
        self.documents = [
            "Barack Obama was born in Hawaii in 1961.",
            "The Earth revolves around the Sun once every 365 days.",
            "India gained independence in 1947 from British rule.",
            "Python is a widely used programming language.",
            "The capital of France is Paris."
        ]

        self.create_index()

    def create_index(self):
        print("[INFO] Creating FAISS index...")
        embeddings = self.model.encode(self.documents)

        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)

        self.index.add(np.array(embeddings))

    def retrieve(self, query, top_k=2):
        query_embedding = self.model.encode([query])

        distances, indices = self.index.search(np.array(query_embedding), top_k)

        results = [self.documents[i] for i in indices[0]]
        return results


if __name__ == "__main__":
    print("🔍 Running RAG Retriever Test...\n")

    retriever = RAGRetriever()

    query = "India became independent in 1947."
    results = retriever.retrieve(query)

    print(f"\n🔎 Query: {query}")
    print("\n📚 Retrieved Evidence:\n")

    for i, res in enumerate(results, 1):
        print(f"{i}. {res}")