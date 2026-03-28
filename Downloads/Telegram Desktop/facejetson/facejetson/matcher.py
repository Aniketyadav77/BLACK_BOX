import numpy as np

class FaceDB:
    def __init__(self, sim_threshold=0.45):
        """
        MobileFaceNet (buffalo_sc) needs a lower threshold than ResNet.
        Start with 0.45 or 0.50.
        """
        self.embeddings = []
        self.labels = []
        self.sim_threshold = sim_threshold

    def load_db(self, path="face_db.npz"):
        try:
            data = np.load(path, allow_pickle=True)
            self.embeddings = list(data['embeddings'])
            self.labels = list(data['names'])
            print(f"Database loaded: {len(self.embeddings)} identities.")
        except Exception as e:
            print(f"Failed to load DB: {e}")

    def add(self, emb, label):
        self.embeddings.append(emb)
        self.labels.append(label)

    def match(self, emb):
        if len(self.embeddings) == 0:
            return "Unknown", 0.0

        # Pure Numpy Cosine Similarity (Faster than sklearn on Jetson)
        # sim = dot(A, B) / (norm(A) * norm(B))
        # Since embeddings are already L2 normalized, we just do dot product.
        
        query_emb = np.array(emb)
        db_embs = np.array(self.embeddings)
        
        # Dot product of query vector with all DB vectors
        sims = np.dot(db_embs, query_emb)
        
        idx = np.argmax(sims)
        score = sims[idx]

        if score >= self.sim_threshold:
            return self.labels[idx], score
        
        return "Unknown", score
