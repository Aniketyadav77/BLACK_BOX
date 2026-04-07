import numpy as np
import pickle
import os

class FaceDB:
    def __init__(self, db_path="known_faces.pkl", sim_threshold=0.45):
        """
        MobileFaceNet (buffalo_sc) needs a lower threshold than ResNet.
        Start with 0.45 or 0.50.
        """
        self.db_path = db_path
        self.sim_threshold = sim_threshold
        
        # We store these as aligned lists so we can use blazing-fast NumPy math
        self.embeddings = []
        self.labels = []

    def load_db(self):
        if not os.path.exists(self.db_path):
            print(f"[WARNING] Database '{self.db_path}' not found. Everyone will be Unknown.")
            return

        try:
            # 1. You MUST open the file in 'rb' (read-binary) mode for pickle
            with open(self.db_path, 'rb') as f:
                data = pickle.load(f)
                
                # 2. enroll.py saves a flat dictionary: {"Name": array(...)}
                # We split it into two aligned lists for the NumPy dot product
                if isinstance(data, dict):
                    self.labels = list(data.keys())
                    self.embeddings = list(data.values())
                    print(f"[SUCCESS] Database loaded: {len(self.embeddings)} identities.")
                else:
                    print("[ERROR] Unrecognized database format. Expected a dictionary.")
                    
        except Exception as e:
            print(f"[ERROR] Failed to load DB: {e}")

    def add(self, emb, label):
        self.embeddings.append(emb)
        self.labels.append(label)

    def match(self, emb):
        if len(self.embeddings) == 0:
            return "Unknown", 0.0

        # Pure Numpy Cosine Similarity (Vectorized for Jetson/GPU optimization)
        # Because the vectors are already L2 normalized in enroll.py, 
        # cosine similarity is just a straight dot product.
        
        query_emb = np.array(emb)
        db_embs = np.array(self.embeddings)
        
        # This calculates the score against EVERY person in the DB simultaneously
        sims = np.dot(db_embs, query_emb)
        
        # Find the index of the highest score
        idx = np.argmax(sims)
        score = float(sims[idx]) # Convert from numpy float to standard python float

        # If the best score beats our threshold, return their name
        if score >= self.sim_threshold:
            return self.labels[idx], score
        
        return "Unknown", score