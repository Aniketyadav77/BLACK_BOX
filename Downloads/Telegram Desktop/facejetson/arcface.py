import os
import numpy as np
from collections import deque
from insightface.model_zoo import get_model
from runtime_utils import runtime_summary

class ArcFaceRecognizer:
    def __init__(self, gpu_id=0):
        # Path where InsightFace downloads models by default
        user_home = os.path.expanduser("~")
        # Ensure this matches the model downloaded by build_db.py
        model_path = os.path.join(user_home, ".insightface", "models", "buffalo_sc", "w600k_mbf.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run build_db.py first!")

        runtime = runtime_summary(prefer_gpu=True, gpu_id=gpu_id)
        print(
            f"ArcFace runtime: {runtime['mode']} | "
            f"providers={runtime['providers']} | "
            f"available={runtime['available_providers']}"
        )
        print(f"Loading Recognition Model: {model_path}")
        self.model = get_model(model_path, providers=runtime['providers'])
        self.model.prepare(ctx_id=runtime['ctx_id'])
        
        # Temporal smoothing buffer to reduce jitter
        self.embedding_buffer = deque(maxlen=5)

    def get_embedding(self, aligned_face):
        if aligned_face is None: 
            return None
            
        # Get feature vector (512-dim)
        try:
            emb = self.model.get_feat(aligned_face)
        except Exception as e:
            print(f"Recognition Error: {e}")
            return None

        if emb is None: 
            return None
            
        # Flatten and Normalize (L2 Norm) - Critical for Cosine Similarity
        emb = emb.flatten()
        norm = np.linalg.norm(emb)
        if norm == 0: 
            return None
        emb = emb / norm

        return emb
