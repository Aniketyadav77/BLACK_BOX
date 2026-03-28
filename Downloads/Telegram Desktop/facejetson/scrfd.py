import numpy as np
from insightface.app import FaceAnalysis
from runtime_utils import runtime_summary

class SCRFDDetector:
    def __init__(self, gpu_id=0, det_size=(320, 320)):
        """
        Initialize SCRFD detector with Buffalo-SC (16MB).
        det_size=(320,320) is critical for 2GB RAM stability.
        """
        runtime = runtime_summary(prefer_gpu=True, gpu_id=gpu_id)
        print(
            f"SCRFD runtime: {runtime['mode']} | "
            f"providers={runtime['providers']} | "
            f"available={runtime['available_providers']}"
        )
        # allowed_modules=['detection'] prevents loading recognition/attributes models
        self.app = FaceAnalysis(name="buffalo_sc", allowed_modules=['detection'])
        self.app.prepare(ctx_id=runtime['ctx_id'], det_size=det_size)

    def detect(self, frame, conf_threshold=0.5):
        """
        Returns list of dicts: {'bbox': [x1,y1,x2,y2], 'landmarks': 5-point-kps, 'score': float}
        """
        try:
            faces = self.app.get(frame)
            results = []
            for f in faces:
                if f.det_score >= conf_threshold:
                    results.append({
                        "bbox": f.bbox.astype(int),
                        "landmarks": f.kps,
                        "score": f.det_score
                    })
            return results
        except Exception as e:
            print(f"Detection Error: {e}")
            return []
