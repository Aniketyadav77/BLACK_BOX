import cv2
import numpy as np

# ArcFace standard reference points (112x112)
REF_PTS = np.array([
    [30.2946, 51.6963], [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655], [62.7299, 92.2041]
], dtype=np.float32)

def align_face(img, landmarks):
    """
    Warp face to standard 112x112 using landmarks.
    """
    if landmarks is None or len(landmarks) != 5:
        return None
        
    src = np.array(landmarks, dtype=np.float32)
    dst = REF_PTS.copy()
    
    # 8.0 offset is standard for InsightFace 112x112 crop
    dst[:, 0] += 8.0 
    
    # Estimate transform matrix
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    
    if M is None:
        return None
        
    # Warp
    aligned = cv2.warpAffine(img, M, (112, 112))
    return aligned
