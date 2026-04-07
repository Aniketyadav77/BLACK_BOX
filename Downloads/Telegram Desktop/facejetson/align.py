# import cv2
# import numpy as np

# # ArcFace standard reference points (112x112)
# REF_PTS = np.array([
#     [30.2946, 51.6963], [65.5318, 51.5014],
#     [48.0252, 71.7366],
#     [33.5493, 92.3655], [62.7299, 92.2041]
# ], dtype=np.float32)

# def align_face(img, landmarks):
#     """
#     Warp face to standard 112x112 using landmarks.
#     """
#     if landmarks is None or len(landmarks) != 5:
#         return None
        
#     src = np.array(landmarks, dtype=np.float32)
#     dst = REF_PTS.copy()
    
#     # 8.0 offset is standard for InsightFace 112x112 crop
#     dst[:, 0] += 8.0 
    
#     # Estimate transform matrix
#     M, _ = cv2.estimateAffinePartial2D(src, dst)
    
#     if M is None:
#         return None
        
#     # Warp
#     aligned = cv2.warpAffine(img, M, (112, 112))
#     return aligned

import cv2
import numpy as np

# Standard reference points for 112x112 ArcFace embedding
REFERENCE_FACIAL_POINTS = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], 
    [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]  
], dtype=np.float32)

def align_face(frame, landmarks=None, bbox=None):
    """
    Attempts to align a face using 5-point landmarks. 
    If landmarks are missing (standard YOLO), it falls back to a bounding box crop.
    """
    h, w = frame.shape[:2]

    # --- FALLBACK: No Landmarks, Use Bounding Box ---
    if landmarks is None or len(landmarks) < 5:
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Boundary protection
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = frame[y1:y2, x1:x2]
            
            # Prevent crashes on empty crops
            if crop.size == 0:
                return None
                
            return cv2.resize(crop, (112, 112))
        else:
            return None

    # --- ADVANCED ALIGNMENT: Use Landmarks ---
    try:
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Squeeze dimensions if YOLO outputs [[[x,y]...]]
        if landmarks.ndim == 3:
            landmarks = landmarks[0]
            
        matrix, _ = cv2.estimateAffinePartial2D(landmarks, REFERENCE_FACIAL_POINTS)
        
        if matrix is None:
            return None
            
        aligned = cv2.warpAffine(frame, matrix, (112, 112), borderMode=cv2.BORDER_REPLICATE)
        return aligned
    except Exception as e:
        print(f"Alignment Error: {e}")
        return None