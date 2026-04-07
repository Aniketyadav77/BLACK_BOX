import cv2
import time
import numpy as np
import math
import threading
import torch
from ultralytics import YOLO

# --- Custom Pipeline Modules ---
from arcface import ArcFaceRecognizer
from matcher import FaceDB
from align import align_face

# --- CONFIGURATION ---
RTSP_URL = "rtsp://admin:mits%251957@192.168.1.245:554/cam/realmonitor?channel=30&subtype=0"
YOLO_MODEL_PATH = "yolov9t-face.pt"
KNOWN_FACES_FILE = "known_faces2.pkl"  
CONF_THRESHOLD = 0.5     
SIM_THRESHOLD = 0.45     
YOLO_IMG_SIZE = 320      
MAX_RETRIES = 10         

# ====================================================================
# BACKGROUND STREAM READER (Crucial for Zero-Lag RTSP)
# ====================================================================
class BackgroundStreamReader:
    def __init__(self, rtsp_url):
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Force minimum buffer
        
        if not self.cap.isOpened():
            print(f"[ERROR] Could not connect to RTSP stream: {rtsp_url}")
            self.running = False
            return
            
        self.ret, self.frame = self.cap.read()
        self.running = True
        
        # Start the background thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Constantly pull frames in the background so the AI never waits
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                time.sleep(0.01) # Tiny sleep to prevent CPU burn if connection blips

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================
def get_center(bbox):
    """Calculates the exact center (x, y) of a bounding box."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def main():
    print("--- System Hardware Check ---")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"[SUCCESS] GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARNING] CUDA not found. Expect severe lag on CPU.")
    print("-----------------------------\n")

    print("Loading AI Pipeline Models...")
    detector = YOLO(YOLO_MODEL_PATH)
    recognizer = ArcFaceRecognizer(gpu_id=0)
    db = FaceDB(db_path=KNOWN_FACES_FILE, sim_threshold=SIM_THRESHOLD)
    db.load_db()

    # --- THE SPATIAL TRACKING CACHE ---
    tracked_faces = []

    print(f"Connecting to CCTV RTSP Stream...")
    stream = BackgroundStreamReader(RTSP_URL)
    
    if not stream.running:
        return

    print("Connected! Starting Security Feed... Press ESC to quit.")

    fps_start = time.time()
    window_name = "CCTV AI Security Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while True:
        frame = stream.read()
        
        if frame is None:
            continue
            
        # We make a copy so we don't draw on the frame while the thread is updating it
        display_frame = frame.copy() 

        # ---------------------------------------------------------
        # AI LOGIC: YOLO runs on EVERY frame for perfectly smooth boxes
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        results = detector(display_frame, conf=CONF_THRESHOLD, imgsz=YOLO_IMG_SIZE, device=0, verbose=False)
        
        current_frame_faces = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            if results[0].keypoints is not None:
                all_keypoints = results[0].keypoints.xy.cpu().numpy()
            else:
                all_keypoints = [None] * len(boxes)

            for box, landmarks in zip(boxes, all_keypoints):
                cx, cy = get_center(box)
                name = "Unknown"
                retries = 0

                # ==========================================
                # 1. SPATIAL MATCHING (No Ghost Boxes)
                # ==========================================
                # CCTV cameras are often high-res (1080p), so we allow a movement of up to 120 pixels
                for tf in tracked_faces:
                    prev_cx, prev_cy = tf['center']
                    distance = math.hypot(cx - prev_cx, cy - prev_cy)
                    
                    if distance < 120:  
                        name = tf['name']
                        retries = tf['retries']
                        break

                # ==========================================
                # 2. RECOGNITION LOGIC (The Grace Period)
                # ==========================================
                if name == "Unknown" and retries < MAX_RETRIES:
                    # Pass display_frame to ensure we extract from the clean image
                    aligned_face = align_face(display_frame, landmarks, box)
                    
                    if aligned_face is not None and aligned_face.size > 0:
                        emb = recognizer.get_embedding(aligned_face)
                        if emb is not None:
                            match_name, score = db.match(emb)
                            if match_name != "Unknown":
                                name = match_name  # Success!
                    
                    if name == "Unknown":
                        retries += 1  # Failed. Increment the retry counter.

                # Save this face's data for the NEXT frame to use
                current_frame_faces.append({
                    'bbox': box,
                    'name': name,
                    'retries': retries,
                    'center': (cx, cy)
                })

        # Update the master tracker for the next loop iteration
        tracked_faces = current_frame_faces

        # ---------------------------------------------------------
        # VISUALIZATION LOGIC
        # ---------------------------------------------------------
        for face_data in tracked_faces:
            x1, y1, x2, y2 = map(int, face_data['bbox'])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(display_frame.shape[1], x2), min(display_frame.shape[0], y2)
            
            name = face_data['name']
            
            if name == "Unknown":
                color = (0, 0, 255)  
                label = "Unknown"
            else:
                color = (0, 255, 0)  
                label = name

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(display_frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ---------------------------------------------------------
        # FPS Counter
        fps_end = time.time()
        time_diff = fps_end - fps_start
        if time_diff > 0:
            fps = 1.0 / time_diff
            cv2.putText(display_frame, f"FPS: {fps:.1f} | RTSP Threaded | GPU", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        fps_start = fps_end

        cv2.imshow(window_name, display_frame)
        
        if cv2.waitKey(1) == 27:
            break

    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()