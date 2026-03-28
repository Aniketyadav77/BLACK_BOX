import cv2
import time
import numpy as np
import platform
from scrfd import SCRFDDetector
from arcface import ArcFaceRecognizer
from simple_tracker import SimpleTracker
from matcher import FaceDB
from align import align_face

# --- CONFIGURATION ---
SKIP_FRAMES = 2          # Process AI every 3rd frame (0, 3, 6...)
CONF_THRESHOLD = 0.5     # Detection confidence threshold
SIM_THRESHOLD = 0.45     # Recognition threshold (Lower for MobileFaceNet)

def get_camera_pipeline():
    """
    Tries to open the best available camera on Jetson Nano.
    1. CSI Camera (Raspberry Pi Cam) via GStreamer
    2. USB Webcam via GStreamer
    3. Standard V4L2 fallback
    """
    # Windows webcam probing (DirectShow / MediaFoundation)
    if platform.system().lower() == "windows":
        candidates = []
        for idx in range(6):
            candidates.append((idx, cv2.CAP_DSHOW, "DirectShow"))
            candidates.append((idx, cv2.CAP_MSMF, "MediaFoundation"))

        for idx, backend, backend_name in candidates:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                print(f"Using Windows camera index {idx} via {backend_name}")
                return cap
            cap.release()

        print("Windows camera probing failed, trying default backend")
        return cv2.VideoCapture(0)

    # 1. CSI Camera Pipeline (Fastest for Pi Cam)
    csi_pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=640, height=360, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=True"
    )
    
    # 2. USB Camera Pipeline (Fastest for USB Cam)
    usb_pipeline = (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw, width=640, height=480, framerate=30/1 ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=True"
    )

    # Try CSI first
    cap = cv2.VideoCapture(csi_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("Using CSI Camera Pipeline")
        return cap

    # Try USB GStreamer
    cap = cv2.VideoCapture(usb_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("Using USB GStreamer Pipeline")
        return cap
        
    # Fallback to standard OpenCV
    print("Using Standard V4L2 Fallback")
    return cv2.VideoCapture(0)

def main():
    # 1. Initialize AI Models (GPU Mode)
    print("Loading Models...")
    detector = SCRFDDetector(gpu_id=0, det_size=(320, 320))
    recognizer = ArcFaceRecognizer(gpu_id=0)
    tracker = SimpleTracker(iou_thresh=0.4)
    db = FaceDB(sim_threshold=SIM_THRESHOLD)
    
    # Load Database
    db.load_db()

    # 2. Start Camera
    cap = get_camera_pipeline()
    if not cap.isOpened():
        print("Error: Could not open any camera.")
        return

    frame_count = 0
    fps_start = time.time()
    
    # Full screen window setup
    window_name = "Jetson Face Security"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Starting Security Feed... Press ESC to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # ---------------------------------------------------------
        # AI LOGIC (Runs every few frames to save FPS)
        # ---------------------------------------------------------
        if frame_count % (SKIP_FRAMES + 1) == 0:
            # A. Detect Faces
            raw_faces = detector.detect(frame, conf_threshold=CONF_THRESHOLD)
            
            detections_for_tracker = []
            
            for face in raw_faces:
                bbox = face['bbox'] # [x1, y1, x2, y2]
                
                # B. Align & Recognize
                aligned_face = align_face(frame, face['landmarks'])
                emb = recognizer.get_embedding(aligned_face)
                
                name = "Unknown"
                if emb is not None:
                    # Matches against DB. Returns "Unknown" if score < SIM_THRESHOLD
                    name, score = db.match(emb)
                
                detections_for_tracker.append({
                    'bbox': bbox,
                    'name': name
                })
            
            # C. Update Tracker (Memories face positions)
            tracks = tracker.update(detections_for_tracker)
        
        else:
            # D. Fast Mode: Use cached tracks for smooth video
            tracks = tracker.tracks

        # ---------------------------------------------------------
        # VISUALIZATION LOGIC (RED vs GREEN)
        # ---------------------------------------------------------
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.bbox)
            
            # --- COLOR SELECTION LOGIC ---
            if t.name == "Unknown":
                color = (0, 0, 255)  # RED (BGR format)
                label = "Unknown"
            else:
                color = (0, 255, 0)  # GREEN (BGR format)
                label = t.name
            # -----------------------------

            # Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Background for Text (Matches box color)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            # Draw Text (White text on top of colored background)
            cv2.putText(frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ---------------------------------------------------------
        # FPS Counter
        # ---------------------------------------------------------
        fps_end = time.time()
        time_diff = fps_end - fps_start
        if time_diff > 0:
            fps = 1.0 / time_diff
            # Draw FPS in top-left corner
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        fps_start = fps_end

        cv2.imshow(window_name, frame)
        frame_count += 1

        # Quit on ESC
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()