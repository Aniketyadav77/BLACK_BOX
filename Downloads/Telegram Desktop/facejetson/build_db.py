import cv2
import numpy as np
import os
import pickle
import torch
from ultralytics import YOLO

# --- Custom Pipeline Modules ---
# Make sure arcface.py and align.py are in the same directory
from arcface import ArcFaceRecognizer
from align import align_face

# --- CONFIGURATION ---
DATASET_DIR = "known_faces"
OUTPUT_DB = "known_faces2.pkl"
YOLO_MODEL_PATH = "yolov9t-face.pt"
CONF_THRESHOLD = 0.6  # Strict confidence so we only extract clear faces

def main():
    print("--- Booting High-Fidelity Enrollment Pipeline ---")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"[SUCCESS] GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARNING] CUDA not found. Falling back to CPU.")
    
    print("Loading Models...")
    # Initialize YOLO (Ultralytics handles GPU assignment automatically)
    detector = YOLO(YOLO_MODEL_PATH)
    recognizer = ArcFaceRecognizer(gpu_id=0)

    # Dictionary to store the final master embeddings
    master_database = {}

    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Directory '{DATASET_DIR}' not found. Please create it and add your folders.")
        return

    # Loop through each person's folder
    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        
        if not os.path.isdir(person_dir):
            continue
            
        print(f"\nProcessing Subject: {person_name}")
        person_embeddings = []

        # Loop through all images for this person
        for img_name in os.listdir(person_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(person_dir, img_name)
            frame = cv2.imread(img_path)
            
            if frame is None:
                print(f"  [!] Failed to read {img_name}")
                continue

            # 1. Run YOLO Detection
            results = detector(frame, conf=CONF_THRESHOLD, verbose=False, device=0)
            
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                # Get the FIRST face detected (Assuming only the target person is in the photo)
                box = results[0].boxes.xyxy[0].cpu().numpy()
                
                if results[0].keypoints is not None:
                    landmarks = results[0].keypoints.xy[0].cpu().numpy()
                else:
                    landmarks = None

                # 2. Align the face using the 5-point landmarks
                aligned_face = align_face(frame, landmarks, box)
                
                # 3. Extract the math embedding
                if aligned_face is not None and aligned_face.size > 0:
                    emb = recognizer.get_embedding(aligned_face)
                    if emb is not None:
                        person_embeddings.append(emb)
                        print(f"  [+] Success: {img_name}")
                    else:
                        print(f"  [-] Failed ArcFace extraction: {img_name}")
            else:
                print(f"  [-] No clear face detected by YOLO in: {img_name}")

        # ==========================================
        # THE CENTROID MATH (High-Quality Averaging)
        # ==========================================
        if len(person_embeddings) > 0:
            print(f"--> Extracted {len(person_embeddings)} valid faces for {person_name}.")
            
            # 1. Average all the vectors together
            mean_embedding = np.mean(person_embeddings, axis=0)
            
            # 2. L2 Normalize the final average so it maps correctly in the ArcFace dimension
            master_embedding = mean_embedding / np.linalg.norm(mean_embedding)
            
            master_database[person_name] = master_embedding
            print(f"--> Master Embedding created for {person_name}!")
        else:
            print(f"--> [ERROR] Could not extract any valid embeddings for {person_name}.")
            # --- SAVE TO DISK ---
    if len(master_database) > 0:
        with open(OUTPUT_DB, 'wb') as f:
            pickle.dump(master_database, f)
        print(f"\n[COMPLETE] Saved {len(master_database)} master identities to {OUTPUT_DB}")
    else:
        print("\n[FAILED] No identities to save.")

if __name__ == "__main__":
    main()


# import cv2
# import numpy as np
# import os
# import pickle
# import torch
# from ultralytics import YOLO

# # --- Custom Pipeline Modules ---
# from arcface import ArcFaceRecognizer
# from align import align_face

# # --- CONFIGURATION ---
# DATASET_DIR = "known_faces"
# OUTPUT_DB = "known_faces.pkl"
# YOLO_MODEL_PATH = "yolov9t-face.pt"
# CONF_THRESHOLD = 0.6  # Strict confidence for reference photos

# def main():
#     print("--- Booting High-Fidelity Enrollment Pipeline ---")
#     if torch.cuda.is_available():
#         DEVICE = 'cuda:0'
#         torch.cuda.set_device(0)
#         print(f"[SUCCESS] GPU Detected: {torch.cuda.get_device_name(0)}")
#     else:
#         DEVICE = 'cpu'
#         print("[WARNING] CUDA not found. Falling back to CPU.")
    
#     print("Loading Models...")
#     detector = YOLO(YOLO_MODEL_PATH)
#     detector.to(DEVICE)
#     recognizer = ArcFaceRecognizer(gpu_id=0)

#     # Dictionary to store the final master embeddings
#     master_database = {}

#     # Ensure dataset directory exists
#     if not os.path.exists(DATASET_DIR):
#         print(f"Error: Directory '{DATASET_DIR}' not found. Please create it and add your photos.")
#         return

#     # Loop through each person's folder
#     for person_name in os.listdir(DATASET_DIR):
#         person_dir = os.path.join(DATASET_DIR, person_name)
        
#         if not os.path.isdir(person_dir):
#             continue
            
#         print(f"\nProcessing Subject: {person_name}")
#         person_embeddings = []

#         # Loop through all images for this person
#         for img_name in os.listdir(person_dir):
#             if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 continue
                
#             img_path = os.path.join(person_dir, img_name)
#             frame = cv2.imread(img_path)
            
#             if frame is None:
#                 print(f"  [!] Failed to read {img_name}")
#                 continue

#             # Run YOLO Detection
#             results = detector(frame, conf=CONF_THRESHOLD, verbose=False, device=DEVICE)
            
#             if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
#                 # Get the FIRST face detected (Assuming only the target person is in the reference photo)
#                 box = results[0].boxes.xyxy[0].cpu().numpy()
                
#                 if results[0].keypoints is not None:
#                     landmarks = results[0].keypoints.xy[0].cpu().numpy()
#                 else:
#                     landmarks = None

#                 # Align and Extract
#                 aligned_face = align_face(frame, landmarks, box)
                
#                 if aligned_face is not None and aligned_face.size > 0:
#                     emb = recognizer.get_embedding(aligned_face)
#                     if emb is not None:
#                         person_embeddings.append(emb)
#                         print(f"  [+] Success: {img_name}")
#                     else:
#                         print(f"  [-] Failed extraction: {img_name}")
#             else:
#                 print(f"  [-] No face detected in: {img_name}")

#         # --- THE CENTROID MATH ---
#         if len(person_embeddings) > 0:
#             print(f"--> Extracted {len(person_embeddings)} valid embeddings for {person_name}.")
            
#             # 1. Average all the vectors together
#             mean_embedding = np.mean(person_embeddings, axis=0)
            
#             # 2. L2 Normalize the final average so it maps correctly on the hypersphere
#             master_embedding = mean_embedding / np.linalg.norm(mean_embedding)
            
#             master_database[person_name] = master_embedding
#             print(f"--> Master Embedding created for {person_name}!")
#         else:
#             print(f"--> [ERROR] Could not extract any valid embeddings for {person_name}.")
#             # --- SAVE TO DISK ---
#     if len(master_database) > 0:
#         with open(OUTPUT_DB, 'wb') as f:
#             pickle.dump(master_database, f)
#         print(f"\n[COMPLETE] Saved {len(master_database)} identities to {OUTPUT_DB}")
#     else:
#         print("\n[FAILED] No identities to save.")

# if __name__ == "__main__":
#     main()