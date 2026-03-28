import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from runtime_utils import runtime_summary

# CONFIG
FACE_DIR = "known_faces"
SAVE_PATH = "face_db.npz"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def build_database():
    # 1. Initialize Buffalo-SC (Downloads models automatically if missing)
    print("Initializing Buffalo-SC for database building...")
    runtime = runtime_summary(prefer_gpu=True, gpu_id=0)
    print(
        f"DB runtime: {runtime['mode']} | "
        f"providers={runtime['providers']} | "
        f"available={runtime['available_providers']}"
    )
    app = FaceAnalysis(name="buffalo_sc") 
    app.prepare(ctx_id=runtime['ctx_id'], det_size=(640, 640))

    embeddings = []
    names = []
    total_files = 0
    unreadable_files = 0
    no_face_files = 0

    if not os.path.exists(FACE_DIR):
        os.makedirs(FACE_DIR)
        print(f"Created {FACE_DIR}. Please add subfolders with images (e.g., known_faces/Obama/1.jpg).")
        return

    print("Scanning images...")
    for person_name in os.listdir(FACE_DIR):
        person_path = os.path.join(FACE_DIR, person_name)
        if not os.path.isdir(person_path): continue

        for img_name in os.listdir(person_path):
            _, ext = os.path.splitext(img_name)
            if ext.lower() not in VALID_EXTENSIONS:
                continue

            total_files += 1
            img_full_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_full_path)
            
            if img is None:
                unreadable_files += 1
                print(f"Skipping unreadable image: {img_full_path}")
                continue

            # Detect faces
            faces = app.get(img)
            
            if len(faces) == 0:
                no_face_files += 1
                print(f"No face detected in {img_name}")
                continue
            
            # Use largest face
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            
            # Store normalized embedding
            embeddings.append(face.normed_embedding)
            names.append(person_name)
            print(f"Encoded: {person_name} from {img_name}")

    if len(embeddings) == 0:
        print("Failed to build DB: no valid face embeddings were generated.")
        print(
            "Summary: "
            f"total_files={total_files}, "
            f"unreadable={unreadable_files}, "
            f"no_face={no_face_files}"
        )
        print(
            "Add clear images under known_faces/<your_name>/ and rerun build_db.py. "
            "Do not use empty or corrupted files."
        )
        return

    # Save to numpy file
    np.savez(SAVE_PATH, embeddings=embeddings, names=names)
    print(
        f"Success! Saved {len(embeddings)} faces from {len(set(names))} identities to {SAVE_PATH}."
    )
    print(
        "Summary: "
        f"total_files={total_files}, "
        f"unreadable={unreadable_files}, "
        f"no_face={no_face_files}"
    )

if __name__ == "__main__":
    build_database()
