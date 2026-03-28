# FaceJetson: Real-time Face Detection & Recognition

SCRFD detector + ArcFace recognition using InsightFace. Works on GPU (CUDA) or CPU. Includes database builder for known faces and a live camera pipeline with simple tracking and cosine-similarity matching.

## Features
- SCRFD detection (from `buffalo_sc`) with confidence filtering.
- ArcFace embeddings (w600k_mbf.onnx) with L2-normalized 512-D vectors.
- Cosine-similarity matcher with configurable threshold.
- Temporal smoothing via simple tracker for stable boxes/labels.
- Database builder for images under `known_faces/`.

## Requirements
- Python 3.8+.
- For GPU: CUDA-capable GPU with CUDA 12 runtime (see `requirements.txt`).
- For CPU-only: use `requirements-cpu.txt`.

## Setup
```ps
# In PowerShell (Windows)
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r facejetson\requirements.txt        # GPU
# or
pip install -r facejetson\requirements-cpu.txt     # CPU
```

## Prepare known faces
1) Add images under `facejetson/known_faces/<person_name>/`.
   - Accepts .jpg/.jpeg/.png/.bmp/.webp
   - Use clear, frontal images; multiple angles improve robustness.
2) Example:
```
facejetson/known_faces/obama/1.jpg
facejetson/known_faces/mary/2.png
```

## Build the embeddings database
This downloads the InsightFace `buffalo_sc` models (SCRFD + ArcFace) on first run and writes `face_db.npz`.
```ps
# From repo root (venv active)
python facejetson\build_db.py
```
- The script reports unreadable images or missing faces.
- Output: `face_db.npz` containing normalized embeddings and labels.

## Run live detection/recognition
```ps
python facejetson\main.py
```
- Press ESC to quit.
- Windows: auto-probes cameras via DirectShow/MSMF, then falls back to default.
- Jetson: uses CSI/USB GStreamer pipelines with a V4L2 fallback.

## Tuning
- Recognition threshold: `SIM_THRESHOLD` in [facejetson/main.py](facejetson/main.py) (default 0.45 for MobileFaceNet). Lower = more matches, higher = stricter.
- Detection confidence: `CONF_THRESHOLD` in [facejetson/main.py](facejetson/main.py) (default 0.5).
- Frame skipping: `SKIP_FRAMES` to trade accuracy vs FPS.
- Tracker IOU + lifetime: `SimpleTracker` params in [facejetson/simple_tracker.py](facejetson/simple_tracker.py).

## Project structure
- [facejetson/main.py](facejetson/main.py): live loop (detect → align → embed → match → track → display).
- [facejetson/build_db.py](facejetson/build_db.py): builds `face_db.npz` from `known_faces/`.
- [facejetson/scrfd.py](facejetson/scrfd.py): SCRFD detector loader.
- [facejetson/arcface.py](facejetson/arcface.py): ArcFace embedding model loader.
- [facejetson/align.py](facejetson/align.py): 5-point alignment to 112×112.
- [facejetson/matcher.py](facejetson/matcher.py): cosine similarity matcher with thresholding.
- [facejetson/simple_tracker.py](facejetson/simple_tracker.py): lightweight IOU-based tracker.
- [facejetson/runtime_utils.py](facejetson/runtime_utils.py): selects CUDA/CPU providers for ONNX Runtime.

## Notes
- If `w600k_mbf.onnx` is missing, run `python facejetson\build_db.py` once; it downloads to `~/.insightface/models/buffalo_sc/`.
- GPU DLL discovery on Windows is handled in `runtime_utils.py` via `os.add_dll_directory`.
- `face_db.npz` contains embeddings; avoid committing it if it has private data.

## GitHub upload quickstart
```ps
git init
git remote add origin https://github.com/Aniketyadav77/BLACK_BOX.git
git add .
git commit -m "feat: add face recognition pipeline"
git push -u origin main
```
(Optional) Split commits:
- Code: `git add facejetson/*.py` then `git commit -m "feat: add face recognition pipeline"`
- Docs: `git add README.md .gitignore` then `git commit -m "docs: add setup and usage guide"`

## Troubleshooting
- "Model not found" → run `build_db.py` to download models.
- "Could not open camera" → try a different index; edit `get_camera_pipeline()` in [facejetson/main.py](facejetson/main.py) to force a backend.
- Low accuracy → add more/better images per person and raise `SIM_THRESHOLD` slightly (e.g., 0.5).
- Slow FPS → increase `SKIP_FRAMES`, lower `det_size` in `SCRFDDetector`, or switch to CPU requirements if GPU drivers are unavailable.
