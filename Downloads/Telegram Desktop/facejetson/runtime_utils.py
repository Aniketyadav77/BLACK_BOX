import os
import site
from pathlib import Path


def _configure_windows_cuda_dll_path():
    """
    Make CUDA/cuDNN DLLs from pip-installed NVIDIA packages discoverable on Windows.
    """
    if os.name != "nt":
        return

    try:
        roots = [Path(p) for p in site.getsitepackages()]
    except Exception:
        return

    seen = set()
    for root in roots:
        nvidia_root = root / "nvidia"
        if not nvidia_root.exists():
            continue

        for dll_dir in list(nvidia_root.glob("*/bin")) + list(nvidia_root.glob("*/lib")):
            dll_path = str(dll_dir)
            if dll_path in seen:
                continue
            seen.add(dll_path)
            try:
                os.add_dll_directory(dll_path)
            except Exception:
                pass

    if seen:
        current_path = os.environ.get("PATH", "")
        prepend = os.pathsep.join(sorted(seen))
        os.environ["PATH"] = f"{prepend}{os.pathsep}{current_path}" if current_path else prepend


_configure_windows_cuda_dll_path()

import onnxruntime as ort


def select_runtime(prefer_gpu=True, gpu_id=0):
    """
    Returns (ctx_id, providers, using_gpu) for InsightFace/ONNX Runtime.
    """
    available = ort.get_available_providers()
    has_cuda = "CUDAExecutionProvider" in available

    if prefer_gpu and has_cuda:
        return gpu_id, ["CUDAExecutionProvider", "CPUExecutionProvider"], True

    return -1, ["CPUExecutionProvider"], False


def runtime_summary(prefer_gpu=True, gpu_id=0):
    available = ort.get_available_providers()
    ctx_id, providers, using_gpu = select_runtime(prefer_gpu=prefer_gpu, gpu_id=gpu_id)
    mode = "GPU" if using_gpu else "CPU"
    return {
        "mode": mode,
        "ctx_id": ctx_id,
        "providers": providers,
        "available_providers": available,
    }
