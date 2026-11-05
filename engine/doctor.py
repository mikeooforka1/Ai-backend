"""Read-only diagnostics. Does not print secrets or download models."""
import importlib.util
import os
from pathlib import Path
import shutil
import sys
print('Python:', sys.version.split()[0])
for tool in ('ffmpeg', 'ffprobe'):
    print(tool + ':', 'FOUND' if shutil.which(tool) else 'MISSING (install FFmpeg and restart PowerShell)')
for name in ('cv2', 'numpy', 'insightface', 'onnxruntime', 'torch', 'librosa', 'scipy'):
    print(name + ':', 'installed' if importlib.util.find_spec(name) else 'missing')
for name in ('FACE_MODEL_PATH', 'WAV2LIP_MODEL_PATH'):
    value = os.getenv(name)
    print(name + ':', 'file found' if value and Path(value).is_file() else 'missing or unconfigured')
value = os.getenv('WAV2LIP_REPO_PATH')
print('WAV2LIP_REPO_PATH (optional for TorchScript):', 'models found' if value and (Path(value) / 'models/wav2lip.py').is_file() else 'not configured; only needed for older state-dictionary checkpoints')
try:
    import main
    main.initialize_face()
    main.lipsync.initialize()
    print('Face swap:', 'READY' if main.FACE_APP is not None else main.FACE_ERROR)
    print('CPU lip sync:', 'READY' if main.lipsync.MODEL is not None else main.lipsync.ERROR)
    sys.exit(0 if main.FACE_APP is not None and main.lipsync.MODEL is not None else 1)
except Exception as error:
    print('Diagnostic error:', str(error))
    sys.exit(1)
