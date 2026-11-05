"""CPU Wav2Lip adapter. Streams frames; never invokes upstream shell commands.
Model implementation/checkpoint are supplied from the official Wav2Lip repository.
"""
import importlib.util
import math
import os
from pathlib import Path
import sys
import zipfile
import numpy as np
import cv2

MODEL = None
ERROR = 'Lip-sync model not configured. Run yarn setup:python --lipsync and configure WAV2LIP_MODEL_PATH.'

def initialize():
    global MODEL, ERROR
    MODEL = None
    checkpoint = os.getenv('WAV2LIP_MODEL_PATH')
    repository = os.getenv('WAV2LIP_REPO_PATH')
    if not checkpoint:
        return
    repository = repository or str(Path(__file__).resolve().parent / 'vendor' / 'Wav2Lip')
    try:
        if not Path(checkpoint).is_file():
            raise ValueError('WAV2LIP_MODEL_PATH does not exist; download the official Wav2Lip checkpoint.')
        import torch
        import librosa
        # The current official GAN download is TorchScript; older .pth files are state dictionaries.
        scripted = False
        if zipfile.is_zipfile(checkpoint):
            with zipfile.ZipFile(checkpoint) as archive:
                scripted = any(name.endswith('/constants.pkl') for name in archive.namelist())
        if scripted:
            model = torch.jit.load(checkpoint, map_location='cpu')
        else:
            models = Path(repository).resolve() / 'models'
            if not (models / '__init__.py').is_file():
                raise ValueError('WAV2LIP_REPO_PATH must contain the official Wav2Lip models folder.')
            spec = importlib.util.spec_from_file_location('vividflow_wav2lip_models', models / '__init__.py', submodule_search_locations=[str(models)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            model = module.Wav2Lip()
            weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
            state = weights.get('state_dict', weights)
            model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
        torch.set_num_threads(max(1, min(4, int(os.getenv('LIPSYNC_CPU_THREADS', '4')))))
        model.cpu().eval()
        # Validate actual inference and checkpoint compatibility before enabling the button.
        with torch.inference_mode():
            output = model(torch.zeros(1, 1, 80, 16), torch.zeros(1, 6, 96, 96))
        if tuple(output.shape) != (1, 3, 96, 96) or not torch.isfinite(output).all():
            raise ValueError('Lip-sync model produced invalid output during its startup test.')
        MODEL, ERROR = model, None
    except Exception as error:
        ERROR = 'Lip-sync unavailable: ' + str(error)
        print(ERROR, file=sys.stderr)

def mel_spectrogram(wav_path):
    """Wav2Lip's training parameters, using current keyword-only librosa APIs."""
    import librosa
    from scipy import signal
    from scipy.io import wavfile
    rate, samples = wavfile.read(str(wav_path))
    if rate != 16000 or samples.ndim != 1:
        raise ValueError('Internal audio must be 16 kHz mono PCM.')
    wav = samples.astype(np.float32) / 32768.0
    if wav.size < 3200 or not np.isfinite(wav).all():
        raise ValueError('Audio is too short or invalid; use at least 0.2 seconds.')
    if np.max(np.abs(wav)) < 1e-4:
        raise ValueError('Uploaded audio is silent. Upload a recording containing speech.')
    spectrum = np.abs(librosa.stft(signal.lfilter([1, -0.97], [1], wav), n_fft=800, hop_length=200, win_length=800, pad_mode="reflect"))
    basis = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600, norm="slaney")
    db = 20 * np.log10(np.maximum(1e-5, basis @ spectrum)) - 20
    mel = np.clip(8 * ((db + 100) / 100) - 4, -4, 4).astype(np.float32)
    if not np.isfinite(mel).all():
        raise ValueError('Audio produced invalid model features.')
    return mel

def mel_window(mel, frame_index, fps):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid lip-sync frame rate.')
    start = min(int(frame_index * 80.0 / fps), max(0, mel.shape[1] - 16))
    chunk = mel[:, start:start + 16]
    if chunk.shape[1] < 16:
        chunk = np.pad(chunk, ((0, 0), (0, 16 - chunk.shape[1])), mode='edge')
    return chunk

def predict(crops, windows):
    if MODEL is None:
        raise ValueError(ERROR)
    import torch
    faces = np.asarray([cv2.resize(crop, (96, 96)) for crop in crops], dtype=np.float32)
    masked = faces.copy()
    masked[:, 48:] = 0
    inputs = np.concatenate([masked, faces], axis=3) / 255.0
    mels = np.asarray(windows, dtype=np.float32)[:, None, :, :]
    with torch.inference_mode():
        output = MODEL(torch.from_numpy(mels), torch.from_numpy(inputs.transpose(0, 3, 1, 2)))
    result = output.cpu().numpy().transpose(0, 2, 3, 1)
    if result.shape != (len(crops), 96, 96, 3) or not np.isfinite(result).all():
        raise ValueError('Lip-sync inference returned invalid pixels.')
    return np.clip(result * 255, 0, 255).astype(np.uint8)

def face_crop_box(bbox, width, height):
    x0, y0, x1, y1 = map(float, bbox)
    w, h = x1 - x0, y1 - y0
    # Include chin without exposing other people to the model.
    x0, y0 = max(0, round(x0 - w * .04)), max(0, round(y0 - h * .03))
    x1, y1 = min(width, round(x1 + w * .04)), min(height, round(y1 + h * .12))
    if min(x1 - x0, y1 - y0) < 24:
        return None
    return [x0, y0, x1, y1]

def blend_mouth(frame, prediction, box):
    """Only the lower face is edited; eyes, hair, surroundings stay original."""
    x0, y0, x1, y1 = box
    roi = frame[y0:y1, x0:x1]
    if not roi.size:
        return
    h, w = roi.shape[:2]
    generated = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_CUBIC)
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (w // 2, round(h * .73)), (max(1, round(w * .43)), max(1, round(h * .25))), 0, 0, 360, 1, -1)
    kernel = max(3, round(min(h, w) * .07) | 1)
    mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    mask[:round(h * .4)] = 0
    # Feather the crop perimeter as well as the ellipse.
    edge = np.minimum(np.minimum(np.arange(w)[None, :] + 1, w - np.arange(w)[None, :]), np.minimum(np.arange(h)[:, None] + 1, h - np.arange(h)[:, None]))
    mask *= np.clip(edge / max(2, min(h, w) * .06), 0, 1)
    # Small bounded color correction avoids obvious seams without erasing expressions.
    ring = (mask > .03) & (mask < .4)
    if np.any(ring):
        correction = np.clip(roi[ring].astype(np.float32).mean(axis=0) - generated[ring].astype(np.float32).mean(axis=0), -12, 12)
        generated = np.clip(generated.astype(np.float32) + correction, 0, 255)
    alpha = mask[:, :, None]
    roi[:] = np.clip(generated * alpha + roi * (1 - alpha), 0, 255).astype(np.uint8)

def render(video, destination, face_boxes, wav_path, fps, progress, mel=None):
    if mel is None:
        mel = mel_spectrogram(wav_path)
    cap = cv2.VideoCapture(str(video))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not cap.isOpened() or not writer.isOpened():
        cap.release(); writer.release()
        raise ValueError('Could not open lip-sync input or video encoder.')
    processed = edited = skipped = 0
    batch_size = max(1, min(8, int(os.getenv('LIPSYNC_BATCH_SIZE', '4'))))
    try:
        while processed < len(face_boxes):
            frames, boxes, indices, crops, windows = [], [], [], [], []
            for _ in range(batch_size):
                if processed + len(frames) >= len(face_boxes):
                    break
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('Lip-sync video ended before its face-track metadata.')
                index = processed + len(frames)
                box = face_boxes[index]
                frames.append(frame); boxes.append(box)
                if box is not None:
                    x0, y0, x1, y1 = box
                    crop = frame[y0:y1, x0:x1]
                    if crop.size:
                        indices.append(len(frames) - 1); crops.append(crop); windows.append(mel_window(mel, index, fps))
            if crops:
                for index, prediction in zip(indices, predict(crops, windows)):
                    blend_mouth(frames[index], prediction, boxes[index])
                    edited += 1
            skipped += len(frames) - len(indices)
            for frame in frames:
                writer.write(frame)
            processed += len(frames)
            progress(processed, len(face_boxes), edited, skipped)
        if edited == 0:
            raise ValueError('No clearly tracked faces were available for lip sync.')
        return {'lipSyncFrames': edited, 'lipSyncSkipped': skipped}
    finally:
        cap.release(); writer.release()
