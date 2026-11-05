# VividFlow CPU upgrade — Windows setup

This update keeps CRA 5.0.1, React 18, Yarn 4.9.2, Express and your existing Python engine. There is no cloud account or service requirement. Keep running the frontend and backend in separate terminals.

## 1. Apply the update without deleting your working models

Stop your backend and frontend using Ctrl+C before replacing code. If Windows asks `Terminate batch job (Y/N)?`, enter `Y`; do not type Python activation commands at that prompt.

Back up your current AI folder. Extract the combined ZIP into a separate folder first, then copy its code into your existing `C:\Users\NO GO NO\Desktop\AI` folder. Alternatively extract `frontend.zip` and `backend.zip` beside each other there. Each archive contains its named top-level folder.

**Keep your existing `backend/.env`, `backend/.venv` and `backend/engine/models`.** The ZIPs exclude them. Replace the whole code directories supplied, including both `engine/main.py` and `engine/lipsync.py`; copying only main.py will fail. Keep your existing FACE_MODEL_PATH, FACE_MODEL_ROOT and FACE_ANALYSIS_NAME settings.

The combined ZIP also updates the root package.json, yarn.lock, .yarnrc.yml and scripts. Standalone ZIPs have their own package.json and Yarn configuration, and can be installed individually when there is no parent workspace.

## 2. Install JavaScript dependencies with Yarn

For the combined project, open a new PowerShell terminal:

```powershell
cd "C:\Users\NO GO NO\Desktop\AI"
corepack yarn install
```

If using the two folders independently, run `corepack yarn install` once in backend and once in frontend instead. Do not use npm. Node 22 or newer is required. CRA may print dependency/deprecation warnings; they are separate from install/build failures. If installation fails, inspect the actual Yarn build.log before changing dependencies.

## 3. Install the NEW optional CPU lip-sync dependencies

First stop the backend. You do **not** need to activate `.venv`. Use Python 3.10, 3.11 or 3.12; the pinned PyTorch dependency does not support every newer Python release.

```powershell
cd "C:\Users\NO GO NO\Desktop\AI\backend"
corepack yarn setup:python --lipsync
```

This reuses the existing `.venv` rather than recreating its locked python.exe, installs CPU PyTorch 2.5.1, librosa and scipy, and checks that the existing CSRT tracker works. It does not download model weights. The dependencies may take several hundred megabytes. No NVIDIA driver, CUDA toolkit, extra C++ installation or paid API key is required for this new lip-sync step.

**Your face swap already works:** you do not need to run `--face` again. On a fresh installation only, use `corepack yarn setup:python --face --lipsync`, install the face models, and configure the same FACE_* variables as before. InsightFace's Windows installation can need C++ tools; that is separate from the new CPU lip-sync dependencies.

If the script reports an unsupported Python version, check it with `backend\.venv\Scripts\python.exe --version`. Keep a backup and create a compatible environment with Python 3.11 or 3.12; do not delete a currently working face environment without preserving it.

## 4. Download the official lip-sync checkpoint

Open the [official Wav2Lip README](https://github.com/Rudrabha/Wav2Lip#non-commercial-open-source-version). In the open-source section, under "Getting the weights", choose **Wav2Lip + GAN**. The current linked file is **Wav2Lip-SD-GAN.pt**, approximately 146 MB. Download the actual model, not a Google Drive webpage or a shortcut.

Current official GAN download: https://drive.google.com/file/d/15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk/view

Place it here:

```text
C:\Users\NO GO NO\Desktop\AI\backend\engine\models\Wav2Lip-SD-GAN.pt
```

Add this ONE new line to your existing `backend/.env`:

```dotenv
WAV2LIP_MODEL_PATH=C:/Users/NO GO NO/Desktop/AI/backend/engine/models/Wav2Lip-SD-GAN.pt
```

Keep your existing FACE_* lines intact. Use forward slashes as shown; spaces in your username are supported. The current official TorchScript checkpoint needs no extra face-detector download, no Git clone, and no WAV2LIP_REPO_PATH. This app already uses your selected InsightFace identity to locate the face.

For **older .pth state-dictionary checkpoints** only: install Git, run `corepack yarn setup:python --lipsync --lipsync-source`, then set WAV2LIP_REPO_PATH to `.../backend/engine/vendor/Wav2Lip`. The source checkout is pinned to `bac9a81e63ecc153202353372e5724b83d9e6322`. The app loads old checkpoints using `weights_only=True` rather than arbitrary pickle loading. Incompatible checkpoints are rejected; use the official current GAN download above.

## 5. Confirm installation, then start both terminals

With the backend still stopped:

```powershell
cd "C:\Users\NO GO NO\Desktop\AI\backend"
corepack yarn doctor
```

Look for `Face swap: READY` and `CPU lip sync: READY`. The doctor loads the real models and runs a Wav2Lip inference check. An unconfigured optional repository is normal for TorchScript. It does not expose API_KEY values or download weights.

**Terminal 1 — backend**

```powershell
cd "C:\Users\NO GO NO\Desktop\AI\backend"
corepack yarn start
```

Leave it running. Express starts Python automatically. Optional models load at startup; give them time to initialize.

**Terminal 2 — CRA frontend**

```powershell
cd "C:\Users\NO GO NO\Desktop\AI\frontend"
corepack yarn start
```

Open http://localhost:3000. From a third PowerShell terminal you can check:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/health | ConvertTo-Json -Depth 5
```

Successful configuration includes `ready: true`, `capabilities.face: true`, and `capabilities.face_lipsync: true`. Optional model failure disables only the corresponding mode; object/person image replacement stays usable. Read `faceError` or `lipSyncError` for the precise reason.

## 6. Run your first face + audio render

1. Choose **AI face swap + audio lip sync** once health reports it ready.
2. Choose a video, one replacement face picture, and a WAV/MP3/M4A speech recording.
3. Start with a clear, front-facing, continuously visible face in a 5–10-second clip at 480p. Avoid cuts, hands over the mouth, extreme profiles and multiple overlapping faces.
4. Click **Upload & prepare selection**. Audio is uploaded with the two other files. If you change audio later, prepare again.
5. Draw a box around exactly one face on the first frame. Choose the original blend to preserve the existing model appearance, or try natural blend for subtle colour matching.
6. Render, review all stages, then load the preview and download the MP4.

CPU lip-sync defaults: **15 seconds maximum, up to 720p and 30 FPS**; audio upload is at most 50 MB. Audio replaces the original soundtrack. Output lasts as long as the shorter of video and uploaded audio. It never repeats video frames to extend the clip. Frames with missing, ambiguous or untracked faces are left unchanged, and warnings tell you how many were skipped.

This machine has Intel UHD graphics and 16 GB RAM. Inference runs on CPU, and speed depends on your CPU, clip and input resolution. The test machine's measured times are not predictions for your PC. If memory usage is high, add `LIPSYNC_BATCH_SIZE=1` and restart. The original full-frame video is preserved in resolution; only the model's face crop is resized internally to 96×96, so lip detail has that model's limitations.

Object/person modes remain tracked image overlays, not full-body reconstruction. This update does not include outfit or generative full-body models.

## Troubleshooting

- **Face model unavailable:** preserve your previously working FACE_* settings and verify their files. Lip-sync requires face swap AND Wav2Lip to be ready.
- **No module named torch/librosa/scipy:** rerun setup:python --lipsync using this backend's script; PYTHON_BIN, if set, must point to the environment you installed into.
- **Lip model load/shape error:** use the official compatible checkpoint above; a saved HTML page or unrelated model will not load.
- **ffmpeg/ffprobe missing:** reopen PowerShell after FFmpeg installation and run `ffmpeg -version` and `ffprobe -version`. Restart VS Code too if its terminals inherited the old PATH.
- **Frontend ECONNREFUSED:** the backend is stopped or not on port 5000. Keep Terminal 1 running. CRA's proxy is still 127.0.0.1:5000.
- **Server access key:** localhost does not require one unless API_KEY is set in backend/.env. If configured, the frontend explicitly asks for it before preparing media. This is your server key, not a model provider key.
- **Audio silent/invalid:** use an actual spoken recording. Pure silence is rejected; supported file extensions alone do not guarantee valid media.
- **Face disappears or turns away:** the stricter identity/movement checks skip unclear frames. After a scene cut or a long loss, later frames are left unchanged. Split the video into clear continuous shots.
- **Cancelled:** cancellation is cooperative. A current neural batch or encoding call may finish before it stops. No partial output is advertised as completed.

## Tests and licensing

Run `corepack yarn test` and `corepack yarn test:python` in backend; optional lip-sync unit tests require the optional dependencies. In frontend run `corepack yarn test --watchAll=false --runInBand` and `corepack yarn build`.

See TEST-REPORT.md for exactly what was tested and the limits of those tests. Wav2Lip's open-source pretrained models are restricted to personal/research/non-commercial use by their authors. InsightFace pretrained face models also have licensing restrictions. For commercial work obtain the appropriate model licenses; a project's MIT-style source license does not automatically license pretrained weights. No pretrained weights are included in these ZIPs.
