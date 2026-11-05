import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const venv = path.join(root, ".venv");
const python = path.join(venv, process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
function run(command, args, cwd = root) {
  const result = spawnSync(command, args, {stdio: "inherit", cwd});
  if (result.error) throw new Error(`${result.error.message}. Check the tool is installed and restart PowerShell.`);
  if (result.status !== 0) process.exit(result.status || 1);
}
const face = process.argv.includes("--face");
const lip = process.argv.includes("--lipsync");
if (!existsSync(python)) {
  run(process.env.PYTHON_BOOTSTRAP || (process.platform === "win32" ? "python" : "python3"), ["-m", "venv", venv]);
} else console.log("Reusing existing backend/.venv; it will not be recreated.");
run(python, ["-c", "import sys; assert (3,10)<=sys.version_info[:2]<=(3,12), 'Use Python 3.10, 3.11 or 3.12 for these pinned optional model dependencies.'"]);
run(python, ["-m", "pip", "install", "--upgrade", "pip"]);
run(python, ["-m", "pip", "install", "-r", path.join(root, "engine/requirements.txt")]);
if (face) {
  run(python, ["-m", "pip", "install", "-r", path.join(root, "engine/requirements-face.txt")]);
  // InsightFace installs another OpenCV distribution; keep only the CSRT-capable one.
  run(python, ["-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless", "opencv-contrib-python"]);
  run(python, ["-m", "pip", "install", "--force-reinstall", "--no-deps", "opencv-contrib-python-headless==4.11.0.86"]);
}
if (lip) {
  run(python, ["-m", "pip", "install", "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cpu"]);
  run(python, ["-m", "pip", "install", "-r", path.join(root, "engine/requirements-lipsync.txt")]);
  const repository = path.join(root, "engine/vendor/Wav2Lip");
  if (process.argv.includes("--lipsync-source") && !existsSync(path.join(repository, "models/wav2lip.py"))) {
    run("git", ["clone", "https://github.com/Rudrabha/Wav2Lip.git", repository]);
    run("git", ["checkout", "bac9a81e63ecc153202353372e5724b83d9e6322"], repository);
  }
  console.log("CPU lip-sync dependencies installed. Download the official GAN TorchScript checkpoint and set WAV2LIP_MODEL_PATH in backend/.env. For older .pth weights add --lipsync-source and set WAV2LIP_REPO_PATH. See SETUP-WINDOWS.md.");
}
run(python, ["-c", "import cv2; assert hasattr(cv2,'TrackerCSRT_create'), 'CSRT tracker is missing'; print('CSRT tracker verified')"]);
console.log("Setup complete. Keep FFmpeg/ffprobe on PATH and restart the backend. No virtualenv activation is needed.");
