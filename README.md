# VividFlow — CPU face and audio studio

CRA 5.0.1 + React 18, Yarn 4, Express 5, and a private Python JSON-line engine automatically started by Express.

Updated features:
- Existing object/person tracked image overlays preserved.
- Face identity confidence and ambiguity checks, movement/scale checks, conservative handling of cuts and loss. Fixed source identity throughout the clip.
- Optional natural colour blending; the original model blend remains the default.
- Face swap followed by uploaded-audio Wav2Lip inference on CPU. Only the selected tracked face's lower region is edited; all other people remain outside the lip-sync inference region.
- Separate render stages, measured stage throughput/ETA, frame statistics, cancellation, readiness diagnosis, and original/result previews.

Read **SETUP-WINDOWS.md** first. Preserve your existing backend/.env, .venv and model files. Install the new optional CPU dependencies and official lip checkpoint before expecting the new mode to become selectable.

Backend and frontend run in separate terminals. At the combined project root, `corepack yarn install`, `corepack yarn build`, `corepack yarn test`, and `corepack yarn setup:python --lipsync` are available. In backend use `corepack yarn start`; in frontend use `corepack yarn start`.

No video or image is sent to a cloud service. Models run locally. No model weights, private .env, environments, uploaded media or node_modules are shipped. Read TEST-REPORT.md for tested behaviours and visual-quality limitations.
