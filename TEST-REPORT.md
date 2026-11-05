# Validation report — 8 October 2026

## Results

| Check | Result | What it establishes |
|---|---|---|
| CRA production build with CI=true | Passed | Current CRA frontend compiles successfully without CI warning failures. |
| React tests | 5 passed | All four readiness-gated modes, audio picker, configuration error display, processing stages, cancellation control and measured stage ETA logic. |
| Express/engine Node tests | 4 passed | Existing selective rendering, authentication, invalid media/selection handling, protocol identifiers, audio upload validation, lip-sync orchestration, timing, progress and cancellation. |
| Python tests | 17 passed | Image alpha/clipping, face selection/identity gates, ambiguity, confidence, movement continuity, natural blend isolation, mel timing/features, silent audio rejection, mouth mask isolation, skipped-face streaming and cancellation markers. |
| Real official Wav2Lip GAN CPU inference | Passed | The current official TorchScript checkpoint loads, passes a forward inference check, renders 10 video frames, changes the mouth region, and preserves regions outside the selected face within re-encoding tolerance. |

The existing object AND person modes were rendered through the actual Express/Python/FFmpeg pipeline on a 40-frame moving-target clip. Every target frame was replaced, the second object was preserved, and H.264 video plus original audio were verified. Cut/disappearance fixtures stopped tracking and produced warnings instead of switching to another object.

The combined face+audio HTTP integration test uses explicitly named **InsightFace and lip model doubles** to test orchestration independently of model files. It verifies a 20-frame output shortened to a 1-second audio recording, preservation of a second target, original 40-frame face-only rendering, model readiness, missing-audio rejection, natural-blend option routing, stage history, cancellation and unavailable downloads. This is not a visual-quality benchmark.

Separately, the **real official Wav2Lip-SD-GAN.pt** downloaded from the authors' current README link was loaded on CPU with PyTorch 2.5.1+cpu. A NASA/scikit-image astronaut portrait fixture and synthesized tone produced a 10-frame video. The test checked actual model output and outside-face isolation, not whether the mouth perceptually matches human speech. The measured lip-render pass took approximately 2.20 seconds on the test machine, excluding startup/model loading and face swapping. Do not use that time to predict performance on the user's Intel PC. Exact machine-readable results are in neural-test-result.json.

## Important limits

- The user's configured InsightFace/inswapper weights were not supplied. Face identity/continuity/blending logic was tested with controlled doubles; realism of the upgraded face swap with those actual weights was not visually benchmarked here.
- The real lip-sync smoke test used a portrait and tone, not the user's moving talking footage. Speech alignment, profiles, occlusions, lighting changes and subjective naturalness still need review on actual footage. The pretrained model operates on 96×96 face inputs and cannot guarantee HD facial detail.
- Validation ran on Linux, Python 3.12, Node 24 and FFmpeg. Windows instructions and path handling are included, but installation was not executed on the user's Windows PC.
- JavaScript build/tests used the previously installed matching CRA/Express dependencies. A fresh Yarn dependency download was not completed in this environment; dependency versions remain the supplied project's versions, and lockfiles/Yarn node-modules configuration are included.
- Model weights and private configuration are excluded from the deliverables. The new mode remains disabled until both face and lip models actually initialize. Startup checks do not falsely report readiness from a filename alone.

## Repeat checks

From backend:

```powershell
corepack yarn test
corepack yarn test:python
corepack yarn doctor
```

From frontend:

```powershell
corepack yarn test --watchAll=false --runInBand
corepack yarn build
```

An optional isolated real model test is provided as backend/tests/neural_smoke.py. Set WAV2LIP_MODEL_PATH, then pass a 512×512-compatible NASA/scikit-image astronaut portrait image to that script; its face coordinates are fixture-specific. It is not a general face detector or an application launch command.
