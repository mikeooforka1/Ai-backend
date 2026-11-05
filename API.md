# Express API

All routes use /api. If API_KEY is configured, send x-api-key on every route except GET /health. Do not expose this local processing server publicly without appropriate deployment hardening. API_KEY remains optional on default 127.0.0.1.

- GET /health: ready, capabilities.object/person/face/face_lipsync, faceError, lipSyncError, device=cpu, busy, queue, authRequired, engineError and CPU lip limits. ready describes the base engine; optional model failure does not disable image modes.
- POST /uploads: multipart video + image + optional audio. MP4/MOV/WebM; PNG/JPEG; WAV/MP3/M4A. No text fields. Video/image up to MAX_UPLOAD_MB (250 MB default), audio up to 50 MB; decoded audio duration 0.2–120s. Response contains upload id, decoded video metadata, audio metadata or null, preview/source routes.
- GET /uploads/:id/preview or /source: authenticated selection/source images.
- POST /jobs JSON: uploadId, mode (object/person/face/face_lipsync), selection={x,y,w,h} normalized 0–1, optional faceQuality=original|natural. Boxes must fit the frame and be at least 2% of each dimension / 12 pixels. Original model blend is the default. Lip-sync needs both models, speech audio and a video meeting the CPU duration/resolution/FPS limits.
- GET /jobs/:id: status queued/processing/completed/failed/cancelled, stage, progress 0–100, detail, history[{stage,at,detail}], created/started/finished/updated timestamps, frame counts, measured stageFps, cancelRequested, warnings, outputDuration and download when completed. Missing face frames remain unchanged. progress reflects stages, not a promised elapsed-time fraction.
- POST /jobs/:id/cancel: remove queued job or request cooperative running cancellation; returns 202. A current inference batch/encoding call can finish first. Finished jobs return 409.
- GET /jobs/:id/download: H.264/AAC MP4 only for completed jobs; 404 before completion, after cancellation or for expired sessions.

Sessions/jobs are in memory. Restarting the backend invalidates prepared uploads and jobs; abandoned UUID session directories are removed on startup. Completed/failed sessions expire after RETENTION_HOURS (24 default). Re-upload after restart. One processing job runs at a time, with a bounded queue. Upload preparation is serialized with rendering.

Lip-sync output ends at the shorter input duration; video is never repeated. Uploaded audio replaces original audio. Object/person/face-only outputs preserve original audio when present. Completed outputs are probed before download is enabled.

The Python protocol is private and contains server-controlled paths. Do not expose it as an HTTP endpoint. The Express gateway validates modes, selection, authentication and uploads before sending commands. Model filenames and external filesystem paths cannot be supplied through job requests.
