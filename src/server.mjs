import dotenv from "dotenv";
import express from "express";
import multer from "multer";
import helmet from "helmet";
import path from "node:path";
import fs from "node:fs/promises";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { randomUUID, timingSafeEqual } from "node:crypto";
import { Engine } from "./engine.mjs";
import { selection, mode, faceQuality } from "./validation.mjs";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
dotenv.config({ path: path.join(root, ".env") });
const data = path.resolve(process.env.DATA_DIR || path.join(root, "data"));
await fs.mkdir(data, { recursive: true });
const host = process.env.HOST || "127.0.0.1";
if (host === "0.0.0.0" && !process.env.API_KEY)
  throw new Error("Set API_KEY before exposing this processing server.");
const python =
  process.env.PYTHON_BIN ||
  path.join(
    root,
    ".venv",
    process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
  );
const engine = new Engine(python, path.join(root, "engine/main.py"), {
  DATA_DIR: data,
});
const app = express();
app.disable("x-powered-by");
app.use(
  helmet({
    contentSecurityPolicy: false,
    crossOriginResourcePolicy: { policy: "same-origin" },
  }),
);
app.use(express.json({ limit: "16kb" }));
app.get("/api/health", (req, res) =>
  res
    .status(engine.ready ? 200 : 503)
    .json({
      ready: engine.ready,
      capabilities: engine.capabilities || {},
      busy: running,
      queue: queue.length,
      authRequired: Boolean(process.env.API_KEY),
      engineError: engine.lastError || null,
      limits: {lipSyncSeconds: Number(process.env.LIPSYNC_MAX_SECONDS || 15), lipSyncPixels: Number(process.env.LIPSYNC_MAX_PIXELS || 921600), lipSyncFps: 30},
    }),
);
app.use("/api", (req, res, next) => {
  const expected = process.env.API_KEY;
  if (expected) {
    const actual = req.get("x-api-key") || "";
    const a = Buffer.from(actual),
      b = Buffer.from(expected);
    if (a.length !== b.length || !timingSafeEqual(a, b))
      return res.status(401).json({ error: "API key required." });
  }
  next();
});
const uploads = new Map(),
  jobs = new Map(),
  queue = [];
let running = false,
  accepting = true;
const upload = multer({
  dest: data,
  limits: {
    fileSize: Number(process.env.MAX_UPLOAD_MB || 250) * 1024 * 1024,
    files: 3,
    fields: 0,
  },
  fileFilter: (req, file, cb) => {
    const allowed =
      file.fieldname === "video"
        ? ["video/mp4", "video/quicktime", "video/webm"]
        : file.fieldname === "audio"
          ? ["audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave", "audio/mpeg", "audio/mp3", "audio/mp4", "audio/x-m4a", "audio/aac"]
          : ["image/png", "image/jpeg"];
    cb(
      allowed.includes(file.mimetype)
        ? null
        : new Error("Upload MP4/MOV/WebM video, PNG/JPEG image and optional WAV/MP3/M4A audio."),
      allowed.includes(file.mimetype),
    );
  },
}).fields([
  { name: "video", maxCount: 1 },
  { name: "image", maxCount: 1 },
  { name: "audio", maxCount: 1 },
]);
app.post("/api/uploads", (req, res, next) => {
  if (!engine.ready)
    return res
      .status(503)
      .json({ error: "Engine is not ready; check backend logs." });
  if (running || queue.length || !accepting)
    return res
      .status(409)
      .json({ error: "Wait for the current processing job before uploading." });
  accepting = false;
  upload(req, res, async (error) => {
    const files = Object.values(req.files || {}).flat();
    let dir;
    try {
      if (error) throw error;
      if (!req.files?.video?.[0] || !req.files?.image?.[0])
        throw new Error("Both a video and a replacement image are required.");
      const id = randomUUID();
      dir = path.join(data, id);
      await fs.mkdir(dir);
      const video = path.join(dir, "input.video"),
        image = path.join(dir, "source.image");
      await fs.rename(req.files.video[0].path, video);
      await fs.rename(req.files.image[0].path, image);
      let audio;
      if (req.files.audio?.[0]) {
        if (req.files.audio[0].size > 50 * 1024 * 1024) throw new Error("Audio upload must be smaller than 50 MB.");
        audio = path.join(dir, "input.audio");
        await fs.rename(req.files.audio[0].path, audio);
      }
      const metadata = await engine.request(
        "inspect",
        { video, image, audio, dir },
        Number(process.env.INSPECT_TIMEOUT_MS || 300000),
      );
      uploads.set(id, { id, dir, video, image, audio, metadata, created: Date.now() });
      res
        .status(201)
        .json({
          id,
          ...metadata,
          preview: `/api/uploads/${id}/preview`,
          source: `/api/uploads/${id}/source`,
        });
    } catch (e) {
      await Promise.all(files.map((f) => fs.rm(f.path, { force: true })));
      if (dir) await fs.rm(dir, { recursive: true, force: true });
      next(e);
    } finally {
      accepting = true;
    }
  });
});
app.get("/api/uploads/:id/:asset", (req, res) => {
  const item = uploads.get(req.params.id);
  const file = { preview: "preview.jpg", source: "source.png" }[
    req.params.asset
  ];
  if (!item || !file)
    return res.status(404).json({ error: "Upload not found." });
  res.sendFile(path.join(item.dir, file));
});
app.post("/api/jobs", (req, res, next) => {
  try {
    if (!engine.ready)
      return res.status(503).json({ error: "Engine unavailable." });
    if (!accepting || queue.length >= 4)
      return res
        .status(429)
        .json({ error: "Processing queue full; try again later." });
    const item = uploads.get(req.body.uploadId);
    if (!item)
      return res
        .status(404)
        .json({
          error: "Upload expired or not found; upload both files again.",
        });
    const box = selection(req.body.selection),
      type = mode(req.body.mode);
    if (["face", "face_lipsync"].includes(type) && !engine.capabilities.face)
      return res
        .status(422)
        .json({
          error: engine.capabilities.faceError || "Face models are not configured. See README face setup.",
        });
    if (type === "face_lipsync") {
      if (!engine.capabilities.face_lipsync) return res.status(422).json({error: engine.capabilities.lipSyncError || "Lip-sync model is unavailable. See SETUP-WINDOWS.md."});
      if (!item.audio) return res.status(400).json({error: "Upload speech audio with your video and picture, then prepare again."});
      const limits = {seconds: Number(process.env.LIPSYNC_MAX_SECONDS || 15), pixels: Number(process.env.LIPSYNC_MAX_PIXELS || 921600)};
      if (item.metadata.duration > limits.seconds || item.metadata.width * item.metadata.height > limits.pixels || item.metadata.fps > 30.01)
        return res.status(422).json({error: `CPU lip sync requires a clip up to ${limits.seconds} seconds, ${limits.pixels} pixels and 30 FPS. Start with 480p.`});
    }
    const quality = faceQuality(req.body.faceQuality);
    if (
      [...jobs.values()].some(
        (j) =>
          j.uploadId === item.id && ["queued", "processing"].includes(j.status),
      )
    )
      return res
        .status(409)
        .json({ error: "This upload already has an active job." });
    const job = {
      id: randomUUID(),
      uploadId: item.id,
      status: "queued",
      progress: 0,
      mode: type,
      selection: box,
      faceQuality: quality,
      stage: "queued",
      history: [{stage: "queued", at: Date.now(), detail: "Waiting for the processing engine."}],
      created: Date.now(),
    };
    jobs.set(job.id, job);
    queue.push(job);
    res.status(202).json(job);
    void drain();
  } catch (e) {
    next(e);
  }
});
engine.on("progress", (m) => {
  const j = jobs.get(m.jobId);
  if (!j || j.status !== "processing") return;
  if (m.stage && m.stage !== j.stage) {
    j.stage = m.stage;
    j.stageStarted = Date.now();
    j.history.push({stage: m.stage, at: j.stageStarted, detail: m.detail || ""});
  }
  for (const field of ["progress", "framesProcessed", "framesTotal", "framesReplaced", "framesSkipped", "lipSyncProcessed", "lipSyncTotal", "lipSyncFrames", "lipSyncSkipped", "stageFps", "detail"])
    if (m[field] !== undefined) j[field] = m[field];
  j.updated = Date.now();
});
async function drain() {
  if (running) return;
  running = true;
  try {
    while (queue.length) {
      const job = queue.shift(),
        item = uploads.get(job.uploadId);
      if (!engine.ready) {
        // Recovery must complete before sending the next queued request.
        const recovered = await engine.waitUntilReady(60000).catch(() => false);
        if (!recovered) {
          job.status = "failed"; job.error = "Python engine did not recover. Check backend logs and restart it."; job.finished = Date.now(); continue;
        }
      }
      if (job.cancelRequested || job.status === "cancelled") continue;
      job.status = "processing";
      job.started = Date.now();
      try {
        const result = await engine.request(
          "process",
          { ...item, jobId: job.id, mode: job.mode, selection: job.selection, faceQuality: job.faceQuality },
          Number(process.env.JOB_TIMEOUT_MS || 1800000),
        );
        if (job.cancelRequested) throw new Error("Processing cancelled by user.");
        Object.assign(job, result, {
          status: "completed",
          progress: 100,
          stage: "completed",
          download: `/api/jobs/${job.id}/download`,
        });
      } catch (e) {
        job.status = job.cancelRequested ? "cancelled" : "failed";
        job.stage = job.status;
        job.error = e.message;
        job.history.push({stage: job.status, at: Date.now(), detail: e.message});
      }
      job.finished = Date.now();
      if (job.status === "completed") job.history.push({stage: "completed", at: job.finished, detail: "Output verified and ready to download."});
      await fs.rm(path.join(item.dir, `${job.id}.cancel`), {force: true});
      if (job.status === "cancelled") await fs.rm(path.join(item.dir, `${job.id}.mp4`), {force: true});
    }
  } finally {
    running = false;
  }
}
app.post("/api/jobs/:id/cancel", async (req, res, next) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({error: "Job not found."});
  if (!["queued", "processing"].includes(job.status)) return res.status(409).json({error: "This job has already finished."});
  try {
    job.cancelRequested = true;
    if (job.status === "queued") {
      const index = queue.indexOf(job);
      if (index !== -1) queue.splice(index, 1);
      job.status = "cancelled"; job.stage = "cancelled"; job.finished = Date.now();
      job.history.push({stage: "cancelled", at: job.finished, detail: "Removed from queue."});
    } else {
      await fs.writeFile(path.join(uploads.get(job.uploadId).dir, `${job.id}.cancel`), "cancel");
    }
    res.status(202).json(job);
  } catch (error) {next(error);}
});
app.get("/api/jobs/:id", (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: "Job not found." });
  res.json(job);
});
app.get("/api/jobs/:id/download", (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job || job.status !== "completed")
    return res.status(404).json({ error: "Output is not ready." });
  res.download(
    path.join(uploads.get(job.uploadId).dir, `${job.id}.mp4`),
    "vividflow-replacement.mp4",
  );
});
const build = path.resolve(root, "../frontend/build");
if (existsSync(build)) {
  app.use(express.static(build));
  app.get("/{*path}", (req, res) =>
    res.sendFile(path.join(build, "index.html")),
  );
}
app.use((error, req, res, next) => {
  console.error(error.message);
  if (res.headersSent) return next(error);
  res
    .status(
      error instanceof multer.MulterError && error.code === "LIMIT_FILE_SIZE"
        ? 413
        : 400,
    )
    .json({ error: error.message });
});
// Crash recovery: no resumable media state is advertised. Remove abandoned sessions on startup.
for (const entry of await fs.readdir(data, { withFileTypes: true }))
  if (entry.isDirectory() && /^[0-9a-f-]{36}$/.test(entry.name))
    await fs.rm(path.join(data, entry.name), { recursive: true, force: true });
const cleanup = setInterval(async () => {
  const cutoff =
    Date.now() - Number(process.env.RETENTION_HOURS || 24) * 3600000;
  for (const [id, item] of uploads) {
    if (
      item.created >= cutoff ||
      [...jobs.values()].some(
        (j) => j.uploadId === id && ["processing", "queued"].includes(j.status),
      )
    )
      continue;
    try {
      await fs.rm(item.dir, { recursive: true, force: true });
      uploads.delete(id);
      for (const [jid, j] of jobs) if (j.uploadId === id) jobs.delete(jid);
    } catch (e) {
      console.error("Cleanup failed:", e.message);
    }
  }
}, 600000);
cleanup.unref();
engine.start();
const server = app.listen(Number(process.env.PORT || 5000), host, () =>
  console.log(
    `VividFlow: http://${host}:${process.env.PORT || 5000} (Python starts automatically)`,
  ),
);
let closing = false;
function shutdown() {
  if (closing) return;
  closing = true;
  clearInterval(cleanup);
  engine.stop();
  server.close();
  setTimeout(() => process.exit(0), 1500).unref();
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
