import { test } from "node:test";
import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const python =
  process.env.PYTHON_BIN ||
  path.join(
    root,
    ".venv",
    process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
  );
test(
  "real Express/Python upload, selective object/person rendering, audio, cuts and errors",
  { timeout: 180000 },
  async (t) => {
    const temp = await mkdtemp(path.join(os.tmpdir(), "vividflow-test-"));
    const port = process.env.TEST_PORT || "5198";
    const base = `http://127.0.0.1:${port}/api`;
    let logs = "";
    const server = spawn(
      process.execPath,
      [path.join(root, "src/server.mjs")],
      {
        cwd: root,
        env: {
          ...process.env,
          PORT: port,
          HOST: "127.0.0.1",
          DATA_DIR: path.join(temp, "data"),
          PYTHON_BIN: python,
          API_KEY: "test-secret",
          FACE_MODEL_PATH: "",
          FACE_MODEL_ROOT: "",
        },
        stdio: ["ignore", "pipe", "pipe"],
      },
    );
    server.stdout.on("data", (d) => (logs += d));
    server.stderr.on("data", (d) => (logs += d));
    t.after(async () => {
      server.kill("SIGTERM");
      await delay(1800);
      if (server.exitCode === null) server.kill("SIGKILL");
      await rm(temp, { recursive: true, force: true });
    });
    const headers = { "x-api-key": "test-secret" };
    async function ready() {
      for (let i = 0; i < 100; i++) {
        try {
          const r = await fetch(base + "/health");
          if (r.ok) return await r.json();
        } catch {}
        await delay(100);
      }
      throw new Error(logs);
    }
    const health = await ready();
    assert.equal(health.ready, true);
    assert.equal(health.capabilities.face, false);
    assert.equal(health.capabilities.face_lipsync, false);
    assert.equal(health.authRequired, true);
    function run(args) {
      const r = spawnSync(python, args, { encoding: "utf8" });
      assert.equal(r.status, 0, r.stderr + r.stdout);
      return r.stdout;
    }
    run([path.join(root, "tests/media.py"), "generate", temp]);
    async function upload(video = "input.mp4", audio) {
      const f = new FormData();
      f.append(
        "video",
        new Blob([await readFile(path.join(temp, video))], {
          type: "video/mp4",
        }),
        video,
      );
      f.append(
        "image",
        new Blob([await readFile(path.join(temp, "source.png"))], {
          type: "image/png",
        }),
        "source.png",
      );
      if (audio) f.append("audio", new Blob([await readFile(path.join(temp, audio))], {type: "audio/wav"}), audio);
      return fetch(base + "/uploads", { method: "POST", headers, body: f });
    }
    assert.equal((await fetch(base + "/jobs/no-such-job")).status, 401);
    assert.equal(
      (await fetch(base + "/jobs/no-such-job", { headers })).status,
      404,
    );
    assert.equal((await upload("invalid.mp4")).status, 400);
    const missing = new FormData();
    missing.append(
      "video",
      new Blob([await readFile(path.join(temp, "input.mp4"))], {
        type: "video/mp4",
      }),
      "x.mp4",
    );
    assert.equal(
      (
        await fetch(base + "/uploads", {
          method: "POST",
          headers,
          body: missing,
        })
      ).status,
      400,
    );
    const withAudio = await upload("input.mp4", "speech.wav");
    assert.equal(withAudio.status, 201, await withAudio.clone().text());
    assert.ok((await withAudio.json()).audio.duration > 0.9);
    assert.equal((await upload("input.mp4", "invalid.wav")).status, 400);
    const res = await upload();
    assert.equal(res.status, 201, await res.clone().text());
    const item = await res.json();
    assert.equal(item.width, 320);
    assert.equal(item.height, 240);
    assert.equal(
      (await fetch(`http://127.0.0.1:${port}${item.preview}`, { headers }))
        .status,
      200,
    );
    const box = { x: 40 / 320, y: 70 / 240, w: 60 / 320, h: 60 / 240 };
    const post = (body) =>
      fetch(base + "/jobs", {
        method: "POST",
        headers: { ...headers, "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    assert.equal(
      (
        await post({
          uploadId: item.id,
          mode: "object",
          selection: { ...box, x: 0.99 },
        })
      ).status,
      400,
    );
    assert.equal(
      (await post({ uploadId: item.id, mode: "face", selection: box })).status,
      422,
    );
    assert.equal((await post({uploadId: item.id, mode: "face_lipsync", selection: box})).status, 422);
    assert.equal((await post({uploadId: item.id, mode: "object", selection: box, faceQuality: "unknown"})).status, 400);
    async function render(item, mode) {
      const r = await post({ uploadId: item.id, mode, selection: box });
      assert.equal(r.status, 202, await r.clone().text());
      const queued = await r.json();
      assert.equal(
        (await post({ uploadId: item.id, mode, selection: box })).status,
        409,
      );
      assert.equal(
        (await fetch(`${base}/jobs/${queued.id}/download`, { headers })).status,
        404,
      );
      for (let i = 0; i < 600; i++) {
        const job = await (
          await fetch(`${base}/jobs/${queued.id}`, { headers })
        ).json();
        if (job.status === "failed") throw new Error(job.error);
        if (job.status === "completed") {
          assert.ok(job.history.some(h => h.stage === "encoding"));
          assert.ok(job.history.some(h => h.stage === "verifying"));
          assert.equal(job.stage, "completed");
          return job;
        }
        await delay(100);
      }
      throw new Error("Job timeout");
    }
    for (const mode of ["object", "person"]) {
      const job = await render(item, mode);
      assert.equal(job.framesTotal, 40);
      assert.equal(job.framesReplaced, 40);
      assert.equal(job.framesSkipped, 0);
      const download = await fetch(`${base}/jobs/${job.id}/download`, {
        headers,
      });
      assert.equal(download.status, 200);
      await writeFile(
        path.join(temp, "output.mp4"),
        Buffer.from(await download.arrayBuffer()),
      );
      console.log(
        mode +
          ": " +
          run([path.join(root, "tests/media.py"), "verify", temp]).trim(),
      );
    }
    for (const file of ["cut.mp4", "occluded.mp4"]) {
      const cut = await (await upload(file)).json();
      const job = await render(cut, "object");
      assert.ok(job.framesSkipped > 0);
      assert.ok(job.trackingLostAt >= 20 && job.trackingLostAt <= 22);
      assert.ok(job.warnings.length > 0);
    }
  },
);
