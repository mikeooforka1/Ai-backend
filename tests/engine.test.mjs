import { test } from "node:test";
import assert from "node:assert/strict";
import { once } from "node:events";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Engine } from "../src/engine.mjs";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const python =
  process.env.PYTHON_BIN ||
  path.join(
    root,
    ".venv",
    process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
  );
test(
  "payload identifiers cannot override private RPC correlation identifiers",
  { timeout: 10000 },
  async (t) => {
    const engine = new Engine(
      python,
      path.join(root, "tests/protocol_fixture.py"),
      {},
    );
    t.after(() => engine.stop());
    engine.start();
    await once(engine, "ready");
    const result = await engine.request(
      "echo",
      { id: "public-upload-id", command: "overridden" },
      2000,
    );
    assert.deepEqual(result, { command: "echo" });
  },
);
