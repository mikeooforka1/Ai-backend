import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { EventEmitter } from "node:events";
import { randomUUID } from "node:crypto";
export class Engine extends EventEmitter {
  constructor(python, script, env) {
    super();
    this.python = python;
    this.script = script;
    this.env = env;
    this.pending = new Map();
    this.ready = false;
    this.stopping = false;
  }
  start() {
    this.ready = false;
    this.lastError = "Python is starting; optional models may take time to load.";
    const child = (this.child = spawn(this.python, ["-u", this.script], {
      env: { ...process.env, ...this.env },
      stdio: ["pipe", "pipe", "pipe"],
    }));
    let diagnostics = "";
    child.stderr.on("data", (d) => {
      diagnostics = (diagnostics + d.toString()).slice(-4000);
      process.stderr.write(d);
    });
    const fail = (error) => {
      if (this.child !== child) return;
      this.ready = false;
      this.lastError = error.message;
      for (const p of this.pending.values()) {
        clearTimeout(p.timer);
        p.reject(error);
      }
      this.pending.clear();
    };
    child.on("error", (e) =>
      fail(
        new Error(
          `Python engine could not start: ${e.message}. Run yarn setup:python.`,
        ),
      ),
    );
    createInterface({ input: child.stdout }).on("line", (line) => {
      let message;
      try {
        message = JSON.parse(line);
      } catch {
        console.error("Invalid engine protocol line");
        return;
      }
      if (message.event === "ready") {
        this.ready = true;
        this.lastError = null;
        this.capabilities = message.capabilities;
        this.emit("ready");
        return;
      }
      if (message.event === "progress") {
        this.emit("progress", message);
        return;
      }
      const p = this.pending.get(message.id);
      if (!p) return;
      clearTimeout(p.timer);
      this.pending.delete(message.id);
      message.error
        ? p.reject(new Error(message.error))
        : p.resolve(message.result);
    });
    child.on("exit", () => {
      fail(new Error(`Python engine exited. ${diagnostics}`));
      if (!this.stopping) this.restart = setTimeout(() => this.start(), 3000);
    });
  }
  waitUntilReady(timeout = 60000) {
    if (this.ready) return Promise.resolve(true);
    return new Promise((resolve, reject) => {
      const finish = () => {clearTimeout(timer); this.off("ready", finish); resolve(true);};
      const timer = setTimeout(() => {this.off("ready", finish); reject(new Error("Engine recovery timed out."));}, timeout);
      this.on("ready", finish);
    });
  }
  request(command, payload, timeout = 30000) {
    if (!this.ready)
      return Promise.reject(
        new Error(
          "Python engine is unavailable. Check backend logs and Python dependencies.",
        ),
      );
    return new Promise((resolve, reject) => {
      const id = randomUUID();
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error("Engine request timed out."));
        this.child.kill();
      }, timeout);
      this.pending.set(id, { resolve, reject, timer });
      this.child.stdin.write(
        JSON.stringify({ ...payload, id, command }) + "\n",
        (e) => {
          if (e) {
            clearTimeout(timer);
            this.pending.delete(id);
            reject(e);
          }
        },
      );
    });
  }
  stop() {
    this.stopping = true;
    clearTimeout(this.restart);
    this.child?.kill();
  }
}
