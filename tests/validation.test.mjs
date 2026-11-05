import { test } from "node:test";
import assert from "node:assert/strict";
import { selection, mode } from "../src/validation.mjs";
test("rejects out-of-frame, non-finite, tiny, and nonnumeric selections", () => {
  for (const value of [
    null,
    {},
    { x: NaN, y: 0, w: 0.2, h: 0.2 },
    { x: 0, y: 0, w: Infinity, h: 0.2 },
    { x: -0.1, y: 0, w: 0.2, h: 0.2 },
    { x: 0.9, y: 0, w: 0.2, h: 0.2 },
    { x: 0, y: 0, w: 0.001, h: 0.2 },
    { x: "0", y: 0, w: 0.2, h: 0.2 },
  ])
    assert.throws(() => selection(value));
  assert.deepEqual(selection({ x: 0, y: 0.1, w: 1, h: 0.9 }), {
    x: 0,
    y: 0.1,
    w: 1,
    h: 0.9,
  });
  assert.throws(() => mode("anything"));
  assert.equal(mode("face"), "face");
});
