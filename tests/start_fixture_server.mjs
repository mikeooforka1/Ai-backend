// Inject the test-only engine at the Engine boundary, leaving production unchanged.
import {Engine} from '../src/engine.mjs';
import {fileURLToPath} from 'node:url';
const start = Engine.prototype.start;
Engine.prototype.start = function() {
  this.script = fileURLToPath(new URL('./lip_fixture.py', import.meta.url));
  return start.call(this);
};
await import('../src/server.mjs');
