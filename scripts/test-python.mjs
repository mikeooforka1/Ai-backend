import {spawnSync} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const python = process.env.PYTHON_BIN || path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const result = spawnSync(python, ['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'], {cwd: root, stdio: 'inherit'});
if (result.error) throw result.error;
process.exit(result.status || 0);
