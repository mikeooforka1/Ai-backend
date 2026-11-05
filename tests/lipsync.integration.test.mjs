import {test} from 'node:test';
import assert from 'node:assert/strict';
import {spawn,spawnSync} from 'node:child_process';
import {mkdtemp,readFile,writeFile,rm} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const python=process.env.PYTHON_BIN||path.join(root,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
test('Express/audio/lip-sync pipeline, original audio retention and cancellation (model doubles)',{timeout:120000},async t=>{
  const directory=await mkdtemp(path.join(os.tmpdir(),'vividflow-lip-test-'));
  const port='5199',base=`http://127.0.0.1:${port}/api`;
  const server=spawn(process.execPath,[path.join(root,'tests/start_fixture_server.mjs')],{cwd:root,env:{...process.env,PORT:port,HOST:'127.0.0.1',PYTHON_BIN:python,DATA_DIR:path.join(directory,'data'),API_KEY:''},stdio:['ignore','pipe','pipe']});
  let logs='';server.stdout.on('data',d=>logs+=d);server.stderr.on('data',d=>logs+=d);
  t.after(async()=>{server.kill('SIGTERM');await delay(1800);if(server.exitCode===null)server.kill('SIGKILL');await rm(directory,{force:true,recursive:true});});
  for(let i=0;i<100;i++){try{const response=await fetch(base+'/health');if(response.ok)break;}catch{}await delay(100);if(i===99)throw new Error(logs);}
  const health=await(await fetch(base+'/health')).json();assert.equal(health.capabilities.face_lipsync,true);
  const generate=spawnSync(python,[path.join(root,'tests/media.py'),'generate',directory],{encoding:'utf8'});assert.equal(generate.status,0,generate.stderr);
  async function upload(audio){const form=new FormData();for(const [field,name,type] of [['video','input.mp4','video/mp4'],['image','source.png','image/png'],...(audio?[['audio','speech.wav','audio/wav']]:[])])form.append(field,new Blob([await readFile(path.join(directory,name))],{type}),name);const response=await fetch(base+'/uploads',{method:'POST',body:form});assert.equal(response.status,201,await response.clone().text());return response.json();}
  const selection={x:40/320,y:70/240,w:60/320,h:60/240};
  async function create(item,mode='face_lipsync'){return fetch(base+'/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({uploadId:item.id,mode,selection,faceQuality:'natural'})});}
  const withoutAudio=await upload(false);assert.equal((await create(withoutAudio)).status,400);
  const item=await upload(true);
  async function wait(id){for(let i=0;i<300;i++){const job=await(await fetch(`${base}/jobs/${id}`)).json();if(['completed','cancelled','failed'].includes(job.status))return job;await delay(100);}throw new Error('Job timed out: '+logs);}
  const created=await create(item);assert.equal(created.status,202,await created.clone().text());const job=await wait((await created.json()).id);assert.equal(job.status,'completed',job.error);
  assert.equal(job.framesTotal,20);assert.equal(job.lipSyncFrames,20);assert.ok(job.outputDuration<=1.01);assert.ok(job.warnings.some(w=>w.includes('ends with')));
  for(const stage of ['face_swap','audio','lip_sync','encoding','verifying'])assert.ok(job.history.some(e=>e.stage===stage),stage);
  const output=path.join(directory,'lip-output.mp4');await writeFile(output,Buffer.from(await(await fetch(`${base}/jobs/${job.id}/download`)).arrayBuffer()));
  const check=spawnSync(python,['-c',`import cv2,json,subprocess; p=${JSON.stringify(output)}; m=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',p])); assert any(s['codec_type']=='audio' for s in m['streams']); c=cv2.VideoCapture(p); assert int(c.get(cv2.CAP_PROP_FRAME_COUNT))==20; ok,f=c.read(); assert ok; assert f[100,255,0]>140; c.release()`],{encoding:'utf8'});assert.equal(check.status,0,check.stderr);
  const face=await create(item,'face');const faceJob=await wait((await face.json()).id);assert.equal(faceJob.status,'completed',faceJob.error);assert.equal(faceJob.framesTotal,40);assert.equal(faceJob.framesSkipped,0);
  const cancelCreate=await create(item);const cancelledId=(await cancelCreate.json()).id;
  const cancel=await fetch(`${base}/jobs/${cancelledId}/cancel`,{method:'POST'});assert.equal(cancel.status,202);
  const cancelled=await wait(cancelledId);assert.equal(cancelled.status,'cancelled');assert.equal((await fetch(`${base}/jobs/${cancelledId}/download`)).status,404);
  console.log('Lip-sync orchestration: 20-frame audio-limited output; second target preserved; progress stages and cancellation verified with explicit model doubles.');
});
