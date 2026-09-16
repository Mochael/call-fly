// Bounded real-speech probe. Saves metrics only; never saves audio or text.
import {parseArgs} from 'node:util';
import {spawnSync} from 'node:child_process';
import {mkdir,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import WebSocket from 'ws';

const {values}=parseArgs({options:{url:{type:'string'},audio:{type:'string'},seconds:{type:'string',default:'30'}}});
assert(values.url && values.audio,'Supply --url and --audio.');
const base=new URL(values.url).origin,seconds=Number(values.seconds);
assert(seconds>0 && seconds<=300);
const frames=Math.round(seconds*12.5),headers={};
if(process.env.VERCEL_AUTOMATION_BYPASS_SECRET)headers['x-vercel-protection-bypass']=process.env.VERCEL_AUTOMATION_BYPASS_SECRET;
const decoded=spawnSync('ffmpeg',['-v','error','-i',values.audio,'-t',String(seconds),'-ar','24000','-ac','1','-f','f32le','pipe:1'],{maxBuffer:40_000_000});
assert.equal(decoded.status,0,'Could not decode the supplied test audio.');
const pcm=Buffer.alloc(frames*7680);decoded.stdout.copy(pcm);
const health=await fetch(base+'/api/health',{headers});assert.equal(health.status,200);
assert.equal((await health.json()).ready,true);
const manifestResponse=await fetch(base+'/api/connectome/manifest.json',{headers});assert.equal(manifestResponse.status,200);
const manifest=await manifestResponse.json();assert.equal(manifest.neurons,166700);
for(const name of ['metadata.json','positions.bin']) {
  const response=await fetch(base+'/api/connectome/'+name,{headers});assert.equal(response.status,200);
  const data=Buffer.from(await response.arrayBuffer());
  assert.equal(createHash('sha256').update(data).digest('hex'),manifest.geometry_files[name]);
}

async function call(count) {
  const started=performance.now(),sent=new Map(),brainHashes=new Set();
  let ready,readyMs,audioFrames=0,modelFrames=0,energy=0,samples=0,nonzeroCorrections=0,metrics;
  const delays=[];let timer,deadline;
  const ws=new WebSocket(base.replace('https:','wss:').replace('http:','ws:')+'/api/conversation',{origin:base,headers,maxPayload:1_048_576});
  try {
    await new Promise((resolve,reject)=>{
      deadline=setTimeout(()=>reject(new Error('Voice probe timed out.')),count*80+210_000);
      function sendFrame(i,begin) {
        if(i>=count)return;
        sent.set(i+1,performance.now());ws.send(pcm.subarray(i*7680,(i+1)*7680));
        timer=setTimeout(()=>sendFrame(i+1,begin),Math.max(0,begin+(i+1)*80-performance.now()));
      }
      ws.on('error',()=>reject(new Error('Hosted WebSocket failed.')));
      ws.on('close',code=>{if(audioFrames<count)reject(new Error(`Call closed early (${code}, ${audioFrames}/${count} audio frames).`));});
      ws.on('message',(raw,binary)=>{
        try {
          if(binary) {
            assert.equal(raw.length,7680);audioFrames++;
            for(let offset=0;offset<raw.length;offset+=4){const x=raw.readFloatLE(offset);assert(Number.isFinite(x));energy+=x*x;samples++;}
            if(audioFrames===count)resolve();return;
          }
          const msg=JSON.parse(raw.toString());
          assert.notEqual(msg.type,'error',msg.message || 'Voice service error.');
          if(msg.type==='ready') {
            ready=msg;readyMs=performance.now()-started;sendFrame(0,performance.now());
            console.log(JSON.stringify({stage:'connected',ready_ms:Math.round(readyMs),gpu:msg.gpu,serving:msg.serving}));
          }
          if(msg.type==='frame') {
            modelFrames++;assert.equal(msg.features.length,64);assert(msg.features.every(Number.isFinite));
            if(msg.reservoir_logit_rms>0)nonzeroCorrections++;
            if(sent.has(msg.index))delays.push(performance.now()-sent.get(msg.index));
            if(msg.brain?.state)brainHashes.add(createHash('sha256').update(msg.brain.state).digest('hex'));
          }
          if(msg.type==='metrics')metrics=msg;
        }catch(error){reject(error);}
      });
    });
    assert.equal(audioFrames,count);assert.equal(modelFrames,count);assert(energy>0);assert(nonzeroCorrections>0);assert(brainHashes.size>5);
    assert.equal(ready.execution,'torch');assert.equal(ready.serving,'upstream-masked-streaming');
    const ordered=delays.toSorted((a,b)=>a-b);
    return {call_id:ready.call_id,ready_ms:readyMs,audio_frames:audioFrames,audio_seconds:samples/24000,
      nonzero_correction_frames:nonzeroCorrections,distinct_brain_states:brainHashes.size,audio_rms:Math.sqrt(energy/samples),
      mean_frame_roundtrip_ms:delays.reduce((a,b)=>a+b,0)/delays.length,p95_frame_roundtrip_ms:ordered[Math.floor(ordered.length*.95)],
      queued_frames:metrics?.queued_frames,model_compute_ms:metrics?.compute_ms};
  }finally {
    clearTimeout(timer);clearTimeout(deadline);
    if(ws.readyState===WebSocket.OPEN){ws.send(JSON.stringify({type:'stop',reason:'user_stop'}));ws.close();}
    else ws.terminate();
  }
}

const report={url:base,geometry_verified:true,call:await call(frames),restart:await call(Math.min(frames,75))};
assert.notEqual(report.call.call_id,report.restart.call_id);
await mkdir('artifacts/vercel',{recursive:true});
await writeFile('artifacts/vercel/voice-probe.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
