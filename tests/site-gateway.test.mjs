import {test} from 'node:test';
import assert from 'node:assert/strict';
import worker from '../deploy/site-worker.js';

test('unconfigured voice service reports unavailable without contacting a GPU',async()=>{
  const response=await worker.fetch(new Request('https://site.example/api/health'),{});
  assert.equal((await response.json()).ready,false);
  assert.equal((await worker.fetch(new Request('https://site.example/api/conversation'),{})).status,503);
});
test('private gateway supplies its credential and strips browser credentials',async()=>{
  const original=globalThis.fetch;let sent;
  globalThis.fetch=async request=>{sent=request;return new Response('ok');};
  try{
    const env={VOICE_BACKEND_URL:'https://backend.example',VOICE_SERVICE_TOKEN:'server-secret'};
    await worker.fetch(new Request('https://site.example/api/call-diagnostics',{method:'POST',body:'{}',headers:{cookie:'private-session',authorization:'Bearer browser-token'}}),env);
    assert.equal(sent.url,'https://backend.example/api/call-diagnostics');
    assert.equal(sent.headers.get('x-voice-service-token'),'server-secret');
    assert.equal(sent.headers.get('cookie'),null);assert.equal(sent.headers.get('authorization'),null);
    assert.equal(await sent.text(),'{}');
  }finally{globalThis.fetch=original;}
});
test('brain geometry stays on the website and does not wake inference',async()=>{
  let path;
  await worker.fetch(new Request('https://site.example/api/connectome/positions.bin'),{ASSETS:{fetch:async req=>{path=new URL(req.url).pathname;return new Response('asset');}}});
  assert.equal(path,'/neural/connectome/positions.bin');
});
