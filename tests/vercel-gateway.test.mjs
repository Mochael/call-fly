import {test} from 'node:test';
import assert from 'node:assert/strict';
import {once} from 'node:events';
import {createServer} from 'node:http';
import WebSocket,{WebSocketServer} from 'ws';
import {backendConfiguration,createConversationServer,diagnosticsHandler,healthHandler} from '../deploy/vercel-gateway.js';

async function listen(server,t) {
  server.listen(0,'127.0.0.1');await once(server,'listening');
  t.after(()=>new Promise(resolve=>server.close(resolve)));
  return `http://127.0.0.1:${server.address().port}`;
}
async function open(url,t,headers={}) {
  const ws=new WebSocket(url.replace('http:','ws:'),{origin:url,...headers});
  t.after(()=>ws.terminate());await once(ws,'open');return ws;
}

test('only an HTTPS Modal origin with a server credential is accepted',()=>{
  const env={VOICE_BACKEND_URL:'https://voice.modal.run',VOICE_SERVICE_TOKEN:'fixture-secret'};
  assert.equal(backendConfiguration(env).url,env.VOICE_BACKEND_URL);
  for(const url of ['http://voice.modal.run','https://evil.example','https://voice.modal.run/path','https://voice.modal.run?x=1','https://user:pass@voice.modal.run']) {
    assert.throws(()=>backendConfiguration({...env,VOICE_BACKEND_URL:url}));
  }
  assert.throws(()=>backendConfiguration({}));
});

test('health exposes configuration status without exposing credentials or contacting inference',async t=>{
  const saved={url:process.env.VOICE_BACKEND_URL,token:process.env.VOICE_SERVICE_TOKEN};
  t.after(()=>{
    for(const [key,value] of Object.entries({VOICE_BACKEND_URL:saved.url,VOICE_SERVICE_TOKEN:saved.token})) {
      if(value===undefined)delete process.env[key];else process.env[key]=value;
    }
  });
  delete process.env.VOICE_BACKEND_URL;delete process.env.VOICE_SERVICE_TOKEN;
  const url=await listen(createServer(healthHandler),t);
  assert.equal((await (await fetch(url)).json()).ready,false);
  process.env.VOICE_BACKEND_URL='https://voice.modal.run';process.env.VOICE_SERVICE_TOKEN='fixture-secret';
  const response=await fetch(url),body=await response.json();assert.equal(body.ready,true);
  assert.equal(body.max_call_seconds,300);assert.ok(!JSON.stringify(body).includes('fixture-secret'));
  assert.equal(response.headers.get('cache-control'),'no-store');
});

test('WebSocket relay preserves text and binary, hides browser credentials, and releases the upstream',async t=>{
  const backend=createServer(),wss=new WebSocketServer({server:backend});
  const backendUrl=await listen(backend,t);let upstream,headers;
  const connected=new Promise(resolve=>wss.once('connection',(ws,req)=>{upstream=ws;headers=req.headers;ws.on('message',(data,isBinary)=>ws.send(data,{binary:isBinary}));resolve();}));
  const gateway=await listen(createConversationServer({configuration:()=>({url:backendUrl,token:'server-fixture'})}),t);
  const client=await open(gateway,t,{headers:{cookie:'browser-private',authorization:'Bearer browser-private','x-voice-service-token':'attacker'}});
  await connected;
  assert.equal(headers['x-voice-service-token'],'server-fixture');assert.equal(headers.cookie,undefined);assert.equal(headers.authorization,undefined);
  let reply=once(client,'message');client.send('{"type":"stop"}');let [data,binary]=await reply;
  assert.equal(binary,false);assert.equal(data.toString(),'{"type":"stop"}');
  reply=once(client,'message');const pcm=Buffer.alloc(7680,17);client.send(pcm);[data,binary]=await reply;
  assert.equal(binary,true);assert.deepEqual(data,pcm);
  const ended=once(upstream,'close');client.close();await ended;
});

test('cross-origin upgrades cannot allocate a backend connection',async t=>{
  let attempts=0;
  const url=await listen(createConversationServer({configuration:()=>{attempts++;return {url:'http://127.0.0.1:1',token:'fixture'};}}),t);
  const ws=new WebSocket(url.replace('http:','ws:'),{origin:'https://evil.example'});
  const status=await new Promise(resolve=>{ws.on('error',()=>{});ws.on('unexpected-response',(_req,res)=>{resolve(res.statusCode);res.resume();ws.terminate();});});
  assert.equal(status,403);assert.equal(attempts,0);
});

test('cancelling before upstream upgrade closes the pending connection',async t=>{
  const backend=createServer();let upstream;
  const seen=new Promise(resolve=>backend.on('upgrade',(_req,socket)=>{upstream=socket;socket.on('error',()=>{});socket.on('end',()=>socket.destroy());resolve();}));
  const backendUrl=await listen(backend,t);
  const url=await listen(createConversationServer({configuration:()=>({url:backendUrl,token:'fixture'})}),t);
  const client=await open(url,t);await seen;
  const ended=once(upstream,'close');upstream.resume();client.close();await ended;
});

test('diagnostics enforce origin/size and forward only the server credential',async t=>{
  let sent,calls=0;
  const url=await listen(createServer((req,res)=>diagnosticsHandler(req,res,{
    configuration:()=>({url:'https://voice.modal.run',token:'server-fixture'}),
    fetcher:async(...args)=>{calls++;sent=args;return new Response('{}');},
  })),t);
  let response=await fetch(url,{method:'POST',headers:{origin:url,cookie:'private',authorization:'Bearer private'},body:'{}'});
  assert.equal(response.status,200);assert.equal(sent[1].headers['x-voice-service-token'],'server-fixture');assert.equal(sent[1].headers.cookie,undefined);
  response=await fetch(url,{method:'POST',headers:{origin:'https://evil.example'},body:'{}'});assert.equal(response.status,403);
  response=await fetch(url,{method:'POST',headers:{origin:url},body:'x'.repeat(4097)});assert.equal(response.status,413);assert.equal(calls,1);
});
