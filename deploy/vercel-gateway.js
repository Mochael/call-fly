// Vercel owns browser access; only this server knows the Modal service token.
import {createServer} from 'node:http';
import WebSocket, {WebSocketServer} from 'ws';

export function backendConfiguration(env=process.env) {
  const url=new URL(env.VOICE_BACKEND_URL || 'https://unconfigured.invalid');
  if(url.protocol!=='https:' || !url.hostname.endsWith('.modal.run') ||
     url.username || url.password || url.pathname!=='/' || url.search || url.hash ||
     !env.VOICE_SERVICE_TOKEN) throw new Error('Voice service is not configured.');
  return {url:url.origin,token:env.VOICE_SERVICE_TOKEN};
}

function sameOrigin(request) {
  try {
    const origin=new URL(request.headers.origin);
    return ['https:','http:'].includes(origin.protocol) && origin.host===request.headers.host;
  } catch {return false;}
}

function json(response,status,body) {
  response.writeHead(status,{'Content-Type':'application/json','Cache-Control':'no-store'});
  response.end(JSON.stringify(body));
}

export function healthHandler(request,response) {
  if(request.method!=='GET')return json(response,405,{message:'Method not allowed.'});
  let configured=true;
  try {backendConfiguration();}catch {configured=false;}
  return json(response,200,{ready:configured,backend:'moshi',execution:'modal',busy:false,
    full_duplex:true,connectome_in_model:true,sample_rate:24000,frame_samples:1920,max_call_seconds:300,
    stage:configured?'Ready for a live conversation.':'Cloud voice service is not configured.',error:null});
}

export async function diagnosticsHandler(request,response,{configuration=backendConfiguration,fetcher=fetch}={}) {
  if(request.method!=='POST')return json(response,405,{message:'Method not allowed.'});
  if(!sameOrigin(request))return json(response,403,{message:'Origin not allowed.'});
  if(Number(request.headers['content-length'])>4096){request.resume();return json(response,413,{message:'Diagnostic too large.'});}
  try {
    const {url,token}=configuration();
    let bytes=0;const chunks=[];
    for await(const chunk of request.iterator({destroyOnReturn:false})) {
      bytes+=chunk.length;
      if(bytes>4096){request.resume();return json(response,413,{message:'Diagnostic too large.'});}
      chunks.push(chunk);
    }
    const upstream=await fetcher(url+'/api/call-diagnostics',{method:'POST',
      headers:{'content-type':'application/json','x-voice-service-token':token},
      body:Buffer.concat(chunks),redirect:'error',signal:AbortSignal.timeout(10_000)});
    // Modal validates the diagnostic schema; never echo upstream error bodies.
    return json(response,upstream.status,upstream.ok?{saved:true}:{message:'Diagnostic was not accepted.'});
  }catch {if(!response.writableEnded)json(response,503,{message:'Diagnostic service unavailable.'});}
}

export function createConversationServer({configuration=backendConfiguration,maxSessionMs=550_000}={}) {
  const server=createServer((_request,response)=>json(response,426,{message:'A WebSocket connection is required.'}));
  const sockets=new WebSocketServer({noServer:true,maxPayload:16_384,perMessageDeflate:false});
  server.on('upgrade',(request,socket,head)=>{
    if(!sameOrigin(request)) {socket.end('HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n');return;}
    let config;
    try {config=configuration();}catch {socket.end('HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n\r\n');return;}
    sockets.handleUpgrade(request,socket,head,client=>{
      const target=new URL('/api/conversation',config.url);target.protocol=target.protocol==='https:'?'wss:':'ws:';
      const upstream=new WebSocket(target,{headers:{'x-voice-service-token':config.token},
        handshakeTimeout:190_000,maxPayload:1_048_576,perMessageDeflate:false,followRedirects:false});
      let finished=false;
      const deadline=setTimeout(()=>fail('The call connection reached its time limit.'),maxSessionMs);
      deadline.unref();
      function cleanup() {
        if(finished)return;
        finished=true;clearTimeout(deadline);
        if(upstream.readyState!==WebSocket.CLOSED)upstream.terminate();
      }
      function fail(message) {
        if(finished)return;
        if(client.readyState===WebSocket.OPEN){client.send(JSON.stringify({type:'error',message}));client.close(1011,'Voice gateway stopped.');}
        cleanup();
      }
      function relay(target,data,isBinary,limit) {
        if(finished || target.readyState!==WebSocket.OPEN)return;
        if(target.bufferedAmount+data.length>limit){fail('The connection cannot keep up with live audio. Please reconnect.');return;}
        target.send(data,{binary:isBinary},error=>{if(error)fail('The voice connection was interrupted.');});
      }
      client.on('message',(data,isBinary)=>{
        if(upstream.readyState!==WebSocket.OPEN){client.close(1000,'Call cancelled.');cleanup();return;}
        relay(upstream,data,isBinary,131_072);
      });
      upstream.on('message',(data,isBinary)=>relay(client,data,isBinary,2_097_152));
      client.on('close',cleanup);
      client.on('error',cleanup);
      upstream.on('error',()=>fail('Could not connect to the GPU voice service. Please try again.'));
      upstream.on('close',code=>{
        if(finished)return;
        if(code!==1000 && code!==1001){fail('The GPU voice connection ended unexpectedly. Please reconnect.');return;}
        client.close(1000,'Voice session ended.');cleanup();
      });
    });
  });
  return server;
}
