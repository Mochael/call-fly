"""Multi-call web gateway and streaming proxy. No model is loaded here.

GPU credentials stay on this server. Health polling and geometry never wake
Modal; only an explicit call connects to remote inference.
"""
import asyncio
from contextlib import asynccontextmanager, AsyncExitStack
import json
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .service_access import ServiceAccess, origin_allowed
from .call_diagnostics import receive as save_diagnostic
from .call_queue import CallQueue, QueueWaitExpired

ROOT=Path(__file__).resolve().parents[1]
CONFIG=Path(os.environ.get('MODAL_SERVICE_CONFIG',ROOT/'.runtime/modal-service.json'))
log=logging.getLogger('uvicorn.error.eric.modal')
CONNECT_TIMEOUT=180
PROGRESS_INTERVAL=5
RETRY_INTERVAL=2
QUEUE_TIMEOUT=300


@asynccontextmanager
async def call_slot(ws,incoming):
    queue=ws.app.state.call_slots
    if queue is None:
        yield True
        return
    async with queue.admission(ws,incoming,timeout=QUEUE_TIMEOUT,progress=PROGRESS_INTERVAL) as admitted:
        yield admitted


def configuration():
    config=({'url':os.environ.get('VOICE_WORKER_URL',''),'token':os.environ.get('VOICE_SERVICE_TOKEN','')}
            if os.environ.get('VOICE_REQUIRE_AUTH')=='1' else json.loads(CONFIG.read_text()))
    url=config.get('url','').rstrip('/')
    parsed=urlparse(url)
    if parsed.scheme!='https' or not (parsed.hostname or '').endswith('.modal.run') or parsed.path or parsed.query or parsed.fragment:
        raise ValueError('Configure the deployed Modal HTTPS origin in .runtime/modal-service.json.')
    if not config.get('token'):raise ValueError('Missing private Modal service credential.')
    return url,config['token']


def allowed_origin(headers):
    return origin_allowed(headers)


@asynccontextmanager
async def lifespan(app):
    capacity=int(os.environ.get('VOICE_CALL_CAPACITY','0'))
    app.state.call_slots=CallQueue(capacity) if capacity else None
    async with httpx.AsyncClient(timeout=10) as client:
        app.state.http=client
        yield

app=FastAPI(lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','[::1]','testserver']
                   + (['*.modal.run'] if os.environ.get('VOICE_REQUIRE_AUTH')=='1' else []))
app.add_middleware(ServiceAccess)

@app.get('/api/health')
async def health():
    try:
        configuration();configured=True;error=None
    except (OSError,ValueError):
        configured=False;error='The Modal voice service has not been configured.'
    return {'ready':configured,'stage':'Ready for a live conversation.' if configured else error,
            'error':error,'busy':False,'backend':'moshi','execution':'modal',
            'model':'kyutai/moshiko-pytorch-bf16','connectome_in_model':True,'full_duplex':True,
            'sample_rate':24000,'frame_samples':1920,'max_call_seconds':300}

@app.get('/api/connectome/{name}')
async def geometry(name:str):
    if name not in {'manifest.json','metadata.json','positions.bin'}:raise HTTPException(404)
    return FileResponse(ROOT/'.runtime/connectome'/name)

@app.post('/api/call-diagnostics')
async def diagnostics(request:Request):
    if not allowed_origin(request.headers):raise HTTPException(403)
    body=await request.body()
    if len(body)>4096:raise HTTPException(413)
    # Save cloud diagnostics on the CPU gateway; never allocate a GPU for logs.
    if os.environ.get('VOICE_REQUIRE_AUTH')=='1':
        return await save_diagnostic(request)
    try:
        url,token=configuration()
        response=await request.app.state.http.post(url+'/api/call-diagnostics',content=body,
            headers={'content-type':'application/json','x-voice-service-token':token})
        return Response(response.content,status_code=response.status_code,media_type='application/json')
    except (OSError,ValueError,httpx.HTTPError):raise HTTPException(503,'Remote diagnostic service unavailable')

@app.websocket('/api/conversation')
async def conversation(ws:WebSocket):
    if not allowed_origin(ws.headers):await ws.close(code=1008);return
    await ws.accept()
    tasks=[]
    try:
        url,token=configuration()
        started=time.monotonic()
        async with AsyncExitStack() as stack:
            incoming=asyncio.create_task(ws.receive())
            tasks=[incoming]
            admitted=await stack.enter_async_context(call_slot(ws,incoming))
            if not admitted:return
            started=time.monotonic()
            await ws.send_json({'type':'warming','phase':'allocating','message':'Connecting your call…'})
            async def connect_remote():
                deadline=time.monotonic()+CONNECT_TIMEOUT
                while True:
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise TimeoutError()
                    try:
                        return await stack.enter_async_context(websockets.connect(
                            url.replace('https://','wss://',1)+'/api/conversation',
                            additional_headers={'x-voice-service-token':token},open_timeout=remaining,
                            max_size=2**20,max_queue=4,ping_interval=20,ping_timeout=30))
                    except websockets.InvalidStatus as exc:
                        if exc.response.status_code not in (429,502,503,504):raise
                        log.info('Call worker temporarily unavailable: HTTP %d',exc.response.status_code)
                        await asyncio.sleep(min(RETRY_INTERVAL,max(0,deadline-time.monotonic())))
            connecting=asyncio.create_task(connect_remote())
            tasks=[incoming,connecting]
            while True:
                done,_=await asyncio.wait(tasks,timeout=PROGRESS_INTERVAL,return_when=asyncio.FIRST_COMPLETED)
                if done:break
                await ws.send_json({'type':'warming','phase':'allocating',
                    'elapsed_s':round(time.monotonic()-started),
                    'message':'Waiting for an available call worker. It may need to start up…'})
            if incoming in done:
                # Before ready the browser can only cancel; release the local
                # call immediately even when Modal is still starting a worker.
                connecting.cancel()
                await asyncio.gather(connecting,return_exceptions=True)
                return
            remote=connecting.result()
            async def upload():
                message=await incoming
                while True:
                    if message['type']=='websocket.disconnect':return
                    payload=message.get('bytes') if message.get('bytes') is not None else message.get('text')
                    if payload is None:return
                    if len(payload)>16384:raise ValueError('Invalid microphone frame size.')
                    await remote.send(payload)
                    message=await ws.receive()
            async def download():
                async for payload in remote:
                    if isinstance(payload,bytes):await ws.send_bytes(payload)
                    else:await ws.send_text(payload)
            relays=[asyncio.create_task(upload()),asyncio.create_task(download())]
            tasks.extend(relays)
            done,_=await asyncio.wait(relays,return_when=asyncio.FIRST_COMPLETED)
            for task in done:task.result()
    except QueueWaitExpired:
        try:await ws.send_json({'type':'error','code':'queue_timeout',
            'message':'Eric is still busy after five minutes. Please try again shortly.'})
        except (RuntimeError,WebSocketDisconnect):pass
    except TimeoutError:
        log.warning('Call worker allocation timed out')
        try:await ws.send_json({'type':'error','code':'capacity_timeout',
            'message':'No call worker became available in time. Please try again shortly.'})
        except (RuntimeError,WebSocketDisconnect):pass
    except (WebSocketDisconnect,websockets.ConnectionClosed):
        pass
    except Exception as exc:
        # Do not log credential-bearing requests or conversation frames.
        log.warning('Modal connection failed: %s',type(exc).__name__)
        try:await ws.send_json({'type':'error','message':'Could not connect to the GPU voice service. Please try again.'})
        except (RuntimeError,WebSocketDisconnect):pass
    finally:
        for task in tasks:task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        try:await ws.close()
        except (RuntimeError,WebSocketDisconnect):pass

app.mount('/',StaticFiles(directory=ROOT/'dist',html=True),name='web')
