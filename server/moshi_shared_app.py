"""Multiple independent WebSocket sessions on one upstream Moshi model batch."""
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import time

import numpy as np
from fastapi import FastAPI,WebSocket,WebSocketDisconnect
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .moshi_batch_engine import SharedMoshiEngine
from .moshi_torch_engine import MODEL,FRAME
from .moshi_engine import MAX_STEPS,RATE
from .shared_sessions import SessionPool
from .service_access import ServiceAccess,origin_allowed
from .call_diagnostics import emit

log=logging.getLogger(__name__)
CAPACITY=int(os.environ.get('MOSHI_SESSION_CAPACITY','2'))
pool=SessionPool(SharedMoshiEngine(CAPACITY))


@asynccontextmanager
async def lifespan(app):
    pool.start()
    yield
    await pool.close()


app=FastAPI(lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','testserver','*.modal.run'])
app.add_middleware(ServiceAccess)


@app.get('/api/health')
async def health():
    return {'ready':pool.engine.ready,'error':pool.error,'stage':pool.engine.stage,
            'capacity':len(pool.slots),'active_sessions':sum(s is not None for s in pool.slots),
            'backend':'moshi','execution':'torch','serving':'upstream-masked-streaming'}


class CallError(Exception):
    def __init__(self,message,code='invalid_input'):
        super().__init__(message);self.code=code


def pcm_frame(data):
    if len(data)!=FRAME*4:raise CallError('Invalid microphone frame size.')
    samples=np.frombuffer(data,dtype='<f4')
    if not np.isfinite(samples).all() or np.max(np.abs(samples))>1.01:raise CallError('Invalid microphone samples.')
    return samples


def stop_control(data):
    if data=='stop':return 'client_stop'
    try:value=json.loads(data)
    except ValueError:raise CallError('Unsupported conversation message.')
    reasons={'user_stop','playback_backlog','transport_backlog','page_closed','microphone_ended',
             'connection_error','session_limit','server_error','audio_unavailable'}
    if isinstance(value,dict) and value.get('type')=='stop' and isinstance(value.get('reason'),str) and value['reason'] in reasons:
        return 'client_'+value['reason']
    raise CallError('Unsupported conversation message.')


@app.websocket('/api/conversation')
async def conversation(ws:WebSocket):
    if not origin_allowed(ws.headers):await ws.close(code=1008);return
    try:await ws.accept()
    except WebSocketDisconnect:return
    except RuntimeError as exc:
        if 'websocket.disconnect' in str(exc):return # cancelled during cold allocation
        raise
    session=None;tasks=[];reason='connection_closed';close_code=None;delivered=0;total_ms=0.
    incoming=asyncio.create_task(ws.receive());tasks.append(incoming)
    try:
        if not pool.engine.ready:
            await ws.send_json({'type':'warming','phase':'loading','message':'Loading the shared voice model…'})
            for _ in range(120):
                if pool.engine.ready or pool.error:break
                done,_=await asyncio.wait([incoming],timeout=1)
                if done:return # Browser cancelled before model readiness.
        if not pool.engine.ready:raise CallError('The shared voice model could not start. Please try again.','model_unavailable')
        try:session=pool.reserve()
        except RuntimeError as exc:raise CallError(str(exc),'capacity_full')
        emit('server_start',call_id=session.call_id,worker_id=os.environ.get('MODAL_TASK_ID','local'),
             slot=session.slot,session_capacity=CAPACITY,max_call_seconds=MAX_STEPS*FRAME/RATE)
        done,_=await asyncio.wait([session.ready,incoming],timeout=15,return_when=asyncio.FIRST_COMPLETED)
        if incoming in done:return
        if session.ready not in done:raise CallError('The speech session could not start in time.','session_start_timeout')
        session.ready.result()
        await ws.send_json({'type':'ready','call_id':session.call_id,'sample_rate':RATE,'frame_samples':FRAME,
            'max_call_seconds':MAX_STEPS*FRAME//RATE,'model':MODEL,'execution':'torch',
            'gpu':pool.engine.device_name,'codec':'torch-cuda','region':os.environ.get('MODAL_REGION'),
            'worker_id':os.environ.get('MODAL_TASK_ID','local'),'session_slot':session.slot,
            'session_capacity':CAPACITY,'serving':'upstream-masked-streaming'})

        async def receive():
            nonlocal reason,close_code
            message=await incoming
            while True:
                if message['type']=='websocket.disconnect':close_code=message.get('code');return
                if message.get('text') is not None:
                    reason=stop_control(message['text']);return
                data=message.get('bytes')
                if data is None:raise CallError('Unsupported conversation message.')
                try:pool.submit(session,pcm_frame(data))
                except asyncio.QueueFull:raise CallError('The speech session fell behind live audio. Please restart this call.','input_backlog')
                except RuntimeError as exc:raise CallError(str(exc),'session_ended')
                try:message=await asyncio.wait_for(ws.receive(),timeout=15)
                except TimeoutError:raise CallError('No microphone audio arrived for 15 seconds. Please restart the call.','input_timeout')

        async def send():
            nonlocal reason,delivered,total_ms
            while True:
                result=await session.outputs.get()
                if 'error' in result:raise CallError(result['error'],'session_failed')
                delivered+=1;total_ms+=result['compute_ms'];index=result['index']
                await ws.send_json({'type':'frame','index':index,'features':result['features'],
                    'brain':result['brain'],'reservoir_logit_rms':result['reservoir_logit_rms'],
                    'text':result['text'],'compute_ms':round(result['compute_ms'],2),
                    'queued_frames':session.inputs.qsize(),'audio_samples':len(result['audio'])})
                await ws.send_bytes(result['audio'].astype('<f4',copy=False).tobytes())
                if index%25==0:
                    emit('server_sample',call_id=session.call_id,slot=session.slot,frames=index,
                         input_frames=session.input_frames,input_queue=session.inputs.qsize(),
                         mean_batch_ms=round(total_ms/delivered,2))
                    await ws.send_json({'type':'metrics','frames':index,'mean_compute_ms':round(total_ms/delivered,2),
                        'input_audio_s':round(index*FRAME/RATE,2),'elapsed_s':round(time.monotonic()-session.started,2),
                        'queued_frames':session.inputs.qsize()})
                if index>=MAX_STEPS:
                    reason='session_limit'
                    await ws.send_json({'type':'session_end','code':reason,
                        'message':'The five-minute call limit was reached. Start a new call to keep chatting.'})
                    return

        transfer=[asyncio.create_task(receive()),asyncio.create_task(send())];tasks.extend(transfer)
        done,_=await asyncio.wait(transfer,return_when=asyncio.FIRST_COMPLETED)
        for task in done:task.result()
    except WebSocketDisconnect as exc:close_code=exc.code
    except CallError as exc:
        reason=exc.code
        try:await ws.send_json({'type':'error','code':exc.code,'message':str(exc)})
        except (RuntimeError,WebSocketDisconnect):pass
    except Exception:
        reason='inference_error';log.exception('Shared Moshi session failed')
        try:await ws.send_json({'type':'error','code':reason,'message':'The voice session failed. Please start a new call.'})
        except (RuntimeError,WebSocketDisconnect):pass
    finally:
        for task in tasks:task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        if session:
            pool.release(session)
            emit('server_end',call_id=session.call_id,slot=session.slot,reason=reason,close_code=close_code,
                 frames=delivered,input_frames=session.input_frames,max_input_queue=session.max_queue,
                 elapsed_s=round(time.monotonic()-session.started,2))
        try:await ws.close()
        except (RuntimeError,WebSocketDisconnect):pass
