"""Full-duplex Moshi with a causal connectome readout and same-state telemetry."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging
import json
import time
import uuid
import os

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .service_access import ServiceAccess, origin_allowed
from .runtime import MoshiEngine, ROOT, MODEL, REVISION, RATE, FRAME, MAX_STEPS
from .connectome import router as connectome_router
from .call_diagnostics import router as diagnostics_router, emit as record_event

log = logging.getLogger('uvicorn.error.eric.moshi')
engine = MoshiEngine()
worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='moshi-inference')
# CUDA model/codec share one stream and must not capture/replay graphs from
# competing threads. MLX retains its separate CPU codec worker.
codec_worker = worker if os.environ.get('MOSHI_BACKEND')=='torch' else ThreadPoolExecutor(max_workers=1, thread_name_prefix='mimi-codec')
call_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app):
    def initialize():
        try:
            engine.initialize()
        except Exception as exc:
            engine.error = str(exc)
            engine.stage = 'Moshi could not start. Check the server log.'
            log.exception('Moshi initialization failed')
    future = asyncio.get_running_loop().run_in_executor(worker, initialize)
    yield
    await future
    worker.shutdown(wait=False, cancel_futures=True)
    if codec_worker is not worker:codec_worker.shutdown(wait=False, cancel_futures=True)


app = FastAPI(lifespan=lifespan)
app.include_router(connectome_router)
app.include_router(diagnostics_router)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', '[::1]', 'testserver', '*.modal.run'] if os.environ.get('MOSHI_BACKEND')=='torch' else ['localhost', '127.0.0.1', '[::1]', 'testserver'])
app.add_middleware(ServiceAccess)


@app.get('/api/health')
async def health():
    return JSONResponse({'ready': engine.ready, 'stage': engine.stage, 'error': engine.error,
                         'busy': call_lock.locked(), 'backend': 'moshi', 'model': MODEL,
                         'revision': REVISION, 'full_duplex': True,
                         'execution':os.environ.get('MOSHI_BACKEND','mlx'),
                         'gpu':getattr(engine,'device_name',None),
                         'codec':getattr(engine,'codec_backend','rust-cpu'),
                         'region':os.environ.get('MODAL_REGION'),
                         'sample_rate': RATE, 'frame_samples': FRAME,
                         'max_call_seconds': MAX_STEPS * FRAME // RATE,
                         'connectome_in_model':True, 'reservoir_update_hz':round(12.5/3,3)})


class CallError(Exception):
    def __init__(self, message, code='invalid_input'):
        super().__init__(message)
        self.code = code


def validate_frame(pcm):
    if len(pcm) != FRAME * 4:
        raise CallError('Invalid audio frame. Refresh the page and try again.')
    frame = np.frombuffer(pcm, dtype='<f4')
    if not np.isfinite(frame).all() or np.max(np.abs(frame)) > 1.01:
        raise CallError('Invalid microphone samples. Please restart the call.')
    return frame


@app.websocket('/api/conversation')
async def conversation(ws: WebSocket):
    if not origin_allowed(ws.headers):
        await ws.close(code=1008)
        return
    await ws.accept()
    if os.environ.get('MOSHI_BACKEND')=='torch' and not engine.ready:
        await ws.send_json({'type':'warming','message':'Starting the voice model…'})
        for _ in range(120):
            if engine.ready or engine.error:break
            await asyncio.sleep(1)
    if not engine.ready or call_lock.locked():
        message = 'Moshi is still warming up.' if not engine.ready else 'Eric is already in a call. End the other call first.'
        await ws.send_json({'type': 'error', 'message': message})
        await ws.close(code=1013)
        return
    await call_lock.acquire()
    loop = asyncio.get_running_loop()
    frames = asyncio.Queue(maxsize=12)  # Fail explicitly rather than accumulate old speech.
    encoded = asyncio.Queue(maxsize=2)
    generated = asyncio.Queue(maxsize=2)
    tasks = []
    started = time.perf_counter()
    count = 0
    compute_total = 0.
    call_id = uuid.uuid4().hex[:8]
    stop_reason = 'connection_closed'
    max_queue = 0
    close_code = None
    last_input_at = started
    max_input_gap_ms = 0.
    received_frames = 0
    stage_ms = {"encode": 0., "generate": 0., "decode": 0.}

    async def receive():
        nonlocal stop_reason, max_queue, close_code, last_input_at, max_input_gap_ms, received_frames
        while True:
            message = await ws.receive()
            if message['type'] == 'websocket.disconnect':
                close_code = message.get('code')
                return
            if message.get('text') is not None:
                value = message['text']
                if value == 'stop':
                    stop_reason = 'client_stop'
                    return
                try:
                    control = json.loads(value)
                except ValueError:
                    raise CallError('Unsupported conversation message.')
                allowed = {'user_stop', 'playback_backlog', 'transport_backlog',
                           'page_closed', 'microphone_ended', 'connection_error', 'session_limit', 'server_error', 'audio_unavailable'}
                if isinstance(control, dict) and control.get('type') == 'stop' and isinstance(control.get('reason'), str) and control['reason'] in allowed:
                    if stop_reason == 'connection_closed':
                        stop_reason = 'client_' + control['reason']
                    return
                raise CallError('Unsupported conversation message.')
            pcm = message.get('bytes')
            if pcm is None:
                raise CallError('Unsupported conversation message.')
            frame = validate_frame(pcm)
            now = time.perf_counter()
            if received_frames:
                max_input_gap_ms = max(max_input_gap_ms, (now-last_input_at)*1000)
            last_input_at = now
            received_frames += 1
            try:
                frames.put_nowait(frame)
                max_queue = max(max_queue, frames.qsize())
            except asyncio.QueueFull:
                message = ('The GPU voice service could not keep up with live audio. Please restart the call.'
                           if os.environ.get('MOSHI_BACKEND')=='torch' else
                           'Moshi is falling behind live audio. Close other heavy apps and restart the call.')
                raise CallError(message, 'input_backlog')

    async def encode():
        while True:
            frame = await frames.get()
            tick = time.perf_counter()
            codes = await loop.run_in_executor(codec_worker, engine.encode, frame)
            stage_ms["encode"] = round((time.perf_counter()-tick)*1000, 2)
            await encoded.put(codes)

    async def generate():
        nonlocal count, compute_total
        while count < MAX_STEPS:
            codes = await encoded.get()
            result = await loop.run_in_executor(worker, engine.generate, codes)
            stage_ms["generate"] = round(result["compute_ms"], 2)
            count += 1
            result['index'] = count
            compute_total += result['compute_ms']
            await generated.put(result)
        await generated.put(None)
        await asyncio.Event().wait()  # Playback drains the final frames before closing.

    async def playback():
        nonlocal stop_reason
        while True:
            result = await generated.get()
            if result is None:
                stop_reason = 'session_limit'
                await ws.send_json({'type': 'session_end', 'code': 'session_limit',
                                    'message': 'The five-minute call limit was reached. Start a new call to keep chatting.'})
                return
            codes = result['audio_codes']
            tick = time.perf_counter()
            audio = await loop.run_in_executor(codec_worker, engine.decode, codes) if codes is not None else None
            stage_ms['decode'] = round((time.perf_counter()-tick)*1000, 2)
            index = result['index']
            await ws.send_json({'type': 'frame', 'index': index, 'features': result['features'],
                                'brain':result.get('brain'), 'reservoir_logit_rms':result.get('reservoir_logit_rms',0),
                                'text': result['text'], 'compute_ms': round(result['compute_ms'], 2),
                                'queued_frames': frames.qsize(),
                                'audio_samples': len(audio) if audio is not None else 0})
            if audio is not None:
                await ws.send_bytes(audio.tobytes())
            if index % 25 == 0:
                record_event('server_sample', call_id=call_id, frames=index, input_frames=received_frames,
                             input_queue=frames.qsize(), encoded_queue=encoded.qsize(), output_queue=generated.qsize(),
                             mean_compute_ms=round(compute_total/count,2), stage_ms=stage_ms.copy(),
                             input_gap_ms=round((time.perf_counter()-last_input_at)*1000,2))
                await ws.send_json({'type': 'metrics', 'frames': index,
                                    'mean_compute_ms': round(compute_total/count, 2),
                                    'input_audio_s': round(index*FRAME/RATE, 2),
                                    'elapsed_s': round(time.perf_counter()-started, 2),
                                    'queued_frames': frames.qsize()})

    try:
        log.info('Call %s starting', call_id)
        record_event('server_start', call_id=call_id, worker_id=os.environ.get('MODAL_TASK_ID','local'),
                     max_call_seconds=MAX_STEPS*FRAME/RATE)
        await loop.run_in_executor(worker, engine.begin)
        # Allocate fresh codec/model caches before accepting live microphone frames.
        # One silent priming frame produces no audio (Moshi's one-frame delay).
        if hasattr(engine, 'step'):
            await loop.run_in_executor(worker, engine.step, np.zeros(FRAME, dtype=np.float32))
        await ws.send_json({'type': 'ready', 'call_id': call_id, 'sample_rate': RATE, 'frame_samples': FRAME,
                            'max_call_seconds': MAX_STEPS*FRAME//RATE, 'model': MODEL,
                            'execution':os.environ.get('MOSHI_BACKEND','mlx'),
                            'gpu':getattr(engine,'device_name',None),
                            'codec':getattr(engine,'codec_backend','rust-cpu'),
                            'region':os.environ.get('MODAL_REGION'),
                            'worker_id':os.environ.get('MODAL_TASK_ID','local')})
        tasks = [asyncio.create_task(receive()), asyncio.create_task(encode()),
                 asyncio.create_task(generate()), asyncio.create_task(playback())]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except WebSocketDisconnect as exc:
        close_code = exc.code
    except CallError as exc:
        stop_reason = exc.code
        try:
            await ws.send_json({'type': 'error', 'code': exc.code, 'message': str(exc)})
        except (RuntimeError, WebSocketDisconnect):
            pass
    except Exception:
        stop_reason = 'inference_error'
        log.exception('Moshi call failed')
        try:
            await ws.send_json({'type': 'error', 'code': 'inference_error', 'message': 'The live speech connection failed. Please start a new call.'})
        except (RuntimeError, WebSocketDisconnect):
            pass
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Queued after any in-flight inference: caches cannot leak into another call.
        try:
            await loop.run_in_executor(codec_worker, lambda: None)
            await loop.run_in_executor(worker, engine.end)
        finally:
            call_lock.release()
            record_event('server_end', call_id=call_id, reason=stop_reason, close_code=close_code,
                         frames=count, input_frames=received_frames, elapsed_s=round(time.perf_counter()-started,2),
                         max_input_queue=max_queue, max_input_gap_ms=round(max_input_gap_ms,2),
                         stage_ms=stage_ms.copy(), mean_compute_ms=round(compute_total/max(count,1),2))
            log.info('Call %s ended: reason=%s frames=%d audio_s=%.2f elapsed_s=%.2f max_input_queue=%d',
                     call_id, stop_reason, count, count*FRAME/RATE, time.perf_counter()-started, max_queue)
        try:
            await ws.close()
        except (RuntimeError, WebSocketDisconnect):
            pass


app.mount('/', StaticFiles(directory=ROOT / 'dist', html=True), name='web')
