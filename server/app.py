"""Local voice chat. One active call shares the inference worker safely."""
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .engine import Engine, ROOT, TTS_MODEL, LLM_MODEL, GREETING
from .voice_quality import SpeechGenerationError

log = logging.getLogger('eric')
engine = Engine()
worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='eric-inference')
call_lock = asyncio.Lock()
MAX_AUDIO_BYTES = 16000 * 2 * 45


@asynccontextmanager
async def lifespan(app):
    def initialize():
        try:
            engine.initialize()
        except Exception as exc:
            engine.error = str(exc)
            engine.stage = 'Eric could not start.'
            log.exception('Initialization failed')
    future = asyncio.get_running_loop().run_in_executor(worker, initialize)
    yield
    await future
    worker.shutdown(wait=False, cancel_futures=True)


app = FastAPI(lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', '[::1]', 'testserver'])


@app.get('/api/health')
async def health():
    return JSONResponse({'ready': engine.ready, 'stage': engine.stage, 'error': engine.error,
                         'busy': call_lock.locked(), 'tts_model': TTS_MODEL, 'llm_model': LLM_MODEL})


@app.websocket('/api/conversation')
async def conversation(ws: WebSocket):
    # Reject cross-site browser requests to this local service.
    origin = ws.headers.get('origin')
    host = ws.headers.get('host')
    if origin and origin not in {f'http://{host}', f'https://{host}'}:
        await ws.close(code=1008)
        return
    await ws.accept()
    if not engine.ready:
        await ws.send_json({'type': 'error', 'message': 'Eric is still warming up. Please try again shortly.'})
        await ws.close(code=1013)
        return
    if call_lock.locked():
        await ws.send_json({'type': 'error', 'message': 'Eric is already in a call. End the other call and try again.'})
        await ws.close(code=1013)
        return
    await call_lock.acquire()
    cancel = threading.Event()
    active = None
    history = []
    loop = asyncio.get_running_loop()

    async def process(pcm):
        queue = asyncio.Queue()
        def emit(event):
            if not cancel.is_set():
                loop.call_soon_threadsafe(queue.put_nowait, event)
        def execute():
            try:
                engine.turn(pcm, history, cancel, emit)
            except SpeechGenerationError:
                log.warning('Voice generation returned invalid audio')
                emit({'type': 'error', 'message': 'Eric could not generate that reply. Please try speaking again.'})
            except Exception:
                log.exception('Conversation turn failed')
                emit({'type': 'error', 'message': 'Something went wrong with that reply. Please try speaking again.'})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)
        future = loop.run_in_executor(worker, execute)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                if isinstance(event, bytes):
                    await ws.send_bytes(event)
                else:
                    await ws.send_json(event)
            await ws.send_json({'type': 'turn_end'})
        finally:
            await future

    try:
        await ws.send_json({'type': 'transcript', 'role': 'assistant', 'text': GREETING})
        await ws.send_json({'type': 'audio_start', 'sample_rate': engine.sample_rate})
        # A cached greeting uses the same cloned voice; user replies are never cached.
        for offset in range(0, len(engine.greeting), 12000):
            await ws.send_bytes(engine.greeting[offset:offset+12000].astype('<f4').tobytes())
        await ws.send_json({'type': 'turn_end'})
        while True:
            message = await ws.receive()
            if message['type'] == 'websocket.disconnect':
                break
            if message.get('text'):
                if message['text'] == 'stop':
                    break
                continue
            pcm = message.get('bytes')
            if pcm is None:
                continue
            if len(pcm) > MAX_AUDIO_BYTES or len(pcm) % 2:
                await ws.send_json({'type': 'error', 'message': 'That recording is too long or invalid. Try a shorter sentence.'})
                await ws.send_json({'type': 'turn_end'})
                continue
            if active and not active.done():
                await ws.send_json({'type': 'busy'})
                continue
            active = asyncio.create_task(process(pcm))
    except WebSocketDisconnect:
        pass
    finally:
        cancel.set()
        if active:
            try:
                await active
            except (WebSocketDisconnect, RuntimeError):
                pass
        call_lock.release()
        try:
            await ws.close()
        except RuntimeError:
            pass


app.mount('/', StaticFiles(directory=ROOT / 'dist', html=True), name='web')
