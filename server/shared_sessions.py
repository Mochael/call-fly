"""Bounded WebSocket session slots around Moshi's upstream batched stream API."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass,field
import time
import uuid

import numpy as np

FRAME=1920
PERIOD=.08


@dataclass
class Session:
    slot:int
    ready:asyncio.Future
    call_id:str=field(default_factory=lambda:uuid.uuid4().hex[:8])
    inputs:asyncio.Queue=field(default_factory=lambda:asyncio.Queue(maxsize=12))
    outputs:asyncio.Queue=field(default_factory=lambda:asyncio.Queue(maxsize=3))
    primed:bool=False
    ended:bool=False
    frames:int=0
    input_frames:int=0
    max_queue:int=0
    started:float=field(default_factory=time.monotonic)


class SessionPool:
    def __init__(self,engine):
        self.engine=engine
        self.slots=[None]*engine.capacity
        self.reset_pending=set()
        self.wake=asyncio.Event()
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='moshi-batch')
        self.task=None
        self.error=None

    def start(self):
        self.task=asyncio.create_task(self.run())

    def reserve(self):
        if self.error:raise RuntimeError(self.error)
        for index,current in enumerate(self.slots):
            if current is None:
                session=Session(index,asyncio.get_running_loop().create_future())
                self.slots[index]=session;self.reset_pending.add(index);self.wake.set()
                return session
        raise RuntimeError('All speech sessions are currently in use. Please try again shortly.')

    def release(self,session):
        session.ended=True
        if not session.ready.done():session.ready.cancel()
        if self.slots[session.slot] is session:
            self.slots[session.slot]=None;self.reset_pending.add(session.slot);self.wake.set()

    def submit(self,session,pcm):
        if session.ended:raise RuntimeError('This call has ended.')
        session.inputs.put_nowait(pcm)
        session.input_frames+=1;session.max_queue=max(session.max_queue,session.inputs.qsize())
        self.wake.set()

    def fail(self,session,message):
        if not session.ready.done():session.ready.set_exception(RuntimeError(message))
        # Bound memory for a slow client; release only that caller's slot.
        while not session.outputs.empty():session.outputs.get_nowait()
        session.outputs.put_nowait({'error':message})
        self.release(session)

    async def run(self):
        loop=asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self.executor,self.engine.initialize)
            deadline=loop.time()
            while True:
                await self.wake.wait()
                if deadline<loop.time()-.16:deadline=loop.time()
                await asyncio.sleep(max(0,deadline-loop.time()))
                self.wake.clear()
                sessions=self.slots.copy()
                resets=list(self.reset_pending);self.reset_pending.clear()
                pcm=np.zeros((len(sessions),FRAME),np.float32)
                active=np.zeros(len(sessions),bool)
                priming=[]
                for index,session in enumerate(sessions):
                    if session is None or session.ended:continue
                    if not session.primed:
                        active[index]=True;priming.append(index)
                    elif not session.inputs.empty():
                        pcm[index]=session.inputs.get_nowait();active[index]=True
                def infer():
                    self.engine.reset_slots(resets)
                    return self.engine.step_batch(pcm,active) if active.any() else [None]*len(sessions)
                results=await loop.run_in_executor(self.executor,infer)
                for index,session in enumerate(sessions):
                    # Disconnect/reuse can occur while the GPU is running. Never
                    # deliver a previous occupant's frame to a reused slot.
                    if session is None or self.slots[index] is not session or session.ended:continue
                    if index in priming:
                        session.primed=True
                        if not session.ready.done():session.ready.set_result(None)
                        continue
                    result=results[index]
                    if result is None:continue
                    session.frames+=1;result['index']=session.frames
                    try:session.outputs.put_nowait(result)
                    except asyncio.QueueFull:self.fail(session,'Audio delivery fell behind. Please restart this call.')
                deadline+=PERIOD
                if self.reset_pending or any(s and (not s.primed or not s.inputs.empty()) for s in self.slots):self.wake.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error=f'Shared speech service failed ({type(exc).__name__}). Please restart the call.'
            self.engine.ready=False;self.engine.error=self.error
            for session in self.slots.copy():
                if session:self.fail(session,self.error)
            # Keep the traceback on the server; never expose raw model errors.
            import logging
            logging.getLogger(__name__).exception('Shared inference loop failed')

    async def close(self):
        for session in self.slots.copy():
            if session:self.fail(session,'The voice server is restarting.')
        if self.task:
            self.task.cancel();await asyncio.gather(self.task,return_exceptions=True)
        await asyncio.get_running_loop().run_in_executor(self.executor,self.engine.close)
        self.executor.shutdown(wait=True)
