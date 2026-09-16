"""Bounded real-GPU test: four active calls and three FIFO waiters, including cancellation."""
import asyncio
from contextlib import AsyncExitStack
import json
from pathlib import Path
import time

import numpy as np
import sphn
import websockets


async def main():
    audio,_=sphn.read('artifacts/browser-microphone.wav',sample_rate=24000)
    audio=np.resize(audio[0],1920*750).reshape(-1,1920)
    calls=[];tasks=[]
    async with AsyncExitStack() as stack:
        async def connect():
            return await stack.enter_async_context(websockets.connect(
                'ws://localhost:8765/api/conversation',open_timeout=20,max_size=2**20))
        async def event(ws,kind=None,phase=None,timeout=180):
            async with asyncio.timeout(timeout):
                while True:
                    data=json.loads(await ws.recv())
                    assert data['type']!='error',data
                    if data.get('phase')==phase and phase is not None:return data
                    if data['type']==kind:return data
                    assert data['type']=='warming',data
        def stream(ws,ready):
            call={'ws':ws,'ready':ready,'frames':0,'closing':False,'peak':0.}
            calls.append(call)
            async def send():
                started=time.monotonic()
                for i,frame in enumerate(audio):
                    await asyncio.sleep(max(0,started+i*.08-time.monotonic()))
                    await ws.send(frame.astype('<f4').tobytes())
            async def receive():
                try:
                    async for payload in ws:
                        if isinstance(payload,bytes):
                            samples=np.frombuffer(payload,dtype='<f4')
                            assert len(samples)==1920 and np.isfinite(samples).all()
                            call['frames']+=1;call['peak']=max(call['peak'],float(np.max(np.abs(samples))))
                        else:assert json.loads(payload)['type'] not in ('error','session_end'),payload
                finally:
                    assert call['closing'],'An active peer disconnected unexpectedly'
            call['tasks']=[asyncio.create_task(send()),asyncio.create_task(receive())]
            tasks.extend(call['tasks']);return call
        async def close(call):
            call['closing']=True
            call['tasks'][0].cancel()
            await asyncio.gather(call['tasks'][0],return_exceptions=True)
            await call['ws'].close()
            await call['tasks'][1]
        try:
            for _ in range(4):
                ws=await connect();ready=await event(ws,kind='ready')
                assert ready['serving']=='upstream-masked-streaming'
                stream(ws,ready)
            workers={c['ready']['worker_id'] for c in calls}
            assert len(workers)==1
            assert len({c['ready']['session_slot'] for c in calls})==4
            cancelled=await connect();first=await event(cancelled,phase='queued')
            assert first['people_ahead']==0
            waiting=await connect();queued=await event(waiting,phase='queued')
            assert queued['people_ahead']==1
            last=await connect();third=await event(last,phase='queued')
            assert third['people_ahead']==2
            await cancelled.send('stop')
            async with asyncio.timeout(5):
                async for message in cancelled:assert json.loads(message)['type']=='warming'
            async def position(ws,expected):
                async with asyncio.timeout(10):
                    while True:
                        update=await event(ws,phase='queued',timeout=10)
                        if update['people_ahead']==expected:return update
            await position(waiting,0)
            await position(last,1)
            # Waiting must not become ready while all four existing calls run.
            try:
                async with asyncio.timeout(2):
                    while True:assert json.loads(await waiting.recv()).get('phase')=='queued'
            except TimeoutError:pass
            before=[c['frames'] for c in calls]
            await close(calls[0])
            ready=await event(waiting,kind='ready',timeout=20)
            assert ready['worker_id'] in workers
            assert ready['session_slot']==calls[0]['ready']['session_slot']
            admitted=stream(waiting,ready)
            await position(last,0)
            await last.close()
            async with asyncio.timeout(10):
                while admitted['frames']<30:await asyncio.sleep(.1)
            assert all(calls[i]['frames']>before[i]+20 for i in (1,2,3))
            assert admitted['peak']>.004
            report={'passed':True,'gpu_workers':1,'active_capacity':4,'waiting_message':queued['message'],
                    'cancelled_waiter_closed':True,'initial_positions':[0,1,2],'positions_after_cancel':[0,1],'last_position_after_admission':0,'admitted_slot':ready['session_slot'],
                    'frames':[c['frames'] for c in calls],'peers_continued':True}
            Path('artifacts/moshi/waiting-room.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report,indent=2))
        finally:
            for call in calls:
                if not call['closing']:await close(call)
            for task in tasks:task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)


if __name__=='__main__':asyncio.run(main())
