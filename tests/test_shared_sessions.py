import asyncio
import threading
import numpy as np
import pytest
from server.shared_sessions import SessionPool,FRAME


class FakeEngine:
    capacity=2;ready=False;error=None
    def __init__(self):
        self.offsets=[0,0];self.history=[0.,0.];self.block=None;self.entered=threading.Event()
    def initialize(self):self.ready=True
    def close(self):pass
    def reset_slots(self,slots):
        for slot in slots:self.offsets[slot]=0;self.history[slot]=0.
    def step_batch(self,pcm,active):
        self.entered.set()
        if self.block:self.block.wait(3)
        results=[]
        for i,enabled in enumerate(active):
            if not enabled:results.append(None);continue
            self.offsets[i]+=1;self.history[i]+=float(pcm[i,0])
            results.append(None if self.offsets[i]==1 else {'audio':np.full(FRAME,self.history[i]),'step':self.offsets[i]})
        return results


async def ready(pool):
    while not pool.engine.ready:await asyncio.sleep(.001)


def test_pause_join_and_reset_are_isolated():
    async def scenario():
        engine=FakeEngine();pool=SessionPool(engine);pool.start();await ready(pool)
        try:
            a=pool.reserve();await a.ready
            pool.submit(a,np.ones(FRAME));first=await asyncio.wait_for(a.outputs.get(),1)
            b=pool.reserve();await b.ready
            assert engine.offsets[a.slot]==2
            pool.submit(a,np.full(FRAME,2));pool.submit(b,np.full(FRAME,7))
            left,right=await asyncio.gather(a.outputs.get(),b.outputs.get())
            assert left['audio'][0]==3 and right['audio'][0]==7
            pool.release(a)
            c=pool.reserve();await c.ready
            assert c.slot==a.slot and c.call_id!=a.call_id
            pool.submit(c,np.full(FRAME,4));pool.submit(b,np.ones(FRAME))
            new,continuing=await asyncio.gather(c.outputs.get(),b.outputs.get())
            assert new['index']==1 and new['audio'][0]==4
            assert continuing['index']==2 and continuing['audio'][0]==8
        finally:await pool.close()
    asyncio.run(scenario())


def test_disconnect_during_inference_cannot_deliver_to_reused_slot():
    async def scenario():
        engine=FakeEngine();pool=SessionPool(engine);pool.start();await ready(pool)
        try:
            a=pool.reserve();await a.ready
            engine.entered.clear();engine.block=threading.Event()
            pool.submit(a,np.full(FRAME,9))
            while not engine.entered.is_set():await asyncio.sleep(.001)
            pool.release(a);b=pool.reserve();engine.block.set()
            await b.ready
            assert b.outputs.empty()
            pool.submit(b,np.full(FRAME,2));result=await b.outputs.get()
            assert result['audio'][0]==2 and result['index']==1
        finally:await pool.close()
    asyncio.run(scenario())


def test_capacity_and_backpressure_affect_only_one_caller():
    async def scenario():
        engine=FakeEngine();pool=SessionPool(engine);pool.start();await ready(pool)
        try:
            a,b=pool.reserve(),pool.reserve();await asyncio.gather(a.ready,b.ready)
            with pytest.raises(RuntimeError,match='currently in use'):pool.reserve()
            for i in range(4):
                pool.submit(a,np.ones(FRAME));pool.submit(b,np.ones(FRAME))
                assert (await b.outputs.get())['index']==i+1
            assert a.ended and not b.ended
            assert 'error' in await a.outputs.get()
            pool.submit(b,np.ones(FRAME));assert (await b.outputs.get())['index']==5
        finally:await pool.close()
    asyncio.run(scenario())
