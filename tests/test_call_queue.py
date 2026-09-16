import asyncio
from types import SimpleNamespace
import pytest
from server.call_queue import CallQueue, QueueWaitExpired


def test_fifo_positions_and_idempotent_release():
    queue=CallQueue(2)
    a,b,c,d,e=[queue.join() for _ in range(5)]
    assert queue.active==2
    assert [queue.people_ahead(t) for t in (c,d,e)]==[0,1,2]
    queue.leave(d)
    assert queue.people_ahead(e)==1
    queue.leave(a)
    assert c.state=='admitted' and queue.people_ahead(e)==0
    # A new arrival cannot jump the line.
    f=queue.join()
    assert queue.people_ahead(f)==1
    queue.leave(b)
    assert e.state=='admitted' and f.state=='waiting'
    queue.leave(b)
    assert queue.active==2
    for ticket in (c,e,f):queue.leave(ticket)
    assert queue.active==0 and not queue.waiting


def test_live_positions_after_cancellation_and_admission():
    async def run():
        queue=CallQueue(1);active=queue.join()
        messages=[asyncio.Queue() for _ in range(3)]
        incoming=[asyncio.get_running_loop().create_future() for _ in range(3)]
        release=[asyncio.Event() for _ in range(3)]
        admitted=[asyncio.Event() for _ in range(3)]
        async def call(i):
            ws=SimpleNamespace(send_json=messages[i].put)
            async with queue.admission(ws,incoming[i],progress=60) as accepted:
                if accepted:
                    admitted[i].set();await release[i].wait()
        tasks=[]
        for i in range(3):
            tasks.append(asyncio.create_task(call(i)))
            event=await asyncio.wait_for(messages[i].get(),1)
            assert event['people_ahead']==i
            assert 'capacity' not in event
        incoming[0].set_result({'type':'websocket.disconnect'})
        await tasks[0]
        assert (await asyncio.wait_for(messages[1].get(),1))['people_ahead']==0
        assert (await asyncio.wait_for(messages[2].get(),1))['people_ahead']==1
        queue.leave(active)
        await asyncio.wait_for(admitted[1].wait(),1)
        event=await asyncio.wait_for(messages[2].get(),1)
        assert event['people_ahead']==0 and "You're next." in event['message']
        release[1].set();await tasks[1]
        await asyncio.wait_for(admitted[2].wait(),1)
        release[2].set();await tasks[2]
        assert queue.active==0 and not queue.waiting
    asyncio.run(run())


def test_timeout_and_send_failure_do_not_leak_or_release_active_slot():
    async def run():
        queue=CallQueue(1);active=queue.join()
        incoming=asyncio.get_running_loop().create_future()
        async def send(_):pass
        with pytest.raises(QueueWaitExpired):
            async with queue.admission(SimpleNamespace(send_json=send),incoming,timeout=.01):
                pytest.fail('Should remain queued')
        assert queue.active==1 and not queue.waiting
        async def fail(_):raise ConnectionError('Disconnected')
        with pytest.raises(ConnectionError):
            async with queue.admission(SimpleNamespace(send_json=fail),incoming):pass
        assert queue.active==1 and not queue.waiting
        queue.leave(active)
    asyncio.run(run())


def test_task_cancellation_racing_slot_handoff_returns_reserved_slot():
    async def run():
        queue=CallQueue(1);active=queue.join()
        sent=asyncio.Event();incoming=asyncio.get_running_loop().create_future()
        async def send(_):sent.set()
        async def wait():
            async with queue.admission(SimpleNamespace(send_json=send),incoming):
                await asyncio.Event().wait()
        task=asyncio.create_task(wait());await sent.wait()
        queue.leave(active);task.cancel()
        await asyncio.gather(task,return_exceptions=True)
        assert queue.active==0 and not queue.waiting
        assert queue.join().state=='admitted'
    asyncio.run(run())
