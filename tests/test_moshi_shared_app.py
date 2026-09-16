import asyncio
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from server import moshi_shared_app as module
from server.shared_sessions import SessionPool,FRAME


class FakeEngine:
    capacity=2;ready=False;error=None;stage='Loading';device_name='Test GPU'
    def __init__(self):self.offset=[0,0];self.total=[0.,0.]
    def initialize(self):self.ready=True
    def close(self):pass
    def reset_slots(self,slots):
        for slot in slots:self.offset[slot]=0;self.total[slot]=0.
    def step_batch(self,pcm,active):
        results=[]
        for i,enabled in enumerate(active):
            if not enabled:results.append(None);continue
            self.offset[i]+=1;self.total[i]+=float(pcm[i,0])
            results.append(None if self.offset[i]==1 else {'audio':np.full(FRAME,self.total[i],dtype=np.float32),
                'features':[.1]*64,'brain':{'model_frame':self.offset[i]},'reservoir_logit_rms':.01,
                'text':' hello','compute_ms':1})
        return results


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(module,'pool',SessionPool(FakeEngine()))
    with TestClient(module.app) as c:
        for _ in range(100):
            if c.get('/api/health').json()['ready']:break
            time.sleep(.005)
        yield c


def frame(ws,value):
    ws.send_bytes(np.full(FRAME,value,dtype='<f4').tobytes())
    metadata=ws.receive_json();audio=np.frombuffer(ws.receive_bytes(),dtype='<f4')
    return metadata,audio


def test_two_sessions_share_server_but_not_audio_or_reset(client):
    with client.websocket_connect('/api/conversation') as a:
        ar=a.receive_json()
        with client.websocket_connect('/api/conversation') as b:
            br=b.receive_json()
            assert ar['worker_id']==br['worker_id'] and ar['session_slot']!=br['session_slot']
            with client.websocket_connect('/api/conversation') as c:
                assert c.receive_json()['code']=='capacity_full'
            assert frame(a,.1)[1][0]==pytest.approx(.1)
            assert frame(b,.2)[1][0]==pytest.approx(.2)
            a.send_text('stop');assert a.receive()['type']=='websocket.close'
            assert frame(b,.3)[1][0]==pytest.approx(.5)
            with client.websocket_connect('/api/conversation') as c:
                cr=c.receive_json()
                assert cr['session_slot']==ar['session_slot'] and cr['call_id']!=ar['call_id']
                meta,audio=frame(c,.4)
                assert meta['index']==1 and audio[0]==pytest.approx(.4)
                c.send_text('stop');assert c.receive()['type']=='websocket.close'
            b.send_text('stop');assert b.receive()['type']=='websocket.close'
    assert client.get('/api/health').json()['active_sessions']==0


def test_invalid_pcm_and_per_call_limit_leave_peer_running(client,monkeypatch):
    monkeypatch.setattr(module,'MAX_STEPS',2)
    with client.websocket_connect('/api/conversation') as a:
        a.receive_json()
        with client.websocket_connect('/api/conversation') as b:
            b.receive_json();b.send_bytes(b'bad')
            assert b.receive_json()['code']=='invalid_input'
            assert b.receive()['type']=='websocket.close'
        assert frame(a,.1)[0]['index']==1
        assert frame(a,.1)[0]['index']==2
        assert a.receive_json()['code']=='session_limit'
        assert a.receive()['type']=='websocket.close'
    assert client.get('/api/health').json()['active_sessions']==0
