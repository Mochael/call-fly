import asyncio
import json
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from server import modal_proxy as module

@pytest.fixture
def client(monkeypatch,tmp_path):
    config=tmp_path/'service.json';config.write_text(json.dumps({'url':'https://test.modal.run','token':'test-secret'}))
    monkeypatch.setattr(module,'CONFIG',config)
    with TestClient(module.app) as client:yield client

def test_health_does_not_wake_gpu_and_origin_is_checked(client,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('Health must not contact Modal')
    monkeypatch.setattr(module.websockets,'connect',forbidden)
    data=client.get('/api/health').json()
    assert data['ready'] and data['execution']=='modal'
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect('/api/conversation',headers={'origin':'https://evil.example'}):pass


def test_duplex_proxy_preserves_frames_and_server_only_auth(client,monkeypatch):
    sent=[]
    class Remote:
        async def __aenter__(self):self.queue=asyncio.Queue();return self
        async def __aexit__(self,*args):pass
        async def send(self,data):sent.append(data);await self.queue.put(data)
        def __aiter__(self):return self.messages()
        async def messages(self):
            yield json.dumps({'type':'ready','call_id':'abc12345'})
            while True:
                value=await self.queue.get()
                if value=='stop':return
                yield json.dumps({'type':'frame','audio_samples':1920,'index':1})
                yield value
    def connect(url,**kwargs):
        assert url=='wss://test.modal.run/api/conversation'
        assert kwargs['additional_headers']=={'x-voice-service-token':'test-secret'}
        return Remote()
    monkeypatch.setattr(module.websockets,'connect',connect)
    for _ in range(2):
        with client.websocket_connect('/api/conversation') as ws:
            assert ws.receive_json()['type']=='warming'
            assert ws.receive_json()['type']=='ready'
            audio=b'\0'*7680;ws.send_bytes(audio)
            assert ws.receive_json()['audio_samples']==1920
            assert ws.receive_bytes()==audio
            ws.send_text('stop');assert ws.receive()['type']=='websocket.close'
    assert len(sent)==4
    assert not client.get('/api/health').json()['busy']

def test_cancel_during_modal_cold_start_releases_call(client,monkeypatch):
    cancelled=[]
    class Starting:
        async def __aenter__(self):
            try:await asyncio.Event().wait()
            finally:cancelled.append(True)
        async def __aexit__(self,*args):pass
    monkeypatch.setattr(module.websockets,'connect',lambda *a,**kw:Starting())
    with client.websocket_connect('/api/conversation') as ws:
        assert ws.receive_json()['type']=='warming'
        ws.send_text('stop')
        assert ws.receive()['type']=='websocket.close'
    assert cancelled
    assert not client.get('/api/health').json()['busy']


def test_simultaneous_calls_are_isolated_and_stopping_one_keeps_other_alive(client,monkeypatch):
    remotes=[]
    class Remote:
        async def __aenter__(self):
            self.queue=asyncio.Queue();self.identifier=len(remotes);remotes.append(self);return self
        async def __aexit__(self,*args):pass
        async def send(self,data):await self.queue.put(data)
        def __aiter__(self):return self.messages()
        async def messages(self):
            yield json.dumps({'type':'ready','call_id':f'{self.identifier:08x}'})
            while True:
                value=await self.queue.get()
                if value=='stop':return
                yield value
    monkeypatch.setattr(module.websockets,'connect',lambda *a,**kw:Remote())
    with client.websocket_connect('/api/conversation') as a:
        a.receive_json();first=a.receive_json()
        with client.websocket_connect('/api/conversation') as b:
            b.receive_json();second=b.receive_json()
            assert first['call_id']!=second['call_id']
            a.send_bytes(b'a'*7680);b.send_bytes(b'b'*7680)
            assert a.receive_bytes()==b'a'*7680
            assert b.receive_bytes()==b'b'*7680
            a.send_text('stop');assert a.receive()['type']=='websocket.close'
            b.send_bytes(b'c'*7680);assert b.receive_bytes()==b'c'*7680
            b.send_text('stop');assert b.receive()['type']=='websocket.close'


def test_allocation_progress_and_timeout_are_explicit(client,monkeypatch):
    class Starting:
        async def __aenter__(self):
            await asyncio.sleep(.04);raise TimeoutError()
        async def __aexit__(self,*args):pass
    monkeypatch.setattr(module,'PROGRESS_INTERVAL',.01)
    monkeypatch.setattr(module.websockets,'connect',lambda *a,**kw:Starting())
    with client.websocket_connect('/api/conversation') as ws:
        assert ws.receive_json()['phase']=='allocating'
        progress=ws.receive_json()
        assert progress['phase']=='allocating' and 'elapsed_s' in progress
        while progress['type']=='warming':progress=ws.receive_json()
        assert progress['code']=='capacity_timeout'
        assert ws.receive()['type']=='websocket.close'


def test_cloud_gateway_auth_and_diagnostics_do_not_contact_gpu(client,monkeypatch):
    from server import call_diagnostics
    monkeypatch.setenv('VOICE_REQUIRE_AUTH','1')
    monkeypatch.setenv('VOICE_WORKER_URL','https://worker.modal.run')
    monkeypatch.setenv('VOICE_SERVICE_TOKEN','test-secret')
    def forbidden(*a,**kw):raise AssertionError('Must not contact GPU')
    monkeypatch.setattr(module.websockets,'connect',forbidden)
    monkeypatch.setattr(client.app.state.http,'post',forbidden)
    monkeypatch.setattr(call_diagnostics,'emit',lambda *a,**kw:None)
    assert client.get('/api/health').status_code==403
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect('/api/conversation'):pass
    headers={'x-voice-service-token':'test-secret','origin':'https://site.example'}
    assert client.get('/api/health',headers=headers).json()['ready']
    record=dict(event='client_end',event_id='a'*8+'-'+ 'a'*4+'-'+ 'a'*4+'-'+ 'a'*4+'-'+ 'a'*12,
                client_id='b'*8+'-'+ 'b'*4+'-'+ 'b'*4+'-'+ 'b'*4+'-'+ 'b'*12,
                call_id='01234567',reason='user_stop',elapsed_ms=9000,input_frames=100,output_frames=95,
                last_server_frame=96,playback_buffer_s=0,socket_buffer_bytes=0,socket_state=1,
                audio_state='running',microphone_state='live',microphone_muted=False,hidden=False)
    assert client.post('/api/call-diagnostics',headers=headers,json=record).status_code==200
    assert client.post('/api/call-diagnostics',headers=headers,json={**record,'transcript':'private'}).status_code==422


def test_transient_allocation_retries_but_auth_errors_do_not(client,monkeypatch):
    from types import SimpleNamespace
    attempts=[]
    class Remote:
        async def __aenter__(self):
            attempts.append(1)
            if len(attempts)==1:raise module.websockets.InvalidStatus(SimpleNamespace(status_code=503))
            self.stopped=asyncio.Event();return self
        async def __aexit__(self,*args):pass
        async def send(self,data):self.stopped.set()
        def __aiter__(self):return self.messages()
        async def messages(self):
            yield json.dumps({'type':'ready','call_id':'12345678'})
            await self.stopped.wait()
    monkeypatch.setattr(module,'RETRY_INTERVAL',.01)
    monkeypatch.setattr(module.websockets,'connect',lambda *a,**kw:Remote())
    with client.websocket_connect('/api/conversation') as ws:
        assert ws.receive_json()['type']=='warming'
        assert ws.receive_json()['type']=='ready'
        ws.send_text('stop');assert ws.receive()['type']=='websocket.close'
    assert len(attempts)==2
    attempts.clear()
    class Forbidden(Remote):
        async def __aenter__(self):
            attempts.append(1)
            raise module.websockets.InvalidStatus(SimpleNamespace(status_code=403))
    monkeypatch.setattr(module.websockets,'connect',lambda *a,**kw:Forbidden())
    with client.websocket_connect('/api/conversation') as ws:
        ws.receive_json();assert ws.receive_json()['type']=='error'
    assert len(attempts)==1
