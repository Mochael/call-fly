import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import server.moshi_app as module

class FakeEngine:
    ready = False
    stage = 'Loading'
    error = None
    def __init__(self):
        self.count = 0
        self.started = threading.Event()
        self.ended = threading.Event()
        self.slow = False
    def initialize(self):
        self.ready = True
    def begin(self):
        self.count = 0; self.ended.clear()
    def encode(self, pcm):
        return pcm
    def generate(self, pcm):
        self.started.set()
        if self.slow: time.sleep(.08)
        self.count += 1
        return {'step': self.count, 'audio_codes': pcm, 'features': [.1]*64,
                'text': ' hello', 'compute_ms': 1}
    def decode(self, pcm):
        return np.full(module.FRAME, .05, dtype='<f4')
    def end(self):
        self.ended.set()

@pytest.fixture
def client(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(module, 'engine', engine)
    monkeypatch.setattr(module, 'worker', ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(module, 'codec_worker', ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(module, 'call_lock', asyncio.Lock())
    with TestClient(module.app) as client:
        for _ in range(100):
            if client.get('/api/health').json()['ready']: break
            time.sleep(.005)
        yield client

PCM = np.zeros(module.FRAME, dtype='<f4').tobytes()

def test_duplex_protocol_and_reset(client):
    for _ in range(2):
        with client.websocket_connect('/api/conversation') as ws:
            assert ws.receive_json()['type'] == 'ready'
            ws.send_bytes(PCM)
            frame = ws.receive_json()
            assert frame['index'] == 1 and len(frame['features']) == 64
            assert len(ws.receive_bytes()) == module.FRAME*4
            # More input is accepted with no turn_end/listen handshake.
            ws.send_bytes(PCM)
            assert ws.receive_json()['index'] == 2
            assert ws.receive_bytes()
            ws.send_text('stop')
            assert module.engine.ended.wait(2)
        assert not client.get('/api/health').json()['busy']

@pytest.mark.parametrize('invalid', [b'bad', np.full(module.FRAME, np.nan, dtype='<f4').tobytes(), np.full(module.FRAME, 2, dtype='<f4').tobytes()])
def test_invalid_pcm_closes_and_releases_call(client, invalid):
    with client.websocket_connect('/api/conversation') as ws:
        ws.receive_json(); ws.send_bytes(invalid)
        assert ws.receive_json()['type'] == 'error'
        assert module.engine.ended.wait(2)
    assert not client.get('/api/health').json()['busy']

def test_origin_exclusivity_and_cleanup_during_inference(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect('/api/conversation', headers={'origin': 'https://example.com'}): pass
    module.engine.slow = True
    with client.websocket_connect('/api/conversation') as ws:
        ws.receive_json()
        with client.websocket_connect('/api/conversation') as other:
            assert 'already in a call' in other.receive_json()['message']
        ws.send_bytes(PCM)
        assert module.engine.started.wait(2)
        ws.send_text('stop')
        assert module.engine.ended.wait(2)
    assert not client.get('/api/health').json()['busy']

def test_limit_drains_last_frame_and_ends_cleanly(client, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=module.log.name)
    monkeypatch.setattr(module, 'MAX_STEPS', 2)
    with client.websocket_connect('/api/conversation') as ws:
        ws.receive_json()
        ws.send_bytes(PCM); ws.send_bytes(PCM)
        for index in [1, 2]:
            assert ws.receive_json()['index'] == index
            assert ws.receive_bytes()
        end = ws.receive_json()
        assert end['type'] == 'session_end' and end['code'] == 'session_limit'
        assert module.engine.ended.wait(2)

    assert 'reason=session_limit frames=2' in caplog.text


def test_client_stop_reason_is_logged_without_conversation_content(client, caplog):
    caplog.set_level(logging.INFO, logger=module.log.name)
    with client.websocket_connect('/api/conversation') as ws:
        ws.receive_json()
        ws.send_json({'type': 'stop', 'reason': 'playback_backlog'})
        assert module.engine.ended.wait(2)
    assert 'reason=client_playback_backlog' in caplog.text


def test_invalid_control_is_rejected(client):
    with client.websocket_connect('/api/conversation') as ws:
        ws.receive_json()
        ws.send_json({'type': 'stop', 'reason': []})
        assert ws.receive_json()['code'] == 'invalid_input'
        assert module.engine.ended.wait(2)
