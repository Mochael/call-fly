import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import server.app as module
from server.engine import pop_phrase
from server.voice_quality import SpeechGenerationError


class FakeEngine:
    def __init__(self):
        self.ready = False
        self.stage = 'loading'
        self.error = None
        self.greeting = np.zeros(240, np.float32)
        self.sample_rate = 24000
        self.cancelled = threading.Event()

    def initialize(self):
        self.ready = True
        self.stage = 'Ready'

    def turn(self, pcm, history, cancel, emit):
        if pcm == b'\x02\x00' * 4000:
            raise SpeechGenerationError('Rejected voice')
        if pcm == b'\x01\x00' * 4000:
            while not cancel.wait(0.01):
                pass
            self.cancelled.set()
            return
        emit({'type': 'transcript', 'role': 'user', 'text': 'Hello'})
        emit({'type': 'audio_start', 'sample_rate': 24000})
        emit(np.ones(240, dtype='<f4').tobytes())
        history.append({'role': 'user', 'content': 'Hello'})


@pytest.fixture
def client(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(module, 'engine', engine)
    monkeypatch.setattr(module, 'worker', ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(module, 'call_lock', asyncio.Lock())
    with TestClient(module.app) as client:
        for _ in range(100):
            if client.get('/api/health').json()['ready']:
                break
            time.sleep(0.005)
        yield client


def greeting(ws):
    assert ws.receive_json()['type'] == 'transcript'
    assert ws.receive_json()['type'] == 'audio_start'
    assert len(ws.receive_bytes()) == 960
    assert ws.receive_json()['type'] == 'turn_end'


def test_phrase_boundaries():
    assert pop_phrase('Your name is Alice. Nice to meet you!') == ('', 'Your name is Alice. Nice to meet you!')
    sentence = 'The strawberries cost 3.14 dollars and I can share them with you.'
    assert pop_phrase(sentence + ' More') == (sentence, 'More')
    assert pop_phrase('I have a thought') == ('', 'I have a thought')
    assert pop_phrase('I have a thought', final=True) == ('I have a thought', '')


def test_real_route_and_host_guard(client):
    assert client.get('/').status_code == 200
    assert 'Start conversation' not in client.get('/').text  # ready state is populated by JS
    assert client.get('/api/health', headers={'host': 'attacker.example'}).status_code == 400


def test_exclusive_call_and_input_validation(client):
    with client.websocket_connect('/api/conversation') as ws:
        greeting(ws)
        with client.websocket_connect('/api/conversation') as other:
            assert 'already in a call' in other.receive_json()['message']
        for invalid in [b'1', bytes(module.MAX_AUDIO_BYTES + 2)]:
            ws.send_bytes(invalid)
            assert ws.receive_json()['type'] == 'error'
            assert ws.receive_json()['type'] == 'turn_end'
        ws.send_bytes(bytes(8000))
        assert ws.receive_json()['type'] == 'transcript'
        assert ws.receive_json()['type'] == 'audio_start'
        assert ws.receive_bytes()
        assert ws.receive_json()['type'] == 'turn_end'
    # Closing a call frees the single inference slot for the next call.
    with client.websocket_connect('/api/conversation') as ws:
        greeting(ws)


def test_disconnect_cancels_inference(client):
    with client.websocket_connect('/api/conversation') as ws:
        greeting(ws)
        ws.send_bytes(b'\x01\x00' * 4000)
        time.sleep(0.03)
        ws.send_text('stop')
        assert module.engine.cancelled.wait(2)


def test_cross_origin_rejected(client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect('/api/conversation', headers={'origin': 'https://attacker.example'}):
            pass
    assert exc.value.code == 1008


def test_voice_generation_failure_keeps_call_usable(client):
    with client.websocket_connect('/api/conversation') as ws:
        greeting(ws)
        ws.send_bytes(b'\x02\x00' * 4000)
        assert 'could not generate' in ws.receive_json()['message']
        assert ws.receive_json()['type'] == 'turn_end'
        ws.send_bytes(bytes(8000))
        assert ws.receive_json()['type'] == 'transcript'
        assert ws.receive_json()['type'] == 'audio_start'
        assert ws.receive_bytes()
        assert ws.receive_json()['type'] == 'turn_end'
