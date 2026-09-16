from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect
from server.service_access import ServiceAccess, origin_allowed


def test_cloud_requires_gateway_secret_for_http_and_websockets(monkeypatch):
    monkeypatch.setenv('MOSHI_BACKEND','torch');monkeypatch.setenv('VOICE_SERVICE_TOKEN','test-private-token')
    app=FastAPI();app.add_middleware(ServiceAccess)
    @app.get('/api/health')
    def health():return {'ready':True}
    @app.websocket('/api/conversation')
    async def call(ws:WebSocket):
        await ws.accept();await ws.send_json({'ready':True});await ws.close()
    with TestClient(app) as client:
        assert client.get('/api/health').status_code==403
        assert client.get('/api/health',headers={'x-voice-service-token':'wrong'}).status_code==403
        headers={'x-voice-service-token':'test-private-token'}
        assert client.get('/api/health',headers=headers).status_code==200
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect('/api/conversation'):pass
        with client.websocket_connect('/api/conversation',headers=headers) as ws:assert ws.receive_json()['ready']
        assert origin_allowed({**headers,'origin':'https://private-site.example','host':'backend.example'})
        assert not origin_allowed({'origin':'https://other.example','host':'backend.example'})
    monkeypatch.delenv('VOICE_SERVICE_TOKEN')
    with TestClient(app) as client:assert client.get('/api/health').status_code==403
