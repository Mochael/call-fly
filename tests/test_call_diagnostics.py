import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server import call_diagnostics as module


def test_diagnostic_endpoint_accepts_only_content_free_fields(monkeypatch):
    events=[]
    monkeypatch.setattr(module,'emit',lambda event,**fields:events.append((event,fields)))
    app=FastAPI();app.include_router(module.router)
    record=dict(event='client_end',event_id='a'*8+'-'+ 'a'*4+'-'+ 'a'*4+'-'+ 'a'*4+'-'+ 'a'*12,
                client_id='b'*8+'-'+ 'b'*4+'-'+ 'b'*4+'-'+ 'b'*4+'-'+ 'b'*12,
                call_id='01234567',reason='playback_backlog',elapsed_ms=9000,input_frames=100,output_frames=95,
                last_server_frame=96,playback_buffer_s=.85,socket_buffer_bytes=0,socket_state=1,
                audio_state='suspended',microphone_state='live',microphone_muted=False,hidden=True)
    with TestClient(app) as client:
        assert client.post('/api/call-diagnostics',json=record).status_code==200
        assert events[0][1]['audio_state']=='suspended'
        assert client.post('/api/call-diagnostics',json={**record,'transcript':'private words'}).status_code==422
        assert client.post('/api/call-diagnostics',json={**record,'reason':'private words'}).status_code==422
        assert client.post('/api/call-diagnostics',json=record,headers={'origin':'https://example.com'}).status_code==403
        assert client.post('/api/call-diagnostics',content=b' '*5000).status_code==413
    assert len(events)==1


def test_rotating_log_is_structured_and_persistent(tmp_path,monkeypatch):
    import logging
    logger=logging.getLogger('diagnostic-file-test');logger.setLevel(logging.INFO)
    monkeypatch.setattr(module,'logger',logger)
    monkeypatch.setattr(module,'PATH',tmp_path/'calls.jsonl')
    module.emit('server_end',call_id='01234567',reason='input_backlog',frames=100)
    record=json.loads((tmp_path/'calls.jsonl').read_text())
    assert record['event']=='server_end' and record['reason']=='input_backlog' and record['time']
    for handler in logger.handlers[:]: handler.close();logger.removeHandler(handler)
