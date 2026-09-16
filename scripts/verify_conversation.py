"""Exercise the production WebSocket and save real speech + measured timings."""
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import websockets
from prepare_test_audio import prepare

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts'


async def receive_turn(ws, name):
    chunks, events = [], []
    start = time.perf_counter()
    first_audio = None
    while True:
        packet = await asyncio.wait_for(ws.recv(), timeout=90)
        if isinstance(packet, bytes):
            first_audio = first_audio or time.perf_counter()-start
            chunks.append(np.frombuffer(packet, dtype='<f4'))
        else:
            event = json.loads(packet)
            events.append(event)
            if event['type'] == 'error':
                raise RuntimeError(event['message'])
            if event['type'] == 'turn_end':
                break
    assert chunks, 'No generated audio received'
    audio = np.concatenate(chunks)
    assert np.isfinite(audio).all() and np.sqrt(np.mean(audio**2)) > 0.001
    sf.write(OUT / f'{name}.wav', audio, 24000)
    user = [e['text'] for e in events if e['type'] == 'transcript' and e['role'] == 'user']
    assistant = [e['text'] for e in events if e['type'] == 'transcript' and e['role'] == 'assistant']
    metrics = next((e for e in events if e['type'] == 'metrics'), {})
    report = dict(user=user, assistant=assistant[-1] if assistant else None,
                  first_audio_s=first_audio, audio_s=len(audio)/24000, metrics=metrics)
    (OUT / f'{name}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return report


async def main():
    prepare()
    async with httpx.AsyncClient() as client:
        for _ in range(180):
            health = (await client.get('http://127.0.0.1:8765/api/health')).json()
            if health['error']:
                raise RuntimeError(health['error'])
            if health['ready']:
                break
            await asyncio.sleep(1)
        else:
            raise TimeoutError('Server did not become ready')
    async with websockets.connect('ws://127.0.0.1:8765/api/conversation') as ws:
        await receive_turn(ws, 'opening')
        await ws.send((OUT / 'user-question.pcm').read_bytes())
        report = await receive_turn(ws, 'conversation-reply')
        assert report['user'] and 'name' in report['user'][0].lower()
        assert 'eric' in report['assistant'].lower()
        await ws.send((OUT / 'user-introduction.pcm').read_bytes())
        await receive_turn(ws, 'conversation-introduction')
        await ws.send((OUT / 'user-recall.pcm').read_bytes())
        report = await receive_turn(ws, 'conversation-recall')
        assert 'alice' in report['assistant'].lower(), 'Conversation memory was lost'
        # Silence must not generate a hallucinated answer.
        await ws.send(bytes(16000))
        assert json.loads(await ws.recv())['type'] == 'no_speech'
        assert json.loads(await ws.recv())['type'] == 'turn_end'
        await ws.send('stop')


if __name__ == '__main__':
    asyncio.run(main())
