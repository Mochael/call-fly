"""Paced full-duplex API probe using real speech and real model inference."""
import argparse
import asyncio
import json
from pathlib import Path
import time

import numpy as np
import sphn
import rustymimi
import websockets

async def main(seconds):
    frames = int(seconds*12.5)
    pcm, _ = sphn.read('artifacts/browser-microphone.wav', sample_rate=24000)
    pcm = np.pad(pcm[0][:frames*1920], (0, max(0, frames*1920-pcm.shape[-1])))
    output, features, metrics, delays, text = [], [], [], [], []
    sent = {}; started = time.perf_counter()
    async with websockets.connect('ws://localhost:8765/api/conversation', origin='http://localhost:8765') as ws:
        async def wait_ready():
            while True:
                message = json.loads(await ws.recv())
                if message['type'] != 'warming':
                    return message
        ready = await asyncio.wait_for(wait_ready(), timeout=180)
        assert ready['type'] == 'ready', ready
        async def send():
            base = time.perf_counter()
            for i in range(frames):
                await asyncio.sleep(max(0, base+i*.08-time.perf_counter()))
                sent[i+1] = time.perf_counter()
                await ws.send(pcm[i*1920:(i+1)*1920].astype('<f4').tobytes())
        async def receive():
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    output.append(np.frombuffer(msg, dtype='<f4'))
                    if len(output) >= frames:
                        return
                    continue
                data = json.loads(msg)
                assert data['type'] != 'error', data
                if data['type'] == 'frame':
                    features.append(data['features']); text.append(data['text'])
                    delays.append(time.perf_counter()-sent[data['index']])
                if data['type'] == 'metrics':
                    metrics.append(data)
                    if data['frames'] % 100 == 0: print(data, flush=True)
        await asyncio.wait_for(asyncio.gather(send(), receive()), timeout=seconds+30)
        try:
            await ws.send('stop')
        except websockets.exceptions.ConnectionClosedOK:
            pass  # The server may already have closed at the five-minute limit.
    Path('artifacts/moshi').mkdir(exist_ok=True)
    audio = np.concatenate(output)
    rustymimi.write_wav('artifacts/moshi/live-api.wav', audio, sample_rate=24000)
    report = {'runtime': ready, 'duration_s': seconds, 'wall_s': time.perf_counter()-started,
              'received_audio_s': len(audio)/24000, 'mean_frame_latency_ms': float(np.mean(delays)*1000),
              'p95_frame_latency_ms': float(np.percentile(delays,95)*1000),
              'last_frame_latency_ms': delays[-1]*1000, 'text': ''.join(text),
              'feature_variation': float(np.mean(np.std(features,axis=0))), 'metrics': metrics}
    Path('artifacts/moshi/live-api.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='metrics'},indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--seconds', type=int, default=40)
    asyncio.run(main(parser.parse_args().seconds))
