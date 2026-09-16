"""Offline real-model speed/codec check; saves a WAV and timings, no microphone."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import sphn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.moshi_engine import MoshiEngine, FRAME, RATE

engine = MoshiEngine()
started = time.perf_counter()
engine.initialize()
print(f'Loaded in {time.perf_counter()-started:.1f}s', flush=True)
engine.begin(seed=299792458)
source = Path('artifacts/browser-microphone.wav')
pcm, _ = sphn.read(str(source), sample_rate=RATE)
pcm = pcm[0]
pcm = np.pad(pcm[:RATE*15], (0, max(0, RATE*20-len(pcm[:RATE*15]))))
times, output, text = [], [], ''
for i in range(len(pcm)//FRAME):
    result = engine.step(pcm[i*FRAME:(i+1)*FRAME])
    times.append(result['compute_ms'])
    text += result['text']
    if result['audio'] is not None:
        output.append(result['audio'])
    if i % 25 == 24:
        print(f'{i+1} frames: {np.mean(times[-25:]):.1f}ms/frame | {text[-90:]}', flush=True)
Path('artifacts/moshi').mkdir(exist_ok=True)
audio = np.concatenate(output)
import rustymimi
rustymimi.write_wav('artifacts/moshi/benchmark.wav', audio, sample_rate=RATE)
report = {'frames': len(times), 'mean_ms': float(np.mean(times)),
          'p95_ms': float(np.percentile(times,95)), 'real_time_factor': float(np.mean(times)/80),
          'text': text, 'peak': float(np.max(np.abs(audio))), 'audio_s': len(audio)/RATE}
Path('artifacts/moshi/benchmark.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
engine.end()
