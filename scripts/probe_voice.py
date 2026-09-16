"""Real cloned-voice smoke test; writes audio and measured timings."""
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.engine import Engine

OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)
start = time.perf_counter()
model = Engine()
model.initialize()
print(f'Model loaded in {time.perf_counter()-start:.2f}s', flush=True)
for index, text in enumerate(['Hello! I am Eric. What is your name?', 'I like strawberries and exploring the forest. What about you?']):
    started = time.perf_counter()
    chunks = []
    first = None
    for chunk in model.speech(text, threading.Event()):
        if not len(chunk):
            continue
        if first is None:
            first = time.perf_counter()-started
            print(f'First audio {first:.2f}s', flush=True)
        chunks.append(chunk)
    audio = np.concatenate(chunks)
    elapsed = time.perf_counter()-started
    sf.write(OUT / f'voice-probe-{index}.wav', audio, model.sample_rate)
    metrics = dict(text=text, first_audio_s=first, elapsed_s=elapsed,
                   audio_s=len(audio)/model.sample_rate,
                   realtime_factor=elapsed/(len(audio)/model.sample_rate))
    print(json.dumps(metrics), flush=True)
    (OUT / f'voice-probe-{index}.json').write_text(json.dumps(metrics, indent=2))
