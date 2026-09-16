"""Exercise the real cloned voice repeatedly, saving audio and recognition diagnostics."""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.engine import Engine, ROOT

TEXTS = [
    'My name is Eric, and I love exploring parks and forests with my friends.',
    'Your name is Alice. Nice meeting you!',
    'Bzzz, I like strawberries and exploring the forest. What about you?',
    'That sounds lovely, zzzz, what did you find in the park today?',
    'I would fly between the trees, stop for a snack, and then visit my friends before heading home.',
    'A quiet afternoon with friends sounds wonderful to me.',
    'Bzzz, those chicken nuggets sound delicious, zzzz!',
    'Your name is Alice. Nice meeting you!',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--passes', type=int, default=1)
    parser.add_argument('--label', default='voice-stability')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.label
    out.mkdir(parents=True, exist_ok=True)
    engine = Engine()
    engine.initialize()
    report = []
    for index, text in enumerate(TEXTS * args.passes):
        start = time.perf_counter()
        chunks, first = [], None
        for chunk in engine.speech(text, threading.Event()):
            if first is None:
                first = time.perf_counter() - start
            chunks.append(chunk)
        elapsed = time.perf_counter() - start
        audio = np.concatenate(chunks)
        assert np.isfinite(audio).all()
        sf.write(out / f'{index:02d}.wav', audio, engine.sample_rate)
        segments, _ = engine.stt.transcribe(str(out / f'{index:02d}.wav'), language='en', beam_size=3, vad_filter=True)
        item = dict(text=text, heard=' '.join(s.text.strip() for s in segments),
                    first_audio_s=round(first, 3), total_s=round(elapsed, 3),
                    audio_s=round(len(audio)/engine.sample_rate, 3),
                    clipped_fraction=float(np.mean(np.abs(audio) >= .999)))
        report.append(item)
        (out / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(item), flush=True)


if __name__ == '__main__':
    main()
