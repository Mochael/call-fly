"""Generate deterministic microphone fixtures using macOS speech synthesis."""
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'artifacts'
PHRASES = {
    'user-question': 'What is your name, and what do you like to do?',
    'user-introduction': 'My name is Alice. Please remember that.',
    'user-recall': 'What is my name?',
}


def prepare():
    OUT.mkdir(exist_ok=True)
    for name, text in PHRASES.items():
        source = OUT / f'{name}.aiff'
        target = OUT / f'{name}.pcm'
        if not target.exists():
            subprocess.run(['say', '-v', 'Samantha', '-o', str(source), text], check=True)
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(source),
                            '-ar', '16000', '-ac', '1', '-f', 's16le', str(target)], check=True)
    microphone = OUT / 'browser-microphone.wav'
    if not microphone.exists():
        # Give the cloned greeting time to finish before the fake user speaks.
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(OUT / 'user-question.aiff'),
                        '-af', 'adelay=7000,apad=pad_dur=12', '-ar', '48000', '-ac', '1',
                        str(microphone)], check=True)


if __name__ == '__main__':
    prepare()
