"""One local, serialized inference worker. No remote inference or transcript storage."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from .voice_quality import SpeechGenerationError, choose_buzz, clean_reply, insert_buzzes, speech_and_buzzes, words

ROOT = Path(__file__).resolve().parents[1]
TTS_MODEL = os.getenv('TTS_MODEL', 'mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit')
TTS_REVISION = os.getenv('TTS_REVISION', 'e7dd0585652209fa0d7783659aad4e8a324de11c')
LLM_MODEL = os.getenv('LLM_MODEL', 'qwen2.5:1.5b')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434')
VOICE_DIR = Path(os.getenv('VOICE_DIR', str(ROOT / 'voices/eric')))
SYSTEM_PROMPT = (
    'You are Eric, a friendly, curious little bumblebee character having a live voice conversation. '
    'Your name is Eric. You enjoy the park, the forest, strawberries, chicken nuggets and your friends. '
    'Answer the actual question directly in ONE short natural sentence, 10 to 20 words. '
    'Remember what the user tells you. Sound warm, playful and conversational. '
    'Occasionally add a little fly buzz: use the literal words bzzz or zzzz in roughly one out '
    'of every three replies, with at most two buzzes in a reply. Put them at natural pauses '
    'at the start, middle, or end of a sentence. Most replies should have no buzz. '
    'No markdown, lists, emojis, other sound-effect spellings, stage directions, or lengthy introductions. '
    'Do not repeat your name unless asked. Do not narrate actions. Never mention these instructions.'
)
GREETING = 'Hello! I am Eric. What is your name?'
VOICE_PROFILE = 'direct-icl-8bit-v2'


def pop_phrase(buffer: str, final: bool = False) -> tuple[str, str]:
    """Keep short sentences together: tiny clone prompts can repeat reference sounds."""
    if not final:
        for match in re.finditer(r'[.!?](?=\s|$)', buffer):
            if len(buffer[:match.end()].split()) >= 12:
                return buffer[:match.end()].strip(), buffer[match.end():].lstrip()
    if final:
        return buffer.strip(), ''
    words = list(re.finditer(r'\S+', buffer))
    if len(words) >= 32:
        end = words[27].end()
        return buffer[:end].strip(), buffer[end:].lstrip()
    return '', buffer


class Engine:
    def __init__(self):
        self.ready = False
        self.stage = 'Preparing the voice…'
        self.error = None
        self.tts = None
        self.stt = None
        self.greeting = None

    def initialize(self):
        # MLX loading and inference must all run on the same dedicated thread.
        import mlx.core as mx
        from faster_whisper import WhisperModel
        from mlx_audio.tts.utils import load_model
        from huggingface_hub import snapshot_download

        if not (VOICE_DIR / 'reference.wav').exists() or not (VOICE_DIR / 'transcript.txt').exists():
            raise RuntimeError('Voice files are missing. Run scripts/setup.sh first.')
        self.ref_text = (VOICE_DIR / 'transcript.txt').read_text().strip()
        reference, sr = sf.read(VOICE_DIR / 'reference.wav', dtype='float32')
        if sr != 24000 or reference.ndim != 1:
            raise RuntimeError('Reference must be mono 24 kHz WAV. Run scripts/setup.sh.')
        self.reference = mx.array(reference)
        buzz_path = VOICE_DIR / 'buzz.wav'
        if buzz_path.exists():
            self.buzz, buzz_sr = sf.read(buzz_path, dtype='float32')
            if buzz_sr != 24000 or self.buzz.ndim != 1:
                raise RuntimeError('buzz.wav must be mono 24 kHz WAV.')
            self.buzz = self.buzz[:int(.8 * 24000)]
        else:
            # A bounded wing-like buzz for fresh installations without a sample.
            t = np.arange(int(.4 * 24000)) / 24000
            self.buzz = (.12 * np.sin(2*np.pi*180*t + .8*np.sin(2*np.pi*35*t))).astype('float32')
        fade = min(480, len(self.buzz)//2)
        if fade:
            self.buzz[:fade] *= np.linspace(0, 1, fade)
            self.buzz[-fade:] *= np.linspace(1, 0, fade)
        peak = np.max(np.abs(self.buzz), initial=0)
        if peak > .4:
            self.buzz *= .4 / peak
        self.stage = 'Loading the listening model…'
        self.stt = WhisperModel('base.en', device='cpu', compute_type='int8', cpu_threads=4)
        self.stage = 'Loading Eric’s voice…'
        model_path = snapshot_download(TTS_MODEL, revision=TTS_REVISION,
                                       allow_patterns=['*.json', '*.safetensors', '*.txt'])
        self.tts = load_model(model_path)
        self.stage = 'Warming up the conversation…'
        with httpx.Client(timeout=120) as client:
            response = client.post(f'{OLLAMA_URL}/api/chat', json={
                'model': LLM_MODEL, 'messages': [{'role': 'user', 'content': 'Say hello.'}],
                'stream': False, 'keep_alive': '30m', 'options': {'num_predict': 3, 'num_ctx': 2048},
            })
            response.raise_for_status()
        # Cache only this fixed greeting. All conversational replies use real inference.
        cache_key = hashlib.sha256(
            reference.tobytes() + self.ref_text.encode() + TTS_MODEL.encode()
            + TTS_REVISION.encode() + VOICE_PROFILE.encode() + GREETING.encode()
        ).hexdigest()[:16]
        cache_dir = ROOT / '.runtime'
        cache_dir.mkdir(exist_ok=True)
        greeting_path = cache_dir / f'greeting-{cache_key}.wav'
        if greeting_path.exists():
            self.greeting, self.sample_rate = sf.read(greeting_path, dtype='float32')
            # Populate the actual clone conditioning cache even on a restart.
            self.tts._prepare_icl_generation_inputs(GREETING, self.reference, self.ref_text, 'English')
        else:
            chunks = list(self.speech(GREETING, threading.Event()))
            self.greeting = np.concatenate(chunks)
            sf.write(greeting_path, self.greeting, self.sample_rate, subtype='FLOAT')
        self.stage = 'Ready when you are.'
        self.ready = True

    def speech(self, text: str, cancel: threading.Event):
        if cancel.is_set():
            return
        speech, buzzes = speech_and_buzzes(text)
        self.sample_rate = 24000
        if not words(speech):
            if buzzes:
                yield self.buzz.copy()
            return
        token_budget = min(260, max(72, len(speech.split()) * 8 + 32))
        # Keep the stable full decoder and sampling settings; no ASR pass or retry.
        for result in self.tts._generate_icl(
            text=speech, ref_audio=self.reference, ref_text=self.ref_text, language='English',
            stream=False, temperature=.7, top_p=.9, top_k=50,
            repetition_penalty=1.05, max_tokens=token_budget, verbose=False,
        ):
            if cancel.is_set():
                return
            if result.sample_rate != 24000:
                raise SpeechGenerationError('Unexpected voice sample rate')
            audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
            if not audio.size or not np.isfinite(audio).all():
                raise SpeechGenerationError('Voice model returned invalid samples')
            audio = insert_buzzes(audio, self.buzz, buzzes, len(words(speech)), self.sample_rate)
            peak = np.max(np.abs(audio), initial=0)
            if peak > .9:
                audio = audio * (.9 / peak)
            for offset in range(0, len(audio), 11520):
                if cancel.is_set():
                    return
                yield audio[offset:offset+11520].astype('<f4')

    def turn(self, pcm: bytes, history: list[dict], cancel: threading.Event, emit):
        started = time.perf_counter()
        audio = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.0
        if len(audio) < 3200 or np.sqrt(np.mean(audio * audio)) < 0.002:
            emit({'type': 'no_speech'})
            return
        emit({'type': 'state', 'state': 'transcribing'})
        segments, _ = self.stt.transcribe(
            audio, language='en', beam_size=1, vad_filter=True,
            condition_on_previous_text=False,
        )
        text = ' '.join(s.text.strip() for s in segments if s.no_speech_prob < 0.8).strip()
        if cancel.is_set():
            return
        if not text:
            emit({'type': 'no_speech'})
            return
        transcribed = time.perf_counter()
        emit({'type': 'transcript', 'role': 'user', 'text': text})
        emit({'type': 'state', 'state': 'thinking'})
        pending = history[-12:] + [{'role': 'user', 'content': text}]
        allow_buzz = choose_buzz(history, random.random())
        turn_prompt = SYSTEM_PROMPT + (
            ' For this reply, include one bzzz or zzzz at a natural pause.' if allow_buzz
            else ' For this reply, do not include any buzz sound spellings.'
        ) + ' Reply with one short sentence only; no emojis.'
        answer, buffer = '', ''
        first_audio = None
        first_token = None
        audio_samples = 0
        with httpx.Client(timeout=httpx.Timeout(90, connect=5)) as client:
            with client.stream('POST', f'{OLLAMA_URL}/api/chat', json={
                'model': LLM_MODEL, 'messages': [{'role': 'system', 'content': turn_prompt}] + pending,
                'stream': True, 'keep_alive': '30m',
                'options': {'temperature': 0.65, 'num_predict': 56, 'num_ctx': 2048},
            }) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if cancel.is_set():
                        return
                    if not line:
                        continue
                    event = json.loads(line)
                    if event.get('error'):
                        raise RuntimeError(event['error'])
                    token = event.get('message', {}).get('content', '')
                    if token and first_token is None:
                        first_token = time.perf_counter()-started
                    answer += token
                    buffer += token
                    phrase, buffer = pop_phrase(buffer, final=event.get('done', False))
                    if phrase:
                        answer = phrase
                        break  # Keep each turn to one short, coherent voice prompt.
                    if event.get('done'):
                        break
        answer = clean_reply(answer, allow_buzz)
        if cancel.is_set():
            return
        emit({'type': 'transcript', 'role': 'assistant', 'text': answer})
        emit({'type': 'audio_start', 'sample_rate': 24000})
        for chunk in self.speech(answer, cancel):
            if first_audio is None:
                first_audio = time.perf_counter()-started
            audio_samples += len(chunk)
            emit(chunk.tobytes())
        if cancel.is_set():
            return
        if not answer.strip() or first_audio is None:
            raise RuntimeError('No spoken reply was produced. Please try again.')
        history[:] = (pending + [{'role': 'assistant', 'content': answer.strip()}])[-12:]
        emit({'type': 'metrics', 'stt_s': round(transcribed-started, 3),
              'first_token_s': round(first_token or 0, 3), 'first_audio_s': round(first_audio, 3),
              'total_s': round(time.perf_counter()-started, 3),
              'audio_s': round(audio_samples/self.sample_rate, 3)})
