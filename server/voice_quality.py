"""Reply formatting and bounded buzz rendering; no speech validation."""
import re
import unicodedata

import numpy as np

BUZZ = re.compile(r'\b(?:bz{2,}|z{3,})\b', re.IGNORECASE)


def clean_reply(text, allow_buzz):
    """Enforce a few buzzes, with no emoji/markup passed to the voice model."""
    text = ''.join(c for c in text if unicodedata.category(c) != 'So' and c not in '\ufe0f\u200d`*#')
    seen = 0
    def marker(match):
        nonlocal seen
        seen += 1
        return match.group() if allow_buzz and seen <= 2 else ''
    text = BUZZ.sub(marker, text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([,;:])\s*([,;:])', r'\1', text)
    text = text.strip(' ,;:')
    if allow_buzz and words(text) and not BUZZ.search(text):
        text = 'Bzzz, ' + text
    return text


class SpeechGenerationError(RuntimeError):
    pass


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower().replace('’', "'"))


def speech_and_buzzes(text):
    """Remove buzz markers from TTS input and remember their word boundaries."""
    positions = []
    parts, last = [], 0
    for match in BUZZ.finditer(text):
        parts.append(text[last:match.start()])
        if len(positions) < 2:
            positions.append(len(words(' '.join(parts))))
        last = match.end()
    parts.append(text[last:])
    speech = re.sub(r'\s+', ' ', ' '.join(parts))
    speech = re.sub(r'([,;:])\s*([,;:])', r'\1', speech)
    return speech.strip(' ,;:'), positions


def choose_buzz(history, draw):
    """Vary buzz turns, avoiding consecutive buzzes and long silent streaks."""
    recent = [item['content'] for item in history if item['role'] == 'assistant'][-2:]
    if recent and BUZZ.search(recent[-1]):
        return False
    return len(recent) == 2 and not any(BUZZ.search(text) for text in recent) or draw < 1/3


def insert_buzzes(audio, buzz, positions, total_words, sample_rate):
    """Insert buzzes near estimated pauses without running speech recognition."""
    if not positions or not len(buzz):
        return audio
    speech_rms = float(np.sqrt(np.mean(audio * audio))) if len(audio) else .012
    buzz_rms = float(np.sqrt(np.mean(buzz * buzz)))
    if buzz_rms > 0:
        buzz = buzz * (max(.012, min(.05, speech_rms * 1.2)) / buzz_rms)
        peak = np.max(np.abs(buzz))
        if peak > .4:
            buzz = buzz * (.4 / peak)
    pad = np.zeros(int(.045 * sample_rate), dtype=np.float32)
    effect = np.concatenate([pad, buzz, pad])
    offsets = []
    for position in positions:
        if position <= 0:
            offset = 0
        elif position >= total_words:
            offset = len(audio)
        else:
            estimate = int(len(audio) * position / max(total_words, 1))
            # Snap to a nearby quiet 20 ms window to avoid interrupting a syllable.
            width = max(1, int(.02 * sample_rate))
            radius = int(.3 * sample_rate)
            candidates = range(max(width, estimate-radius), min(len(audio)-width, estimate+radius)+1, width)
            offset = min(candidates, key=lambda i: float(np.mean(audio[i-width//2:i+width//2+1]**2)), default=estimate)
        offsets.append(offset)
    parts, previous = [], 0
    for offset in sorted(offsets):
        parts.extend([audio[previous:offset], effect])
        previous = offset
    parts.append(audio[previous:])
    return np.concatenate(parts)
