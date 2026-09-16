import threading
from types import SimpleNamespace

import numpy as np
import pytest

from server.engine import Engine
from server.voice_quality import SpeechGenerationError, choose_buzz, clean_reply, insert_buzzes, speech_and_buzzes


def test_buzzes_are_sporadic_bounded_and_not_omitted():
    assert clean_reply('Bzzz, I love strawberries! 🍓', False) == 'I love strawberries!'
    assert clean_reply('Bzzz, I love strawberries! 🍓', True) == 'Bzzz, I love strawberries!'
    assert clean_reply('I love strawberries!', True) == 'Bzzz, I love strawberries!'
    assert clean_reply('bzzz zzzz bzzz zzzz', True).split() == ['bzzz', 'zzzz']
    assert not choose_buzz([{'role': 'assistant', 'content': 'Bzzz, hello!'}], 0)
    assert choose_buzz([{'role': 'assistant', 'content': 'Hello!'}] * 2, 1)
    assert not choose_buzz([], 1)


def test_buzz_markers_are_removed_without_changing_words():
    clean, positions = speech_and_buzzes('Bzzz, I love the forest, zzzz, with my buzzing friends.')
    assert clean == 'I love the forest, with my buzzing friends.'
    assert positions == [0, 4]
    assert speech_and_buzzes('Bzzz zzzz bzzzz zzzzz')[1] == [0, 0]


def test_quiet_buzz_is_audible_and_speech_is_preserved():
    audio = np.full(24000, .01, dtype=np.float32)
    buzz = np.full(9600, .0001, dtype=np.float32)
    mixed = insert_buzzes(audio, buzz, [0], 5, 24000)
    pad = 1080
    assert np.sqrt(np.mean(mixed[pad:pad+9600]**2)) == pytest.approx(.012)
    np.testing.assert_array_equal(mixed[pad*2+9600:], audio)


def test_middle_buzz_uses_quiet_pause_without_dropping_audio():
    audio = np.full(24000, .01, dtype=np.float32)
    audio[11500:12500] = 0
    buzz = np.full(960, .02, dtype=np.float32)
    mixed = insert_buzzes(audio, buzz, [3], 6, 24000)
    # One buzz plus 90 ms padding is added; original samples stay in order.
    assert len(mixed) == len(audio) + 960 + 2160
    np.testing.assert_array_equal(mixed[:11000], audio[:11000])
    np.testing.assert_array_equal(mixed[-11000:], audio[-11000:])


def fake_engine(*, invalid=False):
    engine = Engine()
    engine.reference, engine.ref_text = None, 'reference'
    engine.buzz = np.full(100, .01, np.float32)
    engine.calls = []
    def generate(**kwargs):
        engine.calls.append(kwargs)
        assert 'bzzz' not in kwargs['text'].lower()
        yield SimpleNamespace(audio=np.full(24000, np.nan if invalid else .2, np.float32),
                              sample_rate=24000, token_count=13)
    engine.tts = SimpleNamespace(_generate_icl=generate)
    def transcribe(*args, **kwargs):
        raise AssertionError('Generated speech must never be sent back through ASR')
    engine.stt = SimpleNamespace(transcribe=transcribe)
    return engine


def test_speech_is_generated_once_without_transcription_or_retry():
    engine = fake_engine()
    output = np.concatenate(list(engine.speech('Your name is Alice.', threading.Event())))
    assert len(engine.calls) == 1
    assert len(output) == 24000
    assert np.max(output) == pytest.approx(.2)


def test_buzz_playback_also_needs_no_recognition():
    engine = fake_engine()
    output = np.concatenate(list(engine.speech('Bzzz, your name is Alice.', threading.Event())))
    assert len(output) > 24000
    assert len(engine.calls) == 1


def test_invalid_numeric_samples_raise_without_retry():
    engine = fake_engine(invalid=True)
    with pytest.raises(SpeechGenerationError):
        list(engine.speech('Your name is Alice.', threading.Event()))
    assert len(engine.calls) == 1


def test_cancelled_voice_emits_nothing():
    engine = fake_engine()
    cancel = threading.Event()
    cancel.set()
    assert list(engine.speech('Hello there.', cancel)) == []
    assert engine.calls == []
