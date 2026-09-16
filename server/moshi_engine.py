"""Pinned Moshi MLX inference. Only the serialized inference thread touches MLX.

A fixed connectome and offline-fitted readout modify text logits before sampling.
Moshi audio is conditioned on those sampled text tokens. No online learning.
"""
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'kyutai/moshiko-mlx-q4'
REVISION = '18e4df760a34d5977a34517d7d1580e07acbb2f1'
RATE = 24000
FRAME = 1920
MAX_STEPS = 3750  # Five minutes of model context; start a fresh call afterward.
FILES = ('model.q4.safetensors', 'tokenizer-e351c8d8-checkpoint125.safetensors',
         'tokenizer_spm_32k_3.model')


class MoshiEngine:
    def __init__(self, calibration=False):
        self.calibration = calibration
        self.circuit = None
        self.ready = False
        self.stage = 'Loading Moshi…'
        self.error = None
        self.gen = None

    def initialize(self):
        import mlx.core as mx
        import mlx.nn as nn
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi_mlx import models

        self.mx, self.models = mx, models
        paths = {name: hf_hub_download(MODEL, name, revision=REVISION) for name in FILES}
        self.stage = 'Loading the duplex speech model…'
        self.model = models.Lm(models.config_v0_1())
        self.model.set_dtype(mx.bfloat16)
        nn.quantize(self.model, bits=4, group_size=32)
        self.model.load_weights(paths[FILES[0]], strict=True)
        self.codec_file = paths[FILES[1]]
        self.codec = None
        self.tokenizer = sentencepiece.SentencePieceProcessor(paths[FILES[2]])
        self.stage = 'Warming up live speech…'
        self.model.warmup()
        from .connectome import get_graph
        from .model_reservoir import ModelCircuit, install_mlx_head
        self.circuit = ModelCircuit(get_graph(), MODEL, REVISION, calibration=self.calibration)
        install_mlx_head(self.model, self.circuit)
        self.begin(seed=299792458)
        for _ in range(4):
            self.step(np.zeros(FRAME, dtype=np.float32))
        self.end()
        self.ready = True
        self.stage = 'Ready for a live conversation.'

    def begin(self, seed=None):
        from moshi_mlx import utils
        import rustymimi
        self.end()
        # Reconstruct per call: reset() alone is not trusted to reset every
        # codec positional counter. Reserve ample positional space for 5 min.
        self.codec = rustymimi.Tokenizer(self.codec_file, num_codebooks=8, max_seq_len=16384)
        self.mx.random.seed(seed if seed is not None else time.time_ns() % (2**32))
        self.gen = self.models.LmGen(
            model=self.model, max_steps=MAX_STEPS+8,
            text_sampler=utils.Sampler(temp=.8, top_k=25),
            audio_sampler=utils.Sampler(temp=.8, top_k=250), check=False)

    def end(self):
        self.gen = None
        if self.circuit is not None:
            self.circuit.reset()
        if hasattr(self, 'model'):
            for cache in self.model.transformer_cache + self.model.depformer_cache:
                cache.reset()
        if hasattr(self, 'codec'):
            self.codec = None

    def encode(self, pcm):
        return self.codec.encode_step(np.asarray(pcm, dtype=np.float32).reshape(1, 1, FRAME))

    def decode(self, codes):
        return np.asarray(self.codec.decode_step(codes), dtype='<f4').reshape(-1)

    def generate(self, codes):
        started = time.perf_counter()
        tokens = self.mx.array(codes).transpose(0, 2, 1)[0, :, :8]
        # moshi-mlx 0.3.0 returns (text, normalized temporal-transformer state).
        text_token, hidden = self.gen.step(tokens)
        token = text_token.item()
        audio_tokens = self.gen.last_audio_tokens()
        audio_codes = np.array(audio_tokens[:, :, None], dtype=np.uint32) if audio_tokens is not None else None
        # Fixed signed block projection: 4096 dimensions → 64 bounded channels.
        # This summary is for visualization, not a neuron-to-neuron correspondence.
        values = np.array(hidden.astype(self.mx.float32)).reshape(-1)
        rms = float(np.sqrt(np.mean(values**2)) + 1e-6)
        blocks = values.reshape(64, -1)
        signs = np.where(np.arange(blocks.shape[1]) % 2, -1., 1.)
        features = np.tanh((blocks @ signs) / (np.sqrt(blocks.shape[1])*rms))
        text = self.tokenizer.id_to_piece(token).replace('▁', ' ') if token not in (0, 3) else ''
        return {'audio_codes': audio_codes, 'text': text, 'features': features.tolist(),
                'brain':self.circuit.latest, 'reservoir_logit_rms':self.circuit.last_logit_rms,
                'step': self.gen.step_idx, 'compute_ms': (time.perf_counter()-started)*1000}

    def step(self, pcm):
        """Sequential offline probe. Live serving overlaps codec and model stages."""
        started = time.perf_counter()
        result = self.generate(self.encode(pcm))
        codes = result.pop('audio_codes')
        result['audio'] = self.decode(codes) if codes is not None else None
        result['compute_ms'] = (time.perf_counter()-started)*1000
        return result
