"""FLM-style fixed full graph and an offline-fitted, bias-free readout.

Anatomical recurrence: nftechie/flm (MIT, Alex Wormuth 2026).
The readout affects Moshi text logits before sampling; sampled text conditions
Moshi's audio depformer. No online learning or additional feedback projection.
"""
import base64
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from .connectome import Reservoir, FOLDER

INTERFACE = {'version': 1, 'seed': 7301, 'input_dimensions': 64,
             'readout_dimensions': 128, 'hidden_dimensions': 4096,
             'step_every_frames': 3, 'hidden_scale': 0.03, 'max_logit_rms': 0.10}
DEFAULT_ADAPTER = FOLDER.parent / 'moshi-reservoir' / 'adapter.npz'


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def summarize(hidden):
    values = np.asarray(hidden, np.float32).reshape(-1)
    if values.size != 4096 or not np.isfinite(values).all():
        raise ValueError('Expected a finite 4096-dimensional Moshi state.')
    blocks = values.reshape(64, 64)
    signs = np.where(np.arange(64) % 2, -1., 1.)
    return np.tanh((blocks @ signs) / (8 * (np.sqrt(np.mean(values**2)) + 1e-6))).astype(np.float32)


class ModelReservoir(Reservoir):
    def __init__(self, graph):
        super().__init__(graph)
        rng = np.random.default_rng(7302)
        self.output_bins = rng.integers(0, 128, len(self.state))
        self.output_signs = rng.choice(np.array([-1, 1], np.float32), len(self.state))
        self.output_scale = np.sqrt(np.maximum(1, np.bincount(self.output_bins, minlength=128)))

    def pooled(self):
        values = np.bincount(self.output_bins, weights=self.state*self.output_signs, minlength=128) / self.output_scale
        return (values / np.sqrt(np.mean(values**2)+1e-6)).astype(np.float32)


class ModelCircuit:
    def __init__(self, graph, model_id, revision, adapter=DEFAULT_ADAPTER, calibration=False):
        self.reservoir = ModelReservoir(graph)
        self.calibration = calibration
        self.mode = 'intact'
        self.weights = None
        if not calibration:
            adapter = Path(adapter)
            meta = json.loads(adapter.with_suffix('.json').read_text())
            expected = {'model':model_id, 'revision':revision, 'interface':INTERFACE,
                        'graph_sha256':digest(FOLDER/'manifest.json')}
            for key, value in expected.items():
                if meta.get(key) != value:
                    raise ValueError(f'Reservoir adapter mismatch: {key}')
            if meta['adapter_sha256'] != digest(adapter):
                raise ValueError('Reservoir adapter checksum mismatch.')
            with np.load(adapter, allow_pickle=False) as data:
                self.weights = data['weights'].copy()
            if self.weights.shape != (128,4096) or not np.isfinite(self.weights).all():
                raise ValueError('Invalid reservoir readout weights.')
            if not np.any(self.weights):
                raise ValueError('The reservoir readout must be fitted before serving.')
        self.reset()

    def reset(self):
        self.reservoir.reset()
        self.frame = 0
        self.delta = np.zeros(4096, np.float32)
        self.latest = None
        self.training_sample = None
        self.last_logit_rms = 0.
        self.last_graph_ms = 0.

    def advance(self, hidden):
        self.frame += 1
        self.latest = None
        if (self.frame-1) % INTERFACE['step_every_frames']:
            return self.delta
        started = time.perf_counter()
        features = summarize(hidden)
        if self.mode == 'no_edges':
            self.reservoir.state.fill(0)
            self.reservoir.updates += 1
        else:
            self.reservoir.step(features)
        pooled = self.reservoir.pooled()
        if self.calibration:
            self.training_sample = (pooled.copy(), np.asarray(hidden,np.float32).reshape(-1).copy())
        elif self.mode != 'base':
            self.delta = (INTERFACE['hidden_scale']*np.tanh(pooled @ self.weights)).astype(np.float32)
        else:
            self.delta.fill(0)
        self.last_graph_ms = (time.perf_counter()-started)*1000
        header, _ = self.reservoir.snapshot(self.last_graph_ms)
        header.update({'model_frame':self.frame, 'mode': 'calibration' if self.calibration else self.mode,
                       'source':'moshi_inference', 'state_encoding':'int8-tanh-127',
                       'state':base64.b64encode(np.rint(self.reservoir.state*127).astype(np.int8).tobytes()).decode('ascii')})
        self.latest = header
        return self.delta


def install_mlx_head(model, circuit):
    """Intercept the existing text head; do not fork or patch installed Moshi.

    Both text sampling and audio depformer conditioning remain upstream Moshi.
    The graph is evaluated before text sampling, regardless of browser state.
    """
    import mlx.core as mx
    import mlx.nn as nn

    class CircuitHead(nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head

        def __call__(self, hidden):
            baseline = self.head(hidden)
            values = np.array(hidden.astype(mx.float32)).reshape(-1)
            delta = circuit.advance(values)
            if circuit.calibration or circuit.mode in ('base','no_edges'):
                circuit.last_logit_rms = 0.
                return baseline
            correction = self.head(mx.array(delta).reshape(hidden.shape).astype(hidden.dtype)).astype(mx.float32)
            # Centering removes a softmax-invariant constant. Bound the actual
            # score perturbation, not only the hidden-state projection.
            correction = correction - mx.mean(correction, axis=-1, keepdims=True)
            rms = mx.sqrt(mx.mean(correction**2, axis=-1, keepdims=True))
            correction = correction * mx.minimum(1., INTERFACE['max_logit_rms']/(rms+1e-8))
            circuit.last_logit_rms = float(mx.sqrt(mx.mean(correction**2)).item())
            return baseline.astype(mx.float32) + correction

    model.text_linear = CircuitHead(model.text_linear)
