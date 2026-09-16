"""Full MaleCNS graph and FLM recurrence used inside Moshi inference.

Topology/recurrence adapted from nftechie/flm, MIT (Alex Wormuth, 2026).
All retained edges participate in each reservoir update.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

FOLDER = Path(__file__).resolve().parents[1] / '.runtime' / 'connectome'
router = APIRouter()
_graph = None


class Graph:
    def __init__(self, folder=FOLDER):
        folder = Path(folder)
        self.manifest = json.loads((folder / 'manifest.json').read_text())
        for name, digest in {**self.manifest['arrays'], **self.manifest['geometry_files']}.items():
            with open(folder / name, 'rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise ValueError(f'Connectome integrity check failed: {name}')
        self.ids = np.load(folder / 'ids.npy', mmap_mode='r')
        arrays = [np.load(folder / f'{name}.npy', mmap_mode='r') for name in ['data', 'indices', 'indptr']]
        self.matrix = sparse.csr_matrix(tuple(arrays), shape=(len(self.ids), len(self.ids)), copy=False)
        if self.ids.dtype != np.int64 or np.any(np.diff(self.ids) <= 0):
            raise ValueError('Neuron IDs must be sorted exact int64 values.')
        if self.matrix.nnz != self.manifest['directed_edges'] or len(self.ids) != self.manifest['neurons']:
            raise ValueError('Connectome count mismatch.')


class Reservoir:
    def __init__(self, graph, seed=7301):
        self.graph = graph
        rng = np.random.default_rng(seed)
        # Same seeded interface construction as FLM, with 64 Moshi features as input.
        self.projection = (rng.standard_normal((64, 128)) / 8).astype(np.float32)
        self.bins = rng.integers(0, 128, graph.matrix.shape[0])
        self.signs = rng.choice(np.array([-1, 1], np.float32), graph.matrix.shape[0])
        self.reset()

    def reset(self):
        self.state = np.zeros(self.graph.matrix.shape[0], np.float32)
        self.updates = 0

    def step(self, features, pulse=-1):
        started = time.perf_counter()
        code = np.asarray(features, np.float32) @ self.projection
        code /= np.sqrt(np.mean(code * code) + 1e-6)
        if pulse >= 0:
            code[pulse] += 2.0
        drive = .6 * self.state + .4 * code[self.bins] * self.signs
        self.state = np.tanh(self.graph.matrix @ drive)
        self.updates += 1
        return self.snapshot((time.perf_counter() - started) * 1000)

    def snapshot(self, compute_ms=0):
        return ({'type': 'frame', 'updates': self.updates,
                 'compute_ms': round(compute_ms, 2), 'neurons': len(self.state),
                 'active': int(np.count_nonzero(np.abs(self.state) > .001)),
                 'rms': float(np.sqrt(np.mean(self.state * self.state))),
                 'peak': float(np.max(np.abs(self.state)))},
                self.state.astype('<f4', copy=False).tobytes())


def get_graph():
    global _graph
    if _graph is None:
        _graph = Graph()
    return _graph


def parse_input(message):
    features = message.get('features', [])
    if not isinstance(features, list) or len(features) not in (0, 64):
        raise ValueError('Expected 64 model features.')
    values = np.asarray(features or [0.] * 64, dtype=np.float32)
    if not np.isfinite(values).all() or np.max(np.abs(values)) > 1.01:
        raise ValueError('Invalid model features.')
    pulse = message.get('pulse', -1)
    if type(pulse) is not int or pulse not in (-1, 0, 1):
        raise ValueError('Invalid test pulse.')
    return values, pulse


@router.get('/api/connectome/{asset}')
async def asset(asset: str):
    if asset not in {'manifest.json', 'metadata.json', 'positions.bin'}:
        raise HTTPException(404)
    path = FOLDER / asset
    if not path.is_file():
        raise HTTPException(503, 'Run scripts/setup-connectome.sh to prepare the full connectome.')
    return FileResponse(path, media_type='application/octet-stream' if asset.endswith('.bin') else 'application/json')


