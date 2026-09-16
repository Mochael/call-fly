"""Numerical direction, no-edges control and real transport lifecycle."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from server import connectome as module


def graph(matrix):
    return SimpleNamespace(matrix=sparse.csr_matrix(matrix, dtype=np.float32), ids=np.arange(len(matrix)))


def test_recurrence_uses_presynaptic_state_and_all_edges():
    # Only 0 -> 1 -> 2. This distinguishes incoming vs transposed propagation.
    r = module.Reservoir(graph([[0,0,0],[1,0,0],[0,1,0]]))
    r.state[:] = [1,0,0]
    r.step(np.zeros(64))
    np.testing.assert_allclose(r.state, [0,np.tanh(.6),0], rtol=1e-6)
    r.step(np.zeros(64))
    np.testing.assert_allclose(r.state, [0,0,np.tanh(.6*np.tanh(.6))], rtol=1e-6)
    r.step(np.zeros(64))
    assert not np.any(r.state)


def test_seeded_projection_reset_and_disconnection():
    g=graph([[.2,.8],[.7,.3]])
    a,b=module.Reservoir(g),module.Reservoir(g)
    values=np.linspace(-1,1,64)
    a.step(values); b.step(values)
    np.testing.assert_array_equal(a.state,b.state)
    assert np.any(a.state)
    a.reset()
    assert not np.any(a.state) and a.updates==0
    disconnected=module.Reservoir(graph(np.zeros((2,2))))
    disconnected.step(values, pulse=0)
    assert not np.any(disconnected.state)


@pytest.mark.parametrize('value', [{'features':[1]}, {'features':[float('nan')]*64}, {'features':[2]*64}, {'pulse':9}])
def test_invalid_input(value):
    with pytest.raises(ValueError): module.parse_input(value)


