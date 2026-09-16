import base64
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from server.model_reservoir import ModelCircuit, ModelReservoir, summarize, install_mlx_head


def circuit():
    graph=SimpleNamespace(matrix=sparse.csr_matrix([[.2,.8],[.7,.3]],dtype=np.float32))
    c=ModelCircuit(graph,'test','test',calibration=True)
    c.calibration=False
    c.weights=np.random.default_rng(8).normal(0,.05,(128,4096)).astype(np.float32)
    return c


def test_disconnection_removes_readout_and_reset_removes_memory():
    c=circuit();hidden=np.random.default_rng(3).normal(size=4096).astype(np.float32)
    assert np.any(c.advance(hidden))
    assert c.latest['source']=='moshi_inference'
    packed=np.frombuffer(base64.b64decode(c.latest['state']),dtype=np.int8)/127
    np.testing.assert_allclose(packed,c.reservoir.state,atol=1/127)
    c.reset();c.reservoir.graph.matrix.data.fill(0)
    # Keep mode intact: actual edge removal, not an adapter bypass.
    np.testing.assert_array_equal(c.advance(hidden),np.zeros(4096))
    assert c.reservoir.updates==1
    c.reset();assert c.frame==0 and c.reservoir.updates==0 and not c.delta.any()


def test_rate_is_model_frame_driven_and_bias_free_pooling():
    c=circuit();h=np.ones(4096,dtype=np.float32)
    for _ in range(7):c.advance(h)
    assert c.reservoir.updates==3
    c.reservoir.state.fill(0)
    assert not c.reservoir.pooled().any()


def test_mlx_hook_changes_scores_before_sampling_and_no_edges_matches_base():
    import mlx.core as mx
    import mlx.nn as nn
    mx.random.seed(42)
    model=SimpleNamespace(text_linear=nn.Linear(4096,32,bias=False))
    hidden=mx.array(np.random.default_rng(9).normal(size=(1,1,4096)).astype(np.float32))
    original=model.text_linear(hidden)
    c=circuit();install_mlx_head(model,c)
    adjusted=model.text_linear(hidden)
    delta=np.array(adjusted-original)
    assert np.sqrt(np.mean(delta**2))>1e-6
    assert np.sqrt(np.mean(delta**2))<=.10001
    c.reset();c.reservoir.graph.matrix.data.fill(0)
    np.testing.assert_array_equal(np.array(model.text_linear(hidden)),np.array(original))
