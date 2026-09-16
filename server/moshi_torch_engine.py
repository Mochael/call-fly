"""NVIDIA deployment backend, pinned to Kyutai's native PyTorch implementation.

The reservoir runs outside CUDA graph capture, between temporal prediction and
text sampling. Keep the upstream depformer graph and sampling implementation.
Requires a separately calibrated bf16 readout; MLX q4 artifacts are rejected.
"""
import time
import numpy as np
from .moshi_engine import ROOT, RATE, FRAME, MAX_STEPS
from .model_reservoir import ModelCircuit, DEFAULT_ADAPTER, INTERFACE, summarize
from .connectome import get_graph

MODEL='kyutai/moshiko-pytorch-bf16'
REVISION='2bfc9ae6e89079a5cc7ed2a68436010d91a3d289'
ADAPTER=DEFAULT_ADAPTER.parent/'torch'/'adapter.npz'


class MoshiEngine:
    def __init__(self,calibration=False):
        self.calibration=calibration
        self.ready=False;self.stage='Loading Moshi…';self.error=None
        self.gen=None;self.codec=None;self.circuit=None;self.stream=None;self.codec_stream=None
        self.codec_backend='torch-cuda'
        self.hidden=None

    def initialize(self):
        self.load_model()
        self.stage='Warming up live speech…'
        self.begin(seed=299792458)
        self.end()
        self.ready=True;self.stage='Ready for a live conversation.'

    def load_model(self):
        """Load shared immutable weights; streaming caches are allocated separately."""
        import torch
        from moshi.models import loaders
        self.torch=torch
        if not torch.cuda.is_available():raise RuntimeError('The deployed voice backend requires an NVIDIA GPU.')
        torch.set_num_threads(2)
        self.device_name=torch.cuda.get_device_name(0)
        self.info=loaders.CheckpointInfo.from_hf_repo(MODEL,revision=REVISION)
        self.stage='Loading the duplex speech model…'
        self.model=self.info.get_moshi(device='cuda',dtype=torch.bfloat16).eval()
        self.model.requires_grad_(False)
        self.tokenizer=self.info.get_text_tokenizer()
        self.codec_model=self.info.get_mimi(device='cuda').eval()
        self.codec_model.requires_grad_(False)
        self.circuit=ModelCircuit(get_graph(),MODEL,REVISION,adapter=ADAPTER,calibration=self.calibration)

    def begin(self,seed=None):
        from moshi.models import LMGen
        self.end()
        self.torch.manual_seed(seed if seed is not None else time.time_ns()%(2**32))
        self.codec=self.codec_model
        with self.torch.inference_mode():
            self.codec_stream=self.codec.streaming(1)
            self.codec_stream.__enter__()
            self.gen=LMGen(self.model,temp=.8,temp_text=.8,top_k=250,top_k_text=25)
            self.stream=self.gen.streaming(1)
            self.stream.__enter__()
        # Pinned Moshi API: wrapper executes after CUDA graph replay and before
        # LMGen samples text. CPU recurrence is never captured into a CUDA graph.
        state=self.gen._streaming_state
        original=state.graphed_main
        def coupled(*args):
            hidden,logits=original(*args)
            values=hidden.float().cpu().numpy().reshape(-1)
            self.hidden=values
            delta=self.circuit.advance(values)
            if self.calibration or self.circuit.mode in ('base','no_edges'):
                self.circuit.last_logit_rms=0.
                return hidden,logits
            torch=self.torch
            residual=torch.as_tensor(delta,device=hidden.device,dtype=hidden.dtype).reshape(hidden.shape)
            change=self.model.text_linear(residual).float().unsqueeze(1)
            change=change-change.mean(dim=-1,keepdim=True)
            rms=change.square().mean(dim=-1,keepdim=True).sqrt()
            change=change*torch.clamp(INTERFACE['max_logit_rms']/(rms+1e-8),max=1.)
            self.circuit.last_logit_rms=float(change.square().mean().sqrt().item())
            return hidden,logits.float()+change
        state.graphed_main=coupled
        # New streaming contexts own new CUDA graph wrappers. Their first
        # captures take much longer than a live frame; complete them before
        # announcing readiness, then reset caches without discarding graphs.
        for _ in range(4):self.step(np.zeros(FRAME,np.float32))
        with self.torch.inference_mode():
            self.gen.reset_streaming()
            self.codec.reset_streaming()
        self.circuit.reset();self.hidden=None

    def end(self):
        if self.stream is not None:
            self.stream.close();self.stream=None
        if self.codec_stream is not None:
            self.codec_stream.close();self.codec_stream=None
        self.gen=None
        self.codec=None;self.hidden=None
        if self.circuit is not None:self.circuit.reset()

    def encode(self,pcm):
        with self.torch.inference_mode():
            frame=self.torch.as_tensor(np.asarray(pcm,np.float32).copy(),device='cuda').reshape(1,1,FRAME)
            return self.codec.encode(frame)

    def decode(self,codes):
        with self.torch.inference_mode():
            return self.codec.decode(codes).float().cpu().numpy().reshape(-1)

    def generate(self,codes):
        started=time.perf_counter()
        with self.torch.inference_mode():
            out=self.gen.step(codes)
            token=int(out[0,0,0].item()) if out is not None else 0
            # LMGen reuses buffers; the decoder may be queued behind another
            # encoder operation, so retain this frame's own code tensor.
            audio_codes=out[:,1:,:].clone() if out is not None else None
        text=self.tokenizer.id_to_piece(token).replace('▁',' ') if token not in (0,3) else ''
        return {'audio_codes':audio_codes,'text':text,'features':summarize(self.hidden).tolist(),
                'brain':self.circuit.latest,'reservoir_logit_rms':self.circuit.last_logit_rms,
                'step':self.circuit.frame,'compute_ms':(time.perf_counter()-started)*1000}

    def step(self,pcm):
        result=self.generate(self.encode(pcm));codes=result.pop('audio_codes')
        result['audio']=self.decode(codes) if codes is not None else None
        return result
