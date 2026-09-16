"""Shared Moshi weights using upstream masked streaming and per-row resets.

Pinned Kyutai API: LMGen(support_out_of_sync=True), streaming(B),
set_exec_mask(mask), reset_streaming(reset_mask). No upstream source patches.
Only the existing connectome hook runs between the temporal model and sampling.
All public methods must execute on one inference thread (CUDA graph ownership).
"""
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np

from .moshi_torch_engine import MoshiEngine, MODEL, REVISION, ADAPTER, FRAME
from .model_reservoir import ModelCircuit, INTERFACE, summarize


class SharedMoshiEngine(MoshiEngine):
    def __init__(self, capacity=2, use_sampling=True):
        super().__init__()
        if capacity not in (2,4,8):raise ValueError('Supported capacities: 2, 4, 8')
        self.capacity=capacity
        self.use_sampling=use_sampling
        self.active_mask=np.zeros(capacity,dtype=bool)
        self.circuits=[]
        self.graph_workers=ThreadPoolExecutor(max_workers=capacity,thread_name_prefix='connectome')
        self.last_batch_ms=0.

    def initialize(self):
        from moshi.models import LMGen
        self.load_model()
        torch=self.torch
        self.circuits=[self.circuit]+[ModelCircuit(self.circuit.reservoir.graph,MODEL,REVISION,adapter=ADAPTER)
                                     for _ in range(self.capacity-1)]
        self.hidden_rows=np.zeros((self.capacity,4096),np.float32)
        self.codec=self.codec_model
        # The out-of-sync flag ensures resetting one row cannot suppress output
        # for all other rows through LMGen's global startup delay counter.
        self.gen=LMGen(self.model,use_sampling=self.use_sampling,temp=.8,temp_text=.8,top_k=250,top_k_text=25,support_out_of_sync=True)
        with torch.inference_mode():
            self.codec_stream=self.codec.streaming(self.capacity);self.codec_stream.__enter__()
            self.stream=self.gen.streaming(self.capacity);self.stream.__enter__()
        state=self.gen._streaming_state
        original=state.graphed_main

        def coupled(*args):
            hidden,logits=original(*args)
            values=hidden.float().cpu().numpy().reshape(self.capacity,4096)
            deltas=np.zeros_like(values)
            # Read-only topology/weights are shared. Every row owns its recurrence,
            # cached readout and counters. Inactive rows must not advance.
            indices=np.flatnonzero(self.active_mask)
            jobs=[(int(i),self.graph_workers.submit(self.circuits[i].advance,values[i])) for i in indices]
            for i,job in jobs:
                deltas[i]=job.result();self.hidden_rows[i]=values[i]
            residual=torch.as_tensor(deltas,device=hidden.device,dtype=hidden.dtype).reshape(hidden.shape)
            change=self.model.text_linear(residual).float().unsqueeze(1)
            change-=change.mean(dim=-1,keepdim=True)
            rms=change.square().mean(dim=-1,keepdim=True).sqrt()
            change*=torch.clamp(INTERFACE['max_logit_rms']/(rms+1e-8),max=1.)
            magnitudes=change.square().mean(dim=-1).sqrt().reshape(self.capacity).cpu().numpy()
            for i in indices:self.circuits[i].last_logit_rms=float(magnitudes[i])
            return hidden,logits.float()+change

        state.graphed_main=coupled
        self.stage='Warming shared speech sessions…'
        # Capture both active and masked paths once; subsequent joins reuse the
        # same graphs. Reset only caches, never the loaded weights or graphs.
        zeros=np.zeros((self.capacity,FRAME),np.float32)
        for _ in range(4):self.step_batch(zeros,np.ones(self.capacity,dtype=bool))
        self.step_batch(zeros,np.zeros(self.capacity,dtype=bool))
        self.reset_slots(range(self.capacity))
        self.ready=True;self.stage='Ready for live conversations.'

    def reset_slots(self,slots):
        mask=np.zeros(self.capacity,dtype=bool)
        for slot in slots:mask[slot]=True
        if not mask.any():return
        with self.torch.inference_mode():
            reset=self.torch.as_tensor(mask,device='cuda')
            self.gen.reset_streaming(reset)
            self.codec.reset_streaming(reset)
        for i in np.flatnonzero(mask):
            self.circuits[i].reset();self.hidden_rows[i].fill(0)

    def step_batch(self,pcm,active):
        torch=self.torch
        started=time.perf_counter()
        self.active_mask=np.asarray(active,dtype=bool).copy()
        with torch.inference_mode():
            mask=torch.as_tensor(self.active_mask,device='cuda')
            self.codec.set_exec_mask(mask)
            self.gen.set_exec_mask(mask)
            frame=torch.as_tensor(np.asarray(pcm,np.float32),device='cuda').reshape(self.capacity,1,FRAME)
            codes=self.codec.encode(frame)
            out=self.gen.step(codes)
            # Upstream marks unprimed/inactive rows with negative token IDs.
            # Never pass those into Mimi's codebook lookup.
            assert out is not None
            valid=(out>=0).all(dim=(1,2)) & mask
            self.codec.set_exec_mask(valid)
            audio=self.codec.decode(out[:,1:,:].clamp_min(0)).float().cpu().numpy()[:,0,:]
            tokens=out[:,0,0].cpu().numpy()
            valid_rows=valid.cpu().numpy()
        elapsed=(time.perf_counter()-started)*1000
        self.last_batch_ms=elapsed
        results=[]
        for i in range(self.capacity):
            if not valid_rows[i]:results.append(None);continue
            token=int(tokens[i]);circuit=self.circuits[i]
            results.append({'audio':audio[i].copy(),'text':self.tokenizer.id_to_piece(token).replace('▁',' ') if token not in (0,3) else '',
                            'features':summarize(self.hidden_rows[i]).tolist(),'brain':circuit.latest,
                            'reservoir_logit_rms':circuit.last_logit_rms,'compute_ms':elapsed,
                            'step':circuit.frame})
        return results

    def close(self):
        super().end()
        self.graph_workers.shutdown(wait=True)
