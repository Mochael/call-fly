"""Same audio and RNG seed with intact vs physically disconnected graph."""
import json
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import numpy as np
from scipy import sparse
import sphn
from server.runtime import MoshiEngine, FRAME

parser=argparse.ArgumentParser();parser.add_argument('--audio',default='artifacts/browser-microphone.wav');parser.add_argument('--frames',type=int,default=160);parser.add_argument('--output',default='artifacts/moshi/reservoir-causality.json');args=parser.parse_args()
engine=MoshiEngine();engine.initialize()
pcm,_=sphn.read(args.audio,sample_rate=24000)
pcm=np.pad(pcm[0],(0,FRAME*args.frames))
graph=engine.circuit.reservoir.graph
reports={};outputs={}
for mode in ['intact','no_edges']:
    engine.begin(seed=90210)
    engine.circuit.reservoir.graph=graph if mode=='intact' else SimpleNamespace(matrix=sparse.csr_matrix(graph.matrix.shape,dtype=np.float32))
    scores=[];text=[];audio=[];times=[]
    for i in range(args.frames):
        result=engine.step(pcm[i*FRAME:(i+1)*FRAME])
        text.append(result['text']);scores.append(result['reservoir_logit_rms']);times.append(result['compute_ms'])
        if result['audio'] is not None:audio.append(result['audio'])
    reports[mode]={'text':''.join(text),'mean_logit_delta_rms':float(np.mean(scores)),
                   'max_logit_delta_rms':float(np.max(scores)),'mean_frame_ms':float(np.mean(times)),
                   'p95_frame_ms':float(np.percentile(times,95)),
                   'gpu':getattr(engine,'device_name',None)}
    outputs[mode]=np.concatenate(audio)
    print(mode,reports[mode],flush=True)
    engine.end()
assert reports['intact']['mean_logit_delta_rms']>1e-5
assert reports['no_edges']['max_logit_delta_rms']==0
reports['audio_changed']=bool(np.any(outputs['intact']!=outputs['no_edges']))
reports['text_changed']=reports['intact']['text']!=reports['no_edges']['text']
# Sampling may yield the same tokens despite a nonzero score correction.
# Report this honestly; probability-level causality is checked above.
target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(reports,indent=2))
print(json.dumps(reports,indent=2))
