"""Offline teacher-state calibration of the FLM-style readout (no call learning).

Use generated test speech, never a visitor recording. Bias-free ridge regression
reconstructs the frozen model's hidden state from pooled reservoir activity.
This establishes a trained causal prototype, not a conversational improvement.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import sphn
from server.runtime import MoshiEngine, MODEL, REVISION, FRAME, ADAPTER
from server.model_reservoir import DEFAULT_ADAPTER, INTERFACE, FOLDER, digest


def main(args):
    engine=MoshiEngine(calibration=True)
    engine.initialize()
    datasets=[]
    for run, name in enumerate(args.audio):
        pcm,_=sphn.read(name,sample_rate=24000)
        pcm=pcm.mean(axis=0)
        engine.begin(seed=1000+run)
        rows=[];targets=[]
        limit=min(len(pcm)//FRAME,int(args.seconds*12.5))
        for i in range(limit):
            engine.generate(engine.encode(pcm[i*FRAME:(i+1)*FRAME]))
            c=engine.circuit
            if c.latest:
                x,y=c.training_sample;rows.append(x);targets.append(y)
            if i%125==0: print(f'Calibration clip {run+1}: frame {i}/{limit}',flush=True)
        datasets.append((np.asarray(rows),np.asarray(targets)))
        engine.end()
    if len(datasets)<2: raise ValueError('Supply independent training and validation clips.')
    x=np.concatenate([d[0] for d in datasets[:-1]]).astype(np.float64)
    y=np.concatenate([d[1] for d in datasets[:-1]]).astype(np.float64)
    vx,vy=datasets[-1]
    if len(x)<128 or len(vx)<20: raise ValueError('Insufficient calibration examples.')
    ridge=args.ridge
    weights=np.linalg.solve(x.T@x+ridge*np.eye(128),x.T@y).astype(np.float32)
    pred=vx@weights
    report={'training_samples':len(x),'validation_samples':len(vx),'ridge':ridge,
            'validation_mse':float(np.mean((pred-vy)**2)),
            'zero_baseline_mse':float(np.mean(vy**2)),
            'mean_baseline_mse':float(np.mean((y.mean(axis=0)-vy)**2)),
            'objective':'bias-free ridge reconstruction of frozen Moshi hidden states',
            'quality_claim':'causal prototype only; no demonstrated conversational improvement',
            'audio_sha256':[digest(p) for p in args.audio]}
    if not np.isfinite(weights).all() or report['validation_mse']>=report['zero_baseline_mse']:
        raise ValueError(f'Calibration failed validation: {report}')
    target=ADAPTER
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and not args.replace: raise ValueError('Adapter exists; use --replace explicitly.')
    np.savez(target,weights=weights)
    meta={'model':MODEL,'revision':REVISION,'interface':INTERFACE,
          'graph_sha256':digest(FOLDER/'manifest.json'),'adapter_sha256':digest(target),'calibration':report}
    target.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audio',nargs='+',required=True)
    parser.add_argument('--seconds',type=int,default=60);parser.add_argument('--ridge',type=float,default=100.)
    parser.add_argument('--replace',action='store_true');main(parser.parse_args())
