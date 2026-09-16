"""Isolated evaluation of upstream Moshi shared streaming; not production routing."""
import modal
if modal.is_local():
    from deploy.modal_app import image, volume
else:
    image=modal.Image.debian_slim()
    volume=modal.Volume.from_name('call-fly-model-data')

app=modal.App('call-fly-shared-preview')

@app.function(image=image,cpu=8,memory=16384,gpu='L40S',timeout=900,
              volumes={'/app/.runtime/moshi-reservoir':volume})
def benchmark(capacity:int=2,steps:int=125):
    import json,time
    import numpy as np
    from server.moshi_batch_engine import SharedMoshiEngine
    from server.moshi_torch_engine import FRAME
    engine=SharedMoshiEngine(capacity)
    t=time.perf_counter();engine.initialize();startup=time.perf_counter()-t
    pcm=np.zeros((capacity,FRAME),np.float32)
    mask=np.ones(capacity,bool)
    engine.step_batch(pcm,mask) # per-session priming
    timings=[]
    for _ in range(steps):
        result=engine.step_batch(pcm,mask)
        assert all(r is not None and np.isfinite(r['audio']).all() for r in result)
        timings.append(engine.last_batch_ms)
    # Verify a masked caller's model offsets and brain state do not change.
    before=engine.gen._streaming_state.offsets.cpu().numpy().copy()
    brain=engine.circuits[1].reservoir.state.copy()
    mask[1]=False
    for _ in range(4):engine.step_batch(pcm,mask)
    after=engine.gen._streaming_state.offsets.cpu().numpy()
    assert after[1]==before[1] and after[0]==before[0]+4
    assert np.array_equal(brain,engine.circuits[1].reservoir.state)
    # Reset/rejoin one caller while a peer continues with unchanged offsets.
    old=engine.gen._streaming_state.offsets.cpu().numpy().copy()
    engine.reset_slots([1])
    assert engine.gen._streaming_state.offsets[0].item()==old[0]
    assert engine.circuits[1].frame==0
    mask[1]=True
    result=engine.step_batch(pcm,mask)
    assert result[0] is not None and result[1] is None
    result=engine.step_batch(pcm,mask)
    assert result[0] is not None and result[1] is not None
    report={'capacity':capacity,'gpu':engine.device_name,'startup_s':startup,
            'mean_batch_ms':float(np.mean(timings)),'p95_batch_ms':float(np.percentile(timings,95)),
            'max_batch_ms':max(timings),'model_memory_gb':engine.torch.cuda.memory_allocated()/1e9,
            'mask_and_reset_passed':True}
    engine.close()
    print(json.dumps(report),flush=True)
    return report

@app.function(image=image.env({'MOSHI_SESSION_CAPACITY':'4'}),cpu=8,memory=16384,
              gpu='L40S',timeout=600,region='us',routing_region='us-west',
              min_containers=0,max_containers=1,scaledown_window=120,
              volumes={'/app/.runtime/moshi-reservoir':volume},
              secrets=[modal.Secret.from_name('call-fly-service')])
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def shared():
    from server.moshi_shared_app import app
    return app

@app.function(image=image,cpu=8,memory=16384,gpu='L40S',timeout=900,
              volumes={'/app/.runtime/moshi-reservoir':volume})
def isolation():
    import numpy as np
    import sphn
    from types import SimpleNamespace
    from scipy import sparse
    from server.moshi_batch_engine import SharedMoshiEngine
    from server.moshi_torch_engine import FRAME
    engine=SharedMoshiEngine(4,use_sampling=False);engine.initialize()
    audio,_=sphn.read('/app/calibration/clip-2.wav',sample_rate=24000)
    audio=np.resize(audio[0],FRAME*32)
    def run(change_peer=False):
        engine.reset_slots(range(4))
        pcm=np.zeros((4,FRAME),np.float32);mask=np.ones(4,bool)
        engine.step_batch(pcm,mask)
        outputs=[];features=[]
        for i in range(24):
            pcm[0]=audio[i*FRAME:(i+1)*FRAME]
            pcm[1]=audio[(23-i)*FRAME:(24-i)*FRAME] if change_peer else 0
            if change_peer and i==10:engine.reset_slots([1])
            mask[2]=not change_peer or i%3==0
            result=engine.step_batch(pcm,mask)
            assert result[0] is not None
            outputs.append(result[0]['audio']);features.append(result[0]['features'])
        return np.stack(outputs),np.array(features)
    a,af=run();b,bf=run(True)
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-5)
    np.testing.assert_allclose(af,bf,rtol=0,atol=1e-6)
    # Physically disconnect only row 1's graph, retaining the same readout.
    engine.reset_slots(range(4))
    original=engine.circuits[1].reservoir.graph
    engine.circuits[1].reservoir.graph=SimpleNamespace(matrix=sparse.csr_matrix(original.matrix.shape,dtype=np.float32))
    pcm=np.tile(audio[:FRAME],(4,1));mask=np.ones(4,bool)
    engine.step_batch(pcm,mask);result=engine.step_batch(pcm,mask)
    assert result[1]['reservoir_logit_rms']==0
    assert all(result[i]['reservoir_logit_rms']>0 for i in (0,2,3))
    report={'passed':True,'peer_audio_max_difference':float(np.max(np.abs(a-b))),
            'peer_features_max_difference':float(np.max(np.abs(af-bf))),
            'per_row_graph_ablation_logit_rms':[r['reservoir_logit_rms'] for r in result]}
    engine.close()
    return report

@app.local_entrypoint()
def main(capacity:int=2,steps:int=125,mode:str="benchmark"):
    import json
    from pathlib import Path
    result=isolation.remote() if mode=='isolation' else benchmark.remote(capacity,steps)
    output='shared-isolation' if mode=='isolation' else f'shared-benchmark-{capacity}'
    Path(f'artifacts/moshi/{output}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
