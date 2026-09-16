"""Bounded real-GPU concurrency probe. Checks worker isolation or explicitly requested shared-GPU capacity.

Run from the repo root through the local gateway. Uses prerecorded test speech;
reports hashes and timings, never records transcripts or microphone content.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import sphn
import websockets


async def probe(url, pcm, caller, seconds, barrier=None):
    count=int(seconds*12.5)
    output_hash=hashlib.sha256();brain_hash=hashlib.sha256()
    sent={};delays=[];metrics=[];brain_frames=[];rms=[];received=0;audio_samples=0;peak=0.
    started=time.perf_counter();ready_at=None
    async with websockets.connect(url,open_timeout=20,max_size=2**20,max_queue=16) as ws:
        async with asyncio.timeout(210):
            while True:
                event=json.loads(await ws.recv())
                if event['type']=='ready':break
                assert event['type']=='warming',event
        ready_at=time.perf_counter()
        assert event['execution']=='torch' and event['codec']=='torch-cuda',event
        assert event['worker_id']!='local',event
        print(json.dumps({'caller':caller,'worker':event['worker_id'],'ready_s':round(ready_at-started,2)}),flush=True)
        if barrier:await asyncio.wait_for(barrier.wait(),timeout=210)
        active_at=time.perf_counter()
        async def send():
            # Offset the speech so callers do not all supply identical input.
            samples=np.roll(pcm,caller*7200)
            for i in range(count):
                await asyncio.sleep(max(0,active_at+i*.08-time.perf_counter()))
                sent[i+1]=time.perf_counter()
                await ws.send(samples[i*1920:(i+1)*1920].astype('<f4').tobytes())
        async def receive():
            nonlocal received,audio_samples,peak
            while received<count:
                payload=await ws.recv()
                if isinstance(payload,bytes):
                    audio=np.frombuffer(payload,dtype='<f4')
                    assert len(audio)==1920 and np.isfinite(audio).all()
                    output_hash.update(payload);audio_samples+=len(audio);received+=1
                    peak=max(peak,float(np.max(np.abs(audio))))
                    continue
                data=json.loads(payload)
                assert data['type'] not in ('error','session_end'),data
                if data['type']=='frame':
                    assert data['index']==received+1,data['index']
                    delays.append((time.perf_counter()-sent[data['index']])*1000)
                    rms.append(data['reservoir_logit_rms'])
                    if data.get('brain'):
                        brain_frames.append(data['brain']['model_frame'])
                        brain_hash.update(json.dumps(data['brain'],sort_keys=True).encode())
                elif data['type']=='metrics':metrics.append(data)
        await asyncio.wait_for(asyncio.gather(send(),receive()),timeout=seconds+30)
        await ws.send('stop')
        # Wait for cleanup and socket closure before starting another call.
        async with asyncio.timeout(10):
            async for payload in ws:
                if isinstance(payload,str):
                    data=json.loads(payload)
                    assert data['type']!='error',data
    assert received==count and audio_samples==count*1920
    assert peak>.004 and max(rms)>0
    assert brain_frames and brain_frames[0]<=4,brain_frames[:3]
    assert delays[-1]<1000,delays[-1]
    return {'caller':caller,'call_id':event['call_id'],'worker_id':event['worker_id'],
        'session_slot':event.get('session_slot'),'session_capacity':event.get('session_capacity'),
        'serving':event.get('serving'),'startup_s':round(ready_at-started,3),'active_start':active_at,'active_end':time.perf_counter(),
        'frames':received,'audio_s':audio_samples/24000,'audio_peak':peak,
        'audio_sha256':output_hash.hexdigest(),'brain_sha256':brain_hash.hexdigest(),
        'brain_updates':len(brain_frames),'first_brain_frame':brain_frames[0],
        'mean_latency_ms':float(np.mean(delays)),'p95_latency_ms':float(np.percentile(delays,95)),
        'last_latency_ms':delays[-1],'last_metrics':metrics[-1] if metrics else None}


async def main(args):
    # Each later caller stays connected longer, proving an earlier stop doesn't
    # stop its peers. Start together after every worker has become ready.
    max_seconds=args.seconds+(args.calls-1)*2
    pcm,_=sphn.read('artifacts/browser-microphone.wav',sample_rate=24000)
    pcm=np.resize(pcm[0],int((max_seconds+1)*24000))
    barrier=asyncio.Barrier(args.calls)
    tasks=[asyncio.create_task(probe(args.url,pcm,i,args.seconds+2*i,barrier)) for i in range(args.calls)]
    try:
        calls=await asyncio.gather(*tasks)
    finally:
        for task in tasks:task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
    workers={c['worker_id'] for c in calls}
    if args.max_workers:
        assert len(workers)<=args.max_workers,(len(workers),args.max_workers)
        assert all(c['serving']=='upstream-masked-streaming' for c in calls)
        assert len({(c['worker_id'],c['session_slot']) for c in calls})==args.calls
    else:
        assert len(workers)==args.calls,'Calls shared a worker'
    assert len({c['call_id'] for c in calls})==args.calls
    assert len({c['audio_sha256'] for c in calls})==args.calls
    assert len({c['brain_sha256'] for c in calls})==args.calls
    overlap=min(c['active_end'] for c in calls)-max(c['active_start'] for c in calls)
    assert overlap>=args.seconds-1
    restarted=await probe(args.url,pcm,100,8)
    assert restarted['call_id'] not in {c['call_id'] for c in calls}
    assert restarted['worker_id'] in {c['worker_id'] for c in calls},'Expected an existing worker to be reused'
    result={'passed':True,'simultaneous_calls':args.calls,'gpu_workers':len(workers),'overlap_s':overlap,'calls':calls,'restart':restarted}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--calls',type=int,default=3,choices=range(2,11))
    parser.add_argument('--seconds',type=int,default=24)
    parser.add_argument('--max-workers',type=int,help='Require shared serving on at most this many GPUs')
    parser.add_argument('--url',default='ws://localhost:8765/api/conversation')
    parser.add_argument('--output',default='artifacts/moshi/concurrent-calls.json')
    asyncio.run(main(parser.parse_args()))
