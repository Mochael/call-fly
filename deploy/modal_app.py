"""Explicitly invoked GPU deployment; importing this file provisions nothing.

Preparation: modal run deploy/modal_app.py::calibrate
Serving: modal deploy deploy/modal_app.py
Four independent Moshi sessions per GPU; at most three GPUs, one kept warm.
The stable web_us endpoint is a CPU gateway; diagnostics never consume GPU slots.
"""
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[1]
app=modal.App('call-fly-voice')
SERVING_GPU='L40S'  # L4 measured ~91 ms inference per 80 ms live frame.
volume=modal.Volume.from_name('call-fly-model-data',create_if_missing=True)

def download():
    from huggingface_hub import snapshot_download
    snapshot_download('kyutai/moshiko-pytorch-bf16',revision='2bfc9ae6e89079a5cc7ed2a68436010d91a3d289')

image=(modal.Image.debian_slim(python_version='3.12')
       .apt_install('git','libportaudio2','libgomp1')
       .pip_install('torch==2.8.0',index_url='https://download.pytorch.org/whl/cu128')
       .pip_install_from_requirements(str(ROOT/'deploy/requirements.txt'))
       .env({'MOSHI_BACKEND':'torch','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'})
       .run_function(download)
       .env({'TORCHINDUCTOR_COMPILE_THREADS':'1'})
       .add_local_dir(ROOT/'server','/app/server',copy=True,ignore=['**/__pycache__/**','**/*.pyc'])
       .add_local_dir(ROOT/'scripts','/app/scripts',copy=True,ignore=['**/__pycache__/**','**/*.pyc'])
       .add_local_dir(ROOT/'dist','/app/dist',copy=True,ignore=['client/**','server/**','.openai/**'])
       .add_local_dir(ROOT/'.runtime/connectome','/app/.runtime/connectome',copy=True)
       .add_local_dir(ROOT/'artifacts/calibration','/app/calibration',copy=True)
       .workdir('/app'))

@app.function(image=image,gpu='L4',cpu=2,memory=8192,timeout=900,
              volumes={'/app/.runtime/moshi-reservoir':volume})
def calibrate():
    import subprocess
    subprocess.run(['python','scripts/train_moshi_reservoir.py','--audio',
                    'calibration/clip-0.wav','calibration/clip-1.wav','calibration/clip-2.wav','--seconds','55'],check=True)
    volume.commit()
    return {'calibrated':True}

# Legacy single-call endpoint retained for rollback; it has no warm GPU and
# the web gateway no longer routes new callers here.
@app.function(image=image,gpu=SERVING_GPU,cpu=2,memory=8192,timeout=600,
              region='us',routing_region='us-west',
              min_containers=0,max_containers=1,scaledown_window=60,
              volumes={'/app/.runtime/moshi-reservoir':volume},
              secrets=[modal.Secret.from_name('call-fly-service')])
@modal.asgi_app()
def call_worker():
    from server.moshi_app import app
    return app

# Upstream Moshi/Mimi share weights across four separately masked/reset rows.
# Modal counts each WebSocket as one input, matching the four session slots.
@app.function(image=image.env({'MOSHI_SESSION_CAPACITY':'4'}),gpu=SERVING_GPU,
              cpu=8,memory=16384,timeout=600,region='us',routing_region='us-west',
              min_containers=1,max_containers=3,scaledown_window=600,
              volumes={'/app/.runtime/moshi-reservoir':volume},
              secrets=[modal.Secret.from_name('call-fly-service')])
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def shared_worker():
    from server.moshi_shared_app import app
    return app

# This image deliberately excludes Torch, Moshi and model weights. Health,
# geometry, diagnostics and waiting connections must not occupy GPU workers.
gateway_image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install('fastapi==0.141.1','uvicorn==0.53.0','httpx==0.28.1','websockets==15.0.1')
    .env({'VOICE_REQUIRE_AUTH':'1'})
    .add_local_dir(ROOT/'server','/app/server',copy=True,ignore=['**/__pycache__/**','**/*.pyc'])
    .add_local_dir(ROOT/'dist','/app/dist',copy=True,ignore=['client/**','server/**','.openai/**'])
    .add_local_file(ROOT/'.runtime/connectome/manifest.json','/app/.runtime/connectome/manifest.json',copy=True)
    .add_local_file(ROOT/'.runtime/connectome/metadata.json','/app/.runtime/connectome/metadata.json',copy=True)
    .add_local_file(ROOT/'.runtime/connectome/positions.bin','/app/.runtime/connectome/positions.bin',copy=True)
    .workdir('/app'))

@app.function(image=gateway_image,cpu=0.25,memory=512,timeout=600,
              region='us-west',routing_region='us-west',min_containers=1,max_containers=2,
              secrets=[modal.Secret.from_name('call-fly-service')])
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web_us():
    import os
    os.environ['VOICE_WORKER_URL']=shared_worker.get_web_url()
    from server.modal_proxy import app
    return app

@app.function(image=image,gpu=SERVING_GPU,cpu=2,memory=8192,timeout=600,
              volumes={'/app/.runtime/moshi-reservoir':volume})
def verify():
    import subprocess,json
    subprocess.run(['python','scripts/verify_reservoir_causality.py','--audio',
                    'calibration/clip-2.wav','--frames','375','--output','/tmp/causality.json'],check=True)
    return json.loads(Path('/tmp/causality.json').read_text())
