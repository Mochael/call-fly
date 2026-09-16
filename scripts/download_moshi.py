from pathlib import Path
import sys
from huggingface_hub import snapshot_download
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.moshi_engine import MODEL, REVISION, FILES
snapshot_download(MODEL, revision=REVISION, allow_patterns=[*FILES, 'README.md'])
print(f'Moshi model ready: {MODEL} at {REVISION}')
