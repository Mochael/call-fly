from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel

snapshot_download('mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit',
                  revision='e7dd0585652209fa0d7783659aad4e8a324de11c',
                  allow_patterns=['*.json', '*.safetensors', '*.txt'])
WhisperModel('base.en', device='cpu', compute_type='int8', cpu_threads=4)
print('Qwen voice and Whisper speech recognition are ready.')
