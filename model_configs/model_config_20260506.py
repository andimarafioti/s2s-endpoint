import torch
from huggingface_hub import snapshot_download

torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)

snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
snapshot_download("nvidia/parakeet-tdt-0.6b-v3")
