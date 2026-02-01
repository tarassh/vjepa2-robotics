# V-JEPA 2 on Jetson Orin Nano

Edge deployment of V-JEPA 2 ViT-L (326M params) on NVIDIA Jetson Orin Nano Super (8GB).

## Hardware

- **Device:** Jetson Orin Nano Super (8GB unified memory)
- **GPU:** Orin (8 SMs), shared CPU/GPU memory
- **OS:** Ubuntu 22.04 (L4T R36.4.7, aarch64)
- **CUDA:** 12.6, cuDNN 9.3, TensorRT 10.3

## Performance

All benchmarks use FP16 inference with `torch.no_grad()` + `torch.amp.autocast`.

| Frames | Resolution | Output Tokens | Latency (ms) | FPS |
|--------|-----------|---------------|-------------|-----|
| 4 | 128×128 | 128 | 213 | 4.7 |
| 4 | 224×224 | 392 | 214 | 4.7 |
| 8 | 224×224 | 784 | 242 | **4.1** |
| 2 | 256×256 | 256 | 215 | 4.6 |
| 16 | 128×128 | 512 | 224 | 4.5 |
| 8 | 128×128 | 256 | 216 | 4.6 |
| 16 | 256×256 | 2048 | 452 | 2.2 |

**Recommended config for real-time robotics:** 8 frames @ 224×224 (~4 FPS)

## Feature Quality

Tested with 8 synthetic video patterns (static, motion, circles, checkerboard, diagonal sweep):

- **Within-category similarity:** 0.953 (static↔static, motion↔motion, circle↔circle)
- **Cross-category similarity:** 0.653
- **Discrimination gap:** 0.300
- **Self-similarity:** 1.000
- **Repeatability:** 1.000000

✅ Model produces meaningful, discriminative video features on edge hardware.

## Setup

### 1. Create Python environment

```bash
python3 -m venv ~/vjepa2-env
source ~/vjepa2-env/bin/activate
```

### 2. Install PyTorch for Jetson

```bash
# PyTorch 2.8.0 for JetPack 6 / CUDA 12.6
pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/
```

> ⚠️ PyTorch 2.9.1 requires `libcudss.so.0` which is not available on Jetson. Stick with 2.8.0.

### 3. Install dependencies

```bash
pip install transformers timm einops numpy==1.26.4
```

### 4. Set CUDA paths

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/local/cuda-12.6/lib64
```

### 5. Run benchmarks

```bash
python jetson/benchmark.py        # Speed benchmarks across input sizes
python jetson/feature_quality.py  # Feature discrimination test
```

## Usage

```python
import torch
from transformers import AutoModel

# Load model
model = AutoModel.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256",
    dtype=torch.float16
).to("cuda").eval()

# Inference
video = torch.randn(1, 8, 3, 224, 224, dtype=torch.float16, device="cuda")
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
    features = model.get_vision_features(video)
# features.shape: (1, 784, 1024)
```

## Key Differences from Desktop

| | Desktop | Jetson |
|---|---------|--------|
| **Model loading** | HuggingFace `AutoModel` | Same ✅ |
| **Precision** | FP32 or FP16 | **FP16 required** |
| **Max input** | 16×256×256 easily | 16×256×256 (452ms) |
| **PyTorch** | Latest | **2.8.0** (libcudss constraint) |
| **Memory** | Dedicated VRAM | **Shared** CPU/GPU (7.4 GB total) |

## Notes

- Close unnecessary GUI apps (Warp Terminal, GNOME) to free ~1-2 GB
- The model uses ~0.6 GB GPU memory; rest is for activations
- First inference is slower (CUDA kernel compilation); subsequent are faster
- `torch.cuda.empty_cache()` between runs helps with fragmentation
