# Metal Acceleration for VJEPA2 on Apple Silicon

Guide to using Metal Performance Shaders (MPS) to accelerate VJEPA2 on Mac with Apple Silicon.

## TL;DR

```bash
# Just change --device cpu to --device mps
python robotics_obstacle_detection.py --device mps

# Expected speedup: 3-5x faster than CPU
# Mac M2: ~600ms → ~150ms (13 FPS)
```

## What is Metal/MPS?

**Metal** is Apple's GPU API (like CUDA for NVIDIA)  
**MPS (Metal Performance Shaders)** is PyTorch's backend for Metal

Your Mac M2 has:
- **10-core GPU** (M2 Pro/Max has more)
- **Unified memory** (shared between CPU/GPU)
- **Neural Engine** (for specific operations)

## Quick Benchmark

Run the benchmark to see your performance:

```bash
source .venv/bin/activate
python benchmark_mps.py
```

**Expected results on M2:**

| Configuration | CPU | MPS | Speedup |
|---------------|-----|-----|---------|
| ViT-Large, 16f, 256px | ~2200 ms | ~400-600 ms | **3.5-5.5x** |
| ViT-Large, 8f, 256px | ~1100 ms | ~200-300 ms | **3.5-5.5x** |

## Using MPS in Your Code

### Option 1: Command Line

```bash
# Obstacle detection with MPS
python robotics_obstacle_detection.py \
    --model-size large \
    --device mps

# Collision avoidance with MPS  
python robotics_collision_avoidance.py \
    --device mps
```

### Option 2: Python API

```python
from robotics_obstacle_detection import ObstacleDetector

# Use MPS device
detector = ObstacleDetector(
    model_size='large',
    img_size=256,
    num_frames=16,
    device='mps'  # ← Change this!
)

# Everything else works the same
detector.add_frame(frame)
features, motion_score = detector.detect_obstacles()
```

### Option 3: Manual PyTorch

```python
import torch

# Check MPS availability
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Load model on MPS
model = model.to(device)
video = video.to(device)

# Run inference
with torch.inference_mode():
    output = model(video)
    torch.mps.synchronize()  # Wait for completion
```

## Performance Expectations

### Apple Silicon Performance:

| Mac Model | GPU Cores | Est. FPS (ViT-Large) |
|-----------|-----------|----------------------|
| **M1** | 7-8 | 8-10 FPS |
| **M1 Pro** | 14-16 | 10-12 FPS |
| **M1 Max** | 24-32 | 12-15 FPS |
| **M2** | 8-10 | 8-10 FPS |
| **M2 Pro** | 16-19 | 12-15 FPS |
| **M2 Max** | 30-38 | 15-18 FPS |
| **M3** | 10 | 10-12 FPS |
| **M3 Pro** | 14-18 | 14-18 FPS |
| **M3 Max** | 30-40 | 18-22 FPS |

*Estimates for 16 frames @ 256px resolution*

### Comparison with Other Hardware:

| Hardware | FPS (ViT-Large) | Cost | Power |
|----------|----------------|------|-------|
| **Mac M2 CPU** | 0.5 | $1200 | 15W |
| **Mac M2 MPS** | **10** ✅ | $1200 | 20W |
| **Jetson Orin Nano** | 10-13 | $500 | 15W |
| **Jetson AGX Orin** | 40-50 | $2000 | 40W |
| **RTX 3090** | 50-60 | $1500 | 350W |

## MPS Advantages on Mac

✅ **Already have the hardware** - No need to buy GPU  
✅ **Unified memory** - No CPU↔GPU transfers  
✅ **Good enough for development** - 10-15 FPS usable  
✅ **Power efficient** - Low power draw  
✅ **Silent** - No loud GPU fans  
✅ **Works out of the box** - PyTorch has built-in support

## MPS Limitations

❌ **Not as fast as NVIDIA GPUs** (~3-5x slower than RTX 3090)  
❌ **No FP16 tensor cores** (like NVIDIA)  
❌ **Limited memory** (shared with system)  
❌ **Some ops fall back to CPU** (can be slow)  
❌ **No TensorRT equivalent** (limited optimization)

## Optimizing for MPS

### 1. Reduce Memory Usage

```python
# Use smaller models
model = vit_large_rope(...)  # Instead of vit_giant

# Reduce frames
num_frames = 8  # Instead of 16

# Lower resolution
img_size = 192  # Instead of 256
```

### 2. Avoid CPU↔MPS Transfers

```python
# BAD: Moving data back and forth
for frame in frames:
    frame_mps = frame.to('mps')  # Slow!
    output = model(frame_mps)
    result = output.to('cpu')  # Slow!

# GOOD: Keep data on MPS
frames_mps = frames.to('mps')  # Once
outputs = model(frames_mps)
results = outputs.to('cpu')  # Once at end
```

### 3. Use Batch Processing

```python
# Process multiple frames at once
batch = torch.stack([frame1, frame2, frame3])
batch_mps = batch.to('mps')
outputs = model(batch_mps)  # Faster than one-by-one
```

### 4. Profile MPS Performance

```python
# Time MPS operations
import time

start = time.perf_counter()
output = model(video)
torch.mps.synchronize()  # Important!
elapsed = time.perf_counter() - start
```

## Troubleshooting

### "MPS backend not available"

Check:
```python
import torch
print(torch.backends.mps.is_available())  # Should be True
print(torch.backends.mps.is_built())      # Should be True
```

**Fix**: Update PyTorch
```bash
pip install --upgrade torch torchvision
```

### "RuntimeError: Placeholder storage has not been allocated"

Some operations aren't MPS-optimized and fall back to CPU.

**Fix**: Convert to/from CPU for those ops
```python
try:
    output = model(video_mps)
except RuntimeError:
    # Fall back to CPU for this operation
    output = model(video.cpu()).to('mps')
```

### Slow Performance Despite MPS

Check if operations are falling back to CPU:

```bash
# Enable MPS fallback logging
export PYTORCH_ENABLE_MPS_FALLBACK=1
python robotics_obstacle_detection.py --device mps
```

### Memory Errors

MPS shares memory with system. Close other apps:

```bash
# Check memory usage
Activity Monitor → Memory tab

# Or terminal
vm_stat
```

## Custom Metal Shaders (Advanced)

If you want to write custom Metal shaders:

### Option 1: PyTorch Custom Ops

```python
import torch

# Define custom Metal kernel
metal_source = """
kernel void custom_attention(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
    // Your optimized attention here
}
"""

# Compile and use
custom_op = torch.utils.cpp_extension.load_inline(
    name='custom_attention',
    cpp_sources='',
    extra_cuda_cflags=['-std=c++17'],
    extra_include_paths=['/System/Library/Frameworks/Metal.framework/Headers']
)
```

### Option 2: MLX Framework

Apple's [MLX](https://github.com/ml-explore/mlx) is designed for Metal:

```python
import mlx.core as mx

# MLX has better Metal optimization
# But requires rewriting model code
```

### Option 3: Direct Metal API

Write `.metal` shaders:

```metal
// attention.metal
kernel void optimized_attention(
    device const float4* Q [[buffer(0)]],
    device const float4* K [[buffer(1)]],
    device float4* output [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
    
    // Optimized attention implementation
    float4 q = Q[gid.y];
    float4 k = K[gid.x];
    float score = dot(q, k);
    // ... more logic
}
```

**Compile**: `xcrun -sdk macosx metal -c attention.metal -o attention.air`

**But this requires significant work** - PyTorch MPS is usually good enough!

## When to Use MPS vs When to Get a GPU

### Use MPS if:
✅ Developing/testing on Mac  
✅ 10-15 FPS is acceptable  
✅ Budget constrained  
✅ Need portability (laptop)  
✅ Prototype/research phase

### Get dedicated GPU if:
✅ Need 30+ FPS real-time  
✅ Production deployment  
✅ Running on robot (get Jetson)  
✅ Need maximum performance  
✅ Training models (not just inference)

## Real-World Usage

### Development Workflow:

1. **Develop on Mac with MPS** (10 FPS)
   - Fast iteration
   - Test algorithms
   - Prototype features

2. **Deploy on Jetson/GPU** (30-60 FPS)
   - Real-time performance
   - Production system
   - Actual robot

### Example: My Setup

```python
# Development (Mac M2 with MPS)
if torch.backends.mps.is_available():
    device = 'mps'  # 10 FPS, good for testing
elif torch.cuda.is_available():
    device = 'cuda'  # 50 FPS on my desktop
else:
    device = 'cpu'  # 0.5 FPS, last resort

detector = ObstacleDetector(device=device)
```

## Summary

**MPS gives you 3-5x speedup** on Mac, making development practical:

- **CPU**: 0.5 FPS ❌ (unusable)
- **MPS**: 10 FPS ✅ (development-ready)
- **NVIDIA GPU**: 50+ FPS ✅ (production-ready)

For robotics development on Mac, **MPS is perfect**. For deployment, consider Jetson or desktop GPU.

## Resources

- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [MLX Framework](https://github.com/ml-explore/mlx)
- [Apple Neural Engine](https://github.com/hollance/neural-engine)
