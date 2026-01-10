# Profiling VJEPA2 Performance

Guide to identifying and fixing performance bottlenecks in VJEPA2 models.

## Quick Start

```bash
source .venv/bin/activate

# Profile ViT-Large model (fastest)
python profile_vjepa2.py --model-size large

# Profile with GPU (if available)
python profile_vjepa2.py --model-size large --device cuda

# Quick benchmark only (skip detailed profiling)
python profile_vjepa2.py --model-size large --skip-detailed
```

## Usage

### Basic Profiling

```bash
python profile_vjepa2.py \
    --model-size large \      # Model: large, huge, or giant
    --img-size 256 \           # Input resolution
    --num-frames 16 \          # Number of video frames
    --device cuda \            # Device: cuda or cpu
    --iterations 50            # Benchmark iterations
```

### What Gets Profiled

1. **Simple Benchmark** - Average inference time, FPS, statistics
2. **Layer-by-Layer** - Time spent in each component
3. **Memory Profile** - GPU memory usage (CUDA only)
4. **Precision Comparison** - FP32 vs FP16 performance (CUDA only)
5. **Detailed Profile** - PyTorch profiler with operation-level breakdown

### Example Output

```
Using device: cuda
GPU: NVIDIA GeForce RTX 4090
Loading large model...
Total parameters: 304,714,240
Model size: ~1161.1 MB (FP32)

============================================================
Simple Benchmark (Average over runs)
============================================================
Running benchmark (50 iterations)...

Results:
  Mean:   45.23 ms
  Median: 44.87 ms
  Std:    2.14 ms
  Min:    43.12 ms
  Max:    52.34 ms
  FPS:    22.11

============================================================
Layer-by-Layer Profiling
============================================================

Component timings:
  Patch Embedding:      3.45 ms
  Transformer (tot):   40.12 ms  (24 blocks)
  - Per block (avg):    1.67 ms
  - Per block (std):    0.12 ms
  Normalization:        0.08 ms

Total: 43.65 ms

Slowest 5 transformer blocks:
  Block 23: 1.89 ms
  Block 22: 1.84 ms
  Block 21: 1.79 ms
  Block 20: 1.75 ms
  Block 19: 1.72 ms
```

## Understanding Results

### What to Look For

#### 1. Transformer Blocks Are Slowest
**If**: 90%+ of time in transformer blocks
**Why**: Expected - transformers are compute-heavy
**Fix**: 
- Use smaller model (large instead of giant)
- Reduce input resolution (256 instead of 384)
- Use FP16 precision
- Enable torch.compile() in PyTorch 2.0+

#### 2. High Variance Between Blocks
**If**: Block std > 0.5ms
**Why**: Memory bandwidth issues or thermal throttling
**Fix**:
- Check GPU temperature
- Ensure consistent GPU clocks
- Batch processing to amortize overhead

#### 3. Patch Embedding is Slow
**If**: Patch embedding > 10% of total time
**Why**: Conv3D operations are memory-bound
**Fix**:
- Reduce input resolution
- Use channels-last memory format
- Enable cuDNN autotuner

#### 4. High Memory Usage
**If**: Peak memory > 8GB for large model
**Why**: Intermediate activations stored
**Fix**:
- Use gradient checkpointing (if training)
- Enable activation checkpointing
- Reduce batch size
- Use FP16/BF16

### Interpreting FP32 vs FP16

```
FP32: 45.23 ms
FP16: 28.45 ms
Speedup: 1.59x
```

**Speedup < 1.5x**: CPU-bound or memory-bound
**Speedup 1.5-2x**: Good GPU utilization
**Speedup > 2x**: Excellent tensor core usage

## Advanced Profiling

### Chrome Trace Visualization

The detailed profiler generates a `vjepa2_profile.json` file:

1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Click "Load" and select `vjepa2_profile.json`
4. Explore timeline view of operations

**What to look for:**
- **Gaps**: CPU waiting for GPU (add async operations)
- **Narrow bars**: Short operations with high overhead
- **Memory spikes**: Potential for optimization

### Python Built-in Profiler

For CPU-only detailed profiling:

```bash
python -m cProfile -o profile.prof profile_vjepa2.py --device cpu
python -m pstats profile.prof
```

Then in the pstats shell:
```python
sort cumtime
stats 20
```

### Line Profiler

Install and use line_profiler for per-line profiling:

```bash
pip install line_profiler

# Add @profile decorator to functions in profile_vjepa2.py
kernprof -l -v profile_vjepa2.py
```

## Optimization Strategies

### 1. Model Quantization

```python
import torch

# Post-training quantization
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### 2. TorchScript Compilation

```python
# Compile model with TorchScript
scripted_model = torch.jit.script(model)
scripted_model = torch.jit.optimize_for_inference(scripted_model)
```

### 3. PyTorch 2.0 Compile

```python
# Requires PyTorch 2.0+
compiled_model = torch.compile(model, mode="reduce-overhead")
```

### 4. Mixed Precision

```python
from torch.cuda.amp import autocast

with autocast():
    output = model(input)
```

### 5. Reduce Frame Count

For obstacle detection, you may not need 16 frames:

```python
# Try fewer frames for faster inference
model = vit_large_rope(num_frames=8)  # Instead of 16
```

### 6. Spatial Resolution

Lower resolution = faster inference:

```python
# 256x256 is 2.25x faster than 384x384
model = vit_large_rope(img_size=(256, 256))
```

## Benchmarking Different Configurations

Quick comparison script:

```bash
# Compare all model sizes
for size in large huge giant; do
    echo "Profiling $size..."
    python profile_vjepa2.py \
        --model-size $size \
        --skip-detailed \
        --iterations 20
done

# Compare resolutions
for res in 256 384; do
    echo "Profiling resolution $res..."
    python profile_vjepa2.py \
        --img-size $res \
        --skip-detailed
done

# Compare frame counts
for frames in 8 16 32; do
    echo "Profiling $frames frames..."
    python profile_vjepa2.py \
        --num-frames $frames \
        --skip-detailed
done
```

## Common Bottlenecks & Fixes

| Bottleneck | Symptom | Solution |
|------------|---------|----------|
| **Attention layers** | 70%+ time in attention | Use Flash Attention, reduce sequence length |
| **Memory bandwidth** | Low GPU utilization | Use tensor cores (FP16), increase batch size |
| **Data loading** | Gaps in GPU timeline | Async data loading, pin memory |
| **Small tensors** | Many small operations | Batch operations, fuse kernels |
| **CPU bottleneck** | Low FPS even on GPU | Offload preprocessing to GPU |

## Real-World Optimization Example

**Goal**: Achieve 30 FPS for obstacle detection on RTX 3090

**Starting point**: 12 FPS with ViT-Giant @ 384px, 16 frames

**Optimizations applied**:
1. ✅ Switch to ViT-Large → 22 FPS (1.8x speedup)
2. ✅ Reduce to 256px → 28 FPS (1.3x speedup)
3. ✅ Use FP16 → 35 FPS (1.25x speedup)
4. ✅ Reduce to 12 frames → 38 FPS (1.1x speedup)

**Final**: 38 FPS ✨ (3.2x total speedup)

## Tips

- **Warmup is important**: First few iterations are slower due to CUDA kernel compilation
- **Check GPU utilization**: Use `nvidia-smi dmon` to monitor
- **Profile in realistic conditions**: Use real video data, not random tensors
- **Batch when possible**: Process multiple frames at once
- **CPU profiling**: Don't forget to profile data preprocessing pipeline

## Resources

- [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
