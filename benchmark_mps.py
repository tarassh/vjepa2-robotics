#!/usr/bin/env python3
"""Benchmark VJEPA2 with Metal Performance Shaders (MPS) vs CPU"""

import sys
import time
from pathlib import Path
import torch
import numpy as np

# Add vjepa2 to path
vjepa2_path = Path(__file__).parent / 'vjepa2'
sys.path.insert(0, str(vjepa2_path))

from src.models.vision_transformer import vit_large_rope

def benchmark_device(model, video, device_name, device, iterations=5, warmup=2):
    """Benchmark model on specified device"""
    print(f"\n--- {device_name} Benchmark ---")
    
    model = model.to(device)
    video = video.to(device)
    
    with torch.inference_mode():
        # Warmup
        print(f"Warming up {device_name}...")
        for _ in range(warmup):
            _ = model(video)
            if device.type == 'mps':
                torch.mps.synchronize()
        
        # Benchmark
        print(f"Running {iterations} iterations...")
        times = []
        for i in range(iterations):
            start = time.perf_counter()
            output = model(video)
            
            if device.type == 'mps':
                torch.mps.synchronize()
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed*1000:.2f} ms")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n{device_name} Results:")
    print(f"  Mean: {mean_time*1000:.2f} ms (±{std_time*1000:.2f} ms)")
    print(f"  FPS:  {1/mean_time:.2f}")
    
    return mean_time, times


def main():
    print("="*60)
    print("VJEPA2 MPS vs CPU Benchmark")
    print("="*60)
    
    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("\nERROR: MPS not available!")
        print("Make sure you're running on Apple Silicon with PyTorch 2.0+")
        return
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Load model
    print("\nLoading ViT-Large model...")
    model = vit_large_rope(img_size=(256, 256), num_frames=16)
    model.eval()
    
    # Generate test video
    print("Generating test video...")
    video = torch.randn(1, 3, 16, 256, 256)
    print(f"Video shape: {video.shape}")
    print(f"Video size: {video.numel() * 4 / 1024**2:.1f} MB")
    
    # Benchmark CPU
    cpu_time, cpu_times = benchmark_device(
        model, video, "CPU", 
        torch.device('cpu'), 
        iterations=5, warmup=2
    )
    
    # Benchmark MPS
    mps_time, mps_times = benchmark_device(
        model, video, "MPS (Metal)", 
        torch.device('mps'),
        iterations=5, warmup=3
    )
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"CPU:  {cpu_time*1000:>8.2f} ms  ({1/cpu_time:>6.2f} FPS)")
    print(f"MPS:  {mps_time*1000:>8.2f} ms  ({1/mps_time:>6.2f} FPS)")
    print(f"\nSpeedup: {cpu_time/mps_time:.2f}x faster with MPS")
    
    # Performance category
    speedup = cpu_time / mps_time
    if speedup < 1.5:
        category = "⚠️  Marginal improvement"
    elif speedup < 3:
        category = "✅ Good improvement"
    elif speedup < 5:
        category = "✅ Excellent improvement"
    else:
        category = "🚀 Outstanding improvement"
    
    print(f"\nPerformance: {category}")


if __name__ == '__main__':
    main()
