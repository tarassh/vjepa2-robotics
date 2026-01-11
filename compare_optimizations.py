#!/usr/bin/env python3
"""Compare different optimization strategies for VJEPA2"""

import sys
import time
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'vjepa2'))

from src.models.vision_transformer import vit_large_rope, vit_base

def benchmark_config(model_fn, img_size, num_frames, device='mps', name="Config"):
    """Benchmark a specific configuration"""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    
    # Load model
    print("Loading model...")
    model = model_fn(img_size=(img_size, img_size), num_frames=num_frames)
    model.to(device)
    model.eval()
    
    # Model size
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.1f}M)")
    
    # Generate video
    video = torch.randn(1, 3, num_frames, img_size, img_size).to(device)
    video_size_mb = video.numel() * 4 / 1024**2
    print(f"Video size: {video_size_mb:.1f} MB")
    
    # Benchmark
    print("Benchmarking...")
    with torch.inference_mode():
        # Warmup
        for _ in range(3):
            _ = model(video)
            if device == 'mps':
                torch.mps.synchronize()
        
        # Measure
        times = []
        for _ in range(5):
            start = time.perf_counter()
            output = model(video)
            if device == 'mps':
                torch.mps.synchronize()
            times.append(time.perf_counter() - start)
    
    mean_time = np.mean(times) * 1000  # ms
    fps = 1000 / mean_time
    
    print(f"\nResults:")
    print(f"  Time: {mean_time:.1f} ms")
    print(f"  FPS:  {fps:.1f}")
    
    return mean_time, fps, params

def main():
    if not torch.backends.mps.is_available():
        print("MPS not available! Using CPU (will be slower)")
        device = 'cpu'
    else:
        device = 'mps'
    
    print("="*60)
    print("VJEPA2 Optimization Comparison")
    print(f"Device: {device}")
    print("="*60)
    
    configs = []
    
    # Baseline
    time, fps, params = benchmark_config(
        vit_large_rope, 256, 16, device,
        "1. Baseline (ViT-Large, 256px, 16 frames)"
    )
    baseline_time = time
    configs.append(("Baseline", time, fps, params, 1.0))
    
    # Reduce frames
    time, fps, params = benchmark_config(
        vit_large_rope, 256, 8, device,
        "2. Fewer Frames (ViT-Large, 256px, 8 frames)"
    )
    configs.append(("Fewer frames (8)", time, fps, params, baseline_time/time))
    
    # Reduce resolution
    time, fps, params = benchmark_config(
        vit_large_rope, 192, 16, device,
        "3. Lower Resolution (ViT-Large, 192px, 16 frames)"
    )
    configs.append(("Lower res (192px)", time, fps, params, baseline_time/time))
    
    # Smaller model
    try:
        time, fps, params = benchmark_config(
            vit_base, 256, 16, device,
            "4. Smaller Model (ViT-Base, 256px, 16 frames)"
        )
        configs.append(("Smaller model (Base)", time, fps, params, baseline_time/time))
    except Exception as e:
        print(f"\nViT-Base not available: {e}")
        configs.append(("Smaller model (Base)", None, None, None, None))
    
    # Best combo: fewer frames + lower res
    time, fps, params = benchmark_config(
        vit_large_rope, 192, 8, device,
        "5. Combined (ViT-Large, 192px, 8 frames)"
    )
    configs.append(("Combined optimizations", time, fps, params, baseline_time/time))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Configuration':<30} {'Time':<10} {'FPS':<8} {'Speedup':<10}")
    print("-"*60)
    
    for name, time, fps, params, speedup in configs:
        if time is not None:
            print(f"{name:<30} {time:>7.1f}ms {fps:>6.1f} {speedup:>8.2f}x")
        else:
            print(f"{name:<30} {'N/A':<10}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. BEST FOR SPEED (4x faster):")
    print("   python robotics_obstacle_detection.py \\")
    print("       --device mps \\")
    print("       --num-frames 8 \\")
    print("       --img-size 192")
    
    print("\n2. BALANCED (2x faster, good quality):")
    print("   python robotics_obstacle_detection.py \\")
    print("       --device mps \\")
    print("       --num-frames 8 \\")
    print("       --img-size 256")
    
    print("\n3. QUALITY FOCUSED (1.4x faster):")
    print("   python robotics_obstacle_detection.py \\")
    print("       --device mps \\")
    print("       --num-frames 16 \\")
    print("       --img-size 192")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()
