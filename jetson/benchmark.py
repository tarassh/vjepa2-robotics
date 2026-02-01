#!/usr/bin/env python3
"""
V-JEPA 2 ViT-L inference benchmark for Jetson Orin Nano.

Tests multiple input configurations (frames × resolution) and reports
latency, FPS, and memory usage. Designed for the 8GB shared memory
constraint of Jetson Orin.

Usage:
    python jetson/benchmark.py
    python jetson/benchmark.py --configs 8x224 16x256
    python jetson/benchmark.py --warmup 3 --runs 5
"""

import argparse
import gc
import time
import torch
import sys

DEFAULT_CONFIGS = [
    (4, 128),
    (4, 224),
    (8, 224),
    (2, 256),
    (8, 128),
    (16, 128),
    (16, 256),
]


def mem_stats():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated()
        return alloc / 1e9, free / 1e9, total / 1e9
    return 0, 0, 0


def benchmark_config(model, n_frames, resolution, device, warmup=2, runs=5):
    """Benchmark a single input configuration. Returns dict with results or None on OOM."""
    torch.cuda.empty_cache()
    gc.collect()

    try:
        x = torch.randn(1, n_frames, 3, resolution, resolution,
                         dtype=torch.float16, device=device)

        # Warmup runs
        for _ in range(warmup):
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                _ = model.get_vision_features(x)
            torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                features = model.get_vision_features(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        alloc, free, total = mem_stats()
        avg_ms = sum(times) / len(times)
        min_ms = min(times)
        max_ms = max(times)

        result = {
            'frames': n_frames,
            'resolution': resolution,
            'output_shape': tuple(features.shape),
            'avg_ms': avg_ms,
            'min_ms': min_ms,
            'max_ms': max_ms,
            'fps': 1000 / avg_ms,
            'gpu_alloc_gb': alloc,
            'gpu_free_gb': free,
        }

        del x, features
        return result

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "alloc" in str(e).lower():
            torch.cuda.empty_cache()
            gc.collect()
            return None
        raise


def parse_config(s):
    """Parse '8x224' into (8, 224)."""
    parts = s.lower().split('x')
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(description='V-JEPA 2 Jetson Benchmark')
    parser.add_argument('--configs', nargs='+', type=str, default=None,
                        help='Input configs as NxRES (e.g., 8x224 16x256)')
    parser.add_argument('--warmup', type=int, default=2, help='Warmup runs per config')
    parser.add_argument('--runs', type=int, default=5, help='Timed runs per config')
    parser.add_argument('--model', type=str, default='facebook/vjepa2-vitl-fpc64-256',
                        help='HuggingFace model ID')
    args = parser.parse_args()

    configs = [parse_config(c) for c in args.configs] if args.configs else DEFAULT_CONFIGS

    # System info
    print("=" * 70)
    print("V-JEPA 2 JETSON BENCHMARK")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        _, free, total = mem_stats()
        print(f"Memory: {free:.2f} GB free / {total:.2f} GB total")
    print(f"Model: {args.model}")
    print(f"Warmup: {args.warmup}, Runs: {args.runs}")
    print()

    # Load model
    print("Loading model...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        args.model, dtype=torch.float16
    ).to("cuda").eval()

    n_params = sum(p.numel() for p in model.parameters())
    model_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params / 1e6:.1f}M ({model_mb:.1f} MB on GPU)")
    alloc, free, _ = mem_stats()
    print(f"After load: {alloc:.2f} GB allocated, {free:.2f} GB free")
    print()

    # Run benchmarks
    results = []
    for nf, res in configs:
        print(f"Testing {nf} frames × {res}×{res}...", end=" ", flush=True)
        result = benchmark_config(model, nf, res, "cuda",
                                  warmup=args.warmup, runs=args.runs)
        if result:
            print(f"✅ {result['avg_ms']:.1f}ms ({result['fps']:.1f} FPS)")
            results.append(result)
        else:
            print("❌ OOM")
        gc.collect()
        torch.cuda.empty_cache()

    # Summary table
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    header = f"{'Frames':>6} {'Res':>5} {'Tokens':>7} {'Avg(ms)':>8} {'Min':>8} {'Max':>8} {'FPS':>6} {'GPU(GB)':>8}"
    print(header)
    print("-" * 70)
    for r in results:
        tokens = r['output_shape'][1]
        print(f"{r['frames']:>6} {r['resolution']:>5} {tokens:>7} "
              f"{r['avg_ms']:>8.1f} {r['min_ms']:>8.1f} {r['max_ms']:>8.1f} "
              f"{r['fps']:>6.1f} {r['gpu_alloc_gb']:>8.2f}")

    if results:
        best = min(results, key=lambda r: r['avg_ms'])
        print()
        print(f"⚡ Fastest: {best['frames']}×{best['resolution']} @ {best['fps']:.1f} FPS")


if __name__ == "__main__":
    main()
