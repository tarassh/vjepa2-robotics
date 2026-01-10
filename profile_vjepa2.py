#!/usr/bin/env python3
"""
Profile VJEPA2 models to identify performance bottlenecks
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function

# Add vjepa2 submodule to path
vjepa2_path = Path(__file__).parent / 'vjepa2'
sys.path.insert(0, str(vjepa2_path))

from src.models.vision_transformer import vit_giant_xformers_rope, vit_huge_rope, vit_large_rope
import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def build_transform(img_size):
    """Build video preprocessing transform"""
    short_side_size = int(256.0 / 224 * img_size)
    transform = video_transforms.Compose([
        video_transforms.Resize(short_side_size, interpolation='bilinear'),
        video_transforms.CenterCrop(size=(img_size, img_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    return transform


def load_model(model_size, img_size, num_frames, device):
    """Load VJEPA2 model"""
    print(f"Loading {model_size} model...")
    
    if model_size == 'large':
        model = vit_large_rope(img_size=(img_size, img_size), num_frames=num_frames)
    elif model_size == 'huge':
        model = vit_huge_rope(img_size=(img_size, img_size), num_frames=num_frames)
    elif model_size == 'giant':
        model = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=num_frames)
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    model.to(device)
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024**2:.1f} MB (FP32)")
    
    return model


def generate_dummy_video(num_frames, img_size, batch_size=1):
    """Generate random video tensor for testing"""
    video = torch.randn(batch_size, 3, num_frames, img_size, img_size)
    return video


def simple_benchmark(model, video_tensor, device, num_iterations=50, warmup=5):
    """Simple timing benchmark"""
    video_tensor = video_tensor.to(device)
    
    print(f"\n{'='*60}")
    print("Simple Benchmark (Average over runs)")
    print(f"{'='*60}")
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(video_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running benchmark ({num_iterations} iterations)...")
    times = []
    
    with torch.inference_mode():
        for i in range(num_iterations):
            start = time.perf_counter()
            output = model(video_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    
    times = np.array(times)
    
    print(f"\nResults:")
    print(f"  Mean:   {times.mean()*1000:.2f} ms")
    print(f"  Median: {np.median(times)*1000:.2f} ms")
    print(f"  Std:    {times.std()*1000:.2f} ms")
    print(f"  Min:    {times.min()*1000:.2f} ms")
    print(f"  Max:    {times.max()*1000:.2f} ms")
    print(f"  FPS:    {1/times.mean():.2f}")
    
    return output


def detailed_profile(model, video_tensor, device, output_file="vjepa2_profile.txt"):
    """Detailed profiling with PyTorch profiler"""
    video_tensor = video_tensor.to(device)
    
    print(f"\n{'='*60}")
    print("Detailed Profiling (PyTorch Profiler)")
    print(f"{'='*60}")
    
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.inference_mode():
            with record_function("model_inference"):
                output = model(video_tensor)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
    
    # Print profiler output
    print("\nTop operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    if device.type == 'cuda':
        print("\nTop operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        
        print("\nTop operations by CUDA memory:")
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=20))
    
    # Save detailed profile
    prof.export_chrome_trace(output_file.replace('.txt', '.json'))
    print(f"\nChrome trace saved to: {output_file.replace('.txt', '.json')}")
    print("View it at: chrome://tracing")
    
    return output


def layer_by_layer_profile(model, video_tensor, device):
    """Profile each major component separately"""
    video_tensor = video_tensor.to(device)
    
    print(f"\n{'='*60}")
    print("Layer-by-Layer Profiling")
    print(f"{'='*60}")
    
    timings = {}
    
    with torch.inference_mode():
        # Patch embedding
        start = time.perf_counter()
        x = model.patch_embed(video_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        timings['patch_embed'] = time.perf_counter() - start
        
        # Position embedding
        if model.pos_embed is not None:
            start = time.perf_counter()
            pos_embed = model.interpolate_pos_encoding(video_tensor, model.pos_embed)
            x = x + pos_embed
            if device.type == 'cuda':
                torch.cuda.synchronize()
            timings['pos_embed'] = time.perf_counter() - start
        
        # Transformer blocks
        block_times = []
        for i, blk in enumerate(model.blocks):
            start = time.perf_counter()
            x = blk(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            block_time = time.perf_counter() - start
            block_times.append(block_time)
        
        timings['blocks_total'] = sum(block_times)
        timings['blocks_avg'] = np.mean(block_times)
        timings['blocks_std'] = np.std(block_times)
        
        # Normalization
        start = time.perf_counter()
        x = model.norm(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        timings['norm'] = time.perf_counter() - start
    
    # Print results
    print(f"\nComponent timings:")
    print(f"  Patch Embedding:  {timings['patch_embed']*1000:>8.2f} ms")
    if 'pos_embed' in timings:
        print(f"  Position Embed:   {timings['pos_embed']*1000:>8.2f} ms")
    print(f"  Transformer (tot):{timings['blocks_total']*1000:>8.2f} ms  ({len(model.blocks)} blocks)")
    print(f"  - Per block (avg):{timings['blocks_avg']*1000:>8.2f} ms")
    print(f"  - Per block (std):{timings['blocks_std']*1000:>8.2f} ms")
    print(f"  Normalization:    {timings['norm']*1000:>8.2f} ms")
    print(f"\nTotal: {sum(timings.values())*1000:.2f} ms")
    
    # Show slowest blocks
    print(f"\nSlowest 5 transformer blocks:")
    slowest_indices = np.argsort(block_times)[-5:][::-1]
    for idx in slowest_indices:
        print(f"  Block {idx:2d}: {block_times[idx]*1000:.2f} ms")


def memory_profile(model, video_tensor, device):
    """Profile memory usage"""
    if device.type != 'cuda':
        print("\nMemory profiling only available on CUDA")
        return
    
    video_tensor = video_tensor.to(device)
    
    print(f"\n{'='*60}")
    print("Memory Profiling")
    print(f"{'='*60}")
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    mem_before = torch.cuda.memory_allocated() / 1024**2
    max_mem_before = torch.cuda.max_memory_allocated() / 1024**2
    
    with torch.inference_mode():
        output = model(video_tensor)
        torch.cuda.synchronize()
    
    mem_after = torch.cuda.memory_allocated() / 1024**2
    max_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"\nMemory usage:")
    print(f"  Before inference:     {mem_before:>8.1f} MB")
    print(f"  After inference:      {mem_after:>8.1f} MB")
    print(f"  Peak during inference:{max_mem:>8.1f} MB")
    print(f"  Inference overhead:   {max_mem - mem_before:>8.1f} MB")
    
    # Model parameters memory
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"\nModel parameters:     {param_mem:>8.1f} MB")
    
    # Input/output memory
    input_mem = video_tensor.numel() * video_tensor.element_size() / 1024**2
    output_mem = output.numel() * output.element_size() / 1024**2
    print(f"Input tensor:         {input_mem:>8.1f} MB")
    print(f"Output tensor:        {output_mem:>8.1f} MB")


def compare_precisions(model, video_tensor, device):
    """Compare FP32 vs FP16 performance"""
    if device.type != 'cuda':
        print("\nPrecision comparison only available on CUDA")
        return
    
    print(f"\n{'='*60}")
    print("Precision Comparison")
    print(f"{'='*60}")
    
    video_tensor = video_tensor.to(device)
    
    # FP32
    model.to(torch.float32)
    torch.cuda.empty_cache()
    
    times_fp32 = []
    with torch.inference_mode():
        for _ in range(20):
            start = time.perf_counter()
            _ = model(video_tensor.float())
            torch.cuda.synchronize()
            times_fp32.append(time.perf_counter() - start)
    
    # FP16
    model.to(torch.float16)
    torch.cuda.empty_cache()
    
    times_fp16 = []
    with torch.inference_mode():
        for _ in range(20):
            start = time.perf_counter()
            _ = model(video_tensor.half())
            torch.cuda.synchronize()
            times_fp16.append(time.perf_counter() - start)
    
    print(f"\nFP32: {np.mean(times_fp32)*1000:.2f} ms (±{np.std(times_fp32)*1000:.2f} ms)")
    print(f"FP16: {np.mean(times_fp16)*1000:.2f} ms (±{np.std(times_fp16)*1000:.2f} ms)")
    print(f"Speedup: {np.mean(times_fp32)/np.mean(times_fp16):.2f}x")
    
    # Reset to FP32
    model.to(torch.float32)


def main():
    parser = argparse.ArgumentParser(description='Profile VJEPA2 performance')
    parser.add_argument('--model-size', type=str, default='large', 
                        choices=['large', 'huge', 'giant'],
                        help='Model size to profile')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--num-frames', type=int, default=16,
                        help='Number of frames')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of benchmark iterations')
    parser.add_argument('--skip-detailed', action='store_true',
                        help='Skip detailed profiling (faster)')
    parser.add_argument('--output', type=str, default='vjepa2_profile.txt',
                        help='Output file for detailed profile')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Load model
    model = load_model(args.model_size, args.img_size, args.num_frames, device)
    
    # Generate dummy video
    print(f"\nGenerating dummy video: {args.batch_size}x3x{args.num_frames}x{args.img_size}x{args.img_size}")
    video_tensor = generate_dummy_video(args.num_frames, args.img_size, args.batch_size)
    print(f"Video tensor size: {video_tensor.numel() * 4 / 1024**2:.1f} MB")
    
    # Run profiling
    simple_benchmark(model, video_tensor, device, args.iterations)
    
    if not args.skip_detailed:
        layer_by_layer_profile(model, video_tensor, device)
        
        if device.type == 'cuda':
            memory_profile(model, video_tensor, device)
            compare_precisions(model, video_tensor, device)
        
        detailed_profile(model, video_tensor, device, args.output)
    
    print(f"\n{'='*60}")
    print("Profiling complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
