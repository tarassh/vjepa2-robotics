#!/usr/bin/env python3
"""Quick test of MPS-accelerated obstacle detection"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'vjepa2'))

from robotics_obstacle_detection import ObstacleDetector

def main():
    print("="*60)
    print("Testing MPS-Accelerated Obstacle Detection")
    print("="*60)
    
    # Check MPS
    if not torch.backends.mps.is_available():
        print("\nERROR: MPS not available!")
        return
    
    print("\n✅ MPS is available")
    print(f"PyTorch version: {torch.__version__}")
    
    # Initialize detector with MPS
    print("\nInitializing detector with MPS...")
    detector = ObstacleDetector(
        model_size='large',
        img_size=256,
        num_frames=16,
        device='mps'  # ← Using MPS!
    )
    
    print("\n✅ Detector initialized on MPS")
    
    # Generate fake video frames (simulating camera)
    print("\nGenerating test video frames...")
    for i in range(16):
        # Simulate a 640x480 camera frame
        fake_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detector.add_frame(fake_frame)
        print(f"  Frame {i+1}/16 added")
    
    # Run detection
    print("\nRunning obstacle detection on MPS...")
    import time
    start = time.perf_counter()
    
    features, motion_score = detector.detect_obstacles()
    
    elapsed = time.perf_counter() - start
    
    print(f"\n✅ Detection complete!")
    print(f"   Time: {elapsed*1000:.2f} ms")
    print(f"   FPS: {1/elapsed:.2f}")
    print(f"   Motion score: {motion_score:.3f}")
    
    # Spatial analysis
    spatial_attention = detector.analyze_spatial_features(features)
    if spatial_attention is not None:
        print(f"   Attention map shape: {spatial_attention.shape}")
    
    print("\n" + "="*60)
    print("✅ MPS acceleration working!")
    print("="*60)
    
    print("\nNext step: Run with real camera:")
    print("  python robotics_obstacle_detection.py --device mps")

if __name__ == '__main__':
    main()
