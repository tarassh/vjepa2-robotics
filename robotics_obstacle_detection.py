#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Live obstacle detection from laptop camera using V-JEPA 2 for robotics applications

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add vjepa2 submodule to path for imports
vjepa2_path = Path(__file__).parent / 'vjepa2'
sys.path.insert(0, str(vjepa2_path))

from src.models.attentive_pooler import AttentiveClassifier
from src.models.vision_transformer import vit_giant_xformers_rope, vit_huge_rope, vit_large_rope
import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ObstacleDetector:
    """Live obstacle detection using V-JEPA 2 models"""
    
    def __init__(self, model_size='large', img_size=256, num_frames=16, device='cuda'):
        """
        Initialize obstacle detector
        
        Args:
            model_size: 'large', 'huge', or 'giant'
            img_size: Input image size (256 or 384)
            num_frames: Number of frames to process (default 16)
            device: 'cuda' or 'cpu'
        """
        # Handle device selection (cuda, mps, or cpu)
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = 'cpu'
        elif device == 'mps' and not torch.backends.mps.is_available():
            print("Warning: MPS not available, falling back to CPU")
            self.device = 'cpu'
        else:
            self.device = device
        self.img_size = img_size
        self.num_frames = num_frames
        self.model_size = model_size
        
        # Frame buffer to accumulate video frames
        self.frame_buffer = deque(maxlen=num_frames)
        
        # Load model
        print(f"Loading V-JEPA 2 {model_size} model...")
        self.model = self._load_model(model_size, img_size)
        self.model.to(self.device)
        self.model.eval()
        
        # Build preprocessing transform
        self.transform = self._build_transform(img_size)
        
        print(f"Model loaded on {self.device}")
        print(f"Processing {num_frames} frames at {img_size}x{img_size} resolution")
    
    def _load_model(self, model_size, img_size):
        """Load pretrained V-JEPA 2 model"""
        if model_size == 'large':
            model = vit_large_rope(img_size=(img_size, img_size), num_frames=self.num_frames)
        elif model_size == 'huge':
            model = vit_huge_rope(img_size=(img_size, img_size), num_frames=self.num_frames)
        elif model_size == 'giant':
            model = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=self.num_frames)
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        
        # Try to load PyTorch Hub weights if available
        try:
            if model_size == 'giant':
                model_dump = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant')
            elif model_size == 'huge':
                model_dump = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_huge')
            elif model_size == 'large':
                model_dump = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')
            print("Loaded pretrained weights from PyTorch Hub")
            # Extract model object from tuple if necessary
            if isinstance(model_dump, tuple):
                model = model_dump[0]
            else:
                model = model_dump
        except Exception as e:
            print(f"Could not load pretrained weights from PyTorch Hub: {e}")
            print("Using randomly initialized weights - for best results, download pretrained checkpoints")
        
        return model
    
    def _build_transform(self, img_size):
        """Build video preprocessing transform"""
        short_side_size = int(256.0 / 224 * img_size)
        transform = video_transforms.Compose([
            video_transforms.Resize(short_side_size, interpolation='bilinear'),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
        return transform
    
    def add_frame(self, frame):
        """Add a frame to the buffer (frame should be RGB numpy array)"""
        self.frame_buffer.append(frame)
    
    def detect_obstacles(self):
        """
        Process buffered frames and detect obstacles
        
        Returns:
            features: Encoded video features (can be used for downstream tasks)
            motion_score: Motion/change score (higher = more motion/potential obstacles)
        """
        if len(self.frame_buffer) < self.num_frames:
            return None, 0.0
        
        # Convert frame buffer to video tensor
        video = np.stack(list(self.frame_buffer), axis=0)  # T x H x W x C
        video = torch.from_numpy(video).permute(0, 3, 1, 2)  # T x C x H x W
        
        # Preprocess and add batch dimension
        video_tensor = self.transform(video).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.inference_mode():
            features = self.model(video_tensor)  # B x N x D
        
        # Compute motion score (simplified obstacle detection)
        # Higher variance in features indicates more motion/change
        motion_score = features.var(dim=1).mean().item()
        
        return features, motion_score
    
    def analyze_spatial_features(self, features):
        """
        Analyze spatial distribution of features to identify potential obstacle regions
        
        Returns:
            spatial_attention: H x W attention map
        """
        if features is None:
            return None
        
        # features shape: B x N x D
        # Reshape to spatial grid
        B, N, D = features.shape
        H = W = int(np.sqrt(N / (self.num_frames // 2)))  # Approximate spatial grid
        
        if H * W * (self.num_frames // 2) != N:
            # If not perfect square, use simple attention
            attention = features.norm(dim=-1).squeeze(0)  # N
            return attention.cpu().numpy()
        
        # Reshape to spatial grid and average over time
        spatial_features = features.view(B, self.num_frames // 2, H, W, D)
        spatial_attention = spatial_features.norm(dim=-1).mean(dim=1).squeeze(0)  # H x W
        
        return spatial_attention.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Live obstacle detection using V-JEPA 2')
    parser.add_argument('--model-size', type=str, default='large', choices=['large', 'huge', 'giant'],
                        help='Model size (default: large)')
    parser.add_argument('--img-size', type=int, default=256, choices=[256, 384],
                        help='Input image size (default: 256)')
    parser.add_argument('--num-frames', type=int, default=16,
                        help='Number of frames to buffer (default: 16)')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--motion-threshold', type=float, default=0.5,
                        help='Motion threshold for obstacle warning (default: 0.5)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target FPS for camera capture (default: 30)')
    args = parser.parse_args()
    
    # Initialize detector
    detector = ObstacleDetector(
        model_size=args.model_size,
        img_size=args.img_size,
        num_frames=args.num_frames,
        device=args.device
    )
    
    # Open camera
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_id}")
        return
    
    print("\n=== Live Obstacle Detection ===")
    print(f"Camera ID: {args.camera_id}")
    print("Press 'q' to quit")
    print(f"Motion threshold: {args.motion_threshold}")
    print("\nWarming up... collecting frames...")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add frame to detector
            detector.add_frame(frame_rgb)
            frame_count += 1
            
            # Process when buffer is full
            if frame_count >= args.num_frames:
                features, motion_score = detector.detect_obstacles()
                
                # Analyze spatial features for obstacle localization
                spatial_attention = detector.analyze_spatial_features(features)
                
                # Determine if obstacles detected
                obstacle_detected = motion_score > args.motion_threshold
                
                # Visualize on frame
                display_frame = frame.copy()
                
                # Add status text
                status_text = f"Motion Score: {motion_score:.3f}"
                cv2.putText(display_frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if obstacle_detected:
                    warning_text = "OBSTACLE DETECTED!"
                    cv2.putText(display_frame, warning_text, (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    # Add red border
                    cv2.rectangle(display_frame, (0, 0), 
                                (display_frame.shape[1]-1, display_frame.shape[0]-1),
                                (0, 0, 255), 5)
                else:
                    clear_text = "Clear"
                    cv2.putText(display_frame, clear_text, (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                
                # Overlay spatial attention if available
                if spatial_attention is not None and len(spatial_attention.shape) == 2:
                    # Resize attention map to frame size
                    attention_resized = cv2.resize(spatial_attention, 
                                                  (frame.shape[1], frame.shape[0]))
                    # Normalize to 0-255
                    attention_norm = (attention_resized - attention_resized.min()) / \
                                    (attention_resized.max() - attention_resized.min() + 1e-8)
                    attention_heatmap = (attention_norm * 255).astype(np.uint8)
                    attention_colored = cv2.applyColorMap(attention_heatmap, cv2.COLORMAP_JET)
                    # Blend with original frame
                    display_frame = cv2.addWeighted(display_frame, 0.7, attention_colored, 0.3, 0)
                
                # Display
                cv2.imshow('Obstacle Detection', display_frame)
            else:
                # Still warming up
                warmup_text = f"Buffering frames: {frame_count}/{args.num_frames}"
                cv2.putText(frame, warmup_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('Obstacle Detection', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")


if __name__ == '__main__':
    main()
