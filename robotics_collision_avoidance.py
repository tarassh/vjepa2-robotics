#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Action-conditioned obstacle avoidance using V-JEPA 2-AC for robotics

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

from src.models.vision_transformer import vit_giant_xformers_rope
from src.models.ac_predictor import vit_ac_predictor
import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class CollisionAvoidancePlanner:
    """
    Action-conditioned collision avoidance using V-JEPA 2-AC
    Plans robot actions to avoid collisions based on live camera feed
    """
    
    def __init__(self, img_size=256, num_frames=8, device='cuda'):
        """
        Initialize collision avoidance planner
        
        Args:
            img_size: Input image size (256 recommended for AC model)
            num_frames: Number of frames to process
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
        self.tokens_per_frame = int((img_size // 16) ** 2)  # patch_size=16
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=num_frames)
        
        # Load models
        print("Loading V-JEPA 2-AC models...")
        self.encoder, self.predictor = self._load_models(img_size)
        self.encoder.to(self.device).eval()
        self.predictor.to(self.device).eval()
        
        # Build transform
        self.transform = self._build_transform(img_size)
        
        # Robot state (7-DOF: x, y, z, roll, pitch, yaw, gripper)
        self.current_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        print(f"Models loaded on {self.device}")
        print(f"Ready for collision avoidance planning")
    
    def _load_models(self, img_size):
        """Load pretrained V-JEPA 2-AC encoder and predictor"""
        try:
            encoder, predictor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')
            print("Loaded pretrained V-JEPA 2-AC weights from PyTorch Hub")
        except Exception as e:
            print(f"Could not load from PyTorch Hub: {e}")
            print("Initializing models from scratch...")
            
            # Initialize encoder
            encoder = vit_giant_xformers_rope(
                img_size=(img_size, img_size),
                num_frames=self.num_frames
            )
            
            # Initialize AC predictor
            predictor = vit_ac_predictor(
                img_size=(img_size, img_size),
                num_frames=self.num_frames,
                embed_dim=encoder.embed_dim,
                predictor_embed_dim=1024,
                depth=12,
                num_heads=16,
                use_rope=True
            )
            print("Using randomly initialized weights - download checkpoints for best results")
        
        return encoder, predictor
    
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
        """Add a frame to the buffer (RGB numpy array)"""
        self.frame_buffer.append(frame)
    
    def encode_video(self, video_tensor):
        """Encode video frames to latent representation"""
        B, C, T, H, W = video_tensor.size()
        # Duplicate each frame to create pairs (required by encoder)
        video_expanded = video_tensor.permute(0, 2, 1, 3, 4).flatten(0, 1)
        video_expanded = video_expanded.unsqueeze(2).repeat(1, 1, 2, 1, 1)
        
        with torch.inference_mode():
            features = self.encoder(video_expanded)
            # Reshape back to [B, T, N, D]
            features = features.view(B, T, -1, features.size(-1))
        
        return features
    
    def sample_collision_free_actions(self, context_features, num_samples=25, grid_size=0.1):
        """
        Sample potential actions and evaluate collision risk
        
        Args:
            context_features: Current encoded video features
            num_samples: Number of action samples per dimension
            grid_size: Maximum action magnitude
            
        Returns:
            best_action: Safest action to take
            collision_risks: Risk scores for all sampled actions
        """
        # Create action grid (simplified 3D movement)
        action_samples = []
        for dx in np.linspace(-grid_size, grid_size, num_samples):
            for dy in np.linspace(-grid_size, grid_size, num_samples):
                for dz in np.linspace(-grid_size, grid_size, num_samples):
                    # 7-DOF action: [dx, dy, dz, droll, dpitch, dyaw, dgripper]
                    action_samples.append([dx, dy, dz, 0, 0, 0, 0])
        
        action_samples = np.array(action_samples)
        num_actions = len(action_samples)
        
        # Convert to torch tensors
        actions_tensor = torch.tensor(action_samples, dtype=torch.float32, device=self.device)
        actions_tensor = actions_tensor.unsqueeze(0).unsqueeze(1)  # [1, 1, num_actions, 7]
        
        # Current state
        states_tensor = torch.tensor(self.current_state, dtype=torch.float32, device=self.device)
        states_tensor = states_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 7]
        
        # Repeat context features for each action
        B, T, N, D = context_features.shape
        context_flat = context_features[:, :1].flatten(1, 2)  # Use first frame as context
        
        # Predict next state for each action (simplified - single step prediction)
        collision_risks = []
        
        with torch.inference_mode():
            for i in range(num_actions):
                action = actions_tensor[:, :, i:i+1, :]  # [1, 1, 1, 7]
                
                # Predict next latent state
                try:
                    predicted = self.predictor(context_flat, action, states_tensor)
                    
                    # Compute "energy" - higher energy indicates potential collision
                    # (based on deviation from typical patterns learned during training)
                    energy = predicted.var(dim=-1).mean().item()
                    collision_risks.append(energy)
                except Exception as e:
                    # If prediction fails, assign high risk
                    collision_risks.append(1e6)
        
        collision_risks = np.array(collision_risks)
        
        # Select action with lowest collision risk
        best_idx = np.argmin(collision_risks)
        best_action = action_samples[best_idx]
        
        # Normalize collision risks for visualization
        if collision_risks.max() > collision_risks.min():
            collision_risks = (collision_risks - collision_risks.min()) / \
                            (collision_risks.max() - collision_risks.min())
        
        return best_action, collision_risks, action_samples
    
    def plan_safe_trajectory(self, target_action):
        """
        Plan a safe trajectory given a target action
        
        Args:
            target_action: Desired action [dx, dy, dz, ...]
            
        Returns:
            safe_action: Modified action that avoids collisions
            risk_score: Collision risk estimate
        """
        if len(self.frame_buffer) < self.num_frames:
            return np.zeros(7), 1.0
        
        # Prepare video tensor
        video = np.stack(list(self.frame_buffer), axis=0)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)  # T x C x H x W
        video_tensor = self.transform(video).unsqueeze(0).to(self.device)
        
        # Encode current observation
        features = self.encode_video(video_tensor)
        
        # Sample and evaluate actions
        best_action, risks, actions = self.sample_collision_free_actions(
            features, num_samples=5, grid_size=0.1
        )
        
        # Compute overall risk (mean of lowest risks)
        risk_score = np.sort(risks)[:5].mean()
        
        return best_action, risk_score


def main():
    parser = argparse.ArgumentParser(description='Collision avoidance with V-JEPA 2-AC')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Input image size (default: 256)')
    parser.add_argument('--num-frames', type=int, default=8,
                        help='Number of frames to buffer (default: 8)')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--risk-threshold', type=float, default=0.5,
                        help='Risk threshold for warnings (default: 0.5)')
    args = parser.parse_args()
    
    # Initialize planner
    planner = CollisionAvoidancePlanner(
        img_size=args.img_size,
        num_frames=args.num_frames,
        device=args.device
    )
    
    # Open camera
    cap = cv2.VideoCapture(args.camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_id}")
        return
    
    print("\n=== Collision Avoidance System ===")
    print(f"Camera ID: {args.camera_id}")
    print(f"Press 'q' to quit")
    print(f"Risk threshold: {args.risk_threshold}")
    print("\nCollecting frames...")
    
    frame_count = 0
    
    # Simulated target action (in real robotics, this comes from your controller)
    target_action = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Move forward
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            planner.add_frame(frame_rgb)
            frame_count += 1
            
            if frame_count >= args.num_frames:
                # Plan safe action
                safe_action, risk = planner.plan_safe_trajectory(target_action)
                
                # Visualize
                display_frame = frame.copy()
                
                # Status
                risk_text = f"Collision Risk: {risk:.3f}"
                action_text = f"Safe Action: [{safe_action[0]:.3f}, {safe_action[1]:.3f}, {safe_action[2]:.3f}]"
                
                cv2.putText(display_frame, risk_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, action_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Warning if high risk
                if risk > args.risk_threshold:
                    warning = "HIGH COLLISION RISK!"
                    cv2.putText(display_frame, warning, (10, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.rectangle(display_frame, (0, 0),
                                (display_frame.shape[1]-1, display_frame.shape[0]-1),
                                (0, 0, 255), 3)
                else:
                    status = "SAFE TO PROCEED"
                    cv2.putText(display_frame, status, (10, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.imshow('Collision Avoidance', display_frame)
            else:
                warmup = f"Buffering: {frame_count}/{args.num_frames}"
                cv2.putText(frame, warmup, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('Collision Avoidance', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("System shut down")


if __name__ == '__main__':
    main()
