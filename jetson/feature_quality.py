#!/usr/bin/env python3
"""
V-JEPA 2 feature quality validation for Jetson.

Creates synthetic video patterns with distinct visual characteristics and
measures whether V-JEPA 2 embeddings can discriminate between them. This
validates that the model produces meaningful features on edge hardware.

Test patterns:
  - static_red / static_blue: Static solid colors (no motion)
  - horizontal_motion / vertical_motion: Moving bars in different directions
  - expanding_circle / shrinking_circle: Circles changing size
  - checkerboard_flash: Alternating checkerboard pattern
  - diagonal_sweep: Diagonal gradient sweeping across frame

Pass criteria:
  - Within-category similarity > 0.8
  - Cross-category similarity < within-category
  - Discrimination gap > 0.05

Usage:
    python jetson/feature_quality.py
    python jetson/feature_quality.py --frames 16 --resolution 256
"""

import argparse
import gc
import numpy as np
import torch


def create_video(pattern, n_frames=8, res=224):
    """Create a synthetic video clip with a distinct visual pattern."""
    frames = []
    for t in range(n_frames):
        frame = np.zeros((res, res, 3), dtype=np.float32)

        if pattern == "static_red":
            frame[:, :, 0] = 0.8

        elif pattern == "static_blue":
            frame[:, :, 2] = 0.8

        elif pattern == "horizontal_motion":
            x_pos = int((t / n_frames) * res)
            x0, x1 = max(0, x_pos - 15), min(res, x_pos + 15)
            frame[:, x0:x1, :] = 1.0

        elif pattern == "vertical_motion":
            y_pos = int((t / n_frames) * res)
            y0, y1 = max(0, y_pos - 15), min(res, y_pos + 15)
            frame[y0:y1, :, :] = 1.0

        elif pattern == "expanding_circle":
            cy, cx = res // 2, res // 2
            radius = int((t / n_frames) * res * 0.4) + 5
            Y, X = np.ogrid[:res, :res]
            mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
            frame[mask] = [0.9, 0.9, 0.0]

        elif pattern == "shrinking_circle":
            cy, cx = res // 2, res // 2
            radius = int(((n_frames - t) / n_frames) * res * 0.4) + 5
            Y, X = np.ogrid[:res, :res]
            mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
            frame[mask] = [0.0, 0.9, 0.0]

        elif pattern == "checkerboard_flash":
            block = 32
            for i in range(0, res, block):
                for j in range(0, res, block):
                    if ((i // block) + (j // block) + t) % 2 == 0:
                        frame[i:i + block, j:j + block, :] = 0.9

        elif pattern == "diagonal_sweep":
            for i in range(res):
                for j in range(res):
                    phase = (i + j) / (2 * res) - t / n_frames
                    val = max(0, 1.0 - abs(phase) * 4)
                    frame[i, j] = [val, val * 0.5, val * 0.8]

        frames.append(frame)

    video = np.stack(frames).transpose(0, 3, 1, 2)  # (T, C, H, W)
    return torch.from_numpy(video).unsqueeze(0).half()  # (1, T, C, H, W)


def cosine_sim(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0)
    ).item()


PATTERNS = [
    "static_red",
    "static_blue",
    "horizontal_motion",
    "vertical_motion",
    "expanding_circle",
    "shrinking_circle",
    "checkerboard_flash",
    "diagonal_sweep",
]

# Pairs that should be similar (within-category)
WITHIN_PAIRS = [
    ("static_red", "static_blue"),
    ("horizontal_motion", "vertical_motion"),
    ("expanding_circle", "shrinking_circle"),
]

# Pairs that should be less similar (cross-category)
CROSS_PAIRS = [
    ("static_red", "horizontal_motion"),
    ("horizontal_motion", "expanding_circle"),
]


def main():
    parser = argparse.ArgumentParser(description='V-JEPA 2 Feature Quality Test')
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--resolution', type=int, default=224)
    parser.add_argument('--model', type=str, default='facebook/vjepa2-vitl-fpc64-256')
    args = parser.parse_args()

    print("Loading V-JEPA 2...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        args.model, dtype=torch.float16
    ).to("cuda").eval()

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: {free / 1e9:.2f} GB free / {total / 1e9:.2f} GB total")

    # Extract features
    print(f"\nExtracting features ({args.frames} frames × {args.resolution}×{args.resolution})...")
    features = {}
    for name in PATTERNS:
        video = create_video(name, n_frames=args.frames, res=args.resolution).to("cuda")
        torch.cuda.empty_cache()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            feat = model.get_vision_features(video)
        features[name] = feat.mean(dim=1).cpu()
        print(f"  ✅ {name}: {feat.shape} → pooled {features[name].shape}")
        del video, feat

    # Similarity matrix
    print(f"\n{'=' * 70}")
    print("COSINE SIMILARITY MATRIX")
    print(f"{'=' * 70}")
    short = [p[:8] for p in PATTERNS]
    print(f"{'':>16}" + "".join(f"{n:>10}" for n in short))
    for i, p1 in enumerate(PATTERNS):
        row = f"{PATTERNS[i][:16]:>16}"
        for j, p2 in enumerate(PATTERNS):
            row += f"{cosine_sim(features[p1], features[p2]):>10.3f}"
        print(row)

    # Analysis
    within_sims = [cosine_sim(features[a], features[b]) for a, b in WITHIN_PAIRS]
    cross_sims = [cosine_sim(features[a], features[b]) for a, b in CROSS_PAIRS]
    avg_within = sum(within_sims) / len(within_sims)
    avg_cross = sum(cross_sims) / len(cross_sims)
    gap = avg_within - avg_cross

    print(f"\n{'=' * 70}")
    print("ANALYSIS")
    print(f"{'=' * 70}")
    print("Within-category:")
    for (a, b), sim in zip(WITHIN_PAIRS, within_sims):
        print(f"  {a} ↔ {b}: {sim:.3f}")
    print(f"Cross-category:")
    for (a, b), sim in zip(CROSS_PAIRS, cross_sims):
        print(f"  {a} ↔ {b}: {sim:.3f}")

    self_sim = cosine_sim(features["static_red"], features["static_red"])
    print(f"\nSelf-similarity: {self_sim:.3f}")

    # Repeatability
    v1 = create_video("horizontal_motion", args.frames, args.resolution).to("cuda")
    v2 = create_video("horizontal_motion", args.frames, args.resolution).to("cuda")
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        f1 = model.get_vision_features(v1).mean(dim=1).cpu()
        f2 = model.get_vision_features(v2).mean(dim=1).cpu()
    print(f"Repeatability: {cosine_sim(f1, f2):.6f}")

    print(f"\n{'=' * 70}")
    print(f"Avg within-category:  {avg_within:.3f}")
    print(f"Avg cross-category:   {avg_cross:.3f}")
    print(f"Discrimination gap:   {gap:.3f}")

    if gap > 0.05:
        print("✅ PASS — features are meaningful and discriminative")
    elif gap > 0:
        print("⚠️  MARGINAL — features show weak discrimination")
    else:
        print("❌ FAIL — features are not discriminating")

    return 0 if gap > 0.05 else 1


if __name__ == "__main__":
    exit(main())
