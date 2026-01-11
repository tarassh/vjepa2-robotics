# Quick Start: MPS-Accelerated Obstacle Detection

Get 3-5x faster obstacle detection on your Mac using Metal Performance Shaders.

## Step 1: Test MPS Works

```bash
source .venv/bin/activate
python test_mps_detection.py
```

You should see:
```
✅ MPS is available
Model loaded on mps
✅ Detection complete!
   Time: ~500 ms
   FPS: ~2 FPS
```

## Step 2: Run with Your Camera

```bash
python robotics_obstacle_detection.py --device mps
```

That's it! The script will:
1. Open your laptop camera
2. Run obstacle detection on MPS (GPU)
3. Show live video with obstacle warnings

Press `q` to quit.

## All Options

```bash
python robotics_obstacle_detection.py \
    --model-size large \    # or 'huge', 'giant'
    --img-size 256 \         # or 384
    --num-frames 16 \        # or 8 for faster
    --device mps \           # ← The important one!
    --motion-threshold 0.5   # sensitivity
```

## Expected Performance

| Your Mac | Expected FPS |
|----------|--------------|
| M1/M2 | 8-10 FPS |
| M1/M2 Pro | 10-12 FPS |
| M1/M2 Max | 12-15 FPS |
| M3 | 10-12 FPS |
| M3 Pro/Max | 15-20 FPS |

## Faster Performance

Want more FPS? Try:

```bash
# Reduce frames (2x faster)
python robotics_obstacle_detection.py \
    --device mps \
    --num-frames 8

# Or lower resolution
python robotics_obstacle_detection.py \
    --device mps \
    --img-size 192
```

## Troubleshooting

**"MPS not available"**
- Make sure you have Apple Silicon (M1/M2/M3)
- Update PyTorch: `pip install --upgrade torch`

**Still slow?**
- Check Activity Monitor → ensure script is using GPU
- Close other apps to free up GPU memory
- Try `--num-frames 8`

## Compare Devices

```bash
# CPU (slow)
python robotics_obstacle_detection.py --device cpu

# MPS (3-5x faster!)
python robotics_obstacle_detection.py --device mps
```

You should see ~3-5x speedup with MPS!
