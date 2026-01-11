# How Obstacle Detection Works with VJEPA2

A comprehensive guide to understanding obstacle detection using V-JEPA 2 self-supervised video models.

## Table of Contents

1. [Overview](#overview)
2. [The Problem](#the-problem)
3. [How VJEPA2 Works](#how-vjepa2-works)
4. [Obstacle Detection Pipeline](#obstacle-detection-pipeline)
5. [Understanding the Results](#understanding-the-results)
6. [Interpreting Motion Scores](#interpreting-motion-scores)
7. [Spatial Attention Maps](#spatial-attention-maps)
8. [Practical Guidelines](#practical-guidelines)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Topics](#advanced-topics)

---

## Overview

Traditional obstacle detection relies on:
- **Depth sensors** (LiDAR, stereo cameras) - expensive, power-hungry
- **Object detection models** (YOLO, R-CNN) - require labeled data, specific to object classes
- **Background subtraction** - fragile to lighting changes

**V-JEPA 2 approach:**
- **Self-supervised learning** on internet videos
- **No labels required** - learns from video structure
- **Monocular RGB camera** - works with any camera
- **Motion understanding** - detects change, not specific objects

---

## The Problem

### What is Obstacle Detection?

For robotics, obstacle detection means:

1. **Identify** when something enters the robot's path
2. **Localize** where the obstacle is in the visual field
3. **React quickly** to avoid collision (< 100ms ideally)
4. **Work in real-world conditions** (varying lighting, textures, speeds)

### Challenges:

- **Static objects** after entering (become "background")
- **Fast-moving obstacles** (require low latency)
- **Unknown object types** (never-seen-before obstacles)
- **Varying appearances** (transparent, reflective, small objects)
- **Real-time requirements** (30+ FPS for safety)

---

## How VJEPA2 Works

### What is V-JEPA 2?

**V-JEPA** = Video Joint-Embedding Predictive Architecture

### Training Process (Self-Supervised):

```
1. Take video from internet (millions of videos)
2. Mask out some patches in space and time
3. Predict the masked content from visible context
4. Learn representations that capture:
   - How objects move
   - What changes in scenes
   - Physical relationships
   - Temporal consistency
```

**Key insight**: To predict masked video content, the model must understand physics, motion, and scene dynamics.

### Architecture:

```
Video (T frames) 
    ↓
[Patch Embedding]  ← Breaks video into space-time patches
    ↓
[Transformer Blocks (24 layers)]  ← Self-attention learns relationships
    ↓
Features (B × N × D)  ← Learned representations
```

Where:
- **T** = temporal dimension (number of frames)
- **N** = spatial-temporal tokens (~4096 for 16 frames)
- **D** = feature dimension (1024 for ViT-Large)

### What VJEPA2 Learns:

The model doesn't explicitly learn "obstacle detection" but learns to:

1. **Represent motion patterns** - how things move in the world
2. **Identify changes** - what's different from frame to frame
3. **Understand physics** - objects don't teleport, they move smoothly
4. **Recognize scene structure** - foreground vs background, moving vs static

These learned representations turn out to be **perfect for obstacle detection**!

---

## Obstacle Detection Pipeline

### Step 1: Video Buffering

```python
frame_buffer = deque(maxlen=16)  # Sliding window

# Continuously add frames
frame_buffer.append(new_frame)
```

**Why buffer frames?**
- VJEPA2 needs temporal context (multiple frames)
- 8-16 frames typical (0.25-0.5 seconds at 30 FPS)
- Sliding window for continuous detection

### Step 2: Preprocessing

```python
# Convert to tensor
video = torch.from_numpy(frames)  # T × H × W × C

# Resize and normalize
video = transform(video)  # T × C × H × W, normalized

# Add batch dimension
video = video.unsqueeze(0)  # B × C × T × H × W
```

**Preprocessing steps:**
1. **Resize** to model input size (256×256 or 192×192)
2. **Normalize** with ImageNet mean/std
3. **Convert** to PyTorch tensor format

### Step 3: Feature Extraction

```python
with torch.inference_mode():
    features = model(video)  # B × N × D
```

**What happens inside:**

```
Input video (1 × 3 × 16 × 256 × 256)
    ↓
Patch embedding: breaks into 16×16 patches
    → Creates 16 frames × 16×16 spatial = 4096 tokens
    ↓
24 Transformer layers
    → Each token attends to all others
    → Learns: "this patch changed", "that region is moving"
    ↓
Output features (1 × 4096 × 1024)
    → Rich representation of video content
```

### Step 4: Motion Scoring

```python
motion_score = features.var(dim=1).mean().item()
```

**Why variance?**

Consider a static scene:
```
Token 1: [0.5, 0.3, 0.8, ...]  ─┐
Token 2: [0.5, 0.3, 0.8, ...]   ├─ Very similar
Token 3: [0.5, 0.3, 0.8, ...]  ─┘
→ Low variance = No motion
```

Scene with obstacle entering:
```
Token 1 (background): [0.5, 0.3, 0.8, ...]  ─┐
Token 2 (obstacle):   [0.1, 0.9, 0.2, ...]   ├─ Very different!
Token 3 (background): [0.5, 0.3, 0.8, ...]  ─┘
→ High variance = Motion detected
```

**Mathematical reasoning:**

- Static scene → All tokens have consistent features → Low variance
- Moving object → Tokens in that region differ → High variance
- Variance across spatial dimension captures "how much changed"

### Step 5: Spatial Localization

```python
# Reshape to spatial grid
spatial_features = features.view(B, T, H, W, D)

# Compute attention per location
spatial_attention = spatial_features.norm(dim=-1).mean(dim=1)
```

**Creates a heatmap:**

```
Low attention (blue):  Background, static regions
High attention (red):  Moving regions, obstacles
```

---

## Understanding the Results

### Output Structure:

```python
features, motion_score = detector.detect_obstacles()
```

**Features** (`torch.Tensor`):
- Shape: `(1, 4096, 1024)` for ViT-Large with 16 frames
- Each of 4096 tokens has 1024-dimensional feature vector
- Represents learned understanding of the video

**Motion Score** (`float`):
- Range: Typically 0.0 to 10.0+
- Higher = more motion/change detected
- Lower = static scene

### What the Score Means:

| Motion Score | Interpretation | Action |
|--------------|----------------|--------|
| **< 0.1** | Static scene, no change | Continue |
| **0.1 - 0.5** | Minor motion (lighting change, camera vibration) | Monitor |
| **0.5 - 2.0** | Moderate motion (something moving nearby) | Alert |
| **2.0 - 5.0** | Significant motion (obstacle in path) | ⚠️ Warning |
| **> 5.0** | Major change (obstacle very close or fast-moving) | 🛑 Emergency stop |

**Important**: These are guidelines! Calibrate for your specific:
- Camera setup
- Environment (indoor vs outdoor)
- Robot speed
- Safety requirements

---

## Interpreting Motion Scores

### Factors Affecting Scores:

#### 1. **Obstacle Size**
```
Small obstacle (coffee cup):   Score ~0.5-1.0
Medium obstacle (chair):        Score ~1.0-3.0
Large obstacle (person):        Score ~2.0-5.0
```

#### 2. **Obstacle Speed**
```
Slowly entering frame:   Score increases gradually
Fast motion:            Score spikes suddenly
Stationary after entry: Score decreases over time
```

#### 3. **Environment**
```
Indoor, good lighting:   Stable scores
Outdoor, sunlight:      Variable (shadows cause false positives)
Low light:              Lower sensitivity
```

#### 4. **Camera Motion**
```
Stationary camera:      Clean scores
Moving robot:          Higher baseline (ego-motion)
Vibration:             Noisy scores
```

### Temporal Patterns:

**Obstacle entering frame:**
```
Frame:  1    2    3    4    5    6    7    8
Score: 0.2  0.3  0.8  2.1  3.5  4.2  3.8  3.5
         └────┴────┴────┴── Obstacle detected!
```

**False alarm (lighting change):**
```
Frame:  1    2    3    4    5    6    7    8
Score: 0.2  0.3  0.9  1.2  0.4  0.3  0.2  0.2
         └────┴────┴── Brief spike, then back to normal
```

**Strategy**: Use temporal smoothing!

```python
# Exponential moving average
smoothed_score = 0.7 * prev_score + 0.3 * current_score

# Or median filter
from collections import deque
score_history = deque(maxlen=5)
score_history.append(current_score)
filtered_score = np.median(score_history)
```

---

## Spatial Attention Maps

### What is the Attention Map?

A heatmap showing **where** the model is "attending" (where change is detected).

```python
spatial_attention = detector.analyze_spatial_features(features)
# Shape: (H, W) e.g., (16, 16)
# Values: 0.0 (no attention) to ~1.0 (high attention)
```

### Visualizing:

```python
import cv2
import numpy as np

# Resize to frame size
attention_resized = cv2.resize(spatial_attention, (frame_width, frame_height))

# Normalize
attention_norm = (attention_resized - attention_resized.min()) / \
                 (attention_resized.max() - attention_resized.min())

# Create heatmap
heatmap = cv2.applyColorMap((attention_norm * 255).astype(np.uint8), 
                            cv2.COLORMAP_JET)

# Overlay on frame
display = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
```

### Interpreting Spatial Patterns:

**Localized attention (obstacle detected):**
```
┌─────────────────┐
│ · · · · · · · · │  Background: low attention (blue)
│ · · ███ · · · · │  
│ · · ███ · · · · │  Obstacle: high attention (red)
│ · · ███ · · · · │
│ · · · · · · · · │
└─────────────────┘
```

**Distributed attention (camera motion):**
```
┌─────────────────┐
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  Entire frame has
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  medium attention
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  = Global motion
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
└─────────────────┘
```

### Using Spatial Information:

```python
# Find peak attention location
peak_y, peak_x = np.unravel_index(spatial_attention.argmax(), 
                                   spatial_attention.shape)

# Convert to frame coordinates
obstacle_x = (peak_x / spatial_attention.shape[1]) * frame_width
obstacle_y = (peak_y / spatial_attention.shape[0]) * frame_height

# Determine obstacle direction
if obstacle_x < frame_width / 3:
    direction = "LEFT"
elif obstacle_x > 2 * frame_width / 3:
    direction = "RIGHT"
else:
    direction = "CENTER"

print(f"Obstacle detected at {direction}")
```

---

## Practical Guidelines

### 1. **Calibrate Your Threshold**

Don't use fixed threshold! Test in your environment:

```python
# Calibration procedure
print("Calibrating... Point camera at empty space")
baseline_scores = []
for i in range(100):
    frame = capture_frame()
    detector.add_frame(frame)
    if i >= 16:  # After buffer full
        _, score = detector.detect_obstacles()
        baseline_scores.append(score)

baseline_mean = np.mean(baseline_scores)
baseline_std = np.std(baseline_scores)

# Set threshold: mean + 3 * std
threshold = baseline_mean + 3 * baseline_std
print(f"Recommended threshold: {threshold:.2f}")
```

### 2. **Temporal Filtering**

Reduce false positives with smoothing:

```python
class SmoothedDetector:
    def __init__(self, detector, window_size=5):
        self.detector = detector
        self.score_history = deque(maxlen=window_size)
    
    def detect(self, frame):
        self.detector.add_frame(frame)
        features, score = self.detector.detect_obstacles()
        
        self.score_history.append(score)
        
        # Median filter
        smoothed_score = np.median(self.score_history)
        
        return features, smoothed_score
```

### 3. **Multi-Level Alerts**

Graduated response based on severity:

```python
def classify_threat(score, threshold):
    if score < threshold:
        return "CLEAR", None
    elif score < threshold * 2:
        return "CAUTION", "slow_down"
    elif score < threshold * 4:
        return "WARNING", "prepare_stop"
    else:
        return "DANGER", "emergency_stop"

level, action = classify_threat(motion_score, threshold)
```

### 4. **Region of Interest (ROI)**

Focus on relevant parts of frame:

```python
# Only check bottom half (where ground-level obstacles are)
roi_mask = np.zeros_like(spatial_attention)
roi_mask[spatial_attention.shape[0]//2:, :] = 1

# Apply mask
relevant_attention = spatial_attention * roi_mask
roi_score = relevant_attention.mean()
```

### 5. **Combine with Other Sensors**

VJEPA2 is vision-based. Combine with:

```python
# Sensor fusion
vision_confidence = motion_score > threshold
ultrasonic_distance = read_ultrasonic()

if vision_confidence and ultrasonic_distance < 30:
    # Both sensors agree: definitely an obstacle
    emergency_stop()
elif vision_confidence or ultrasonic_distance < 20:
    # One sensor triggered: proceed with caution
    slow_down()
```

---

## Troubleshooting

### Issue: Constant High Scores

**Cause**: Camera vibration, auto-exposure changes

**Solution**:
```python
# Increase baseline threshold
threshold *= 1.5

# Or disable auto-exposure
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
```

### Issue: Missing Slow-Moving Obstacles

**Cause**: Gradual change not detected as motion

**Solution**:
```python
# Use longer frame buffer
num_frames = 32  # Instead of 16

# Or compare to baseline
first_frame_features = features_at_t0
current_features = features_at_t_now
change = (current_features - first_frame_features).abs().mean()
```

### Issue: False Positives from Shadows

**Cause**: Lighting changes trigger motion detection

**Solution**:
```python
# Spatial filtering: ignore diffuse changes
attention_threshold = 0.6
localized_attention = spatial_attention > attention_threshold
if localized_attention.sum() > 10:  # At least 10 pixels
    # Localized motion = real obstacle
    alert()
```

### Issue: Detection Lag

**Cause**: Frame buffer needs to fill

**Solution**:
```python
# Reduce frame count
num_frames = 8  # Faster response

# Pre-fill buffer
for _ in range(num_frames):
    detector.add_frame(initial_frame)
```

---

## Advanced Topics

### 1. **Distinguishing Static from Moving Obstacles**

```python
class PersistentObstacleTracker:
    def __init__(self):
        self.obstacle_history = {}
    
    def track(self, spatial_attention, frame_id):
        # Find blobs in attention map
        from scipy import ndimage
        labeled, num_features = ndimage.label(spatial_attention > 0.5)
        
        for i in range(1, num_features + 1):
            blob = (labeled == i)
            centroid = ndimage.center_of_mass(blob)
            
            # Track over time
            if centroid in self.obstacle_history:
                # Check if moved
                prev_centroid = self.obstacle_history[centroid]['pos']
                distance = np.linalg.norm(np.array(centroid) - prev_centroid)
                
                if distance < 2:  # pixels
                    self.obstacle_history[centroid]['static_frames'] += 1
                else:
                    self.obstacle_history[centroid]['static_frames'] = 0
            else:
                self.obstacle_history[centroid] = {
                    'pos': centroid,
                    'static_frames': 0
                }
```

### 2. **Depth Estimation from Motion**

While VJEPA2 doesn't output depth directly, motion parallax provides cues:

```python
# Closer objects have larger optical flow
# This is implicit in VJEPA2's learned features

# Approximate: higher attention magnitude = closer
distance_estimate = 1.0 / (spatial_attention.max() + 0.1)
```

### 3. **Predicting Future Collisions**

```python
# Track obstacle trajectory
obstacle_positions = []  # [(x, y, t), ...]

# Fit velocity
if len(obstacle_positions) >= 3:
    velocities = np.diff(obstacle_positions, axis=0)
    avg_velocity = velocities.mean(axis=0)
    
    # Extrapolate
    future_position = obstacle_positions[-1] + avg_velocity * timesteps
    
    # Check if in robot's path
    if is_collision_path(future_position, robot_trajectory):
        alert("Collision predicted in {timesteps * dt:.1f}s")
```

### 4. **Fine-Tuning for Your Robot**

Collect data and fine-tune:

```python
# Collect examples
examples = []
for scene in ["empty_room", "person_entering", "chair_moving"]:
    video, label = record_scenario(scene)
    features = detector.model(video)
    examples.append((features, label))

# Train simple classifier on top
from sklearn.svm import SVC
X = [f.mean(dim=1).cpu().numpy() for f, _ in examples]
y = [label for _, label in examples]
classifier = SVC().fit(X, y)

# Use in production
features, _ = detector.detect_obstacles()
X_new = features.mean(dim=1).cpu().numpy()
prediction = classifier.predict(X_new)
```

---

## Summary

### Key Takeaways:

1. **VJEPA2 learns motion understanding** from self-supervised video training
2. **Motion score = feature variance** across spatial-temporal dimensions
3. **Spatial attention shows where** obstacles are located
4. **Calibrate threshold** for your specific environment and use case
5. **Temporal filtering** reduces false positives
6. **Combine with other sensors** for robust detection

### Best Practices:

✅ Start with default threshold (0.5), calibrate for your setup  
✅ Use temporal smoothing (median filter over 5 frames)  
✅ Visualize spatial attention during development  
✅ Test in various lighting conditions  
✅ Combine with ultrasonic/LiDAR when safety-critical  
✅ Monitor frame rate - 10+ FPS minimum for safety

### Limitations to Remember:

❌ No explicit depth information (monocular)  
❌ Static obstacles after entering become "background"  
❌ Requires temporal context (frame buffer latency)  
❌ Performance depends on lighting conditions  
❌ Not trained specifically for obstacle detection

Despite limitations, VJEPA2 provides a **powerful, flexible, no-training-required** solution for robotics obstacle detection!

---

## References

- [V-JEPA 2 Paper](https://arxiv.org/abs/2506.09985)
- [V-JEPA 2 GitHub](https://github.com/facebookresearch/vjepa2)
- [This Repository](https://github.com/tarassh/vjepa2-robotics)
