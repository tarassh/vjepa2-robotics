# V-JEPA2 for Robotics Obstacle Detection

This guide shows how to use V-JEPA2 for live obstacle detection and collision avoidance from a laptop camera feed.

## Overview

Two applications are provided:

1. **`robotics_obstacle_detection.py`** - Basic obstacle detection using V-JEPA2 encoder
   - Detects motion and potential obstacles in the camera view
   - Provides spatial attention maps showing where obstacles are located
   - Good for general obstacle awareness

2. **`robotics_collision_avoidance.py`** - Advanced collision avoidance using V-JEPA2-AC (Action-Conditioned)
   - Plans robot actions to avoid collisions
   - Uses the action-conditioned world model to predict outcomes
   - Samples multiple possible actions and selects the safest one

## Installation

First, ensure you have the vjepa2 environment set up:

```bash
conda activate vjepa2-312
```

Install OpenCV for camera support:

```bash
pip install opencv-python
```

## Quick Start

### Basic Obstacle Detection

Run with default settings (uses 'large' model, runs on GPU if available):

```bash
python robotics_obstacle_detection.py
```

With custom options:

```bash
python robotics_obstacle_detection.py \
    --model-size large \
    --img-size 256 \
    --num-frames 16 \
    --camera-id 0 \
    --device cuda \
    --motion-threshold 0.5
```

### Advanced Collision Avoidance

Run the action-conditioned planner:

```bash
python robotics_collision_avoidance.py
```

With custom options:

```bash
python robotics_collision_avoidance.py \
    --img-size 256 \
    --num-frames 8 \
    --camera-id 0 \
    --device cuda \
    --risk-threshold 0.5
```

## Arguments

### Obstacle Detection

- `--model-size`: Model size (`large`, `huge`, `giant`) - default: `large`
- `--img-size`: Input image resolution (256 or 384) - default: `256`
- `--num-frames`: Number of frames to buffer - default: `16`
- `--camera-id`: Camera device ID - default: `0`
- `--device`: Device to use (`cuda` or `cpu`) - default: `cuda`
- `--motion-threshold`: Motion detection threshold - default: `0.5`
- `--fps`: Target camera FPS - default: `30`

### Collision Avoidance

- `--img-size`: Input image resolution - default: `256`
- `--num-frames`: Number of frames to buffer - default: `8`
- `--camera-id`: Camera device ID - default: `0`
- `--device`: Device to use (`cuda` or `cpu`) - default: `cuda`
- `--risk-threshold`: Risk threshold for warnings - default: `0.5`

## Using Pretrained Weights

For best results, download pretrained model checkpoints:

### V-JEPA2 Models (for obstacle detection)

```bash
# ViT-Large (300M params, fastest)
wget https://dl.fbaipublicfiles.com/vjepa2/vitl.pt -P ./checkpoints/

# ViT-Huge (600M params)
wget https://dl.fbaipublicfiles.com/vjepa2/vith.pt -P ./checkpoints/

# ViT-Giant (1B params, best accuracy)
wget https://dl.fbaipublicfiles.com/vjepa2/vitg.pt -P ./checkpoints/
```

### V-JEPA2-AC Model (for collision avoidance)

```bash
# Action-conditioned ViT-Giant
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt -P ./checkpoints/
```

The scripts will automatically try to load pretrained weights via PyTorch Hub. If that fails, they'll work with randomly initialized weights (but performance will be limited).

## Integration with Robotics Systems

### As a ROS Node

To integrate with ROS (Robot Operating System):

1. Subscribe to camera topics for video input
2. Use the `ObstacleDetector` or `CollisionAvoidancePlanner` class
3. Publish obstacle warnings or safe actions to your robot controller

Example skeleton:

```python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from robotics_obstacle_detection import ObstacleDetector

class ObstacleDetectionNode:
    def __init__(self):
        self.detector = ObstacleDetector(model_size='large')
        self.bridge = CvBridge()
        
        rospy.Subscriber('/camera/image_raw', Image, self.callback)
        self.pub = rospy.Publisher('/obstacles/detected', Bool, queue_size=10)
    
    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.detector.add_frame(frame)
        features, motion_score = self.detector.detect_obstacles()
        
        if motion_score > 0.5:
            self.pub.publish(True)
```

### Direct Integration

For direct integration into your robotics code:

```python
from robotics_collision_avoidance import CollisionAvoidancePlanner

# Initialize planner
planner = CollisionAvoidancePlanner(img_size=256, num_frames=8)

# In your control loop:
while robot_running:
    # Get camera frame
    frame = camera.capture()
    planner.add_frame(frame)
    
    # Get desired action from your controller
    target_action = get_target_action()
    
    # Plan safe action
    safe_action, risk = planner.plan_safe_trajectory(target_action)
    
    # Execute safe action if risk is acceptable
    if risk < 0.5:
        robot.execute_action(safe_action)
    else:
        robot.stop()  # Emergency stop
```

## How It Works

### Obstacle Detection

1. **Frame Buffering**: Accumulates video frames from the camera
2. **Video Encoding**: Uses V-JEPA2 encoder to extract spatio-temporal features
3. **Motion Analysis**: Computes variance in features to detect motion/changes
4. **Spatial Localization**: Generates attention maps showing where obstacles are

### Collision Avoidance

1. **Video Encoding**: Encodes current camera view with V-JEPA2
2. **Action Sampling**: Generates multiple candidate robot actions
3. **Prediction**: Uses AC predictor to simulate outcome of each action
4. **Risk Evaluation**: Scores each action based on predicted future state
5. **Safe Action Selection**: Chooses the action with lowest collision risk

## Performance Notes

- **Model Size**: `large` is fastest but less accurate, `giant` is most accurate but slower
- **Frame Buffer**: Smaller buffers (8 frames) are faster, larger (16-32) more accurate
- **GPU**: Strongly recommended - CPU will be very slow
- **Resolution**: 256px is faster, 384px more accurate for fine details
- **MacOS**: Requires `eva-decord` or `decord2` (see main README)

## Customization

### Adjust Motion Sensitivity

Modify `motion_threshold` to be more or less sensitive:
- Lower values (0.1-0.3): More sensitive, more false positives
- Higher values (0.7-0.9): Less sensitive, may miss obstacles

### Custom Action Space

Edit `sample_collision_free_actions()` in `robotics_collision_avoidance.py` to match your robot's action space:

```python
# Example: 2D navigation robot (x, y, theta)
action_samples = []
for dx in np.linspace(-0.5, 0.5, num_samples):
    for dy in np.linspace(-0.5, 0.5, num_samples):
        for dtheta in np.linspace(-0.3, 0.3, num_samples):
            action_samples.append([dx, dy, dtheta])
```

### Add Goal-Directed Planning

Modify `plan_safe_trajectory()` to bias towards a goal:

```python
def plan_safe_trajectory(self, target_action, goal_position):
    best_action, risks, actions = self.sample_collision_free_actions(...)
    
    # Compute distance to goal for each action
    goal_distances = [np.linalg.norm(action[:3] - goal_position) 
                     for action in actions]
    
    # Combine risk and goal distance
    scores = risks + 0.5 * np.array(goal_distances)
    best_idx = np.argmin(scores)
    
    return actions[best_idx], risks[best_idx]
```

## Troubleshooting

### Camera Not Found
```
Error: Could not open camera 0
```
- Try different camera IDs: `--camera-id 1` or `--camera-id 2`
- Check camera permissions in System Preferences (macOS)
- Ensure no other application is using the camera

### Out of Memory
```
CUDA out of memory
```
- Use smaller model: `--model-size large`
- Reduce resolution: `--img-size 256`
- Reduce frame buffer: `--num-frames 8`
- Use CPU: `--device cpu` (will be slower)

### Slow Performance
- Use GPU if available
- Reduce `num_frames`
- Use `large` model instead of `giant`
- Lower camera resolution

### Import Errors
```
ModuleNotFoundError: No module named 'decord'
```
- On macOS: `pip install eva-decord` or `pip install decord2`
- On Linux/Windows: `pip install decord`

## Citations

If you use this for research, please cite:

```bibtex
@article{assran2025vjepa2,
  title={V-JEPA~2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author={Assran, Mahmoud and Bardes, Adrien and Fan, David and Garrido, Quentin and ...},
  journal={arXiv preprint arXiv:2506.09985},
  year={2025}
}
```

## License

MIT License (see main LICENSE file)
