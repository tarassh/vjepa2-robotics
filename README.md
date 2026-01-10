# VJEPA2-Robotics: Obstacle Detection and Collision Avoidance

Real-time obstacle detection and collision avoidance for robotics using [V-JEPA 2](https://github.com/facebookresearch/vjepa2) vision models.

## Overview

This repository provides plug-and-play obstacle detection and collision avoidance tools for robotics applications using V-JEPA 2 self-supervised video models. Process live camera feeds to detect obstacles and plan safe robot actions.

### Features

- 🎥 **Live Camera Processing** - Real-time video analysis from laptop/robot cameras
- 🚨 **Obstacle Detection** - Motion-based detection with spatial attention maps
- 🤖 **Collision Avoidance** - Action-conditioned planning for safe navigation
- ⚡ **GPU Accelerated** - Fast inference on CUDA-enabled devices
- 🔧 **Easy Integration** - Simple APIs for ROS and custom robotics systems

## Demo

<p align="center">
  <img src="docs/demo.gif" width="600" alt="Obstacle Detection Demo">
</p>

## Installation

### Prerequisites

- Python 3.12+
- PyTorch with CUDA support (recommended)
- OpenCV

### Setup

```bash
# Clone this repository
git clone https://github.com/tarassh/vjepa2-robotics.git
cd vjepa2-robotics

# Create conda environment
conda create -n vjepa2-robotics python=3.12
conda activate vjepa2-robotics

# Install V-JEPA 2
git clone https://github.com/facebookresearch/vjepa2.git
cd vjepa2
pip install -e .
cd ..

# Install additional dependencies
pip install opencv-python
```

### Download Pretrained Models (Optional but Recommended)

```bash
mkdir -p checkpoints

# For obstacle detection (choose one)
wget https://dl.fbaipublicfiles.com/vjepa2/vitl.pt -P checkpoints/  # Fast (300M)
wget https://dl.fbaipublicfiles.com/vjepa2/vitg.pt -P checkpoints/  # Best (1B)

# For collision avoidance
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt -P checkpoints/
```

## Quick Start

### Obstacle Detection

```bash
python robotics_obstacle_detection.py
```

The system will:
1. Open your laptop camera
2. Buffer video frames
3. Detect obstacles in real-time
4. Show spatial attention maps
5. Alert when obstacles are detected

Press `q` to quit.

### Collision Avoidance Planning

```bash
python robotics_collision_avoidance.py
```

This mode:
1. Captures live video
2. Samples possible robot actions
3. Predicts outcomes using V-JEPA 2-AC
4. Selects safest actions
5. Displays collision risk scores

## Usage

### Command Line Options

#### Obstacle Detection
```bash
python robotics_obstacle_detection.py \
    --model-size large \          # Model size: large, huge, or giant
    --img-size 256 \               # Input resolution: 256 or 384
    --num-frames 16 \              # Frame buffer size
    --camera-id 0 \                # Camera device ID
    --device cuda \                # Device: cuda or cpu
    --motion-threshold 0.5         # Detection sensitivity
```

#### Collision Avoidance
```bash
python robotics_collision_avoidance.py \
    --img-size 256 \               # Input resolution
    --num-frames 8 \               # Frame buffer size
    --camera-id 0 \                # Camera device ID
    --device cuda \                # Device: cuda or cpu
    --risk-threshold 0.5           # Risk threshold
```

### Python API

#### Basic Obstacle Detection

```python
from robotics_obstacle_detection import ObstacleDetector
import cv2

# Initialize detector
detector = ObstacleDetector(model_size='large', img_size=256)

# Capture and process frames
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    detector.add_frame(frame_rgb)
    features, motion_score = detector.detect_obstacles()
    
    if motion_score > 0.5:
        print("Obstacle detected!")
```

#### Collision Avoidance Planning

```python
from robotics_collision_avoidance import CollisionAvoidancePlanner
import numpy as np

# Initialize planner
planner = CollisionAvoidancePlanner(img_size=256, num_frames=8)

# In your control loop
target_action = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
safe_action, risk = planner.plan_safe_trajectory(target_action)

if risk < 0.5:
    robot.execute_action(safe_action)
else:
    robot.stop()  # Emergency stop
```

## Integration Examples

### ROS Integration

```python
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from robotics_collision_avoidance import CollisionAvoidancePlanner

class CollisionAvoidanceNode:
    def __init__(self):
        self.planner = CollisionAvoidancePlanner()
        self.bridge = CvBridge()
        
        rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.planner.add_frame(frame)
        
        target = get_navigation_goal()
        safe_action, risk = self.planner.plan_safe_trajectory(target)
        
        if risk < 0.5:
            self.publish_cmd(safe_action)

if __name__ == '__main__':
    rospy.init_node('collision_avoidance')
    node = CollisionAvoidanceNode()
    rospy.spin()
```

### Custom Robot Integration

See [ROBOTICS_README.md](ROBOTICS_README.md) for detailed integration examples.

## How It Works

### Obstacle Detection Pipeline

1. **Video Capture** - Buffers frames from camera
2. **Feature Extraction** - V-JEPA 2 encoder processes video
3. **Motion Analysis** - Computes spatio-temporal feature variance
4. **Spatial Localization** - Generates attention maps for obstacle regions

### Collision Avoidance Pipeline

1. **Scene Encoding** - Encodes current view with V-JEPA 2
2. **Action Sampling** - Generates candidate robot actions
3. **Forward Prediction** - V-JEPA 2-AC predicts future states
4. **Risk Assessment** - Evaluates collision likelihood
5. **Safe Selection** - Chooses lowest-risk action

## Performance

| Model | Params | FPS (GPU) | FPS (CPU) | Accuracy |
|-------|--------|-----------|-----------|----------|
| ViT-Large | 300M | ~30 | ~2 | Good |
| ViT-Huge | 600M | ~20 | ~1 | Better |
| ViT-Giant | 1B | ~15 | ~0.5 | Best |

*Tested on NVIDIA RTX 4090 and Apple M2 Pro*

## Customization

### Custom Action Space

Modify the action space for your robot:

```python
# In robotics_collision_avoidance.py
def sample_collision_free_actions(self, ...):
    action_samples = []
    for vx in np.linspace(-0.5, 0.5, num_samples):
        for vy in np.linspace(-0.5, 0.5, num_samples):
            for omega in np.linspace(-0.3, 0.3, num_samples):
                action_samples.append([vx, vy, omega])  # 2D navigation
```

### Goal-Directed Planning

Add goal-seeking behavior:

```python
def plan_with_goal(self, target_action, goal_position):
    best_action, risks, actions = self.sample_collision_free_actions(...)
    
    # Balance safety and progress to goal
    goal_costs = [np.linalg.norm(a[:3] - goal_position) for a in actions]
    combined_scores = risks + 0.3 * np.array(goal_costs)
    
    return actions[np.argmin(combined_scores)]
```

## Troubleshooting

**Camera not opening?**
- Try different camera IDs: `--camera-id 1`
- Check permissions in System Settings
- Close other apps using the camera

**Out of memory?**
- Use smaller model: `--model-size large`
- Reduce resolution: `--img-size 256`
- Fewer frames: `--num-frames 8`

**Slow performance?**
- Enable GPU acceleration
- Use ViT-Large model
- Lower camera resolution

See [ROBOTICS_README.md](ROBOTICS_README.md) for more troubleshooting tips.

## Citation

If you use this in your research, please cite:

```bibtex
@article{assran2025vjepa2,
  title={V-JEPA~2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author={Assran, Mahmoud and Bardes, Adrien and Fan, David and Garrido, Quentin and 
          Howes, Russell and Komeili, Mojtaba and Muckley, Matthew and Rizvi, Ammar and 
          Roberts, Claire and Sinha, Koustuv and others},
  journal={arXiv preprint arXiv:2506.09985},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) file

## Acknowledgments

Built on [V-JEPA 2](https://github.com/facebookresearch/vjepa2) by Meta FAIR.

## Contributing

Contributions welcome! Please open an issue or submit a PR.

## Contact

- GitHub: [@tarassh](https://github.com/tarassh)
- Issues: [GitHub Issues](https://github.com/tarassh/vjepa2-robotics/issues)
