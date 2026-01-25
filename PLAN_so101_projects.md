# SO101 Arm Projects Plan

## Context / First steps
Hello, I have an SO101 follower arm, would like to try some fun project with it, currently don't really have a working camera,
also I don't have a leader arm, so collecting data and using learning methods or computer vision stuff is not really an option.
In theory could just use something very simple like laptop webcam, but I am guessing first would be best to start with inverse kinematics.

We are currently inside a cloned and fetched `lerobot` repository, also there is the calibration file of the arm in: `my_awesome_follower_arm.json`

Starting up todo:
  - [x] write a plan of what we need to do some inverse kinematics / be able to follow trajectories / execute movement with the arm:
      - the plan should explain how will we be using the lerobot library / what interfaces it provides which we can use
      - what will we need to setup in terms of code and functionality

---

## Plan: Inverse Kinematics & Trajectory Control for SO101

### 1. LeRobot Library Interfaces Overview

The lerobot library provides several key interfaces for controlling the SO101 arm:

#### Core Classes
| Class | Location | Purpose |
|-------|----------|---------|
| `SO101Follower` | `src/lerobot/robots/so_follower/` | Main robot interface for the follower arm |
| `SO101FollowerConfig` | `src/lerobot/robots/so_follower/config_so_follower.py` | Configuration dataclass |
| `RobotKinematics` | `src/lerobot/model/kinematics.py` | FK/IK solver using `placo` library |
| `FeetechMotorsBus` | `src/lerobot/motors/feetech/` | Low-level motor communication |

#### Joint Names (6 DOF)
1. `shoulder_pan` (ID: 1) - Base rotation
2. `shoulder_lift` (ID: 2) - Shoulder vertical
3. `elbow_flex` (ID: 3) - Elbow bend
4. `wrist_flex` (ID: 4) - Wrist pitch
5. `wrist_roll` (ID: 5) - Wrist rotation
6. `gripper` (ID: 6) - End effector

---

### 2. What We Need to Set Up

#### Step 1: Install Dependencies
```bash
# Core lerobot install (if not done)
pip install -e .

# IK requires the placo library
pip install placo
```

#### Step 2: Copy Calibration File to Expected Location
The robot expects calibration at `~/.cache/huggingface/lerobot/calibration/robots/so_follower/`
```bash
mkdir -p ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
cp my_awesome_follower_arm.json ~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json
```

#### Step 3: Find Your USB Port
```bash
lerobot-find-port
# OR
ls /dev/ttyACM* /dev/ttyUSB*
```

---

### 3. Basic Robot Control (Joint Space)

```python
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# Configuration
config = SO101FollowerConfig(
    port="/dev/ttyACM0",  # Your USB port
    id="my_awesome_follower_arm",  # Matches calibration filename
    use_degrees=True,  # Use degrees (easier to work with)
)

# Connect
robot = SO101Follower(config)
robot.connect()

# Read current joint positions
obs = robot.get_observation()
print(obs)  # {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": 10.5, ...}

# Send joint command
action = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 50.0,
}
robot.send_action(action)

# Disconnect
robot.disconnect()
```

---

### 4. Inverse Kinematics Setup

The IK system uses the `placo` library with a URDF model.

#### URDF Location
The SO101 URDF should be at: `SO101/so101_new_calib.urdf` (or similar path in the repo)

```python
from lerobot.model.kinematics import RobotKinematics
import numpy as np

# Initialize kinematics solver
kinematics = RobotKinematics(
    urdf_path="./SO101/so101_new_calib.urdf",  # Path to URDF
    target_frame_name="gripper_frame_link",    # End-effector frame
    joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
)

# Forward Kinematics: joints -> end-effector pose
joint_positions_deg = np.array([0, 0, 0, 0, 0])
ee_pose_4x4 = kinematics.forward_kinematics(joint_positions_deg)
print(f"End-effector position: {ee_pose_4x4[:3, 3]}")

# Inverse Kinematics: end-effector pose -> joints
current_joints = np.array([0, 0, 0, 0, 0])
desired_pose = np.eye(4)
desired_pose[:3, 3] = [0.15, 0.0, 0.10]  # Target position (x, y, z)

new_joints = kinematics.inverse_kinematics(
    current_joints,
    desired_pose,
    position_weight=1.0,
    orientation_weight=0.01  # Low = position only, High = also match orientation
)
```

---

### 5. Trajectory Following

For smooth trajectory execution, interpolate between waypoints:

```python
import numpy as np
import time

def interpolate_trajectory(start_joints, end_joints, num_steps=50):
    """Linear interpolation between joint configurations"""
    trajectory = []
    for i in range(num_steps):
        t = i / (num_steps - 1)
        interpolated = start_joints + t * (end_joints - start_joints)
        trajectory.append(interpolated)
    return trajectory

def execute_trajectory(robot, trajectory, dt=0.02):
    """Execute trajectory on robot"""
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", 
                   "wrist_flex", "wrist_roll", "gripper"]
    
    for joints in trajectory:
        action = {f"{name}.pos": float(joints[i]) for i, name in enumerate(joint_names)}
        robot.send_action(action)
        time.sleep(dt)
```

---

### 6. Complete Example: Move to Cartesian Position

```python
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.model.kinematics import RobotKinematics
import numpy as np
import time

# Setup
config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="my_awesome_follower_arm",
    use_degrees=True,
)
robot = SO101Follower(config)

kinematics = RobotKinematics(
    urdf_path="./SO101/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
)

# Connect
robot.connect()

# Get current position
obs = robot.get_observation()
current_joints = np.array([
    obs["shoulder_pan.pos"],
    obs["shoulder_lift.pos"],
    obs["elbow_flex.pos"],
    obs["wrist_flex.pos"],
    obs["wrist_roll.pos"],
])

# Define target Cartesian position
target_pose = np.eye(4)
target_pose[:3, 3] = [0.15, 0.05, 0.12]  # x, y, z in meters

# Compute IK
target_joints = kinematics.inverse_kinematics(current_joints, target_pose)

# Interpolate trajectory
trajectory = interpolate_trajectory(current_joints, target_joints, num_steps=100)

# Execute
for joints in trajectory:
    action = {
        "shoulder_pan.pos": joints[0],
        "shoulder_lift.pos": joints[1],
        "elbow_flex.pos": joints[2],
        "wrist_flex.pos": joints[3],
        "wrist_roll.pos": joints[4],
        "gripper.pos": 50.0,  # Keep gripper at 50%
    }
    robot.send_action(action)
    time.sleep(0.02)

robot.disconnect()
```

---

### 7. Suggested Project Progression

| Step | Task | Complexity |
|------|------|------------|
| 1 | Basic connection & joint reading | Easy |
| 2 | Joint-space position control | Easy |
| 3 | Setup kinematics (FK verification) | Medium |
| 4 | IK to specific Cartesian points | Medium |
| 5 | Smooth trajectory interpolation | Medium |
| 6 | Draw shapes (circle, square) in Cartesian space | Fun! |
| 7 | Pick-and-place with hardcoded positions | Medium |

---

### 8. Useful CLI Commands

```bash
# Find USB port
lerobot-find-port

# Re-calibrate if needed
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_awesome_follower_arm

# Test with teleoperation (if you had a leader arm)
lerobot-teleoperate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_awesome_follower_arm
```

---

### 9. Files to Check/Create

- [ ] Verify URDF exists: `SO101/so101_new_calib.urdf`
- [ ] Copy calibration: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json`
- [ ] Create test script: `scripts/test_so101_ik.py`

---

---

## COMPLETED: Inverse Kinematics & Teleop Control

**Status**: DONE (Jan 2026)

### What Was Built

Created a unified IK control script at `scripts/test_so101_ik.py` with the following capabilities:

#### IK Solvers
- **Generic IK** (`solve_ik_generic`): Maintains current orientation, good for general movement
- **Pointing-down IK** (`solve_ik_pointing_down`): Constrains gripper to point downward (within configurable tilt tolerance), essential for drawing/writing tasks

#### Motion Modes
| Mode | Command | Description |
|------|---------|-------------|
| Joint | `mode j` | Fast joint-space interpolation, curved Cartesian paths |
| Cartesian | `mode c` | Interpolates in XYZ space, solves IK at each step - straighter paths |
| Closed-loop | `mode cl` | Moves, reads actual position, corrects if error > 2mm - most accurate |

#### Gripper Lock Feature
- `g <0-100>` - Set gripper position AND lock it (maintains grip during arm movements)
- `grip` - Show current lock status
- `release` - Unlock gripper

#### Shape Drawing
- `draw <shape> [size] [x] [y] [z]` - Draw square, circle, or triangle
- Uses pointing-down IK with relaxed tilt constraint (up to 25° from vertical)

### Key Technical Insights

1. **5 DOF limitation**: SO101 has 5 joints + gripper, cannot achieve arbitrary position AND orientation simultaneously
2. **Gripper-down constraint**: For gripper pointing down, `shoulder_lift + elbow_flex + wrist_flex ≈ 90°`
3. **Trajectory accuracy**: Joint interpolation causes curved paths; Cartesian interpolation fixes this
4. **Servo precision**: STS3215 servos have ~0.1° resolution, closed-loop correction helps for precision tasks

### Usage

```bash
cd /home/snakeonex/fun/lerobot
source .venv/bin/activate
python scripts/test_so101_ik.py --port /dev/ttyACM0 -i
```

### Interactive Commands Reference

```
Movement:
  x/y/z <mm>              Move axis by delta
  goto <x> <y> <z>        Move to position (mm)
  home                    Return to home position

Drawing:
  draw <shape> [size] [x] [y] [z]   Draw with gripper down
  down                    Point gripper down at current XY

Gripper:
  g <0-100>               Set and lock gripper
  grip                    Show lock status
  release                 Unlock gripper

Settings:
  mode [j|c|cl]           Set motion mode
  tilt [deg]              Set max tilt from vertical (default 25°)
  r                       Refresh display
  q                       Quit
```

---

## Next Steps / Future Projects

Now that IK and basic control are working, potential directions:

### 1. Vision Integration (Laptop Webcam)
- Simple object detection with webcam
- Hand-eye calibration to map camera coordinates to robot coordinates
- Visual servoing for pick-and-place

### 2. Handwriting/Drawing Improvements
- Better path planning for smoother curves
- Pressure/force control (if servo current feedback available)
- SVG or G-code interpreter for complex drawings

### 3. Autonomous Tasks
- Hardcoded pick-and-place sequences
- Simple state machines for repetitive tasks
- Workspace scanning patterns

### 4. Data Collection (for future learning)
- Record trajectories during manual guidance
- Replay and refine recorded motions
- Build dataset for imitation learning (when camera available)

### 5. Simulation
- Test trajectories in simulation before real robot
- Visualize workspace and reachability

