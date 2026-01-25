"""
SO101 Reaching Environment using PyBullet.

Simplified version with:
- Position control (action = target joint positions)
- Option for static goal (easier to learn)

Observation (11D):
  - joint_positions (5): normalized to [-1, 1]
  - ee_position (3): end-effector XYZ in meters
  - target_position (3): target XYZ in meters

Action (5D):
  - joint_position_deltas: continuous [-1, 1], scaled to position change per step

Reward:
  - -distance to target (dense)
  - +bonus when within threshold
"""

import gymnasium
import numpy as np
import pybullet as p
import pybullet_data
from pathlib import Path


class SO101ReachEnv(gymnasium.Env):
    """SO101 arm reaching task in PyBullet simulation."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    # Joint configuration
    JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    NUM_JOINTS = 5
    
    # Joint limits (degrees) - from real robot
    JOINT_LIMITS_DEG = {
        "shoulder_pan": (-135, 135),
        "shoulder_lift": (-90, 90),
        "elbow_flex": (-135, 135),
        "wrist_flex": (-90, 90),
        "wrist_roll": (-135, 135),
    }
    
    # Task parameters
    SUCCESS_THRESHOLD = 0.02  # 2cm
    MAX_STEPS = 100  # Shorter episodes with position control
    
    # Position control parameters
    MAX_DELTA_DEG = 5.0  # Max degrees per step
    
    def __init__(
        self, 
        render_mode=None, 
        urdf_path=None,
        static_goal=True,  # Use fixed goal position
        goal_pos=None,     # Custom goal position [x, y, z] in meters
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.static_goal = static_goal
        
        # Default goal: in front of the robot, easy to reach
        if goal_pos is not None:
            self.fixed_goal = np.array(goal_pos)
        else:
            self.fixed_goal = np.array([0.25, 0.0, 0.20])  # 25cm forward, 20cm up
        
        # Find URDF
        if urdf_path is None:
            candidates = [
                Path(__file__).parent.parent.parent / "SO101" / "so101_new_calib.urdf",
                Path("SO101/so101_new_calib.urdf"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    urdf_path = str(candidate.resolve())
                    break
            else:
                raise FileNotFoundError(f"Could not find SO101 URDF. Tried: {candidates}")
        
        self.urdf_path = urdf_path
        
        # Observation: joint_pos(5) + ee_pos(3) + target(3) = 11
        # Removed velocities since we're doing position control
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        
        # Action: joint position deltas normalized to [-1, 1]
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(self.NUM_JOINTS,), dtype=np.float32
        )
        
        # PyBullet state
        self.physics_client = None
        self.robot_id = None
        self.joint_indices = None
        self.ee_link_index = None
        self.target_pos = None
        self.target_visual = None
        self.step_count = 0
        self.current_joint_pos = None  # Track commanded positions
        
        # Convert joint limits to radians
        self.joint_limits_rad = {
            name: (np.radians(lo), np.radians(hi))
            for name, (lo, hi) in self.JOINT_LIMITS_DEG.items()
        }
        
        self.max_delta_rad = np.radians(self.MAX_DELTA_DEG)
        
    def _connect_physics(self):
        """Connect to PyBullet physics server."""
        if self.physics_client is not None:
            return
        
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        
    def _load_robot(self):
        """Load the SO101 URDF."""
        p.loadURDF("plane.urdf")
        
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )
        
        # Find joint indices by name
        self.joint_indices = []
        num_joints = p.getNumJoints(self.robot_id)
        
        joint_name_to_index = {}
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode("utf-8")
            joint_name_to_index[joint_name] = i
        
        for name in self.JOINT_NAMES:
            if name in joint_name_to_index:
                self.joint_indices.append(joint_name_to_index[name])
            else:
                raise ValueError(f"Joint '{name}' not found in URDF.")
        
        # Find end-effector link
        ee_candidates = ["gripper_frame_link", "gripper_link", "tool_link"]
        self.ee_link_index = num_joints - 1
        
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            link_name = info[12].decode("utf-8")
            if link_name in ee_candidates:
                self.ee_link_index = i
                break
            
    def _create_target_visual(self):
        """Create a visual sphere for the target."""
        if self.target_visual is not None:
            p.removeBody(self.target_visual)
        
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.02,
            rgbaColor=[1, 0, 0, 0.7]
        )
        self.target_visual = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.target_pos
        )
        
    def _get_joint_positions(self):
        """Get current joint positions in radians."""
        positions = []
        for idx in self.joint_indices:
            state = p.getJointState(self.robot_id, idx)
            positions.append(state[0])
        return np.array(positions)
    
    def _get_ee_position(self):
        """Get end-effector position."""
        state = p.getLinkState(self.robot_id, self.ee_link_index)
        return np.array(state[0])
    
    def _normalize_positions(self, positions):
        """Normalize joint positions to [-1, 1]."""
        normalized = []
        for i, name in enumerate(self.JOINT_NAMES):
            lo, hi = self.joint_limits_rad[name]
            norm = 2.0 * (positions[i] - lo) / (hi - lo) - 1.0
            normalized.append(np.clip(norm, -1.0, 1.0))
        return np.array(normalized)
    
    def _get_obs(self):
        """Construct observation vector."""
        joint_pos = self._get_joint_positions()
        ee_pos = self._get_ee_position()
        
        obs = np.concatenate([
            self._normalize_positions(joint_pos),  # 5
            ee_pos,  # 3
            self.target_pos,  # 3
        ]).astype(np.float32)
        
        return obs
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._connect_physics()
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        self._load_robot()
        
        # Initialize joints to zero (home position)
        self.current_joint_pos = np.zeros(self.NUM_JOINTS)
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, idx, 0.0, 0.0)
        
        # Set target
        if self.static_goal:
            self.target_pos = self.fixed_goal.copy()
        else:
            # Random target in workspace
            self.target_pos = np.array([
                np.random.uniform(0.18, 0.32),
                np.random.uniform(-0.10, 0.10),
                np.random.uniform(0.10, 0.28),
            ])
        
        if self.render_mode == "human":
            self._create_target_visual()
        
        self.step_count = 0
        self._prev_distance = np.linalg.norm(self._get_ee_position() - self.target_pos)
        
        obs = self._get_obs()
        info = {"distance": self._prev_distance}
        
        return obs, info
    
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        
        # Convert action to position delta
        delta = action * self.max_delta_rad
        
        # Update target positions with clipping to joint limits
        new_pos = self.current_joint_pos + delta
        for i, name in enumerate(self.JOINT_NAMES):
            lo, hi = self.joint_limits_rad[name]
            new_pos[i] = np.clip(new_pos[i], lo, hi)
        
        self.current_joint_pos = new_pos
        
        # Apply position control
        for i, idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id, idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self.current_joint_pos[i],
                force=50.0,
                maxVelocity=2.0,
            )
        
        # Step simulation
        for _ in range(8):  # More substeps for position control to settle
            p.stepSimulation()
        
        self.step_count += 1
        
        # Get state
        obs = self._get_obs()
        ee_pos = self._get_ee_position()
        distance = np.linalg.norm(ee_pos - self.target_pos)
        
        # IMPROVED REWARD: Simpler and more stable
        # 1. Negative distance penalty (normalized to ~[-1, 0])
        #    Max workspace distance ~0.4m, so divide by 0.4
        reward = -distance / 0.4  # Range: ~[-1, 0]
        
        # 2. Small bonus for getting close (shaped reward)
        if distance < 0.10:  # Within 10cm
            reward += 0.2
        if distance < 0.05:  # Within 5cm
            reward += 0.3
        
        # 3. Success bonus (large but not overwhelming)
        terminated = False
        if distance < self.SUCCESS_THRESHOLD:
            reward += 5.0  # Clear success signal
            terminated = True
        
        # 4. Small action penalty to encourage efficiency
        action_cost = 0.01 * np.sum(action ** 2)
        reward -= action_cost
        
        truncated = self.step_count >= self.MAX_STEPS
        
        info = {
            "distance": distance,
            "success": distance < self.SUCCESS_THRESHOLD,
            "step": self.step_count,
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            width, height = 640, 480
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.2, 0, 0.1],
                distance=0.6,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=width/height, nearVal=0.1, farVal=100
            )
            _, _, rgb, _, _ = p.getCameraImage(
                width, height, view_matrix, proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            return np.array(rgb[:, :, :3], dtype=np.uint8)
        return None
    
    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


# Quick test
if __name__ == "__main__":
    import time
    
    print("Testing SO101ReachEnv (position control, static goal)...")
    
    # Test headless
    env = SO101ReachEnv(render_mode=None, static_goal=True)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Target position: {env.target_pos}")
    print(f"Initial EE position: {obs[5:8]}")
    print(f"Initial distance: {info['distance']*1000:.1f}mm")
    
    # Random actions
    total_reward = 0
    min_dist = info['distance']
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info['distance'] < min_dist:
            min_dist = info['distance']
        if terminated or truncated:
            print(f"Episode ended at step {i+1}, success={info['success']}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final distance: {info['distance']*1000:.1f}mm")
    print(f"Best distance: {min_dist*1000:.1f}mm")
    env.close()
    
    print("\nTest passed!")
