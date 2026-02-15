#!/usr/bin/env python
"""
SO101 Inverse Kinematics Control - Move the arm to a Cartesian position.

Usage:
    python scripts/test_so101_ik.py --port /dev/ttyACM0 goto 400 0 200
    python scripts/test_so101_ik.py --port /dev/ttyACM0 --no-cameras goto 400 0 200
    python scripts/test_so101_ik.py --port /dev/ttyACM0 --dataset user/so101_recover recover 10 40
    python scripts/test_so101_ik.py --port /dev/ttyACM0 --no-cameras deploy outputs/train/.../last/pretrained_model 5 30
"""

import argparse
import math
import shutil
import time

import cv2
import numpy as np

IK_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Analytical kinematics constants (from URDF geometry analysis)
PAN_CENTER_X = 38.8   # Pan rotation center X offset (mm)
PIVOT_R = 30.4         # Shoulder pivot radial offset from pan center (mm)
PIVOT_Z = 116.6        # Shoulder pivot height (mm)
L1 = 116.0             # Upper arm length (mm)
L2 = 135.0             # Lower arm length (mm)
L3 = 160.5             # Wrist to gripper tip (mm)
A1_BASE = 76.0         # Upper arm angle at SL=0 (deg from horizontal)
A2_OFFSET = 73.8       # Angle offset between L1 and L2 at EF=0
A3_OFFSET = 5.0        # Angle offset between L2 and L3 at WF=0

# Camera configuration
CAMERAS = {
    "front": {"index": 4, "width": 320, "height": 240, "fps": 30},
    "wrist": {"index": 6, "width": 640, "height": 360, "fps": 30},
}


# =============================================================================
# Kinematics
# =============================================================================

def joints_from_obs(obs) -> np.ndarray:
    """Extract IK joint angles from a robot observation."""
    return np.array([obs[f"{j}.pos"] for j in IK_JOINTS])


def analytical_fk(joints_deg):
    """Forward kinematics using analytical model. Returns XYZ in mm."""
    pan, sl, ef, wf, wr = joints_deg

    a1 = math.radians(A1_BASE - sl)
    a2 = math.radians(A1_BASE - A2_OFFSET - sl - ef)
    a3 = math.radians(A1_BASE - A2_OFFSET - A3_OFFSET - sl - ef - wf)

    tip_r = PIVOT_R + L1*math.cos(a1) + L2*math.cos(a2) + L3*math.cos(a3)
    tip_z = PIVOT_Z + L1*math.sin(a1) + L2*math.sin(a2) + L3*math.sin(a3)

    pan_rad = math.radians(pan)
    x = PAN_CENTER_X + tip_r * math.cos(pan_rad)
    y = -tip_r * math.sin(pan_rad)
    return np.array([x, y, tip_z])


def analytical_ik(target_pos_mm, current_wf, current_wr):
    """
    Analytical IK for SO101 arm. Returns joint angles in degrees.

    Solves shoulder_pan analytically, then uses effective-link 2-link IK
    for shoulder_lift and elbow_flex. Keeps wrist_flex and wrist_roll unchanged.

    Deterministic: same input always gives the same output.
    """
    tx, ty, tz = target_pos_mm

    pan = math.degrees(math.atan2(-ty, tx - PAN_CENTER_X))

    pan_rad = math.radians(pan)
    if abs(math.cos(pan_rad)) > 0.01:
        tip_r = (tx - PAN_CENTER_X) / math.cos(pan_rad)
    else:
        tip_r = -ty / math.sin(pan_rad)

    wf = current_wf
    delta = math.radians(A3_OFFSET + wf)
    L_eff = math.sqrt(L2**2 + L3**2 + 2*L2*L3*math.cos(delta))
    gamma = math.atan2(L3*math.sin(delta), L2 + L3*math.cos(delta))

    dr = tip_r - PIVOT_R
    dz = tz - PIVOT_Z
    d = math.sqrt(dr**2 + dz**2)

    max_reach = L1 + L_eff - 1
    if d > max_reach:
        dr *= max_reach / d
        dz *= max_reach / d
        d = max_reach

    cos_beta = max(-1.0, min(1.0, (L1**2 + L_eff**2 - d**2) / (2*L1*L_eff)))
    beta = math.acos(cos_beta)
    phi = math.atan2(dz, dr)
    cos_alpha = max(-1.0, min(1.0, (L1**2 + d**2 - L_eff**2) / (2*L1*d)))
    alpha = math.acos(cos_alpha)

    a1 = phi + alpha
    sl = A1_BASE - math.degrees(a1)

    a_eff = a1 - (math.pi - beta)
    a2 = a_eff + gamma
    ef = (A1_BASE - A2_OFFSET) - sl - math.degrees(a2)

    return np.array([pan, sl, ef, wf, current_wr])


# =============================================================================
# Display
# =============================================================================

def render_cameras(obs, joints, cam_names):
    """Build combined camera frame with state overlay. Returns the frame or None."""
    frames = [cv2.cvtColor(obs[n], cv2.COLOR_RGB2BGR) for n in cam_names if n in obs]
    if not frames:
        return None

    resized = [cv2.resize(f, (320, 240)) for f in frames]
    combined = np.hstack(resized)

    pos = analytical_fk(joints)
    names = ["pan", "sl", "ef", "wf", "wr"]
    lines = [
        f"x={pos[0]:.1f}  y={pos[1]:.1f}  z={pos[2]:.1f}",
        "  ".join(f"{n}={joints[i]:.1f}" for i, n in enumerate(names)),
    ]
    for i, line in enumerate(lines):
        cv2.putText(combined, line, (10, 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("SO101", combined)
    cv2.waitKey(1)
    return combined


# =============================================================================
# Routines
# =============================================================================

# Target loop rate in Hz. Controls how fast we run.
TARGET_HZ = 20
# Max degrees per second. Controls smoothness of motion.
MAX_DEG_PER_SEC = 15.0
# Convergence threshold in degrees. Servo jitter means we can't expect exact positioning.
# Shoulder_lift under gravity can be off by ~6°, so needs to be generous.
CONVERGE_DEG = 8.0


class JointRoutine:
    """Move joints to target with velocity-limited smooth interpolation.

    Only the joints specified in `target` are moved; the rest hold their current position.
    Limits movement speed to MAX_DEG_PER_SEC for smooth motion.
    Applies drift correction to compensate for servo offset.
    """
    def __init__(self, target: dict[str, float], obs: dict, hold_s: float = 1.0):
        self.target = {j: obs[f"{j}.pos"] for j in ALL_JOINTS}
        self.target.update(target)
        self.command = {j: obs[f"{j}.pos"] for j in ALL_JOINTS}  # starts at current position
        self.hold_s = hold_s
        self.converged_at = None

    def step(self, robot, obs):
        """Send velocity-limited action. Returns (done, action_sent)."""
        dt = 1.0 / TARGET_HZ  # fixed dt matching our target loop rate
        max_step = MAX_DEG_PER_SEC * dt

        current = {j: obs[f"{j}.pos"] for j in ALL_JOINTS}

        action = {}
        for j in ALL_JOINTS:
            # Move command toward target, limited by max velocity
            diff = self.target[j] - self.command[j]
            step = max(-max_step, min(max_step, diff))
            self.command[j] += step
            action[f"{j}.pos"] = self.command[j]

        sent = robot.send_action(action)

        max_err = max(abs(current[j] - self.target[j]) for j in ALL_JOINTS)
        if self.converged_at is None and max_err < CONVERGE_DEG:
            self.converged_at = time.time()
        done = self.converged_at is not None and time.time() >= self.converged_at + self.hold_s
        return done, sent


class GotoRoutine:
    """Move to a Cartesian position using IK + JointRoutine."""
    def __init__(self, target_mm, obs):
        joints = joints_from_obs(obs)
        target_joints = analytical_ik(target_mm, joints[3], joints[4])
        achieved = analytical_fk(target_joints)
        err = np.linalg.norm(achieved - target_mm)
        if err > 10:
            print(f"  Warning: IK error {err:.1f}mm - may be unreachable")
        target = {j: target_joints[k] for k, j in enumerate(IK_JOINTS)}
        self._inner = JointRoutine(target, obs, hold_s=5.0)

    def step(self, robot, obs):
        return self._inner.step(robot, obs)


class PolicyRoutine:
    """Run a trained policy to control the robot. Stops after max_steps."""
    def __init__(self, policy, preprocessor, postprocessor, max_steps=100):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.max_steps = max_steps
        self.step_count = 0
        self.target = {j: 0.0 for j in ALL_JOINTS}  # for debug display

    def step(self, robot, obs):
        import torch
        from lerobot.utils.control_utils import predict_action

        # Build observation in the format the policy expects
        state = np.array([obs[f"{j}.pos"] for j in ALL_JOINTS], dtype=np.float32)
        policy_obs = {
            "observation.state": state,
            "observation.environment_state": state.copy(),
        }

        action = predict_action(
            observation=policy_obs,
            policy=self.policy,
            device=torch.device(self.policy.config.device),
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_amp=False,
        )
        # action is a tensor — may have batch dim, squeeze it
        action = action.squeeze()
        action_dict = {f"{j}.pos": action[i].item() for i, j in enumerate(ALL_JOINTS)}
        self.target = {j: action_dict[f"{j}.pos"] for j in ALL_JOINTS}
        sent = robot.send_action(action_dict)
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return done, sent


# =============================================================================
# Run loop
# =============================================================================

def run_routine(routine, task, robot, dataset, cameras, args):
    """Run a routine until done, optionally recording to dataset and displaying cameras."""
    build_dataset_frame = None
    if dataset is not None:
        from lerobot.datasets.utils import build_dataset_frame

    step_count = 0
    t_start = time.monotonic()
    while True:
        t0 = time.monotonic()
        obs = robot.get_observation()
        done, action_sent = routine.step(robot, obs)

        if dataset is not None:
            obs_frame = build_dataset_frame(dataset.features, obs, prefix="observation")
            act_frame = build_dataset_frame(dataset.features, action_sent, prefix="action")
            dataset.add_frame({**obs_frame, **act_frame, "task": task})

        if cameras:
            joints = joints_from_obs(obs)
            render_cameras(obs, joints, list(cameras))

        step_count += 1

        # Enforce target loop rate
        sleep_s = (1.0 / TARGET_HZ) - (time.monotonic() - t0)
        if sleep_s > 0:
            time.sleep(sleep_s)
        dt = time.monotonic() - t0
        fps = 1.0 / dt if dt > 0 else 0

        # Print status every 10 steps
        if step_count % 10 == 0:
            cur = {j: obs[f"{j}.pos"] for j in ALL_JOINTS}
            tgt = routine.target if hasattr(routine, 'target') else routine._inner.target
            errs = " ".join(f"{j[:3]}={cur[j]:+.1f}→{tgt[j]:+.1f}" for j in ALL_JOINTS)
            max_err_j = max(ALL_JOINTS, key=lambda j: abs(cur[j] - tgt[j]))
            max_err_v = abs(cur[max_err_j] - tgt[max_err_j])
            elapsed = time.monotonic() - t_start
            print(f"    step={step_count:4d}  fps={fps:4.0f}  {errs}  worst={max_err_j[:3]}({max_err_v:.1f}°)  elapsed={elapsed:.1f}s")

        if done:
            break


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SO101 IK Control")
    parser.add_argument("--port", required=True, help="USB port")
    parser.add_argument("--id", default="my_awesome_follower_arm", help="Robot ID")
    parser.add_argument("--no-cameras", action="store_true", help="Disable cameras")
    parser.add_argument("--dataset", type=str, help="Record LeRobot dataset (e.g. user/so101_goto)")
    parser.add_argument("command", help="Command to run (goto, recover, deploy)")
    parser.add_argument("args", nargs="*", help="Command arguments")
    args = parser.parse_args()

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    cameras = {}
    if not args.no_cameras:
        cameras = {name: OpenCVCameraConfig(index_or_path=c["index"], fps=c["fps"],
                                            width=c["width"], height=c["height"])
                   for name, c in CAMERAS.items()}

    config = SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True, cameras=cameras)
    robot = SO101Follower(config)
    robot.connect(calibrate=False)

    # --- Dataset setup ---
    dataset = None
    if args.dataset:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
        from lerobot.datasets.utils import combine_feature_dicts
        from lerobot.processor.factory import make_default_processors

        teleop_proc, action_proc, obs_proc = make_default_processors()
        features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_proc,
                initial_features=create_initial_features(action=robot.action_features),
                use_videos=bool(cameras),
            ),
            aggregate_pipeline_dataset_features(
                pipeline=obs_proc,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=bool(cameras),
            ),
        )
        # Add environment_state (copy of observation.state) so ACT can train without images
        if "observation.state" in features:
            features["observation.environment_state"] = {
                **features["observation.state"],
            }
        dataset = LeRobotDataset.create(
            repo_id=args.dataset,
            fps=20,
            features=features,
            robot_type=robot.name,
            use_videos=bool(cameras),
            image_writer_threads=len(cameras) * 4 if cameras else 0,
        )
        print(f"  Dataset: {args.dataset} ({dataset.root})")

    try:
        if args.command == "goto":
            target = np.array([float(x) for x in args.args[:3]])
            task = f"goto {' '.join(args.args[:3])}"
            obs = robot.get_observation()
            routine = GotoRoutine(target, obs)
            run_routine(routine, task, robot, dataset, cameras, args)
            if dataset is not None:
                dataset.save_episode()

        elif args.command == "recover":
            n_episodes = int(args.args[0]) if args.args else 10
            pan_min = float(args.args[1]) if len(args.args) > 1 else 10.0
            pan_max = float(args.args[2]) if len(args.args) > 2 else 35.0
            rng = np.random.default_rng()
            task = "recover_pan"

            for ep in range(n_episodes):
                # Reset to home (all joints to 0) before each episode
                print(f"  Going home...")
                obs = robot.get_observation()
                home = JointRoutine({j: 0.0 for j in ALL_JOINTS}, obs, hold_s=1.0)
                run_routine(home, task, robot, None, cameras, args)

                # Phase 1: move to random pan angle (not recorded)
                angle = float(rng.uniform(pan_min, pan_max))
                sign = rng.choice([-1, 1])
                pan_target = sign * angle
                print(f"  Episode {ep+1}/{n_episodes}: pan -> {pan_target:.1f}°")
                obs = robot.get_observation()
                setup = JointRoutine({"shoulder_pan": pan_target}, obs, hold_s=0.5)
                run_routine(setup, task, robot, None, cameras, args)

                # Phase 2: recover to center (recorded)
                print(f"    Recording recovery: pan {pan_target:.1f}° -> 0°")
                obs = robot.get_observation()
                routine = JointRoutine({"shoulder_pan": 0.0}, obs, hold_s=1.0)
                run_routine(routine, task, robot, dataset, cameras, args)

                if dataset is not None:
                    dataset.save_episode()
                    print(f"    Saved episode {ep+1} ({dataset.num_frames} total frames)")

        elif args.command == "deploy":
            # deploy <checkpoint_path> [n_episodes] [pan_min] [pan_max]
            if not args.args:
                print("Usage: deploy <checkpoint_path> [n_episodes=5] [pan_min=10] [pan_max=35]")
                return
            checkpoint_path = args.args[0]
            n_episodes = int(args.args[1]) if len(args.args) > 1 else 5
            pan_min = float(args.args[2]) if len(args.args) > 2 else 10.0
            pan_max = float(args.args[3]) if len(args.args) > 3 else 35.0

            import torch
            from lerobot.policies.act.modeling_act import ACTPolicy
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.pretrained import PreTrainedConfig

            policy_cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
            policy_cfg.pretrained_path = checkpoint_path
            policy = ACTPolicy.from_pretrained(checkpoint_path)
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=checkpoint_path,
            )
            print(f"  Loaded policy from {checkpoint_path}")

            rng = np.random.default_rng()
            task = "deploy_recover_pan"

            for ep in range(n_episodes):
                # Reset to home
                print(f"  Going home...")
                obs = robot.get_observation()
                home = JointRoutine({j: 0.0 for j in ALL_JOINTS}, obs, hold_s=1.0)
                run_routine(home, task, robot, None, cameras, args)

                # Move to random pan angle
                angle = float(rng.uniform(pan_min, pan_max))
                sign = rng.choice([-1, 1])
                pan_target = sign * angle
                print(f"  Episode {ep+1}/{n_episodes}: pan -> {pan_target:.1f}°")
                obs = robot.get_observation()
                setup = JointRoutine({"shoulder_pan": pan_target}, obs, hold_s=0.5)
                run_routine(setup, task, robot, None, cameras, args)

                # Let policy recover
                print(f"    Policy recovering from {pan_target:.1f}°...")
                preprocessor.reset()
                postprocessor.reset()
                routine = PolicyRoutine(policy, preprocessor, postprocessor, max_steps=60)
                run_routine(routine, task, robot, None, cameras, args)

                # Check result
                obs = robot.get_observation()
                final_pan = obs["shoulder_pan.pos"]
                print(f"    Result: pan={final_pan:+.1f}° (target=0.0°, error={abs(final_pan):.1f}°)")

        else:
            print(f"Unknown command '{args.command}'. Available: goto, recover, deploy")
    finally:
        if dataset is not None:
            print(f"  Dataset done: {dataset.num_episodes} episode(s), {dataset.num_frames} frame(s)")
        if cameras:
            cv2.destroyAllWindows()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
