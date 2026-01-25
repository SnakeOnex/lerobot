#!/usr/bin/env python
"""
SO101 Inverse Kinematics Control - Move the arm using Cartesian positions.

Usage:
    python scripts/test_so101_ik.py --port /dev/ttyACM0
    python scripts/test_so101_ik.py --port /dev/ttyACM0 --interactive
"""

import argparse
import time

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

URDF_PATH = "SO101/so101_new_calib.urdf"
IK_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


# =============================================================================
# Kinematics
# =============================================================================

def get_kinematics():
    return RobotKinematics(urdf_path=URDF_PATH, target_frame_name="gripper_frame_link", joint_names=IK_JOINTS)


def get_current_joints(robot) -> np.ndarray:
    obs = robot.get_observation()
    return np.array([obs[f"{j}.pos"] for j in IK_JOINTS])


def get_ee_pose(kin, joints):
    T = kin.forward_kinematics(joints)
    return T[:3, 3], T


def solve_ik_generic(kin, current_joints, target_pos, iterations=5):
    """Generic IK - maintains current orientation."""
    T_target = np.eye(4)
    T_target[:3, 3] = target_pos
    T_target[:3, :3] = kin.forward_kinematics(current_joints)[:3, :3]

    result = current_joints.copy()
    for _ in range(iterations):
        result = kin.inverse_kinematics(result, T_target, position_weight=1.0, orientation_weight=0.01)
    return result


def solve_ik_pointing_down(kin, target_xyz, max_tilt_deg: float = 25.0, initial_guess=None):
    """
    IK with gripper pointing down. Allows tilt up to max_tilt_deg from vertical.
    Returns: (joints, error_mm, tilt_deg)
    """
    best_joints = initial_guess if initial_guess is not None else np.zeros(5)
    best_error = float('inf')
    
    # Coarse search
    for pan in np.linspace(-70, 70, 22):
        for lift in np.linspace(0, 100, 22):
            for elbow in np.linspace(-80, 80, 22):
                for wrist_flex in np.linspace(-100, 100, 22):
                    pitch_sum = lift + elbow + wrist_flex
                    if abs(pitch_sum - 90) > max_tilt_deg:
                        continue
                    
                    joints = np.array([pan, lift, elbow, wrist_flex, 0.0])
                    T = kin.forward_kinematics(joints)
                    error = np.linalg.norm(T[:3, 3] - target_xyz)
                    cost = error + abs(pitch_sum - 90) * 0.00005
                    
                    if cost < best_error:
                        best_error = cost
                        best_joints = joints.copy()
    
    # Refine
    p0, l0, e0, w0 = best_joints[:4]
    for pan in np.linspace(p0-6, p0+6, 12):
        for lift in np.linspace(l0-6, l0+6, 12):
            for elbow in np.linspace(e0-6, e0+6, 12):
                for wrist_flex in np.linspace(w0-6, w0+6, 12):
                    pitch_sum = lift + elbow + wrist_flex
                    if abs(pitch_sum - 90) > max_tilt_deg:
                        continue
                    
                    joints = np.array([pan, lift, elbow, wrist_flex, 0.0])
                    T = kin.forward_kinematics(joints)
                    error = np.linalg.norm(T[:3, 3] - target_xyz)
                    
                    if error < best_error:
                        best_error = error
                        best_joints = joints.copy()
    
    T_final = kin.forward_kinematics(best_joints)
    z_axis = T_final[:3, 2]
    tilt = np.degrees(np.arccos(max(-1, min(1, -z_axis[2])))) if z_axis[2] < 0 else 90
    
    return best_joints, best_error * 1000, tilt


# =============================================================================
# Motion
# =============================================================================

# Global gripper target - when set, all moves will maintain this grip
_gripper_target = None


def set_gripper_target(value):
    """Set the desired gripper position. All moves will maintain this."""
    global _gripper_target
    _gripper_target = value


def get_gripper_target(robot):
    """Get gripper value to command - uses target if set, else current position."""
    global _gripper_target
    if _gripper_target is not None:
        return _gripper_target
    return robot.get_observation()["gripper.pos"]


def move_smooth(robot, target_joints, duration=1.0, steps=20):
    """Basic joint-space interpolation (open-loop)."""
    current = get_current_joints(robot)
    gripper = get_gripper_target(robot)

    for i in range(steps + 1):
        t = (i / steps) ** 2 * (3 - 2 * i / steps)
        interp = current + (target_joints - current) * t
        action = {f"{j}.pos": interp[k] for k, j in enumerate(IK_JOINTS)}
        action["gripper.pos"] = gripper
        robot.send_action(action)
        time.sleep(duration / steps)


def move_cartesian(robot, kin, target_xyz, ik_func, duration=1.0, steps=15, **ik_kwargs):
    """
    Cartesian-space interpolation - interpolates in XYZ, solves IK at each step.
    Produces straighter paths than joint interpolation.
    """
    current_joints = get_current_joints(robot)
    start_xyz, _ = get_ee_pose(kin, current_joints)
    gripper = get_gripper_target(robot)
    
    prev_joints = current_joints
    for i in range(steps + 1):
        t = i / steps
        t = t * t * (3 - 2 * t)  # ease in-out
        
        # Interpolate in Cartesian space
        xyz = start_xyz + (target_xyz - start_xyz) * t
        
        # Solve IK for this intermediate point
        if ik_func == solve_ik_pointing_down:
            joints, _, _ = ik_func(kin, xyz, initial_guess=prev_joints, **ik_kwargs)
        else:
            joints = ik_func(kin, prev_joints, xyz, **ik_kwargs)
        
        action = {f"{j}.pos": joints[k] for k, j in enumerate(IK_JOINTS)}
        action["gripper.pos"] = gripper
        robot.send_action(action)
        prev_joints = joints
        time.sleep(duration / steps)
    
    return prev_joints


def move_corrected(robot, kin, target_xyz, ik_func, threshold_mm=2.0, max_attempts=5, **ik_kwargs):
    """
    Closed-loop correction - moves to target, reads actual position, corrects if needed.
    More accurate but slower than open-loop.
    """
    gripper = get_gripper_target(robot)
    
    for attempt in range(max_attempts):
        # Read actual position
        actual_joints = get_current_joints(robot)
        actual_xyz, _ = get_ee_pose(kin, actual_joints)
        
        error = np.linalg.norm(actual_xyz - target_xyz) * 1000
        if error < threshold_mm:
            return actual_joints, error, attempt + 1
        
        # Compute correction from actual position
        if ik_func == solve_ik_pointing_down:
            target_joints, _, _ = ik_func(kin, target_xyz, initial_guess=actual_joints, **ik_kwargs)
        else:
            target_joints = ik_func(kin, actual_joints, target_xyz, **ik_kwargs)
        
        # Move (shorter duration for corrections)
        dur = 0.5 if attempt == 0 else 0.3
        current = actual_joints
        for i in range(10):
            t = (i + 1) / 10
            interp = current + (target_joints - current) * t
            action = {f"{j}.pos": interp[k] for k, j in enumerate(IK_JOINTS)}
            action["gripper.pos"] = gripper
            robot.send_action(action)
            time.sleep(dur / 10)
        
        time.sleep(0.1)  # Let it settle
    
    # Final check
    actual_joints = get_current_joints(robot)
    actual_xyz, _ = get_ee_pose(kin, actual_joints)
    error = np.linalg.norm(actual_xyz - target_xyz) * 1000
    return actual_joints, error, max_attempts


# =============================================================================
# Shapes
# =============================================================================

def generate_shape(shape, cx, cy, z, size):
    """Generate waypoints (meters)."""
    if shape == "square":
        hs = size / 2
        return [
            np.array([cx - hs, cy - hs, z]), np.array([cx - hs, cy + hs, z]),
            np.array([cx + hs, cy + hs, z]), np.array([cx + hs, cy - hs, z]),
            np.array([cx - hs, cy - hs, z]),
        ]
    elif shape == "circle":
        angles = np.linspace(0, 2 * np.pi, 17)
        r = size / 2
        return [np.array([cx + r * np.cos(a), cy + r * np.sin(a), z]) for a in angles]
    elif shape == "triangle":
        r = size / np.sqrt(3)
        return [np.array([cx + r * np.cos(np.pi/2 + i * 2*np.pi/3), 
                         cy + r * np.sin(np.pi/2 + i * 2*np.pi/3), z]) for i in range(4)]
    else:
        raise ValueError(f"Unknown shape: {shape}")


# =============================================================================
# Display
# =============================================================================

def show_status(kin, robot):
    global _gripper_target
    joints = get_current_joints(robot)
    pos, T = get_ee_pose(kin, joints)
    gripper = robot.get_observation()["gripper.pos"]
    
    z_axis = T[:3, 2]
    tilt = np.degrees(np.arccos(max(-1, min(1, -z_axis[2])))) if z_axis[2] < 0 else 90

    # Gripper status with lock indicator
    if _gripper_target is not None:
        grip_str = f"Gripper: {gripper:.0f}% [LOCKED @ {_gripper_target:.0f}%]"
    else:
        grip_str = f"Gripper: {gripper:.0f}%"

    print("\n" + "=" * 60)
    print("Joints (deg):", " ".join(f"{j[:3]}={joints[i]:.1f}" for i, j in enumerate(IK_JOINTS)))
    print(grip_str)
    print(f"EE (mm): X={pos[0]*1000:.1f}  Y={pos[1]*1000:.1f}  Z={pos[2]*1000:.1f}  Tilt={tilt:.0f}°")
    print("=" * 60)


def print_help():
    print("""
Commands:
  x/y/z <mm>                 Move axis by delta
  goto <x> <y> <z>           Move to position (mm)
  home                       Return to home position
  
  draw <shape> [size] [x] [y] [z]
                             Draw shape with gripper down
                             Shapes: square, circle, triangle
                             Default: 50mm at current XY, Z=-30
  down                       Point gripper down at current XY
  
  g <0-100>                  Set gripper (and lock it there)
  grip                       Show current grip target
  release                    Release grip lock (gripper follows current pos)
  tilt [deg]                 Set/show max tilt (default 25)
  mode [j|c|cl]              Set motion mode:
                               j  = joint interp (fast, curved paths)
                               c  = cartesian interp (straight paths)
                               cl = closed-loop (accurate, slower)
  r                          Refresh
  q                          Quit
""")


# =============================================================================
# Interactive mode
# =============================================================================

def run_interactive(robot):
    kin = get_kinematics()
    max_tilt = 25.0
    motion_mode = "c"  # Default to cartesian interpolation
    
    print("\n" + "=" * 60)
    print(" SO101 Interactive IK Control")
    print("=" * 60)
    print_help()

    try:
        while True:
            show_status(kin, robot)
            mode_names = {"j": "joint", "c": "cartesian", "cl": "closed-loop"}
            print(f"[mode: {mode_names.get(motion_mode, motion_mode)}, tilt: {max_tilt}°]")
            cmd = input("\n> ").strip().split()

            if not cmd:
                continue
            c = cmd[0].lower()
            
            if c == "q":
                break
            if c == "r":
                continue
            if c == "help":
                print_help()
                continue

            # Settings
            if c == "tilt":
                if len(cmd) >= 2:
                    max_tilt = float(cmd[1])
                print(f"Max tilt: {max_tilt}°")
                continue
            
            if c == "mode":
                if len(cmd) >= 2 and cmd[1] in ["j", "c", "cl"]:
                    motion_mode = cmd[1]
                print(f"Motion mode: {mode_names.get(motion_mode, motion_mode)}")
                continue

            # Basic movement
            if c == "home":
                print("Moving to home...")
                move_smooth(robot, np.zeros(5), duration=1.5)
                continue

            # Gripper commands
            if c == "g" and len(cmd) >= 2:
                grip_val = float(cmd[1])
                set_gripper_target(grip_val)
                obs = robot.get_observation()
                action = {f"{j}.pos": obs[f"{j}.pos"] for j in IK_JOINTS}
                action["gripper.pos"] = grip_val
                robot.send_action(action)
                time.sleep(0.3)
                print(f"Gripper locked at {grip_val:.0f}%")
                continue
            
            if c == "grip":
                if _gripper_target is not None:
                    print(f"Gripper locked at {_gripper_target:.0f}%")
                else:
                    obs = robot.get_observation()
                    print(f"Gripper unlocked (currently {obs['gripper.pos']:.0f}%)")
                continue
            
            if c == "release":
                set_gripper_target(None)
                print("Gripper lock released - will follow current position")
                continue

            # Point down
            if c == "down":
                joints = get_current_joints(robot)
                pos, _ = get_ee_pose(kin, joints)
                target = np.array([max(pos[0], 0.18), pos[1], -0.03])
                print(f"Pointing down at ({target[0]*1000:.0f}, {target[1]*1000:.0f}, -30)mm...")
                
                if motion_mode == "cl":
                    _, err, attempts = move_corrected(robot, kin, target, solve_ik_pointing_down, 
                                                       max_tilt_deg=max_tilt)
                    print(f"  Final error: {err:.1f}mm ({attempts} attempts)")
                else:
                    new_joints, err, tilt = solve_ik_pointing_down(kin, target, max_tilt)
                    print(f"  Planned error: {err:.1f}mm, Tilt: {tilt:.1f}°")
                    if motion_mode == "c":
                        move_cartesian(robot, kin, target, solve_ik_pointing_down, max_tilt_deg=max_tilt)
                    else:
                        move_smooth(robot, new_joints, duration=1.0)
                continue

            # Draw
            if c == "draw":
                shape = cmd[1] if len(cmd) >= 2 else "square"
                size = float(cmd[2]) / 1000 if len(cmd) >= 3 else 0.05
                
                joints = get_current_joints(robot)
                pos, _ = get_ee_pose(kin, joints)
                
                # Parse optional position args
                cx = float(cmd[3]) / 1000 if len(cmd) >= 4 else max(pos[0], 0.18)
                cy = float(cmd[4]) / 1000 if len(cmd) >= 5 else pos[1]
                z = float(cmd[5]) / 1000 if len(cmd) >= 6 else -0.03
                
                waypoints = generate_shape(shape, cx, cy, z, size)
                print(f"\nDrawing {shape} ({size*1000:.0f}mm) at ({cx*1000:.0f}, {cy*1000:.0f}, {z*1000:.0f})mm")
                print(f"Mode: {mode_names.get(motion_mode, motion_mode)}")
                
                # Plan
                trajectory = []
                prev = None
                for i, wp in enumerate(waypoints):
                    jts, err, tilt = solve_ik_pointing_down(kin, wp, max_tilt, prev)
                    marker = " (!)" if err > 5 else ""
                    print(f"  WP{i}: err={err:.1f}mm tilt={tilt:.0f}°{marker}")
                    trajectory.append((wp, jts))
                    prev = jts
                
                try:
                    input("\nEnter to execute, Ctrl+C to cancel...")
                    
                    for i, (wp, jts) in enumerate(trajectory):
                        if motion_mode == "cl":
                            _, err, attempts = move_corrected(robot, kin, wp, solve_ik_pointing_down,
                                                               max_tilt_deg=max_tilt)
                            print(f"  WP{i}: err={err:.1f}mm ({attempts} corrections)")
                        elif motion_mode == "c":
                            move_cartesian(robot, kin, wp, solve_ik_pointing_down,
                                          duration=1.0 if i == 0 else 0.5, max_tilt_deg=max_tilt)
                        else:
                            move_smooth(robot, jts, duration=1.0 if i == 0 else 0.5)
                    
                    print("Done!")
                except KeyboardInterrupt:
                    print("\nCancelled")
                continue

            # Cartesian movement
            try:
                joints = get_current_joints(robot)
                pos, _ = get_ee_pose(kin, joints)
                target = pos.copy()

                if c in ["x", "y", "z"] and len(cmd) >= 2:
                    axis = {"x": 0, "y": 1, "z": 2}[c]
                    target[axis] += float(cmd[1]) / 1000
                elif c == "goto" and len(cmd) >= 4:
                    target = np.array([float(cmd[1]), float(cmd[2]), float(cmd[3])]) / 1000
                else:
                    print("Unknown command. Type 'help' for commands.")
                    continue

                print(f"Target: ({target[0]*1000:.1f}, {target[1]*1000:.1f}, {target[2]*1000:.1f})mm")
                
                if motion_mode == "cl":
                    _, err, attempts = move_corrected(robot, kin, target, solve_ik_generic)
                    print(f"  Final error: {err:.1f}mm ({attempts} attempts)")
                elif motion_mode == "c":
                    move_cartesian(robot, kin, target, solve_ik_generic, duration=0.8)
                else:
                    new_joints = solve_ik_generic(kin, joints, target)
                    achieved, _ = get_ee_pose(kin, new_joints)
                    err = np.linalg.norm(achieved - target) * 1000
                    if err > 10:
                        print(f"Warning: Error {err:.1f}mm - may be unreachable")
                    move_smooth(robot, new_joints, duration=0.8)

            except (ValueError, IndexError) as e:
                print(f"Error: {e}")

    except KeyboardInterrupt:
        print("\nExiting...")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SO101 IK Control")
    parser.add_argument("--port", required=True, help="USB port")
    parser.add_argument("--id", default="my_awesome_follower_arm", help="Robot ID")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    config = SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True)
    robot = SO101Follower(config)
    robot.connect(calibrate=False)

    try:
        if args.interactive:
            run_interactive(robot)
        else:
            # Default: show status
            kin = get_kinematics()
            show_status(kin, robot)
            print("\nUse --interactive (-i) for control mode")
    finally:
        robot.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
