#!/usr/bin/env python
"""
SO101 Basic Control Script - Connect, read, and control your SO101 arm.

Usage:
    python scripts/test_so101_basic.py --port /dev/ttyACM0
    python scripts/test_so101_basic.py --port /dev/ttyACM0 --mode interactive
"""

import argparse
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def show_status(robot):
    """Display arm status with both normalized and raw values."""
    bus = robot.bus
    voltage = bus.read("Present_Voltage", "shoulder_pan", normalize=False)
    print(f"\n[Status] Port: {robot.config.port} | Voltage: {voltage/10:.1f}V")
    
    print("\n" + "-" * 78)
    print(f"{'Joint':<16} | {'ID':>3} | {'Position':>12} | {'Raw':>6} | {'Min':>6} | {'Max':>6} | {'Temp':>4}")
    print("-" * 78)
    
    pos_norm = bus.sync_read("Present_Position", normalize=True)
    pos_raw = bus.sync_read("Present_Position", normalize=False)
    cal = robot.calibration
    
    for name, motor in bus.motors.items():
        temp = bus.read("Present_Temperature", name, normalize=False)
        r_min = cal[name].range_min if name in cal else "?"
        r_max = cal[name].range_max if name in cal else "?"
        unit = "%" if name == "gripper" else "deg"
        print(f"{name:<16} | {motor.id:>3} | {pos_norm[name]:>9.1f} {unit} | {pos_raw[name]:>6} | {r_min:>6} | {r_max:>6} | {temp:>3}°C")
    print("-" * 78)


def mode_all(robot):
    """Show full arm status."""
    show_status(robot)


def mode_interactive(robot):
    """Interactive joint control."""
    joints = list(robot.bus.motors.keys())
    print("\nInteractive mode - Enter: <joint_num> <value>  |  'r'=refresh  |  'q'=quit\n")
    
    try:
        while True:
            show_status(robot)
            print("\nJoints:", ", ".join(f"[{i}]{j}" for i, j in enumerate(joints)))
            
            cmd = input("> ").strip().lower()
            if cmd == 'q':
                break
            if cmd == 'r':
                continue
            
            try:
                idx, target = int(cmd.split()[0]), float(cmd.split()[1])
                if 0 <= idx < len(joints):
                    obs = robot.get_observation()
                    action = {f"{j}.pos": obs[f"{j}.pos"] for j in joints}
                    action[f"{joints[idx]}.pos"] = target
                    print(f"Moving {joints[idx]} to {target}...")
                    robot.send_action(action)
                    time.sleep(0.3)
            except (ValueError, IndexError):
                print("Invalid. Use: <joint_num> <value>")
    except KeyboardInterrupt:
        print("\nExiting...")


def main():
    parser = argparse.ArgumentParser(description="SO101 Control")
    parser.add_argument("--port", required=True, help="USB port (e.g., /dev/ttyACM0)")
    parser.add_argument("--id", default="my_awesome_follower_arm", help="Robot ID")
    parser.add_argument("--mode", default="all", choices=["all", "interactive"])
    args = parser.parse_args()
    
    print(f"\n{'='*60}\n SO101 Control | Port: {args.port} | Mode: {args.mode}\n{'='*60}")
    
    config = SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True)
    robot = SO101Follower(config)
    robot.connect(calibrate=False)
    
    try:
        {"all": mode_all, "interactive": mode_interactive}[args.mode](robot)
    finally:
        robot.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
