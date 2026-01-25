"""
Evaluate and visualize a trained SO101 reaching policy.

Usage:
    python scripts/so101_sim/evaluate.py --checkpoint so101_reach_final.pt
    python scripts/so101_sim/evaluate.py --checkpoint so101_reach_final.pt --episodes 10
"""

import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])

from env import SO101ReachEnv
from train_simple import Policy


def evaluate(args):
    """Run evaluation with visualization."""

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Create environment with rendering
    env = SO101ReachEnv(render_mode="human")

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    saved_args = checkpoint.get("args", {})
    hidden_size = saved_args.get("hidden_size", 128)

    policy = Policy(obs_dim, action_dim, hidden_size=hidden_size)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    print(f"Loaded policy from step {checkpoint.get('global_step', 'unknown')}")
    print(f"Hidden size: {hidden_size}")

    # Run episodes
    episode_returns = []
    episode_lengths = []
    episode_successes = []

    for episode in range(args.episodes):
        obs, info = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        total_reward = 0
        step = 0
        done = False

        while not done:
            with torch.no_grad():
                if args.deterministic:
                    # Use mean action
                    hidden = policy.net(obs)
                    action = policy.actor_mean(hidden)
                else:
                    # Sample from distribution
                    action, _, _, _ = policy.get_action_and_value(obs)

                action = action.squeeze(0).numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            total_reward += reward
            step += 1
            done = terminated or truncated

            # Slow down for visualization
            time.sleep(1.0 / 60)

        success = info.get("success", False)
        episode_returns.append(total_reward)
        episode_lengths.append(step)
        episode_successes.append(success)

        print(
            f"Episode {episode + 1}/{args.episodes}: "
            f"Return={total_reward:.2f}, "
            f"Length={step}, "
            f"Success={success}, "
            f"Final dist={info['distance'] * 1000:.1f}mm"
        )

    env.close()

    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Mean return: {np.mean(episode_returns):.2f} +/- {np.std(episode_returns):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
    print(f"Success rate: {np.mean(episode_successes):.1%}")


def demo_random(args):
    """Demo with random actions (no checkpoint needed)."""
    print("Running random policy demo...")

    env = SO101ReachEnv(render_mode="human")

    for episode in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        step = 0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated
            time.sleep(1.0 / 60)

        print(
            f"Episode {episode + 1}: Return={total_reward:.2f}, Length={step}, "
            f"Success={info.get('success', False)}"
        )

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate SO101 reaching policy")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic (mean) actions")
    parser.add_argument("--random", action="store_true", help="Use random actions (no checkpoint)")

    args = parser.parse_args()

    if args.random:
        demo_random(args)
    elif args.checkpoint:
        evaluate(args)
    else:
        print("Please specify --checkpoint or --random")
        print("  --checkpoint <path>  : Evaluate trained policy")
        print("  --random             : Demo with random actions")


if __name__ == "__main__":
    main()
