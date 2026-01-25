"""
Simple PPO training script for SO101 reaching task.
No PufferLib dependency - uses basic PyTorch + Gymnasium.

Usage:
    python scripts/so101_sim/train_simple.py
    python scripts/so101_sim/train_simple.py --render
    python scripts/so101_sim/train_simple.py --total-timesteps 100000
"""

import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from collections import deque

import sys
sys.path.insert(0, str(__file__).rsplit('/', 1)[0])
from env import SO101ReachEnv


class Policy(nn.Module):
    """Simple MLP policy for continuous control."""
    
    def __init__(self, obs_dim, action_dim, hidden_size=128):
        super().__init__()
        
        # Shared network
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        
        # Actor head
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head
        self.critic = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        
    def get_value(self, obs):
        return self.critic(self.net(obs)).squeeze(-1)
    
    def get_action_and_value(self, obs, action=None):
        hidden = self.net(obs)
        
        action_mean = self.actor_mean(hidden)
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(hidden).squeeze(-1)
        
        return action, log_prob, entropy, value


def make_vec_env(num_envs, render=False):
    """Create multiple environment instances."""
    envs = []
    for i in range(num_envs):
        render_mode = "human" if (render and i == 0) else None
        envs.append(SO101ReachEnv(render_mode=render_mode))
    return envs


class VecEnv:
    """Simple vectorized environment wrapper."""
    
    def __init__(self, envs):
        self.envs = envs
        self.num_envs = len(envs)
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        
    def reset(self):
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        return np.stack(obs_list)
    
    def step(self, actions):
        obs_list, rewards, terminateds, truncateds, infos = [], [], [], [], []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                # Auto-reset
                final_info = info.copy()
                obs, _ = env.reset()
                info["final_info"] = final_info
            
            obs_list.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        
        return np.stack(obs_list), np.array(rewards), np.array(terminateds), np.array(truncateds), infos
    
    def close(self):
        for env in self.envs:
            env.close()


def train(args):
    """Main training loop."""
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create vectorized environment
    envs = make_vec_env(args.num_envs, render=args.render)
    vec_env = VecEnv(envs)
    
    obs_dim = vec_env.observation_space.shape[0]
    action_dim = vec_env.action_space.shape[0]
    
    print(f"Created {args.num_envs} environments")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create policy
    policy = Policy(obs_dim, action_dim, args.hidden_size).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Training state
    global_step = 0
    num_updates = args.total_timesteps // (args.num_envs * args.num_steps)
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    
    # Buffers
    obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs, action_dim), device=device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    
    # Tracking
    episode_returns = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_successes = deque(maxlen=100)
    best_success_rate = 0.0
    
    # Initialize
    obs = vec_env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    done = torch.zeros(args.num_envs, device=device)
    
    start_time = time.time()
    
    print(f"\nStarting training for {args.total_timesteps:,} timesteps ({num_updates} updates)")
    print("=" * 70)
    
    for update in range(1, num_updates + 1):
        # Rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            
            obs_buf[step] = obs
            dones_buf[step] = done
            
            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(obs)
            
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value
            
            # Step
            next_obs, reward, terminated, truncated, infos = vec_env.step(action.cpu().numpy())
            rewards_buf[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            
            # Track episodes
            for i, info in enumerate(infos):
                if "final_info" in info:
                    final = info["final_info"]
                    # Estimate return from final distance
                    episode_successes.append(final.get("success", False))
                    episode_lengths.append(final.get("step", args.num_steps))
            
            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            done = torch.tensor(np.logical_or(terminated, truncated), dtype=torch.float32, device=device)
        
        # Compute advantages (GAE)
        with torch.no_grad():
            next_value = policy.get_value(obs)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0
            
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                
                delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values_buf
        
        # Flatten batch
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_actions = actions_buf.reshape(-1, action_dim)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # PPO update
        clipfracs = []
        for epoch in range(args.update_epochs):
            indices = torch.randperm(batch_size, device=device)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = indices[start:end]
                
                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                
                # Policy loss
                pg_loss1 = -b_advantages[mb_inds] * ratio
                pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
        
        # Logging
        if update % args.log_interval == 0 or update == 1:
            elapsed = time.time() - start_time
            sps = global_step / elapsed
            
            success_rate = np.mean(episode_successes) if episode_successes else 0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0
            avg_reward = rewards_buf.mean().item()
            
            print(f"Update {update:4d}/{num_updates} | "
                  f"Steps: {global_step:7,} | "
                  f"SPS: {sps:5.0f} | "
                  f"Reward: {avg_reward:6.3f} | "
                  f"Success: {success_rate:5.1%} | "
                  f"Clip: {np.mean(clipfracs):.3f}")
        
        # Save checkpoint
        if update % args.save_interval == 0:
            checkpoint_path = f"so101_reach_step{global_step}.pt"
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "args": vars(args),
            }, checkpoint_path)
            print(f"  -> Saved: {checkpoint_path}")
        
        # Save best model
        success_rate = np.mean(episode_successes) if episode_successes else 0
        if success_rate > best_success_rate and len(episode_successes) >= 20:
            best_success_rate = success_rate
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "success_rate": success_rate,
                "args": vars(args),
            }, "so101_reach_best.pt")
            print(f"  -> New best! Success rate: {success_rate:.1%}")
    
    # Final save
    final_path = "so101_reach_final.pt"
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "args": vars(args),
    }, final_path)
    
    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"Training complete! {global_step:,} steps in {elapsed:.1f}s ({global_step/elapsed:.0f} SPS)")
    print(f"Final model saved to: {final_path}")
    
    vec_env.close()


def main():
    parser = argparse.ArgumentParser(description="Train SO101 reaching policy (simple)")
    
    # Environment
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of environments")
    
    # Training
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Total timesteps")
    parser.add_argument("--num-steps", type=int, default=128, help="Steps per rollout")
    parser.add_argument("--num-minibatches", type=int, default=4, help="Minibatches")
    parser.add_argument("--update-epochs", type=int, default=4, help="PPO epochs")
    
    # Algorithm
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip")
    parser.add_argument("--ent-coef", type=float, default=0.005, help="Entropy coef")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value coef")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max grad norm")
    
    # Architecture
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size")
    
    # Misc
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval")
    parser.add_argument("--save-interval", type=int, default=50, help="Save interval")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
