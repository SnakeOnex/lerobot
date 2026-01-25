"""
Training script for SO101 reaching task using PufferLib.

Usage:
    python scripts/so101_sim/train.py
    python scripts/so101_sim/train.py --render  # With visualization
    python scripts/so101_sim/train.py --num-envs 16 --total-timesteps 1000000
"""

import argparse
import torch
import torch.nn as nn
import numpy as np

import pufferlib
import pufferlib.emulation
import pufferlib.vector
import pufferlib.pytorch

from env import SO101ReachEnv


class Policy(nn.Module):
    """Simple MLP policy for continuous control."""
    
    def __init__(self, env, hidden_size=128):
        super().__init__()
        
        obs_size = env.single_observation_space.shape[0]
        action_size = env.single_action_space.shape[0]
        
        # Shared network
        self.net = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(obs_size, hidden_size)),
            nn.Tanh(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        
        # Actor head (outputs mean of Gaussian)
        self.actor_mean = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, action_size), std=0.01
        )
        # Learnable log std
        self.actor_logstd = nn.Parameter(torch.zeros(action_size))
        
        # Critic head
        self.critic = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1.0
        )
        
    def forward(self, obs, state=None):
        """Forward pass for training."""
        hidden = self.net(obs)
        
        # Actor
        action_mean = self.actor_mean(hidden)
        action_std = torch.exp(self.actor_logstd)
        
        # Critic
        value = self.critic(hidden)
        
        return action_mean, action_std, value.squeeze(-1)
    
    def forward_eval(self, obs, state=None):
        """Forward pass for evaluation (same as forward for this simple policy)."""
        return self.forward(obs, state)
    
    def get_value(self, obs):
        """Get value estimate only."""
        hidden = self.net(obs)
        return self.critic(hidden).squeeze(-1)
    
    def get_action_and_value(self, obs, action=None):
        """Get action, log prob, entropy, and value."""
        hidden = self.net(obs)
        
        action_mean = self.actor_mean(hidden)
        action_std = torch.exp(self.actor_logstd)
        
        # Create distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(hidden).squeeze(-1)
        
        return action, log_prob, entropy, value


def make_env(render=False):
    """Create a single environment instance."""
    render_mode = "human" if render else None
    return SO101ReachEnv(render_mode=render_mode)


def train(args):
    """Main training loop."""
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create vectorized environment
    def env_creator():
        gym_env = make_env(render=args.render)
        return pufferlib.emulation.GymnasiumPufferEnv(gym_env)
    
    vecenv = pufferlib.vector.make(
        env_creator,
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        backend=pufferlib.vector.Multiprocessing,
    )
    
    print(f"Created {args.num_envs} environments with {args.num_workers} workers")
    print(f"Observation space: {vecenv.single_observation_space}")
    print(f"Action space: {vecenv.single_action_space}")
    
    # Create policy
    policy = Policy(vecenv, hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Training state
    global_step = 0
    num_updates = args.total_timesteps // (args.num_envs * args.num_steps)
    
    # Storage
    obs_buf = torch.zeros((args.num_steps, args.num_envs) + vecenv.single_observation_space.shape).to(device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs) + vecenv.single_action_space.shape).to(device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # Initialize
    obs, _ = vecenv.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    done = torch.zeros(args.num_envs, device=device)
    
    # Tracking
    episode_returns = []
    episode_lengths = []
    episode_successes = []
    
    print(f"\nStarting training for {args.total_timesteps:,} timesteps ({num_updates} updates)")
    print("=" * 60)
    
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
            
            # Step environment
            action_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, infos = vecenv.step(action_np)
            
            rewards_buf[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            
            # Track episodes
            done_mask = np.logical_or(terminated, truncated)
            for i, d in enumerate(done_mask):
                if d and "episode" in infos:
                    ep_info = infos["episode"]
                    if i < len(ep_info.get("r", [])):
                        episode_returns.append(ep_info["r"][i])
                        episode_lengths.append(ep_info["l"][i])
                # Track success from info
                if d and "success" in infos:
                    if isinstance(infos["success"], (list, np.ndarray)):
                        episode_successes.append(infos["success"][i])
                    else:
                        episode_successes.append(infos["success"])
            
            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            done = torch.tensor(done_mask, dtype=torch.float32, device=device)
        
        # Compute advantages using GAE
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
        b_obs = obs_buf.reshape((-1,) + vecenv.single_observation_space.shape)
        b_actions = actions_buf.reshape((-1,) + vecenv.single_action_space.shape)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # PPO update
        batch_size = args.num_envs * args.num_steps
        minibatch_size = batch_size // args.num_minibatches
        
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
            avg_return = np.mean(episode_returns[-100:]) if episode_returns else 0
            avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0
            success_rate = np.mean(episode_successes[-100:]) if episode_successes else 0
            
            print(f"Update {update}/{num_updates} | "
                  f"Steps: {global_step:,} | "
                  f"Return: {avg_return:.2f} | "
                  f"Length: {avg_length:.0f} | "
                  f"Success: {success_rate:.1%} | "
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
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Final save
    final_path = "so101_reach_final.pt"
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "args": vars(args),
    }, final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")
    
    vecenv.close()


def main():
    parser = argparse.ArgumentParser(description="Train SO101 reaching policy")
    
    # Environment
    parser.add_argument("--render", action="store_true", help="Enable rendering during training")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for vectorization")
    
    # Training
    parser.add_argument("--total-timesteps", type=int, default=500_000, help="Total timesteps")
    parser.add_argument("--num-steps", type=int, default=128, help="Steps per rollout")
    parser.add_argument("--num-minibatches", type=int, default=4, help="Number of minibatches")
    parser.add_argument("--update-epochs", type=int, default=4, help="PPO update epochs")
    
    # Algorithm
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    
    # Architecture
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden layer size")
    
    # Misc
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N updates")
    parser.add_argument("--save-interval", type=int, default=50, help="Save every N updates")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
