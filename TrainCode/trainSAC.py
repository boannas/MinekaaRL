import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import VecTransposeImage

import gym
from gym import spaces
from gym.vector import SyncVectorEnv
from stable_baselines3.common.buffers import ReplayBuffer

from ursina_env_SAC_zigzag import UrsinaParkourEnv

class ImageReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device, n_envs=1):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.n_envs = n_envs
        
        self.pos = 0
        self.full = False
        
        # Initialize buffers
        obs_shape = observation_space.shape
        self.observations = np.zeros((buffer_size, *obs_shape), dtype=np.uint8)
        self.next_observations = np.zeros((buffer_size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((buffer_size, 1), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
    
    def add(self, obs, next_obs, actions, rewards, dones, infos):
        # Add transitions to the buffer
        for i in range(self.n_envs):
            self.observations[self.pos] = np.array(obs)[i]
            self.next_observations[self.pos] = np.array(next_obs)[i]
            self.actions[self.pos] = np.array([actions])[i]
            self.rewards[self.pos] = np.array([rewards])[i]
            self.dones[self.pos] = np.array([dones])[i]
            
            self.pos += 1
            if self.pos == self.buffer_size:
                self.full = True
                self.pos = 0
    
    def sample(self, batch_size):
        # Sample a batch of transitions
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        
        from collections import namedtuple
        SampleBatch = namedtuple("SampleBatch", ["observations", "next_observations", "actions", "rewards", "dones"])
        
        return SampleBatch(
            observations=torch.as_tensor(self.observations[batch_inds], dtype=torch.float32, device=self.device) / 255.0,
            next_observations=torch.as_tensor(self.next_observations[batch_inds], dtype=torch.float32, device=self.device) / 255.0,
            actions=torch.as_tensor(self.actions[batch_inds], device=self.device),
            rewards=torch.as_tensor(self.rewards[batch_inds], device=self.device),
            dones=torch.as_tensor(self.dones[batch_inds], device=self.device)
        )

class EnsureImageSpace(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

    def observation(self, obs):
        return obs

@dataclass
class Args:
    exp_name: str = "sac_ursina_zigzag"
    seed: int = 42
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False

    total_timesteps: int = 20000
    buffer_size: int = int(1e4)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 64
    learning_starts: int = 1000
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    update_frequency: int = 1
    target_network_frequency: int = 500
    alpha: float = 0.2
    autotune: bool = True
    target_entropy_scale: float = 0.7


def make_env(seed):
    def thunk():
        env = UrsinaParkourEnv()

        class EnsureImageSpace(gym.ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                self.observation_space = gym.spaces.Box(
                    low=0, high=255, shape=(3, 84, 84), dtype=np.uint8
                )

            def observation(self, obs):
                if obs.shape != (3, 84, 84):
                    obs = np.transpose(obs, (2, 0, 1))
                return obs

        env = EnsureImageSpace(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SoftQNetwork(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )
        with torch.inference_mode():
            test_input = torch.zeros(1, *obs_shape, dtype=torch.float32)
            output_dim = self.conv(test_input).shape[1]
        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_q = layer_init(nn.Linear(512, action_dim))

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        return self.fc_q(x)

class Actor(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), 
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )
        with torch.inference_mode():
            test_input = torch.zeros(1, *obs_shape, dtype=torch.float32)
            output_dim = self.conv(test_input).shape[1]
        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_logits = layer_init(nn.Linear(512, action_dim))

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        return self.fc_logits(x)

    def get_action(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        logits = self(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = F.log_softmax(logits, dim=-1)
        probs = dist.probs
        return action, log_prob, probs


def train():
    args = Args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"cnn_log/{run_name}")
    writer.add_text("hyperparameters", "\n".join([f"{k}: {v}" for k, v in vars(args).items()]))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create the vectorized environment
    envs = SyncVectorEnv([make_env(args.seed)])
    obs_shape = envs.single_observation_space.shape
    action_dim = envs.single_action_space.n

    # Initialize actor and critic networks
    actor = Actor(obs_shape, action_dim).to(device)
    qf1 = SoftQNetwork(obs_shape, action_dim).to(device)
    qf2 = SoftQNetwork(obs_shape, action_dim).to(device)
    qf1_target = SoftQNetwork(obs_shape, action_dim).to(device)
    qf2_target = SoftQNetwork(obs_shape, action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr, eps=1e-4)

    # Initialize alpha
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(torch.tensor(1.0 / action_dim))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
        alpha = log_alpha.exp().item()
    else:
        alpha = args.alpha

    rb = ImageReplayBuffer(args.buffer_size, envs.single_observation_space, envs.single_action_space, device)

    # Reset the environment
    obs, _ = envs.reset()
    start_time = time.time()

    # Initialize loss variables
    qf1_loss = torch.tensor(0.0)
    qf2_loss = torch.tensor(0.0)
    actor_loss = torch.tensor(0.0)
    alpha_loss = torch.tensor(0.0)

    # Training loop
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device) / 255.0
                actions, _, _ = actor.get_action(obs_tensor)
            actions = actions.cpu().numpy()

        # Step the environment
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Store transition in replay buffer
        rb.add(obs, next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # Log episode returns
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None and "episode" in info:
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    print(f"Step {global_step}: Return={info['episode']['r']}")
                    break

        # Update networks
        if global_step > args.learning_starts and global_step % args.update_frequency == 0:
            data = rb.sample(args.batch_size)

            # --- Update Q networks ---
            with torch.no_grad():
                _, next_log_pi, next_probs = actor.get_action(data.next_observations)
                qf1_next = qf1_target(data.next_observations)
                qf2_next = qf2_target(data.next_observations)
                min_qf_next = next_probs * (torch.min(qf1_next, qf2_next) - alpha * next_log_pi)
                target_q = data.rewards.flatten() + args.gamma * (1 - data.dones.flatten()) * min_qf_next.sum(dim=1)

            q1 = qf1(data.observations).gather(1, data.actions.long()).squeeze()
            q2 = qf2(data.observations).gather(1, data.actions.long()).squeeze()
            qf1_loss = F.mse_loss(q1, target_q)
            qf2_loss = F.mse_loss(q2, target_q)

            q_optimizer.zero_grad()
            (qf1_loss + qf2_loss).backward()
            q_optimizer.step()

            # --- Update Actor ---
            _, log_pi, probs = actor.get_action(data.observations)
            with torch.no_grad():
                qf1_values = qf1(data.observations)
                qf2_values = qf2(data.observations)
                min_qf = torch.min(qf1_values, qf2_values)
            actor_loss = (probs * (alpha * log_pi - min_qf)).sum(dim=1).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # --- Update alpha ---
            if args.autotune:
                alpha_loss = (probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).sum(dim=1).mean()
                a_optimizer.zero_grad()
                alpha_loss.backward()
                a_optimizer.step()
                alpha = log_alpha.exp().item()

            if global_step % 1000 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                writer.add_scalar("charts/alpha", alpha, global_step)
                print(f"Step {global_step}: Q1 Loss = {qf1_loss.item():.3f}, Actor Loss = {actor_loss.item():.3f}")

                # --- Save model checkpoints ---
                checkpoint_dir = f"checkpoints/{run_name}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                torch.save(actor.state_dict(), os.path.join(checkpoint_dir, f"actor_{global_step}.pth"))
                torch.save(qf1.state_dict(), os.path.join(checkpoint_dir, f"qf1_{global_step}.pth"))
                torch.save(qf2.state_dict(), os.path.join(checkpoint_dir, f"qf2_{global_step}.pth"))
                if args.autotune:
                    torch.save(log_alpha, os.path.join(checkpoint_dir, f"log_alpha_{global_step}.pth"))

        # Update target networks
        if global_step % args.target_network_frequency == 0:
            for p, tp in zip(qf1.parameters(), qf1_target.parameters()):
                tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
            for p, tp in zip(qf2.parameters(), qf2_target.parameters()):
                tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)

    envs.close()
    writer.close()


if __name__ == "__main__":
    train()
