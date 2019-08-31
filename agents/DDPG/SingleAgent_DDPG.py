import os
import pickle
from collections import deque
from functools import reduce
import numpy as np
import torch
import torch.distributions
import torch.nn as nn
import torch.optim.optimizer
from tensorboardX import SummaryWriter

import utility.constants as constants
from agents.GenericAgent import GenericAgent
from utility.PrioritisedExperienceReplayBuffer_cython import PrioritizedReplayBuffer
from utility.ReplayMemory import ExperienceReplayMemory
from utility.Scheduler import Scheduler
from utility.noise import OUNoise
from utility.td_buffer import TDBuffer
import torch.nn.functional as F
from munch import DefaultMunch


class AgentDDPG(GenericAgent):
    """Interacts with and learns from the environment."""

    def __init__(self, config: DefaultMunch):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super().__init__(config)
        self.actor: torch.nn.Module = config.actor_fn()
        self.critic: torch.nn.Module = config.critic_fn()
        self.target_actor: torch.nn.Module = config.actor_fn()  # clones the actor
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic: torch.nn.Module = config.critic_fn()  # clones the critic
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.input_dim: int = config[constants.input_dim]
        self.output_dim: int = config[constants.output_dim]
        self.n_episodes: int = config[constants.n_episodes]
        self.gamma: float = config[constants.gamma]
        self.tau: float = config[constants.tau]
        self.device = config[constants.device]
        self.optimiser_actor: torch.optim.optimizer.Optimizer = config.optimiser_actor_fn(self.actor)
        self.optimiser_critic: torch.optim.optimizer.Optimizer = config.optimiser_critic_fn(self.critic)
        self.ending_condition = config[constants.ending_condition]
        self.batch_size = config.batch_size
        self.action_size = config.action_size
        self.n_agents = config.n_agents
        self.beta_start = 0.4
        self.beta_end = 1.0
        self.train_every = config.train_every
        self.train_n_times = config.train_n_times
        self.buffer_size = config.buffer_size
        self.n_step_td = config.n_step_td
        self.learn_start: int = config.learn_start if config.learn_start is not None else 0
        self.writer: SummaryWriter = config.summary_writer_fn()
        self.use_priority = config.use_priority
        self.replay_buffer = config.replay_buffer_fn() if config.replay_buffer_fn else self.buffer_fn()
        self.td_buffer = self.local_td_buffer_fn()
        self.config = config
        self.max_action = 1
        self.tag = config.tag if config.tag is not None else ""
        self.global_step = 0  # used internally for loading checkpoints from save file
        self.noise = OUNoise(self.action_size, config.seed)

    def buffer_fn(self):
        if self.use_priority:
            buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=0.6)
        else:
            buffer = ExperienceReplayMemory(self.buffer_size)
        return buffer

    def local_td_buffer_fn(self):
        if self.use_priority:
            return TDBuffer(n_steps=self.n_step_td, gamma=self.gamma, memory=self.replay_buffer, evaluate_fn=self.calculate_td_errors, device=self.device)
        else:
            return TDBuffer(n_steps=self.n_step_td, gamma=self.gamma, memory=self.replay_buffer, evaluate_fn=lambda *args: 1.0, device=self.device)

    # self.t_update_target_step = 0
    def required_properties(self):
        return [constants.input_dim,
                constants.output_dim,
                constants.max_t,
                constants.n_episodes,
                constants.gamma,
                constants.tau,
                constants.device,
                "summary_writer_fn"]

    def step(self, memory):
        self.td_buffer.add(memory)
        self.global_step = (self.global_step + 1)
        if self.global_step % self.train_every == 0:
            if len(self.replay_buffer) >= self.batch_size:
                experiences, is_values, indices = self.replay_buffer.sample(self.batch_size, beta=0.5)
                self.learn(self.global_step)

    def reset(self):
        # self.global_step = 0  # maybe wrong place
        self.noise.reset()  # todo resets the noise generator/or maybe the internals of noisy nets

    def act(self, states: torch.Tensor, noise_magnitude=0):
        # noise_magnitude = 0.2 if self.noise_scheduler is None else self.noise_scheduler.get(i_episode)
        if self.global_step < self.learn_start:
            actions = (torch.rand(states.size()[0], self.action_size) * 2).to(self.device) - self.max_action
        else:
            self.actor.eval()
            with torch.no_grad():
                actions: torch.Tensor = self.actor(states)
                # noise = torch.normal(torch.zeros_like(actions), torch.ones_like(actions) * noise_magnitude)  # adds exploratory noise
                if noise_magnitude != 0:
                    noise = torch.tensor(self.noise.sample(), dtype=torch.float, device=self.device)
                else:
                    noise = torch.zeros_like(actions)
                actions = torch.clamp(actions + noise, -self.max_action, self.max_action).to(self.device)  # clips the action to the allowed boundaries
            self.actor.train()
        return actions

    def learn(self, i_episode: int):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        for i in range(self.train_n_times):
            beta = (self.beta_end - self.beta_start) * i_episode / self.n_episodes + self.beta_start
            experiences, is_values, indexes = self.replay_buffer.sample(self.batch_size, beta=beta)
            states, actions, rewards, dones, next_states = zip(*experiences)
            states = torch.stack([torch.stack(state) for state in states])
            actions = torch.stack([torch.stack(action) for action in actions])
            rewards = torch.stack(rewards)
            next_states = torch.stack([torch.stack(state) for state in next_states])
            is_values = torch.from_numpy(is_values).float().to(self.device)
            dones = 1 - torch.stack([torch.stack(done).any() for done in dones]).float()
            policy_noise = 0.2  #
            noise_clip = 0.5
            # Select action according to policy and add clipped noise
            batch_size = states.size()[0]
            # noise = torch.normal(torch.zeros_like(actions), torch.ones_like(actions) * policy_noise)
            # noise = noise.clamp(-noise_clip, noise_clip).to(self.device)
            all_next_actions = (self.target_actor(next_states.view(-1, states.size()[-1])).view(actions.size()))

            next_states_next_actions = torch.cat((next_states, all_next_actions), dim=2)
            state_action = torch.cat((states, actions), dim=2)
            next_states_next_actions_size = next_states_next_actions.size()
            target_Q = self.target_critic(next_states_next_actions.view(next_states_next_actions_size[0], -1))
            y = rewards + (self.gamma ** self.n_step_td * target_Q * dones.unsqueeze(dim=1))  # sets 0 to the entries which are done
            Qs_a1 = self.critic(state_action.view(state_action.size()[0], -1))

            # update critic
            self.critic.train()
            self.actor.train()
            td_error = y.detach() - Qs_a1 + 1e-5
            if self.use_priority:
                self.replay_buffer.update_priorities(indexes, abs(td_error))
            loss_critic = F.mse_loss(y.detach(), Qs_a1)  # * is_values.detach()
            self.optimiser_critic.zero_grad()
            loss_critic.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            self.optimiser_critic.step()

            self.writer.add_scalar(f'loss/critic{self.tag}', loss_critic.mean(), i_episode)
            mu_s = self.actor(states[:, 0])
            s_mu_s = torch.cat((states[:, 0], mu_s), dim=1)
            other_mu_s = self.actor(states[:, 1]).detach()
            state_action2 = torch.cat((states, torch.stack([mu_s, other_mu_s], dim=1)), dim=2)
            Qs_mu_s1 = self.critic(state_action2.view(state_action2.size()[0], -1))

            # update actor
            loss_actor = -Qs_mu_s1  # gradient ascent , use only the first of the critic's outputs
            self.optimiser_actor.zero_grad()
            loss_actor.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            self.optimiser_actor.step()

            self.writer.add_scalar(f'loss/actor{self.tag}', loss_critic.mean(), i_episode)
            self.soft_update(self.critic, self.target_critic, self.tau)
            self.soft_update(self.actor, self.target_actor, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def calculate_td_errors(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        concat_states = torch.cat([states, actions], dim=1)
        suggested_next_action = self.target_actor(next_states)
        concat_next_states = torch.cat([next_states, suggested_next_action], dim=1)
        dones = (1 - dones.float())
        target_Q1, target_Q2 = self.target_critic(concat_next_states)
        Q1, Q2 = self.critic(concat_states)
        y = rewards + np.power(self.gamma, self.n_step_td) * torch.min(target_Q1, target_Q2) * dones
        td_errors = torch.min(y - Q1, y - Q2)
        return td_errors  # calculate the td-errors, maybe use GAE

    def save(self, path, global_step):
        torch.save({
            "global_step": global_step,
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "optimiser_actor": self.optimiser_actor.state_dict(),
            "optimiser_critic": self.optimiser_critic.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.target_actor.load_state_dict(checkpoint["target_actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.optimiser_actor.load_state_dict(checkpoint["optimiser_actor"])
        self.optimiser_critic.load_state_dict(checkpoint["optimiser_critic"])
        self.replay_buffer = checkpoint["optimiser_critic"]
        self.global_step = checkpoint["global_step"]
        print(f'Loading complete')
