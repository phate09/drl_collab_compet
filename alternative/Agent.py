#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement a model-free approach called Deep DPG (DDPG)


@author: udacity, ucaiado

Created on 10/07/2018
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

from munch import DefaultMunch

from alternative.models import Actor, Critic
from utility.noise import OUNoise


class Agent(object):
    '''
    Implementation of a DQN agent that interacts with and learns from the
    environment
    '''

    def __init__(self, config1: DefaultMunch, rand_seed, meta_agent):
        '''Initialize an MetaAgent object.

        :param nb_agents: int. number of agents to use
        :param rand_seed: int. random seed
        :param memory: ReplayBuffer object.
        '''
        self.config = config1
        self.action_size = self.config.action_size
        self.state_size = self.config.state_size
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.config.action_size, rand_seed).to(self.config.device)
        self.actor_target = Actor(self.state_size, self.config.action_size, rand_seed).to(self.config.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.config.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.state_size, self.config.action_size, meta_agent.n_agents, rand_seed).to(self.config.device)
        self.critic_target = Critic(self.state_size, self.config.action_size, meta_agent.n_agents, rand_seed).to(self.config.device)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.config.lr_critic)

        # Noise process
        self.noise = OUNoise(self.config.action_size, rand_seed)

        # Replay memory
        self.memory = meta_agent.memory

        # Initialize time step (for updating every update_every steps)
        self.t_step = 0

    def step(self):

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.config.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset
            # and learn
            if len(self.memory) > self.config.batch_size:
                # source: Sample a random minibatch of N transitions from R
                experiences, _, _ = self.memory.sample(self.config.batch_size)
                states, actions, rewards, next_states, dones, others_states, others_actions, others_next_states = zip(*experiences)
                self.learn((torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(next_states), torch.stack(dones), torch.stack(others_states).squeeze(), torch.stack(others_actions).squeeze(), torch.stack(others_next_states).squeeze()), self.config.gamma)

    def act(self, states, add_noise=True):
        '''Returns actions for given states as per current policy.

        :param states: array_like. current states
        :param add_noise: Boolean. If should add noise to the action
        '''
        # states = torch.from_numpy(states).float().to(self.config.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states)
        self.actor_local.train()
        # source: Select action at = μ(st|θμ) + Nt according to the current
        # policy and exploration noise
        if add_noise:
            sample_np = self.noise.sample()
            sample = torch.tensor(sample_np, dtype=torch.float, device=self.config.device)
            actions += sample
        clipped = torch.clamp(actions, -1, 1)
        return clipped

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):

        (states, actions, rewards, next_states, dones, others_states, others_actions, others_next_states) = experiences
        all_states = torch.cat((states, others_states), dim=1).to(self.config.device)
        all_actions = torch.cat((actions, others_actions), dim=1).to(self.config.device)
        all_next_states = torch.cat((next_states, others_next_states), dim=1).to(self.config.device)

        # --------------------------- update critic ---------------------------
        all_next_actions = torch.cat([self.actor_target(states), self.actor_target(others_states)], dim=1).to(self.config.device)
        Q_targets_next = self.critic_target(all_next_states, all_next_actions)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones.float()))
        Q_expected = self.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # --------------------------- update actor ---------------------------
        this_actions_pred = self.actor_local(states)
        others_actions_pred = self.actor_local(others_states)
        others_actions_pred = others_actions_pred.detach()
        actions_pred = torch.cat((this_actions_pred, others_actions_pred), dim=1).to(self.config.device)
        actor_loss = -self.critic_local(all_states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------- update target networks ----------------------
        self.soft_update(self.critic_local, self.critic_target, self.config.tau)
        self.soft_update(self.actor_local, self.actor_target, self.config.tau)

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

    def save(self, path, global_step):
        torch.save({
            "global_step": global_step,
            "actor": self.actor_local.state_dict(),
            "target_actor": self.actor_target.state_dict(),
            "critic": self.critic_local.state_dict(),
            "target_critic": self.critic_target.state_dict(),
            "optimiser_actor": self.actor_optimizer.state_dict(),
            "optimiser_critic": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor_local.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["target_actor"])
        self.critic_local.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["target_critic"])
        self.actor_optimizer.load_state_dict(checkpoint["optimiser_actor"])
        self.critic_optimizer.load_state_dict(checkpoint["optimiser_critic"])
        self.replay_buffer = checkpoint["optimiser_critic"]
        self.global_step = checkpoint["global_step"]
        print(f'Loading complete')
