#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement a model-free approach called Deep DPG (DDPG)


@author: udacity, ucaiado

Created on 10/07/2018
"""

import numpy as np
import random
import copy
import os
import yaml
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

import pdb

from munch import DefaultMunch

from alternative.models import Actor, Critic
from utility.noise import OUNoise


class MultiAgent(object):
    '''
    '''

    def __init__(self, config: DefaultMunch, state_size, action_size, nb_agents, rand_seed):
        '''Initialize an MultiAgent object.

        :param state_size: int. dimension of each state
        :param action_size: int. dimension of each action
        :param nb_agents: int. number of agents to use
        :param rand_seed: int. random seed
        '''
        self.config = config
        # Replay memory
        self.memory = ReplayBuffer(self.config.buffer_size, rand_seed)
        self.n_agents = nb_agents
        self.na_idx = np.arange(self.n_agents)
        self.action_size = action_size
        self.act_size = action_size * nb_agents
        self.state_size = state_size * nb_agents
        self.agents = [Agent(config, state_size, action_size, rand_seed, self) for i in range(nb_agents)]

    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add((states[0], actions[0], rewards[0], next_states[0], dones[0], states[1], actions[1], next_states[1]))
        self.agents[0].step()
        self.memory.add((states[1], actions[1], rewards[1], next_states[1], dones[1], states[0], actions[0], next_states[0]))
        self.agents[1].step()

    def act(self, states, add_noise=True):
        actions1: torch.Tensor = self.agents[0].act(states[0], add_noise)
        actions2: torch.Tensor = self.agents[1].act(states[1], add_noise)
        actions = torch.stack([actions1, actions2], dim=0)
        return actions

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def save(self, path, episode):
        for i, agent in enumerate(self.agents):
            agent.save(path + str(i), episode)

    def load(self, path):
        for i, agent in enumerate(self.agents):
            agent.load(path + str(i))


class Agent(object):
    '''
    Implementation of a DQN agent that interacts with and learns from the
    environment
    '''

    def __init__(self, config1: DefaultMunch, state_size, action_size, rand_seed, meta_agent):
        '''Initialize an MetaAgent object.

        :param state_size: int. dimension of each state
        :param action_size: int. dimension of each action
        :param nb_agents: int. number of agents to use
        :param rand_seed: int. random seed
        :param memory: ReplayBuffer object.
        '''
        self.config = config1
        self.action_size = action_size

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, rand_seed).to(self.config.device)
        self.actor_target = Actor(state_size, action_size, rand_seed).to(self.config.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.config.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, meta_agent.n_agents, rand_seed).to(self.config.device)
        self.critic_target = Critic(state_size, action_size, meta_agent.n_agents, rand_seed).to(self.config.device)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.config.lr_critic)

        # Noise process
        self.noise = OUNoise(action_size, rand_seed)

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
                experiences = self.memory.sample(self.config.batch_size)
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
        '''
        Update policy and value params using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        :param experiences: Tuple[torch.Tensor]. tuple of (s, a, r, s', done)
        :param gamma: float. discount factor
        '''
        (states, actions, rewards, next_states, dones, others_states,
         others_actions, others_next_states) = experiences
        # rewards_ = torch.clamp(rewards, min=-1., max=1.)
        rewards_ = rewards
        all_states = torch.cat((states, others_states), dim=1).to(self.config.device)
        all_actions = torch.cat((actions, others_actions), dim=1).to(self.config.device)
        all_next_states = torch.cat((next_states, others_next_states), dim=1).to(self.config.device)

        # --------------------------- update critic ---------------------------
        # Get predicted next-state actions and Q values from target models
        l_all_next_actions = []
        l_all_next_actions.append(self.actor_target(states))
        l_all_next_actions.append(self.actor_target(others_states))
        all_next_actions = torch.cat(l_all_next_actions, dim=1).to(self.config.device)

        Q_targets_next = self.critic_target(all_next_states, all_next_actions)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards_ + (gamma * Q_targets_next * (1 - dones.float()))
        # Compute critic loss: L = 1/N SUM{(yi − Q(si, ai|θQ))^2}
        Q_expected = self.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # suggested by Attempt 3, from Udacity
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # --------------------------- update actor ---------------------------
        # Compute actor loss: ∇θμ J ≈1/N  ∇aQ(s, a|θQ)|s=si,a=μ(si)∇θμ μ(s|θμ)
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
        # Update the critic target networks: θQ′ ←τθQ +(1−τ)θQ′
        # Update the actor target networks: θμ′ ←τθμ +(1−τ)θμ′
        self.soft_update(self.critic_local, self.critic_target, self.config.tau)
        self.soft_update(self.actor_local, self.actor_target, self.config.tau)

    def soft_update(self, local_model, target_model, tau):
        '''Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model: PyTorch model. weights will be copied from
        :param target_model: PyTorch model. weights will be copied to
        :param tau: float. interpolation parameter
        '''
        iter_params = zip(target_model.parameters(), local_model.parameters())
        for target_param, local_param in iter_params:
            tensor_aux = tau * local_param.data + (1.0 - tau) * target_param.data
            target_param.data.copy_(tensor_aux)

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


class ReplayBuffer(object):
    '''Fixed-size buffer to store experience tuples.'''

    def __init__(self, buffer_size, seed):
        '''Initialize a ReplayBuffer object.

        :param action_size: int. dimension of each action
        :param buffer_size: int: maximum size of buffer
        :param batch_size (int): size of each training batch
        :param seed (int): random seed
        '''
        self.memory = deque(maxlen=buffer_size)
        self.seed = random.seed(seed)

    def add(self, transition):
        '''Add a new experience to memory.'''
        self.memory.append(transition)

    def sample(self, batch_size):
        '''Randomly sample a batch of experiences from memory.'''
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)
