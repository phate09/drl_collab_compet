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

from alternative import param_table
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
        self.memory = ReplayBuffer(self.config.BUFFER_SIZE, rand_seed)
        self.nb_agents = nb_agents
        self.na_idx = np.arange(self.nb_agents)
        self.action_size = action_size
        self.act_size = action_size * nb_agents
        self.state_size = state_size * nb_agents
        self.l_agents = [Agent(config, state_size, action_size, rand_seed, self) for i in range(nb_agents)]

    def step(self, states, actions, rewards, next_states, dones):
        experience = zip(self.l_agents, states, actions, rewards, next_states, dones)
        for i, e in enumerate(experience):
            agent, state, action, reward, next_state, done = e
            na_filtered = self.na_idx[self.na_idx != i]
            others_states = states[na_filtered]
            others_actions = actions[na_filtered]
            others_next_states = next_states[na_filtered]
            self.memory.add(state, action, reward, next_state, done, others_states, others_actions, others_next_states)
            agent.step()

    def act(self, states, add_noise=True):
        actions1: torch.Tensor = self.l_agents[0].act(states[0],add_noise)
        actions2: torch.Tensor = self.l_agents[1].act(states[1],add_noise)
        actions = torch.stack([actions1, actions2], dim=0)
        return actions.cpu().numpy()

    def reset(self):
        for agent in self.l_agents:
            agent.reset()


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
        self.actor_local = Actor(state_size, action_size, rand_seed).to(self.config.DEVC)
        self.actor_target = Actor(state_size, action_size, rand_seed).to(self.config.DEVC)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.config.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, meta_agent.nb_agents, rand_seed).to(self.config.DEVC)
        self.critic_target = Critic(state_size, action_size, meta_agent.nb_agents, rand_seed).to(self.config.DEVC)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.config.LR_CRITIC)

        # Noise process
        self.noise = OUNoise(action_size, rand_seed)

        # Replay memory
        self.memory = meta_agent.memory

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self):

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.config.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset
            # and learn
            if len(self.memory) > self.config.BATCH_SIZE:
                # source: Sample a random minibatch of N transitions from R
                experiences = self.memory.sample(self.config.BATCH_SIZE)
                states, actions, rewards, next_states, dones, others_states, others_actions, others_next_states = zip(*experiences)
                states = torch.from_numpy(np.array(states)).float().to(self.config.DEVC)
                actions = torch.from_numpy(np.array(actions)).float().to(self.config.DEVC)
                rewards = torch.from_numpy(np.array(rewards)).float().to(self.config.DEVC)
                next_states = torch.from_numpy(np.array(next_states)).float().to(self.config.DEVC)
                dones = torch.from_numpy(np.array(dones)).float().to(self.config.DEVC)

                others_states = torch.from_numpy(np.array(others_states)).float().to(self.config.DEVC).squeeze()
                others_actions = torch.from_numpy(np.array(others_actions)).float().to(self.config.DEVC).squeeze()
                others_next_states = torch.from_numpy(np.array(others_next_states)).float().to(self.config.DEVC).squeeze()
                self.learn((states, actions, rewards, next_states, dones, others_states, others_actions, others_next_states), self.config.GAMMA)

    def act(self, states, add_noise=True):
        '''Returns actions for given states as per current policy.

        :param states: array_like. current states
        :param add_noise: Boolean. If should add noise to the action
        '''
        states = torch.from_numpy(states).float().to(self.config.DEVC)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states)
        self.actor_local.train()
        # source: Select action at = μ(st|θμ) + Nt according to the current
        # policy and exploration noise
        if add_noise:
            sample_np = self.noise.sample()
            sample = torch.tensor(sample_np, dtype=torch.float, device=self.config.DEVC)
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
        all_states = torch.cat((states, others_states), dim=1).to(self.config.DEVC)
        all_actions = torch.cat((actions, others_actions), dim=1).to(self.config.DEVC)
        all_next_states = torch.cat((next_states, others_next_states), dim=1).to(self.config.DEVC)

        # --------------------------- update critic ---------------------------
        # Get predicted next-state actions and Q values from target models
        l_all_next_actions = []
        l_all_next_actions.append(self.actor_target(states))
        l_all_next_actions.append(self.actor_target(others_states))
        all_next_actions = torch.cat(l_all_next_actions, dim=1).to(self.config.DEVC)

        Q_targets_next = self.critic_target(all_next_states, all_next_actions).squeeze()
        # Compute Q targets for current states (y_i)
        Q_targets = rewards_ + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss: L = 1/N SUM{(yi − Q(si, ai|θQ))^2}
        Q_expected = self.critic_local(all_states, all_actions).squeeze()
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
        actions_pred = torch.cat((this_actions_pred, others_actions_pred), dim=1).to(self.config.DEVC)
        actor_loss = -self.critic_local(all_states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------- update target networks ----------------------
        # Update the critic target networks: θQ′ ←τθQ +(1−τ)θQ′
        # Update the actor target networks: θμ′ ←τθμ +(1−τ)θμ′
        self.soft_update(self.critic_local, self.critic_target, self.config.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.config.TAU)

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
        # self.experience = namedtuple("Experience",
        #                              field_names=["state", "action", "reward",
        #                                           "next_state", "done",
        #                                           "others_states",
        #                                           "others_actions",
        #                                           "others_next_states"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, others_states,
            others_actions, others_next_states):
        '''Add a new experience to memory.'''
        # e = self.experience(state, action, reward, next_state, done, others_states, others_actions, others_next_states)
        self.memory.append((state, action, reward, next_state, done, others_states, others_actions, others_next_states))

    def sample(self, batch_size):
        '''Randomly sample a batch of experiences from memory.'''
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)
