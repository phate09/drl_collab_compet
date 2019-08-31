import os
import pickle
from collections import deque
from functools import reduce
import numpy as np
import torch
import torch.distributions
import torch.nn as nn
import torch.optim.optimizer
from munch import DefaultMunch
from tensorboardX import SummaryWriter

import utility.constants as constants
from agents.GenericAgent import GenericAgent
from agents.DDPG.SingleAgent_DDPG import AgentDDPG
from utility.ReplayMemory import ExperienceReplayMemory
from utility.PrioritisedExperienceReplayBuffer_cython import PrioritizedReplayBuffer
from utility.Scheduler import Scheduler
from utility.td_buffer_TD3 import TDBuffer
import torch.nn.functional as F


class MultiAgentDDPG(GenericAgent):
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
        self.input_dim: int = config.input_dim
        self.output_dim: int = config.output_dim
        self.max_t: int = config.max_t
        self.n_episodes: int = config.n_episodes
        self.gamma: float = config.gamma
        self.tau: float = config.tau
        self.device = config.device
        self.ending_condition = config.ending_condition
        self.batch_size = config.batch_size
        self.action_size = config.action_size
        self.n_agents = config.n_agents
        self.beta_start = 0.4
        self.beta_end = 1.0
        self.train_every = config.train_every
        self.train_n_times = config.train_n_times
        self.log_dir = config.log_dir if constants.log_dir in config else None
        self.n_step_td = config.n_step_td
        self.use_noise = config.use_noise if constants.use_noise in config else False
        self.noise_scheduler: Scheduler = config.noise_scheduler if constants.noise_scheduler in config else None
        self.learn_start: int = config.learn_start if config.learn_start is not None else 0
        self.evaluate_every = config.evaluate_every if config.evaluate_every is not None else 100
        self.config = config
        self.starting_episode = 0  # used for loading checkpoints from save file
        self.max_action = 1
        self.d = 2
        self.use_priority: bool = config.use_priority
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.optimiser_actors = []
        self.optimiser_critics = []
        # self.replay_buffers = []
        self.td_buffers = []
        self.agents = []
        for i, agent_config in enumerate(config.agent_configs):
            merged_config = DefaultMunch.fromDict({**agent_config, **config})
            merged_config.pop('agent_configs', None)  # remove the agent configs entry
            merged_config.tag=i
            if config.use_shared_memory:
                if config.use_priority:
                    buffer = PrioritizedReplayBuffer(config.buffer_size, alpha=0.6)
                else:
                    buffer = ExperienceReplayMemory(config.buffer_size)
                replay_buffer_fn = lambda: buffer
                merged_config.replay_buffer_fn = replay_buffer_fn
            # td_buffer_fn = lambda :TDBuffer(n_steps=self.n_step_td, gamma=self.gamma, memory=buffer, critic=agent_config.critic, target_critic=target_critic, target_actor=target_actor, device=self.device)
            # self.td_buffers.append(TDBuffer(n_steps=self.n_step_td, gamma=self.gamma, memory=buffer, critic=agent_config.critic, target_critic=target_critic, target_actor=target_actor, device=self.device))
            agent = AgentDDPG(merged_config)
            self.agents.append(agent)

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

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def act(self, states, noise_magnitude=0) -> torch.Tensor:
        """
        Returns the action(s) to take given the current state(s)
        :param states:
        :param noise_magnitude:
        :return:
        """
        na_rtn = torch.zeros((self.n_agents, self.action_size)).to(self.device)
        for idx, agent in enumerate(self.agents):
            agent: AgentDDPG
            na_rtn[idx, :] = agent.act(states[idx].unsqueeze(dim=0), noise_magnitude)
        return na_rtn

    def step(self, states, actions, rewards, dones, next_states):
        # adds the transitions to the corresponding td_buffer from the point of view of the agent
        queue = deque(maxlen=self.n_agents)
        for i in range(self.n_agents):
            queue.append((states[i], actions[i], rewards[i], dones[i], next_states[i]))
        for i in range(self.n_agents):
            self.agents[i].step(list(queue))
            queue.rotate(1)  # rotate by 1 so it basically shifts the point of view

    def save(self, path, episode):
        for agent in self.agents:
            agent.save(path, episode)

    def load(self, path):
        for agent in self.agents:
            agent.load(path)
