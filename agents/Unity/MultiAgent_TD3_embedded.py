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
from agents.Unity.SingleAgent_TD3 import AgentTD3
from utility.ReplayMemory import ExperienceReplayMemory
from utility.PrioritisedExperienceReplayBuffer_cython import PrioritizedReplayBuffer
from utility.Scheduler import Scheduler
from utility.td_buffer import TDBuffer
import torch.nn.functional as F


class MultiAgentTD3(GenericAgent):
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

        self.actor1 = config.actor_fn()
        self.actor2 = config.actor_fn()
        self.target_actor1 = config.actor_fn()
        self.target_actor2 = config.actor_fn()
        self.critic1 = config.critic_fn()
        self.critic2 = config.critic_fn()
        self.target_critic1 = config.critic_fn()
        self.target_critic2 = config.critic_fn()
        self.optimiser_actor1 = config.optimiser_actor_fn(self.actor1)
        self.optimiser_actor2 = config.optimiser_actor_fn(self.actor2)
        self.optimiser_critic1 = config.optimiser_critic_fn(self.critic1)
        self.optimiser_critic2 = config.optimiser_critic_fn(self.critic2)
        self.replay_buffer = ExperienceReplayMemory(config.buffer_size)
        self.td_buffer = TDBuffer(n_steps=self.n_step_td, gamma=self.gamma, memory=self.replay_buffer, evaluate_fn=lambda *args: 1.0, device=self.device)
        self.global_step = 0

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
        if self.global_step < self.learn_start:
            actions = (torch.rand(self.n_agents, self.action_size) * 2).to(self.device) - self.max_action
            return actions
        else:
            self.actor1.eval()
            self.actor2.eval()
            with torch.no_grad():
                actions1: torch.Tensor = self.actor1(states)
                actions2: torch.Tensor = self.actor2(states)
                actions = torch.stack([actions1, actions2], dim=0)
                if noise_magnitude != 0:
                    noise = torch.normal(torch.zeros_like(actions), torch.ones_like(actions) * noise_magnitude).clamp(-self.max_action / 2, self.max_action / 2).to(device=self.device)  # adds exploratory noise
                    # noise = torch.tensor(self.noise.sample(), dtype=torch.float, device=self.device)
                else:
                    noise = torch.zeros_like(actions)
                actions = torch.clamp(actions + noise, -self.max_action, self.max_action).to(self.device)  # clips the action to the allowed boundaries
            self.actor1.train()
            self.actor2.train()
            return actions

    def step(self, states, actions, rewards, dones, next_states):
        # adds the transitions to the corresponding td_buffer from the point of view of the agent
        queue = deque(maxlen=self.n_agents)
        for i in range(self.n_agents):
            queue.append((states[i], actions[i], rewards[i], dones[i], next_states[i]))
        # for i in range(self.n_agents):
        #     self.agents[i].step(list(queue))
        #     queue.rotate(1)  # rotate by 1 so it basically shifts the point of view
        self.td_buffer.add(list(queue))
        self.global_step = (self.global_step + 1)
        if self.global_step % self.train_every == 0:
            if len(self.replay_buffer) >= self.batch_size:
                experiences, is_values, indices = self.replay_buffer.sample(self.batch_size, beta=0.5)
                self.learn(self.global_step)

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
            # all_next_actions = []
            # for i in range(self.n_agents):
            #     noise = torch.normal(torch.zeros_like(actions[:, i]), torch.ones_like(actions[:, i]) * policy_noise)
            #     noise = noise.clamp(-noise_clip, noise_clip).to(self.device)
            #     target_action_next = (self.target_actor(next_states[:, i]) + noise).clamp(-self.max_action, self.max_action)
            #     all_next_actions.append(target_action_next)
            #
            # all_next_actions = torch.stack(all_next_actions,dim=1)
            noise = torch.normal(torch.zeros_like(actions), torch.ones_like(actions) * policy_noise)
            noise = noise.clamp(-noise_clip, noise_clip).to(self.device)
            all_next_actions = (self.target_actor(next_states.view(-1, states.size()[-1])).view(actions.size()) + noise).clamp(-self.max_action, self.max_action)

            next_states_next_actions = torch.cat((next_states, all_next_actions), dim=2)
            state_action = torch.cat((states, actions), dim=2)
            next_states_next_actions_size = next_states_next_actions.size()
            target_Q1, target_Q2 = self.target_critic(next_states_next_actions.view(next_states_next_actions_size[0], -1))
            target_Q = torch.min(target_Q1, target_Q2)  # takes the minimum of the two critics
            y = rewards + (self.gamma ** self.n_step_td * target_Q * dones.unsqueeze(dim=1))  # sets 0 to the entries which are done
            Qs_a1, Qs_a2 = self.critic(state_action.view(state_action.size()[0], -1))

            # update critic
            self.critic.train()
            self.actor.train()
            td_error = torch.min(y.detach() - Qs_a1, y.detach() - Qs_a2) + 1e-5
            if self.use_priority:
                self.replay_buffer.update_priorities(indexes, abs(td_error))
            loss_critic = F.mse_loss(y.detach(), Qs_a1) + F.mse_loss(y.detach(), Qs_a2)  # * is_values.detach()
            self.optimiser_critic.zero_grad()
            loss_critic.mean().backward()
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            self.optimiser_critic.step()

            self.writer.add_scalar(f'loss/critic{self.tag}', loss_critic.mean(), i_episode)
            if i % self.d == 0:
                mu_s = self.actor(states[:, 0])
                s_mu_s = torch.cat((states[:, 0], mu_s), dim=1)
                other_mu_s = self.actor(states[:, 1]).detach() if self.other_actor_fn is None else self.other_actor_fn(states[:, 1]).detach()
                state_action2 = torch.cat((states, torch.stack([mu_s, other_mu_s], dim=1)), dim=2)
                Qs_mu_s1, Qs_mu_s2 = self.critic(state_action2.view(state_action2.size()[0], -1))

                # update actor
                loss_actor = -Qs_mu_s1  # gradient ascent , use only the first of the critic's outputs
                self.optimiser_actor.zero_grad()
                loss_actor.mean().backward()
                # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                self.optimiser_actor.step()

                self.writer.add_scalar(f'loss/actor{self.tag}', loss_critic.mean(), i_episode)
                self.soft_update(self.critic, self.target_critic, self.tau)
                self.soft_update(self.actor, self.target_actor, self.tau)

    def save(self, path, episode):
        for agent in self.agents:
            agent.save(path, episode)

    def load(self, path):
        for agent in self.agents:
            agent.load(path)
