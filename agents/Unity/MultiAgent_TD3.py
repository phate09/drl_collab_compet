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
from utility.td_buffer_TD3 import TDBuffer
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
            agent = AgentTD3(merged_config)
            self.agents.append(agent)
        for i,agent in enumerate(self.agents):
            agent.other_actor_fn = self.agents[(i+1)%self.n_agents].act

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
            agent: AgentTD3
            na_rtn[idx, :] = agent.act(states[idx].unsqueeze(dim=0), noise_magnitude)
        return na_rtn

    def step(self, states, actions, rewards, dones, next_states):
        # adds the transitions to the corresponding td_buffer from the point of view of the agent
        queue = deque(maxlen=self.n_agents)
        for i in range(self.n_agents):
            queue.append((states[i], actions[i], rewards[i], dones[i], next_states[i]))
        for i in range(self.n_agents):
            self.agents[i].step(list(queue))
            # queue.rotate(1)  # rotate by 1 so it basically shifts the point of view

    # def train(self, env, ending_condition):
    #     """
    #     :param env:
    #     :param writer:
    #     :param ending_condition: a method that given a score window returns true or false
    #     :return:
    #     """
    #     brain_name = env.brain_names[0]
    #     scores = []  # list containing scores from each episode
    #     scores_window = deque(maxlen=100)  # last 100 scores
    #     scores_max_window = deque(maxlen=100)  # last 100 scores
    #     scores_min_window = deque(maxlen=100)  # last 100 scores
    #     i_steps = 0
    #     for i_episode in range(self.starting_episode, self.n_episodes):
    #         # todo reset the noise
    #         if i_episode != 0 and i_episode % self.evaluate_every == 0:
    #             print("")
    #             eval_scores: list = self.evaluate(env, n_episodes=10, train_mode=True)
    #             print("")
    #             self.writer.add_scalar('eval/score_average', np.mean(eval_scores), i_episode)
    #         env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    #         # Reset the memory of the agent
    #         states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=self.device)  # get the current state
    #         score = 0
    #         score_max = 0
    #         score_min = 0
    #         noise_magnitude = 0.2 if self.noise_scheduler is None else self.noise_scheduler.get(i_steps)
    #         for t in range(self.max_t):
    #             with torch.no_grad():
    #                 if i_steps == self.learn_start:
    #                     print("Starting to learn")
    #                 if i_steps < self.learn_start:
    #                     actions = np.random.randn(self.n_agents, self.action_size)  # select an action (for each agent)
    #                     actions = torch.tensor(np.clip(actions, -self.max_action, self.max_action), dtype=torch.float, device=self.device)  # all actions between -1 and 1
    #                 else:
    #                     actions = []
    #                     for i in range(self.n_agents):
    #                         self.actors[i].eval()
    #                         actions.append(self.actors[i](states[i].unsqueeze(dim=0)).squeeze())
    #                         self.actors[i].train()
    #                     actions = torch.stack(actions, dim=0)
    #                     # actions: torch.Tensor = actor(states)
    #                     noise = torch.normal(torch.zeros_like(actions), torch.ones_like(actions) * noise_magnitude) if self.use_noise else 0  # adds exploratory noise
    #                     actions = torch.clamp(actions + noise, -self.max_action, self.max_action)  # clips the action to the allowed boundaries
    #                 # todo actions should be a list of n_agent elements containing 2 values each
    #                 env_info = env.step(actions.cpu().detach().numpy())[brain_name]  # send the action to the environment
    #                 next_states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=self.device)  # get the next state
    #                 rewards = torch.tensor(env_info.rewards, dtype=torch.float, device=self.device).unsqueeze(dim=1)  # get the reward
    #                 dones = torch.tensor(env_info.local_done, dtype=torch.uint8, device=self.device).unsqueeze(dim=1)  # see if episode has finished
    #                 for i in range(self.n_agents):
    #                     self.td_buffers[i].add((states[i], actions[i], rewards[i], dones[i], next_states[i]))
    #             # train the agent
    #             if len(self.replay_buffers[t % self.n_agents]) > self.batch_size and i_steps > self.learn_start and i_steps != 0 and i_steps % self.train_every == 0:
    #                 self.learn(i_episode=i_episode)
    #             states = next_states
    #             score_max += rewards.max().item()
    #             score_min += rewards.min().item()
    #             score += rewards.mean().item()
    #             i_steps += self.n_agents
    #             if dones.any():
    #                 for buffer in self.td_buffers:
    #                     buffer.flush()  # flushes the remaining transitions in the buffer to memory
    #                 # break
    #             # if np.any(env_info.global_done):
    #             #     break
    #         scores_window.append(score)  # save most recent score
    #         scores_max_window.append(score_max)  # save most recent score
    #         scores_min_window.append(score_min)  # save most recent score
    #         scores.append(score)  # save most recent score
    #         self.writer.add_scalar('data/score', score, i_episode)
    #         self.writer.add_scalar('data/score_average', np.mean(scores_window), i_episode)
    #         self.writer.add_scalar('data/score_max_average', np.mean(scores_max_window), i_episode)
    #         self.writer.add_scalar('data/score_min_average', np.mean(scores_min_window), i_episode)
    #         self.writer.flush()
    #         print(f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} ', end="")
    #         if (i_episode + 1) % 100 == 0:
    #             print(f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} ')
    #         # if (i_episode + 1) % 10 == 0:
    #         #     self.save(os.path.join(self.log_dir, f"checkpoint_{i_episode + 1}.pth"), i_episode)
    #         result = {"mean": np.mean(scores_window)}
    #         if ending_condition(result):
    #             print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
    #             # self.save(os.path.join(self.log_dir, f"checkpoint_success.pth"), i_episode)
    #             break
    #     return scores
    #
    # def evaluate(self, env, n_episodes=1, train_mode=False):
    #     brain_name = env.brain_names[0]
    #     scores = []  # list containing scores from each episode
    #     scores_window = deque(maxlen=100)  # last 100 scores
    #     i_steps = 0
    #     for i in range(self.n_agents):
    #         self.actors[i].eval()
    #         self.critics[i].eval()
    #     for i_episode in range(n_episodes):
    #         env_info = env.reset(train_mode=train_mode)[brain_name]  # reset the environment
    #         n_agents = len(env_info.agents)
    #         states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=self.device)  # get the current state
    #         score = 0
    #         for t in range(self.max_t):
    #             with torch.no_grad():
    #                 actions: torch.Tensor = self.actors[t % self.n_agents](states)
    #                 # noise = torch.normal(torch.zeros_like(actions), torch.ones_like(actions) * 0.2)
    #                 noise = 0  # no noise during evaluation
    #                 actions = torch.clamp(actions + noise, -1, 1)  # clips the action to the allowed boundaries
    #                 env_info = env.step(actions.cpu().detach().numpy())[brain_name]  # send the action to the environment
    #                 next_states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=self.device)  # get the next state
    #                 rewards = torch.tensor(env_info.rewards, dtype=torch.float, device=self.device).unsqueeze(dim=1)  # get the reward
    #                 dones = torch.tensor(env_info.local_done, dtype=torch.uint8, device=self.device).unsqueeze(dim=1)  # see if episode has finished
    #             states = next_states
    #             score += rewards.mean().item()
    #             i_steps += n_agents
    #             if dones.any():
    #                 break
    #         scores_window.append(score)  # save most recent score
    #         scores.append(score)  # save most recent score
    #         print(f'\rEval Episode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} ', end="")
    #         if (i_episode + 1) % 100 == 0:
    #             print(f'\rEval Episode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} ')
    #     for i in range(self.n_agents):
    #         self.actors[i].train()
    #         self.critics[i].train()
    #     return scores

    def save(self, path, episode):
        for agent in self.agents:
            agent.save(path, episode)

    def load(self, path):
        for agent in self.agents:
            agent.load(path)
