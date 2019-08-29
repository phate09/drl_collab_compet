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
        self.n_agents = len(config.agent_configs)
        self.beta_start = 0.4
        self.beta_end = 1.0
        self.train_every = config.train_every
        self.train_n_times = config.train_n_times
        self.log_dir = config.log_dir if constants.log_dir in config else None
        self.n_step_td = config.n_step_td
        self.use_noise = config.use_noise if constants.use_noise in config else False
        self.noise_scheduler: Scheduler = config.noise_scheduler if constants.noise_scheduler in config else None
        self.learn_start: int = config.learn_start if config.learn_start is not None else 0
        self.writer: SummaryWriter = config.summary_writer
        self.evaluate_every = config.evaluate_every if config.evaluate_every is not None else 100
        self.config = config
        self.starting_episode = 0  # used for loading checkpoints from save file
        self.max_action = 1
        self.d = 2
        self.use_priority = config.use_priority
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.optimiser_actors = []
        self.optimiser_critics = []
        self.replay_buffers = []
        self.td_buffers = []
        for i, agent_config in enumerate(config.agent_configs):
            self.actors.append(agent_config.actor)
            target_actor = pickle.loads(pickle.dumps(agent_config.actor))
            target_actor.eval()  # sets the target to eval
            self.target_actors.append(target_actor)
            self.critics.append(agent_config.critic)
            target_critic = pickle.loads(pickle.dumps(agent_config.critic))
            target_critic.eval()  # sets the target to eval
            self.target_critics.append(target_critic)
            self.optimiser_actors.append(agent_config.optimiser_actor)
            self.optimiser_critics.append(agent_config.optimiser_critic)
            if config.use_priority:
                buffer = PrioritizedReplayBuffer(config.buffer_size, alpha=0.6)
            else:
                buffer = ExperienceReplayMemory(config.buffer_size)
            self.replay_buffers.append(buffer)
            self.td_buffers.append(TDBuffer(n_steps=self.n_step_td, gamma=self.gamma, memory=buffer, critic=agent_config.critic, target_critic=target_critic, target_actor=target_actor, device=self.device))

    # self.t_update_target_step = 0
    def required_properties(self):
        return [constants.input_dim,
                constants.output_dim,
                constants.max_t,
                constants.n_episodes,
                constants.gamma,
                constants.tau,
                constants.device,
                constants.summary_writer]

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

    def train(self, env, ending_condition):
        """
        :param env:
        :param writer:
        :param ending_condition: a method that given a score window returns true or false
        :return:
        """
        brain_name = env.brain_names[0]
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        scores_max_window = deque(maxlen=100)  # last 100 scores
        scores_min_window = deque(maxlen=100)  # last 100 scores
        i_steps = 0
        for i_episode in range(self.starting_episode, self.n_episodes):
            # todo reset the noise
            if i_episode != 0 and i_episode % self.evaluate_every == 0:
                print("")
                eval_scores: list = self.evaluate(env, n_episodes=10, train_mode=True)
                print("")
                self.writer.add_scalar('eval/score_average', np.mean(eval_scores), i_episode)
            env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
            # Reset the memory of the agent
            states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=self.device)  # get the current state
            score = 0
            score_max = 0
            score_min = 0
            noise_magnitude = 0.2 if self.noise_scheduler is None else self.noise_scheduler.get(i_steps)
            for t in range(self.max_t):
                with torch.no_grad():
                    if i_steps == self.learn_start:
                        print("Starting to learn")
                    if i_steps < self.learn_start:
                        actions = np.random.randn(self.n_agents, self.action_size)  # select an action (for each agent)
                        actions = torch.tensor(np.clip(actions, -self.max_action, self.max_action), dtype=torch.float, device=self.device)  # all actions between -1 and 1
                    else:
                        actions = []
                        for i in range(self.n_agents):
                            self.actors[i].eval()
                            actions.append(self.actors[i](states[i].unsqueeze(dim=0)).squeeze())
                            self.actors[i].train()
                        actions = torch.stack(actions, dim=0)
                        # actions: torch.Tensor = actor(states)
                        noise = torch.normal(torch.zeros_like(actions), torch.ones_like(actions) * noise_magnitude) if self.use_noise else 0  # adds exploratory noise
                        actions = torch.clamp(actions + noise, -self.max_action, self.max_action)  # clips the action to the allowed boundaries
                    # todo actions should be a list of n_agent elements containing 2 values each
                    env_info = env.step(actions.cpu().detach().numpy())[brain_name]  # send the action to the environment
                    next_states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=self.device)  # get the next state
                    rewards = torch.tensor(env_info.rewards, dtype=torch.float, device=self.device).unsqueeze(dim=1)  # get the reward
                    dones = torch.tensor(env_info.local_done, dtype=torch.uint8, device=self.device).unsqueeze(dim=1)  # see if episode has finished
                    for i in range(self.n_agents):
                        self.td_buffers[i].add((states[i], actions[i], rewards[i], dones[i], next_states[i]))
                # train the agent
                if len(self.replay_buffers[t % self.n_agents]) > self.batch_size and i_steps > self.learn_start and i_steps != 0 and i_steps % self.train_every == 0:
                    self.learn(i_episode=i_episode)
                states = next_states
                score_max += rewards.max().item()
                score_min += rewards.min().item()
                score += rewards.mean().item()
                i_steps += self.n_agents
                if dones.any():
                    for buffer in self.td_buffers:
                        buffer.flush()  # flushes the remaining transitions in the buffer to memory
                    # break
                # if np.any(env_info.global_done):
                #     break
            scores_window.append(score)  # save most recent score
            scores_max_window.append(score_max)  # save most recent score
            scores_min_window.append(score_min)  # save most recent score
            scores.append(score)  # save most recent score
            self.writer.add_scalar('data/score', score, i_episode)
            self.writer.add_scalar('data/score_average', np.mean(scores_window), i_episode)
            self.writer.add_scalar('data/score_max_average', np.mean(scores_max_window), i_episode)
            self.writer.add_scalar('data/score_min_average', np.mean(scores_min_window), i_episode)
            self.writer.flush()
            print(f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} ', end="")
            if (i_episode + 1) % 100 == 0:
                print(f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} ')
            # if (i_episode + 1) % 10 == 0:
            #     self.save(os.path.join(self.log_dir, f"checkpoint_{i_episode + 1}.pth"), i_episode)
            result = {"mean": np.mean(scores_window)}
            if ending_condition(result):
                print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
                # self.save(os.path.join(self.log_dir, f"checkpoint_success.pth"), i_episode)
                break
        return scores

    def learn(self, i_episode: int):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        for i in range(self.train_n_times):
            beta = (self.beta_end - self.beta_start) * i_episode / self.n_episodes + self.beta_start
            for agent_i in range(self.n_agents):
                experiences, is_values, indexes = self.replay_buffers[agent_i].sample(self.batch_size, beta=beta)
                states, actions, rewards, next_states, dones = zip(*experiences)
                states = torch.stack(states)
                actions = torch.stack(actions)
                rewards = torch.stack(rewards)
                next_states = torch.stack(next_states)
                is_values = torch.from_numpy(is_values).float().to(self.device)
                dones = 1 - torch.stack(dones).float()
                policy_noise = 0.2  #
                noise_clip = 0.5
                # Select action according to policy and add clipped noise
                noise = torch.normal(torch.zeros_like(actions), torch.ones_like(actions) * policy_noise)
                # torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
                noise = noise.clamp(-noise_clip, noise_clip)

                target_mu_sprime = (self.target_actors[agent_i](next_states) + noise).clamp(-self.max_action, self.max_action)
                mu_s = self.actors[agent_i](states)
                target_sprime_target_mu_sprime = torch.cat((next_states, target_mu_sprime), dim=1)
                s_a = torch.cat((states, actions), dim=1)
                s_mu_s = torch.cat((states, mu_s), dim=1)
                target_Q1, target_Q2 = self.target_critics[agent_i](target_sprime_target_mu_sprime)
                target_Q = torch.min(target_Q1, target_Q2)  # takes the minimum of the two critics
                y = rewards + (self.gamma ** self.n_step_td * target_Q * dones)  # sets 0 to the entries which are done
                Qs_a1, Qs_a2 = self.critics[agent_i](s_a)
                Qs_mu_s1, Qs_mu_s2 = self.critics[agent_i](s_mu_s)

                self.critics[agent_i].train()
                self.actors[agent_i].train()
                td_error = torch.min(y.detach() - Qs_a1, y.detach() - Qs_a2) + 1e-5
                if self.use_priority:
                    self.replay_buffers[agent_i].update_priorities(indexes, abs(td_error.detach().cpu().numpy()))
                loss_critic = F.mse_loss(y.detach(), Qs_a1) + F.mse_loss(y.detach(), Qs_a2) * is_values.detach()
                self.optimiser_critics[agent_i].zero_grad()
                loss_critic.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.critics[agent_i].parameters(), 1)
                self.optimiser_critics[agent_i].step()

                if i % self.d == 0:
                    loss_actor = -Qs_mu_s1  # gradient ascent , use only the first of the critic
                    self.optimiser_actors[agent_i].zero_grad()
                    loss_actor.mean().backward()
                    torch.nn.utils.clip_grad_norm_(self.actors[agent_i].parameters(), 1)
                    self.optimiser_actors[agent_i].step()

                    self.writer.add_scalar('loss/actor', loss_critic.mean(), i_episode)
                    self.writer.add_scalar('loss/critic', loss_critic.mean(), i_episode)
                    self.soft_update(self.critics[agent_i], self.target_critics[agent_i], self.tau)
                    self.soft_update(self.actors[agent_i], self.target_actors[agent_i], self.tau)

    def evaluate(self, env, n_episodes=1, train_mode=False):
        brain_name = env.brain_names[0]
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        i_steps = 0
        for i in range(self.n_agents):
            self.actors[i].eval()
            self.critics[i].eval()
        for i_episode in range(n_episodes):
            env_info = env.reset(train_mode=train_mode)[brain_name]  # reset the environment
            n_agents = len(env_info.agents)
            states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=self.device)  # get the current state
            score = 0
            for t in range(self.max_t):
                with torch.no_grad():
                    actions: torch.Tensor = self.actors[t % self.n_agents](states)
                    # noise = torch.normal(torch.zeros_like(actions), torch.ones_like(actions) * 0.2)
                    noise = 0  # no noise during evaluation
                    actions = torch.clamp(actions + noise, -1, 1)  # clips the action to the allowed boundaries
                    env_info = env.step(actions.cpu().detach().numpy())[brain_name]  # send the action to the environment
                    next_states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=self.device)  # get the next state
                    rewards = torch.tensor(env_info.rewards, dtype=torch.float, device=self.device).unsqueeze(dim=1)  # get the reward
                    dones = torch.tensor(env_info.local_done, dtype=torch.uint8, device=self.device).unsqueeze(dim=1)  # see if episode has finished
                states = next_states
                score += rewards.mean().item()
                i_steps += n_agents
                if dones.any():
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            print(f'\rEval Episode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} ', end="")
            if (i_episode + 1) % 100 == 0:
                print(f'\rEval Episode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} ')
        for i in range(self.n_agents):
            self.actors[i].train()
            self.critics[i].train()
        return scores

    def calculate_discounted_rewards(self, reward_list: list) -> np.ndarray:
        rewards = reward_list.copy()
        rewards.reverse()
        previous_rewards = 0
        for i in range(len(rewards)):
            rewards[i] = rewards[i] + self.gamma * previous_rewards
            previous_rewards = rewards[i]
        rewards.reverse()
        rewards_array = np.asanyarray(rewards)
        return rewards_array

    def save(self, path, episode):
        torch.save({
            "episode": episode,
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
        self.starting_episode = checkpoint["episode"]
        print(f'Loading complete')
