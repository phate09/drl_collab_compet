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
from utility.Scheduler import Scheduler
from utility.td_buffer import TDBuffer
import torch.nn.functional as F


class AgentTD3(GenericAgent):
    """Interacts with and learns from the environment."""

    def __init__(self, config):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super().__init__(config)
        self.input_dim: int = config[constants.input_dim]
        self.output_dim: int = config[constants.output_dim]
        self.max_t: int = config[constants.max_t]
        self.n_episodes: int = config[constants.n_episodes]
        self.gamma: float = config[constants.gamma]
        self.tau: float = config[constants.tau]
        self.device = config[constants.device]
        self.actor: torch.nn.Module = config[constants.model_actor]
        self.critic1: torch.nn.Module = config[constants.model_critic]
        self.target_actor: torch.nn.Module = pickle.loads(pickle.dumps(self.actor))  # clones the actor
        self.target_critic: torch.nn.Module = pickle.loads(pickle.dumps(self.critic1))  # clones the critic
        self.optimiser_actor: torch.optim.optimizer.Optimizer = config[constants.optimiser_actor]
        self.optimiser_critic: torch.optim.optimizer.Optimizer = config[constants.optimiser_critic]
        self.ending_condition = config[constants.ending_condition]
        self.batch_size = config[constants.batch_size]
        self.action_size = config[constants.action_size]
        self.n_agents = config[constants.n_agents]
        self.n_games = 1
        self.beta_start = 0.4
        self.beta_end = 1.0
        self.train_every = config[constants.train_every]
        self.train_n_times = config[constants.train_n_times]
        self.replay_buffer = PrioritizedReplayBuffer(config[constants.buffer_size], alpha=0.6)
        self.log_dir = config[constants.log_dir] if constants.log_dir in config else None
        self.n_step_td = config[constants.n_step_td]
        self.use_noise = config[constants.use_noise] if constants.use_noise in config else False
        self.noise_scheduler: Scheduler = config[constants.noise_scheduler] if constants.noise_scheduler in config else None
        self.learn_start: int = config[constants.learn_start] if constants.learn_start in config else 0
        self.writer: SummaryWriter = config[constants.summary_writer]
        self.evaluate_every = config[constants.evaluate_every] if constants.evaluate_every in config else 100
        self.td_buffer = TDBuffer(n_steps=self.n_step_td, gamma=self.gamma, memory=self.replay_buffer, evaluate_fn=self.calculate_td_errors, device=self.device)
        self.config = config
        self.starting_episode = 0  # used for loading checkpoints from save file
        self.max_action = 1
        self.d = 2

    # self.t_update_target_step = 0
    def required_properties(self):
        return [constants.input_dim,
                constants.output_dim,
                constants.max_t,
                constants.n_episodes,
                constants.gamma,
                constants.tau,
                constants.device,
                constants.model_actor,
                constants.model_critic,
                constants.optimiser_actor,
                constants.optimiser_critic,
                constants.ending_condition,
                constants.summary_writer]

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

            target_mu_sprime = (self.target_actor(next_states) + noise).clamp(-self.max_action, self.max_action)
            mu_s = self.actor(states)
            target_sprime_target_mu_sprime = torch.cat((next_states, target_mu_sprime), dim=1)
            s_a = torch.cat((states, actions), dim=1)
            s_mu_s = torch.cat((states, mu_s), dim=1)
            target_Q = torch.min(self.target_critic(target_sprime_target_mu_sprime))  # takes the minimum of the two critics
            y = rewards + (self.gamma ** self.n_step_td * target_Q * dones)  # sets 0 to the entries which are done
            Qs_a1, Qs_a2 = self.critic1(s_a)
            Qs_mu_s1, Qs_mu_s2 = self.critic1(s_mu_s)

            self.critic1.train()
            self.actor.train()
            td_error = min(y.detach() - Qs_a1, y.detach() - Qs_a2) + 1e-5
            self.update_priorities(indexes, abs(td_error))
            loss_critic = F.mse_loss(y.detach(), Qs_a1) + F.mse_loss(y.detach(), Qs_a2) * is_values.detach()
            self.optimiser_critic.zero_grad()
            loss_critic.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1)
            self.optimiser_critic.step()

            if i % self.d == 0:
                loss_actor = -Qs_mu_s1  # gradient ascent , use only the first of the critic
                self.optimiser_actor.zero_grad()
                loss_actor.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                self.optimiser_actor.step()

                self.writer.add_scalar('loss/actor', loss_critic.mean(), i_episode)
                self.writer.add_scalar('loss/critic', loss_critic.mean(), i_episode)
                self.soft_update(self.critic1, self.target_critic, self.tau)
                self.soft_update(self.actor, self.target_actor, self.tau)

    # calls = 0
    def update_priorities(self, indexes, td_error):
        self.replay_buffer.update_priorities(indexes, abs(td_error).detach().cpu().numpy())
        # self.calls+=1
        # if self.calls>100:
        #     exit()

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
        i_steps = 0
        for i_episode in range(self.starting_episode, self.n_episodes):
            if i_episode != 0 and i_episode % self.evaluate_every == 0:
                print("")
                eval_scores: list = self.evaluate(env, train_mode=True)
                print("")
                self.writer.add_scalar('eval/score_average', np.mean(eval_scores), i_episode)
            env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
            n_agents = len(env_info.agents)
            # Reset the memory of the agent
            state_list = []
            next_state_list = []
            action_list = []
            reward_list = []
            done_list = []
            states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=self.device)  # get the current state
            score = 0
            # self.actor.eval()
            # self.critic.eval()
            noise_magnitude = 0.2 if self.noise_scheduler is None else self.noise_scheduler.get(i_episode)
            for t in range(self.max_t):
                with torch.no_grad():
                    if i_steps == self.learn_start:
                        print("Starting to learn")
                    if i_steps < self.learn_start:
                        actions = (torch.rand(self.n_agents, self.action_size) * 2).to(self.device) - self.max_action
                    else:
                        actions: torch.Tensor = self.actor(states)
                        noise = torch.normal(torch.zeros_like(actions), torch.ones_like(actions) * noise_magnitude) if self.use_noise else 0  # adds exploratory noise
                        actions = torch.clamp(actions + noise, -self.max_action, self.max_action)  # clips the action to the allowed boundaries
                    env_info = env.step(actions.cpu().detach().numpy())[brain_name]  # send the action to the environment
                    next_states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=self.device)  # get the next state
                    rewards = torch.tensor(env_info.rewards, dtype=torch.float, device=self.device).unsqueeze(dim=1)  # get the reward
                    dones = torch.tensor(env_info.local_done, dtype=torch.uint8, device=self.device).unsqueeze(dim=1)  # see if episode has finished
                    state_list.append(states)
                    action_list.append(actions)
                    reward_list.append(rewards)
                    done_list.append(dones)
                    next_state_list.append(next_states)
                    self.td_buffer.add((states, actions, rewards, dones, next_states))
                # train the agent
                if len(self.replay_buffer) > self.batch_size and i_steps > self.learn_start and i_steps != 0 and i_steps % self.train_every == 0:
                    self.learn(i_episode=i_episode)
                states = next_states
                score += rewards.mean().item()
                i_steps += n_agents
                if dones.any():
                    self.td_buffer.flush()  # flushes the remaining transitions in the buffer to memory
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            self.writer.add_scalar('data/score', score, i_episode)
            self.writer.add_scalar('data/score_average', np.mean(scores_window), i_episode)
            self.writer.flush()
            print(f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} ', end="")
            if (i_episode + 1) % 100 == 0:
                print(f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} ')
            if (i_episode + 1) % 10 == 0:
                self.save(os.path.join(self.log_dir, f"checkpoint_{i_episode + 1}.pth"), i_episode)
            result = {"mean": np.mean(scores_window)}
            if ending_condition(result):
                print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
                self.save(os.path.join(self.log_dir, f"checkpoint_success.pth"), i_episode)
                break
        return scores

    def evaluate(self, env, n_episodes=1, train_mode=False):
        brain_name = env.brain_names[0]
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        i_steps = 0
        self.actor.eval()
        self.critic1.eval()
        for i_episode in range(n_episodes):
            env_info = env.reset(train_mode=train_mode)[brain_name]  # reset the environment
            n_agents = len(env_info.agents)
            states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=self.device)  # get the current state
            score = 0
            self.actor.eval()
            self.critic1.eval()
            for t in range(self.max_t):
                with torch.no_grad():
                    actions: torch.Tensor = self.actor(states)
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
        self.actor.train()
        self.critic1.train()
        return scores

    def calculate_td_errors(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        concat_states = torch.cat([states, actions], dim=1)
        suggested_next_action = self.target_actor(next_states)
        concat_next_states = torch.cat([next_states, suggested_next_action], dim=1)
        dones = (1 - dones).float()
        target_Q1, target_Q2 = self.target_critic(concat_next_states)
        Q1, Q2 = self.critic1(concat_states)
        y = rewards + np.power(self.gamma, self.n_step_td) * torch.min(target_Q1, target_Q2) * dones
        td_errors = torch.min(y - Q1, y - Q2)
        return td_errors  # calculate the td-errors, maybe use GAE

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
            "critic": self.critic1.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "optimiser_actor": self.optimiser_actor.state_dict(),
            "optimiser_critic": self.optimiser_critic.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.target_actor.load_state_dict(checkpoint["target_actor"])
        self.critic1.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.optimiser_actor.load_state_dict(checkpoint["optimiser_actor"])
        self.optimiser_critic.load_state_dict(checkpoint["optimiser_critic"])
        self.replay_buffer = checkpoint["optimiser_critic"]
        self.starting_episode = checkpoint["episode"]
        print(f'Loading complete')
