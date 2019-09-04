import torch
from munch import DefaultMunch

from alternative.Agent import Agent


class MultiAgent(object):
    def __init__(self, config: DefaultMunch):
        self.config = config
        self.memory = self.config.memory
        self.n_agents = self.config.n_agents
        self.action_size = self.config.action_size
        self.agents = [Agent(self.config) for i in range(self.n_agents)]

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
