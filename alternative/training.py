#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement ...


@author: ucaiado

Created on 10/07/2018
"""
from alternative.agent import MultiAgent, PARAMS
from collections import deque
from unityagents import UnityEnvironment
import numpy as np
import platform
import time
import pickle
import pdb
import torch
from alternative.make_env import make

'''
Begin help functions and variables
'''
SOLVED = False

'''
End help functions and variables
'''

if __name__ == '__main__':

    seed = 2
    torch.manual_seed(seed)
    np.random.seed(seed)
    worker_id = 1
    print(f'Worker_id={worker_id}')
    env = UnityEnvironment("../environment/Tennis_Linux/Tennis.x86_64", worker_id=worker_id, seed=seed, no_graphics=True)
    num_agents = 2
    brain = env.brains[env.brain_names[0]]
    env_info = env.reset(train_mode=True)[env.brain_names[0]]
    n_agents = len(env_info.agents)
    print('Number of agents:', n_agents)
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size
    state_multiplier = brain.num_stacked_vector_observations
    # from drlnd.ddpg_agent import Agent
    episodes = 10000
    rand_seed = 0

    scores = []
    scores_std = []
    scores_avg = []
    scores_window = deque(maxlen=100)  # last 100 scores

    Agent = MultiAgent
    agent = Agent(state_size * state_multiplier, action_size, num_agents, rand_seed)

    print('\n')
    print(PARAMS)

    print('\nNN ARCHITECURES:')
    for ii in range(len(agent)):
        print('agent %i:' % (ii + 1))
        print(agent[ii].actor_local)
        print(agent[ii].critic_local)
        print('\n')

    print('\nTRAINING:')
    for episode in range(episodes):
        env_info = env.reset(train_mode=True)[env.brain_names[0]]
        states = env_info.vector_observations
        score = np.zeros(len(agent))
        for i in range(1000):
            actions = agent.act(states, add_noise=True)
            env_info = env.step(actions)[env.brain_names[0]]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            # next_states, rewards, dones = env.step(actions)
            agent.step(states, actions, rewards, next_states, dones)
            score += rewards
            states = next_states
            if np.any(dones):
                break
        scores.append(np.max(score))
        scores_window.append(np.max(score))
        scores_avg.append(np.mean(scores_window))
        scores_std.append(np.std(scores_window))
        s_msg = '\rEpisode {}\tAverage Score: {:.3f}\tσ: {:.3f}\tScore: {:.3f}'
        print(s_msg.format(episode, np.mean(scores_window),
                           np.std(scores_window), np.max(score)), end="")
        if episode % 100 == 0:
            print(s_msg.format(episode, np.mean(scores_window),
                               np.std(scores_window), np.max(score)))
        if np.mean(scores_window) >= 0.5:
            SOLVED = True
            s_msg = '\n\nEnvironment solved in {:d} episodes!\tAverage '
            s_msg += 'Score: {:.3f}\tσ: {:.3f}'
            print(s_msg.format(episode, np.mean(scores_window),
                               np.std(scores_window)))
            # todo save the models

    # save data to use later
    if not SOLVED:
        s_msg = '\n\nEnvironment not solved =/'
        print(s_msg.format(episode, np.mean(scores_window),
                           np.std(scores_window)))
