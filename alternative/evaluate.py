import random
from collections import deque
from datetime import datetime

import numpy as np
import torch
from munch import DefaultMunch
from unityagents import UnityEnvironment

from alternative.MultiAgent import MultiAgent
from utility.ReplayMemory import ExperienceReplayMemory

if __name__ == '__main__':

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    current_time = now.strftime('%b%d_%H-%M-%S')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: ", device)
    seed = 2
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    worker_id = 10
    print(f'Worker_id={worker_id}')
    env = UnityEnvironment("../environment/Tennis_Linux/Tennis.x86_64", worker_id=worker_id, seed=seed, no_graphics=False)
    brain = env.brains[env.brain_names[0]]
    env_info = env.reset(train_mode=False)[env.brain_names[0]]
    n_agents = len(env_info.agents)
    print('Number of agents:', n_agents)
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size
    state_multiplier = brain.num_stacked_vector_observations
    action_type = brain.vector_action_space_type
    comment = f"TD3 Unity Tennis"
    # actor1.test(device)
    rand_seed = 0
    config = DefaultMunch()
    config.seed = seed
    config.n_episodes = 10
    config.max_t = 1000
    config.buffer_size = 100000  # replay buffer size
    config.batch_size = 200  # minibatch size
    config.gamma = 0.99  # discount factor
    config.tau = 0.001  # for soft update of targt params
    config.lr_actor = 0.0001  # learning rate of the actor
    config.lr_critic = 0.001  # learning rate of the critic
    config.n_agents = n_agents
    config.state_size = state_size * state_multiplier
    config.action_size = action_size
    config.learn_start = 3000
    config.max_action = 1
    config.memory = ExperienceReplayMemory(config.buffer_size, rand_seed)
    config.update_every = 2  # steps to update
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rand_seed = 0
    scores = []
    scores_std = []
    scores_avg = []
    scores_window = deque(maxlen=100)  # last 100 scores

    agent = MultiAgent(config)
    agent.load("../runs/Sep04_18-36-00_TD3 Unity Tennis/checkpoint_success.pth")
    print('\nEVALUATING:')
    # start the training
    global_steps = 0
    noise_scheduler = config.noise_scheduler
    for i_episode in range(config.n_episodes):
        env_info = env.reset(train_mode=False)[env.brain_names[0]]  # reset the environment
        states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=device)  # get the current state
        score = 0
        # noise_magnitude = noise_scheduler.get(global_steps)
        for i in range(config.max_t):
            actions = agent.act(states, add_noise=False)
            env_info = env.step(actions.cpu().numpy())[env.brain_names[0]]
            next_states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=device)  # get the next state
            rewards = torch.tensor(env_info.rewards, dtype=torch.float, device=device).unsqueeze(dim=1)  # get the reward
            dones = torch.tensor(env_info.local_done, dtype=torch.uint8, device=device).unsqueeze(dim=1)  # see if episode has finished
            global_steps += 1
            score += torch.max(rewards).item()
            states = next_states
            if dones.any():
                break
        scores.append(score)
        scores_window.append(score)
        scores_std.append(np.std(scores_window))
        scores_avg.append(np.mean(scores_window))
        s_msg = '\rEpisode {}\tAverage Score: {:.3f}\tσ: {:.3f}\tStep: {:.3f}'
        print(s_msg.format(i_episode, np.mean(scores_window), np.std(scores_window), global_steps), end="")
    print("Finished.")
