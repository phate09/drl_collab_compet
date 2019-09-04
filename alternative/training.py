import json
import os
import random
from collections import deque
from datetime import datetime

import jsonpickle
import numpy as np
import torch
from munch import DefaultMunch
from tensorboardX import SummaryWriter
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
    worker_id = 2
    print(f'Worker_id={worker_id}')
    env = UnityEnvironment("../environment/Tennis_Linux/Tennis.x86_64", worker_id=worker_id, seed=seed, no_graphics=True)
    brain = env.brains[env.brain_names[0]]
    env_info = env.reset(train_mode=True)[env.brain_names[0]]
    n_agents = len(env_info.agents)
    print('Number of agents:', n_agents)
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size
    state_multiplier = brain.num_stacked_vector_observations
    action_type = brain.vector_action_space_type
    comment = f"TD3 Unity Tennis"
    log_dir = os.path.join('../runs', current_time + '_' + comment)
    os.mkdir(log_dir)
    print(f"logging to {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)
    config = DefaultMunch()
    config.seed = seed
    config.n_episodes = 40000
    config.max_t = 1000
    config.buffer_size = 100000
    config.batch_size = 200
    config.gamma = 0.99
    config.tau = 0.001
    config.lr_actor = 0.0001
    config.lr_critic = 0.0001
    config.n_agents = n_agents
    config.state_size = state_size * state_multiplier
    config.action_size = action_size
    config.learn_start = 10000
    config.max_action = 1  # maximum value allowed for each action
    config.memory = ExperienceReplayMemory(config.buffer_size, seed)
    config.update_every = 2
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_file = open(os.path.join(log_dir, "config.json"), "w+")
    config_file.write(json.dumps(json.loads(jsonpickle.encode(config, unpicklable=False, max_depth=1)), indent=4, sort_keys=True))
    config_file.close()

    scores = []
    scores_std = []
    scores_avg = []
    scores_window = deque(maxlen=100)  # last 100 scores

    agent = MultiAgent(config)

    global_steps = 0
    noise_scheduler = config.noise_scheduler
    for i_episode in range(config.n_episodes):
        env_info = env.reset(train_mode=True)[env.brain_names[0]]  # reset the environment
        states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=device)  # get the current state
        score = 0
        # noise_magnitude = noise_scheduler.get(global_steps)
        for i in range(config.max_t):
            if global_steps < config.learn_start:
                actions = (torch.rand(n_agents, action_size) * 2).to(device) - config.max_action
            else:
                actions = agent.act(states, add_noise=True)
            env_info = env.step(actions.cpu().numpy())[env.brain_names[0]]
            next_states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=device)  # get the next state
            rewards = torch.tensor(env_info.rewards, dtype=torch.float, device=device).unsqueeze(dim=1)  # get the reward
            dones = torch.tensor(env_info.local_done, dtype=torch.uint8, device=device).unsqueeze(dim=1)  # see if episode has finished
            agent.step(states, actions, rewards, next_states, dones)
            global_steps += 1
            score += torch.max(rewards).item()
            states = next_states
            if dones.any():
                break
        scores.append(score)
        scores_window.append(score)
        scores_std.append(np.std(scores_window))
        scores_avg.append(np.mean(scores_window))
        writer.add_scalar('data/score', score, i_episode)
        writer.add_scalar('data/score_average', np.mean(scores_window), i_episode)
        writer.add_scalar('data/score_max', np.max(scores_window), i_episode)
        writer.add_scalar('data/score_min', np.min(scores_window), i_episode)
        writer.add_scalar('data/score_std', np.std(scores_window), i_episode)
        s_msg = '\rEpisode {}\tAverage Score: {:.3f}\tσ: {:.3f}\tStep: {:}'
        print(s_msg.format(i_episode, np.mean(scores_window), np.std(scores_window), global_steps), end="")
        if i_episode % 100 == 0 and i_episode != 0:
            print(s_msg.format(i_episode, np.mean(scores_window), np.std(scores_window), global_steps))
            agent.save(os.path.join(log_dir, f"checkpoint_{i_episode}.pth"), i_episode)
        if np.mean(scores_window) >= 0.9:
            s_msg = '\n\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}\tσ: {:.3f}'
            print(s_msg.format(i_episode, np.mean(scores_window), np.std(scores_window)))
            agent.save(os.path.join(log_dir, f"checkpoint_success.pth"), i_episode)
            break
    print("Finished.")
