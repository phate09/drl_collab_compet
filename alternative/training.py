import yaml

from alternative.agent import MultiAgent
from unityagents import UnityEnvironment
import os
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from munch import Munch, DefaultMunch
from tensorboardX import SummaryWriter
from alternative import param_table

'''
Begin help functions and variables
'''
SOLVED = False


def set_global_parms(d_table)->DefaultMunch:
    '''
    convert statsmodel tabel to the agent parameters

    :param d_table: Dictionary. Parameters of the agent
    '''
    l_table = [(a, [b]) for a, b in d_table.items()]
    d_params = dict([[x[0], x[1][0]] for x in l_table])
    table = param_table.generate_table(l_table[:int(len(l_table) / 2)],
                                       l_table[int(len(l_table) / 2):],
                                       'DDPG PARAMETERS')
    config = DefaultMunch()
    config.BUFFER_SIZE = d_params['BUFFER_SIZE']  # replay buffer size
    config.BATCH_SIZE = d_params['BATCH_SIZE']  # minibatch size
    config.GAMMA = d_params['GAMMA']  # discount factor
    config.TAU = d_params['TAU']  # for soft update of targt params
    config.LR_ACTOR = d_params['LR_ACTOR']  # learning rate of the actor
    config.LR_CRITIC = d_params['LR_CRITIC']  # learning rate of the critic
    config.WEIGHT_DECAY = d_params['WEIGHT_DECAY']  # L2 weight decay
    config.UPDATE_EVERY = d_params['UPDATE_EVERY']  # steps to update
    config.DEVC = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.PARAMS = table
    return config


# PATH = os.path.dirname(os.path.realpath(__file__))
PATH = "/home/edoardo/Development/drl_collab_compet/alternative/config.yaml"  # PATH.replace('ddpg', 'config.yaml')

'''
End help functions and variables
'''

if __name__ == '__main__':

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    current_time = now.strftime('%b%d_%H-%M-%S')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: ", device)
    seed = 2
    torch.manual_seed(seed)
    np.random.seed(seed)
    worker_id = 1
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
    # actor1.test(device)
    optimizer_actor_fn = lambda actor: optim.Adam(actor.parameters(), lr=1e-4)
    optimizer_critic_fn = lambda critic: optim.Adam(critic.parameters(), lr=1e-4)
    ending_condition = lambda result: result['mean'] >= 300.0
    log_dir = os.path.join('../runs', current_time + '_' + comment)
    os.mkdir(log_dir)
    print(f"logging to {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    config2 = set_global_parms(yaml.load(open(PATH, 'r'))['DDPG'])
    config = DefaultMunch()
    config.seed = seed
    config.n_episodes = 40000
    config.max_t = 1000
    rand_seed = 0
    config.update(config2)
    scores = []
    scores_std = []
    scores_avg = []
    scores_window = deque(maxlen=100)  # last 100 scores

    Agent = MultiAgent
    agent = Agent(config,state_size * state_multiplier, action_size, n_agents, rand_seed)

    print('\nTRAINING:')
    # start the training
    global_steps = 0
    noise_scheduler = config.noise_scheduler
    for i_episode in range(config.n_episodes):
        env_info = env.reset(train_mode=True)[env.brain_names[0]]  # reset the environment
        states = env_info.vector_observations  # torch.tensor(env_info.vector_observations, dtype=torch.float, device=device)  # get the current state
        score = 0
        # noise_magnitude = noise_scheduler.get(global_steps)
        for i in range(config.max_t):
            actions = agent.act(states, add_noise=True)
            env_info = env.step(actions)[env.brain_names[0]]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            # next_states, rewards, dones = env.step(actions)
            agent.step(states, actions, rewards, next_states, dones)
            global_steps += 1
            score += np.max(rewards)
            states = next_states
            if np.any(dones):
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
        s_msg = '\rEpisode {}\tAverage Score: {:.3f}\tσ: {:.3f}\tScore: {:.3f}'
        print(s_msg.format(i_episode, np.mean(scores_window), np.std(scores_window), np.max(score)), end="")
        if i_episode % 100 == 0:
            print(s_msg.format(i_episode, np.mean(scores_window), np.std(scores_window), np.max(score)))
            # agent.save(os.path.join(log_dir, f"checkpoint_{i_episode}.pth"), i_episode)
        if np.mean(scores_window) >= 0.5:
            SOLVED = True
            s_msg = '\n\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}\tσ: {:.3f}'
            print(s_msg.format(i_episode, np.mean(scores_window), np.std(scores_window)))
            # agent.save(os.path.join(log_dir, f"checkpoint_success.pth"), i_episode)
            break
    print("Finished.")
