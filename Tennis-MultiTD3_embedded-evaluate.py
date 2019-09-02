import json
import os
from collections import deque
from datetime import datetime

import jsonpickle
import numpy as np
import torch
import torch.optim as optim
from munch import Munch, DefaultMunch
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment

import utility.constants as constants
from agents.Unity.MultiAgent_TD3_embedded import MultiAgentTD3
from networks.actor_critic.Policy_actor import Policy_actor
from networks.actor_critic.Policy_critic_twin import Policy_critic
from utility.Scheduler import Scheduler


def main():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    current_time = now.strftime('%b%d_%H-%M-%S')
    device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: ", device)
    seed = 2
    torch.manual_seed(seed)
    np.random.seed(seed)
    worker_id = 10
    print(f'Worker_id={worker_id}')
    env = UnityEnvironment("./environment/Tennis_Linux/Tennis.x86_64", worker_id=worker_id, seed=seed, no_graphics=False)
    brain = env.brains[env.brain_names[0]]
    env_info = env.reset(train_mode=False)[env.brain_names[0]]
    n_agents = len(env_info.agents)
    print('Number of agents:', n_agents)
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size
    state_multiplier = brain.num_stacked_vector_observations
    action_type = brain.vector_action_space_type
    comment = f"TD3 Unity Tennis"
    actor_fn = lambda: Policy_actor(state_size * state_multiplier, action_size, hidden_layer_size=200).to(device)
    critic_fn = lambda: Policy_critic((state_size * state_multiplier + action_size) * n_agents, hidden_layer_size=200).to(device)
    # actor1.test(device)
    optimizer_actor_fn = lambda actor: optim.Adam(actor.parameters(), lr=1e-4)
    optimizer_critic_fn = lambda critic: optim.Adam(critic.parameters(), lr=1e-4)
    ending_condition = lambda result: result['mean'] >= 300.0
    # log_dir = os.path.join('runs', current_time + '_' + comment)
    # os.mkdir(log_dir)
    # print(f"logging to {log_dir}")
    # writer = SummaryWriter(log_dir=log_dir)
    optimiser_actor_fn = lambda actor: optim.Adam(actor.parameters(), lr=1e-4)
    optimiser_critic_fn = lambda critic: optim.Adam(critic.parameters(), lr=1e-4)
    config = DefaultMunch()
    config.seed = seed
    config.n_episodes = 40000
    config.batch_size = 256
    config.buffer_size = int(1e6)
    config.max_t = 2000  # just > 1000
    config.input_dim = state_size * state_multiplier
    config.output_dim = action_size
    config.gamma = 0.99  # discount
    config.tau = 0.005  # soft merge
    config.device = device
    config.train_every = 4
    config.train_n_times = 2
    config.n_step_td = 1
    config.ending_condition = ending_condition
    config.learn_start = 0  # training starts after this many transitions
    config.evaluate_every = 100
    config.use_noise = False
    config.use_priority = False
    config.noise_scheduler = Scheduler(1.0, 0.1, config.max_t * 10, warmup_steps=0)
    config.n_agents = n_agents
    config.action_size = action_size
    # config.log_dir = log_dir
    config.use_shared_memory = False
    # config.summary_writer_fn = lambda: writer
    config.actor_fn = actor_fn
    config.critic_fn = critic_fn
    config.optimiser_actor_fn = optimiser_actor_fn
    config.optimiser_critic_fn = optimiser_critic_fn

    # config_file = open(os.path.join(log_dir, "config.json"), "w+")
    # config_file.write(json.dumps(json.loads(jsonpickle.encode(config, unpicklable=False, max_depth=1)), indent=4, sort_keys=True))
    # config_file.close()
    agent = MultiAgentTD3(config)
    agent.load("runs/Sep01_17-48-13_TD3 Embedded Unity Tennis/checkpoint_1000.pth")
    scores = []  # list containing scores from each episode
    scores_std = []
    scores_avg = []
    scores_window = deque(maxlen=100)  # last 100 scores

    # start the training
    global_steps = 0
    noise_scheduler = config.noise_scheduler
    for i_episode in range(config.n_episodes):
        env_info = env.reset(train_mode=False)[env.brain_names[0]]  # reset the environment
        states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=device)  # get the current state
        score = 0
        noise_magnitude=noise_scheduler.get(global_steps)
        for i in range(config.max_t):
            with torch.no_grad():
                actions = agent.act(states, 0)  # no noise magnitude
                env_info = env.step(actions.cpu().numpy())[env.brain_names[0]]
                next_states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=device)  # get the next state
                rewards = torch.tensor(env_info.rewards, dtype=torch.float, device=device).unsqueeze(dim=1)  # get the reward
                dones = torch.tensor(env_info.local_done, dtype=torch.uint8, device=device).unsqueeze(dim=1)  # see if episode has finished
            agent.step(states, actions, rewards, next_states, dones)
            global_steps += 1
            score += rewards.max().item()
            states = next_states
            if dones.any():
                break
        scores.append(score)
        scores_window.append(score)
        scores_std.append(np.std(scores_window))
        scores_avg.append(np.mean(scores_window))
        s_msg = '\rEpisode {}\tAverage Score: {:.3f}\tnoise: {:.3f}\tScore: {:.3f}'
        print(s_msg.format(i_episode, np.mean(scores_window), noise_magnitude, np.max(score)), end="")
    print("Finished.")


if __name__ == '__main__':
    main()
