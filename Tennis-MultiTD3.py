import json
import os
from datetime import datetime

import jsonpickle
import numpy as np
import torch
import torch.optim as optim
from munch import Munch, DefaultMunch
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment

import utility.constants as constants
from agents.Unity.MultiAgent_TD3 import MultiAgentTD3
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
    worker_id = 1
    print(f'Worker_id={worker_id}')
    env = UnityEnvironment("./environment/Tennis_Linux/Tennis.x86_64", worker_id=worker_id, seed=seed, no_graphics=True)
    brain = env.brains[env.brain_names[0]]
    env_info = env.reset(train_mode=True)[env.brain_names[0]]
    print('Number of agents:', len(env_info.agents))
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size
    state_multiplier = brain.num_stacked_vector_observations
    action_type = brain.vector_action_space_type
    comment = f"TD3 Unity Tennis"
    actor1 = Policy_actor(state_size * state_multiplier, action_size, hidden_layer_size=200).to(device)
    actor2 = Policy_actor(state_size * state_multiplier, action_size, hidden_layer_size=200).to(device)
    critic1 = Policy_critic(state_size * state_multiplier + action_size, hidden_layer_size=200).to(device)
    critic2 = Policy_critic(state_size * state_multiplier + action_size, hidden_layer_size=200).to(device)
    # actor1.test(device)
    optimizer_actor1 = optim.Adam(actor1.parameters(), lr=1e-4)
    optimizer_critic1 = optim.Adam(critic1.parameters(), lr=1e-4)
    optimizer_actor2 = optim.Adam(actor2.parameters(), lr=1e-4)
    optimizer_critic2 = optim.Adam(critic2.parameters(), lr=1e-4)
    ending_condition = lambda result: result['mean'] >= 300.0
    log_dir = os.path.join('runs', current_time + '_' + comment)
    os.mkdir(log_dir)
    print(f"logging to {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    config = DefaultMunch()
    config.n_episodes = 4000
    config.batch_size = 100
    config.buffer_size = int(1e6)
    config.max_t = 2000  # just > 1000
    config.input_dim = state_size * state_multiplier
    config.output_dim = action_size
    config.gamma = 0.99  # discount
    config.tau = 0.005  # soft merge
    config.device = device
    config.train_every = 50
    config.train_n_times = 2
    config.n_step_td = 1
    config.ending_condition = ending_condition
    config.learn_start = config.max_t  # training starts after this many transitions
    config.evaluate_every = 100
    config.use_noise = True
    config.use_priority = False
    config.noise_scheduler = Scheduler(1.0, 0.1, config.max_t*1000, warmup_steps=config.max_t)
    # config.n_agents = len(env_info.agents)
    config.action_size = action_size
    config.log_dir = log_dir
    config.summary_writer = writer
    agent_config1 = DefaultMunch(actor=actor1, critic=critic1, optimiser_actor=optimizer_actor1, optimiser_critic=optimizer_critic1)
    agent_config2 = DefaultMunch(actor=actor2, critic=critic2, optimiser_actor=optimizer_actor2, optimiser_critic=optimizer_critic2)
    config.agent_configs = [agent_config1, agent_config2]

    config_file = open(os.path.join(log_dir, "config.json"), "w+")
    config_file.write(json.dumps(json.loads(jsonpickle.encode(config, unpicklable=False, max_depth=1)), indent=4, sort_keys=True))
    config_file.close()
    agent = MultiAgentTD3(config)
    # agent.save("/home/edoardo/PycharmProjects/ProximalPolicyOptimisation/runs/Aug13_16-46-32_DDPG Unity Reacher multi/checkpoint_50.pth",1)
    # agent.load("/home/edoardo/PycharmProjects/ProximalPolicyOptimisation/runs/Aug13_16-46-32_DDPG Unity Reacher multi/checkpoint_50.pth")
    agent.train(env, ending_condition)
    print("Finished.")


if __name__ == '__main__':
    main()
