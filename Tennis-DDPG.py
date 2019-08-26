import json
import os
from datetime import datetime

import jsonpickle
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment

import utility.constants as constants
from agents.Unity.Agent_DDPG import AgentDDPG
from networks.actor_critic.Policy_actor import Policy_actor
from networks.actor_critic.Policy_critic import Policy_critic
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
    worker_id = 0
    print(f'Worker_id={worker_id}')
    env = UnityEnvironment("./environment/Tennis_Linux/Tennis.x86_64", worker_id=worker_id, seed=seed, no_graphics=True)
    brain = env.brains[env.brain_names[0]]
    env_info = env.reset(train_mode=True)[env.brain_names[0]]
    print('Number of agents:', len(env_info.agents))
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size
    state_multiplier = brain.num_stacked_vector_observations
    action_type = brain.vector_action_space_type
    comment = f"DDPG Unity Tennis"
    actor = Policy_actor(state_size * state_multiplier, action_size, hidden_layer_size=512).to(device)
    critic = Policy_critic(state_size * state_multiplier + action_size, hidden_layer_size=512).to(device)
    # actor.test(device)
    optimizer_actor = optim.Adam(actor.parameters(), lr=1e-4)
    optimizer_critic = optim.Adam(critic.parameters(), lr=1e-4)
    ending_condition = lambda result: result['mean'] >= 300.0
    log_dir = os.path.join('runs', current_time + '_' + comment)
    os.mkdir(log_dir)
    print(f"logging to {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)
    config = {
        constants.optimiser_actor: optimizer_actor,
        constants.optimiser_critic: optimizer_critic,
        constants.model_actor: actor,
        constants.model_critic: critic,
        constants.n_episodes: 20000,
        constants.batch_size: 512,
        constants.buffer_size: int(1e6),
        constants.max_t: 2000,  # just > 1000
        constants.input_dim: state_size * state_multiplier,
        constants.output_dim: action_size,
        constants.gamma: 0.99,  # discount
        constants.tau: 0.001,  # soft merge
        constants.device: device,
        constants.train_every: 4,
        constants.train_n_times: 1,
        constants.n_step_td: 1,
        constants.ending_condition: ending_condition,
        constants.learn_start: 1600,  # training starts after this many transitions
        constants.log_dir: log_dir,
        constants.use_noise: True,
        constants.noise_scheduler: Scheduler(1.0, 0.1, 700, warmup_steps=1600),
        constants.summary_writer: writer
    }
    config_file = open(os.path.join(log_dir, "config.json"), "w+")
    config_file.write(json.dumps(json.loads(jsonpickle.encode(config, unpicklable=False, max_depth=1)), indent=4, sort_keys=True))
    config_file.close()
    agent = AgentDDPG(config)
    # agent.save("/home/edoardo/PycharmProjects/ProximalPolicyOptimisation/runs/Aug13_16-46-32_DDPG Unity Reacher multi/checkpoint_50.pth",1)
    # agent.load("/home/edoardo/PycharmProjects/ProximalPolicyOptimisation/runs/Aug13_16-46-32_DDPG Unity Reacher multi/checkpoint_50.pth")
    agent.train(env, ending_condition)
    print("Finished.")


if __name__ == '__main__':
    main()
