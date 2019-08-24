import numpy as np
import torch
import torch.optim as optim
from unityagents import UnityEnvironment

import utility.constants as constants
from agents.Unity.Agent_DDPG import AgentDDPG
from networks.actor_critic.Policy_actor import Policy_actor
from networks.actor_critic.Policy_critic import Policy_critic


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: ", device)
    seed = 2
    torch.manual_seed(seed)
    np.random.seed(seed)
    worker_id = 0
    print(f'Worker_id={worker_id}')
    env = UnityEnvironment("./environment/Reacher_Linux/Reacher.x86_64", worker_id=worker_id, seed=seed, no_graphics=False)
    brain = env.brains[env.brain_names[0]]
    env_info = env.reset(train_mode=False)[env.brain_names[0]]
    print('Number of agents:', len(env_info.agents))
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size
    action_type = brain.vector_action_space_type
    actor = Policy_actor(state_size, action_size).to(device)
    critic = Policy_critic(state_size + action_size).to(device)
    # actor.test(device)
    optimizer_actor = optim.Adam(actor.parameters(), lr=1e-4)
    optimizer_critic = optim.Adam(critic.parameters(), lr=1e-4)
    ending_condition = lambda result: result['mean'] >= 300.0
    config = {
        constants.optimiser_actor: optimizer_actor,
        constants.optimiser_critic: optimizer_critic,
        constants.model_actor: actor,
        constants.model_critic: critic,
        constants.n_episodes: 2000,
        constants.batch_size: 64,
        constants.buffer_size: int(1e6),
        constants.max_t: 2000,  # just > 1000
        constants.input_dim: state_size,
        constants.output_dim: action_size,
        constants.gamma: 0.99,  # discount
        constants.tau: 0.001,  # soft merge
        constants.device: device,
        constants.train_every: 20 * 6,
        constants.train_n_times: 4,
        constants.n_step_td: 1,
        constants.ending_condition: ending_condition,
    }
    agent = AgentDDPG(config)
    agent.load("./runs/Aug13_17-36-37_DDPG Unity Reacher multi/checkpoint_2000.pth")
    agent.evaluate(env,100)
    print("Finished.")


if __name__ == '__main__':
    main()
