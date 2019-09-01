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
from agents.DDPG.MultiAgent_DDPG import MultiAgentDDPG
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
    comment = f"DDPG Unity Tennis"
    actor_fn = lambda: Policy_actor(state_size * state_multiplier, action_size, hidden_layer_size=200).to(device)
    critic_fn = lambda: Policy_critic((state_size * state_multiplier + action_size) * n_agents, hidden_layer_size=200).to(device)
    # actor1.test(device)
    optimizer_actor_fn = lambda actor: optim.Adam(actor.parameters(), lr=1e-4)
    optimizer_critic_fn = lambda critic: optim.Adam(critic.parameters(), lr=1e-4)
    ending_condition = lambda result: result['mean'] >= 300.0
    log_dir = os.path.join('runs', current_time + '_' + comment)
    os.mkdir(log_dir)
    print(f"logging to {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    config = DefaultMunch()
    config.seed = seed
    config.n_episodes = 40000
    config.batch_size = 32
    config.buffer_size = int(1e6)
    config.max_t = 2000  # just > 1000
    config.input_dim = state_size * state_multiplier
    config.output_dim = action_size
    config.gamma = 0.99  # discount
    config.tau = 0.005  # soft merge
    config.device = device
    config.train_every = 4
    config.train_n_times = 2
    config.n_step_td = 10
    config.ending_condition = ending_condition
    config.learn_start = 0  # training starts after this many transitions
    config.evaluate_every = 100
    config.use_noise = True
    config.use_priority = False
    config.noise_scheduler = Scheduler(1.0, 0.1, config.max_t * 10, warmup_steps=config.max_t)
    config.n_agents = n_agents
    config.action_size = action_size
    config.log_dir = log_dir
    config.summary_writer_fn = lambda: writer
    agent_config1 = DefaultMunch(actor_fn=actor_fn, critic_fn=critic_fn, optimiser_actor_fn=optimizer_actor_fn, optimiser_critic_fn=optimizer_critic_fn)
    agent_config2 = DefaultMunch(actor_fn=actor_fn, critic_fn=critic_fn, optimiser_actor_fn=optimizer_actor_fn, optimiser_critic_fn=optimizer_critic_fn)
    config.agent_configs = [agent_config1, agent_config2]

    config_file = open(os.path.join(log_dir, "config.json"), "w+")
    config_file.write(json.dumps(json.loads(jsonpickle.encode(config, unpicklable=False, max_depth=1)), indent=4, sort_keys=True))
    config_file.close()
    agent = MultiAgentDDPG(config)
    agent.load("runs/Aug31_15-27-30_TD3 Unity Tennis/checkpoint_17300.pth")
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
        noise_magnitude = noise_scheduler.get(global_steps)
        for i in range(config.max_t):
            with torch.no_grad():
                actions = agent.act(states, 0)
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
        writer.add_scalar('data/score', score, i_episode)
        writer.add_scalar('data/score_average', np.mean(scores_window), i_episode)
        writer.add_scalar('data/score_max', np.max(scores_window), i_episode)
        writer.add_scalar('data/score_min', np.min(scores_window), i_episode)
        writer.add_scalar('data/score_std', np.std(scores_window), i_episode)
        s_msg = '\rEpisode {}\tAverage Score: {:.3f}\tnoise: {:.3f}\tScore: {:.3f}'
        print(s_msg.format(i_episode, np.mean(scores_window), noise_magnitude, np.max(score)), end="")
        if i_episode % 100 == 0:
            print(s_msg.format(i_episode, np.mean(scores_window), noise_magnitude, np.max(score)))
            agent.save(os.path.join(log_dir, f"checkpoint_{i_episode}.pth"), i_episode)
        if np.mean(scores_window) >= 0.5:
            SOLVED = True
            s_msg = '\n\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}\tσ: {:.3f}'
            print(s_msg.format(i_episode, np.mean(scores_window), np.std(scores_window)))
            agent.save(os.path.join(log_dir, f"checkpoint_success.pth"), i_episode)
            # save the models
            # s_name = agent.__name__
            # s_aux = '%scheckpoint-%s.%s.%i.pth'
            # for ii in range(len(agent)):
            #     s_actor_path = s_aux % (DATA_PREFIX, s_name, 'actor', ii)
            #     s_critic_path = s_aux % (DATA_PREFIX, s_name, 'critic', ii)
            #     torch.save(agent[ii].actor_local.state_dict(), s_actor_path)
            #     torch.save(agent[ii].critic_local.state_dict(), s_critic_path)
            break

    # agent.save("/home/edoardo/PycharmProjects/ProximalPolicyOptimisation/runs/Aug13_16-46-32_DDPG Unity Reacher multi/checkpoint_50.pth",1)
    # agent.load("/home/edoardo/PycharmProjects/ProximalPolicyOptimisation/runs/Aug13_16-46-32_DDPG Unity Reacher multi/checkpoint_50.pth")
    # agent.train(env, ending_condition)
    print("Finished.")


if __name__ == '__main__':
    main()
