from munch import DefaultMunch

from agents.Unity.MultiAgent_TD3_embedded import MultiAgentTD3
from networks.actor_critic.Policy_actor import Policy_actor
from networks.actor_critic.Policy_critic_twin import Policy_critic
import torch
import torch.optim as optim
import numpy as np

from utility.Scheduler import Scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device: ", device)
seed = 4
torch.manual_seed(seed)
np.random.seed(seed)
action_size = 2
state_size = 8
state_multiplier = 3
n_agents = 2
batch_size = 256
# action_type = brain.vector_action_space_type
comment = f"TD3 Embedded Unity Tennis"
actor_fn = lambda: Policy_actor(state_size * state_multiplier, action_size, hidden_layer_size=200).to(device)
critic_fn = lambda: Policy_critic((state_size * state_multiplier + action_size) * n_agents, hidden_layer_size=200).to(device)
# actor1.test(device)
optimiser_actor_fn = lambda actor: optim.Adam(actor.parameters(), lr=1e-3)
optimiser_critic_fn = lambda critic: optim.Adam(critic.parameters(), lr=1e-3)
ending_condition = lambda result: result['mean'] >= 300.0
writer = None

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
config.learn_start = config.max_t  # training starts after this many transitions
config.evaluate_every = 100
config.use_noise = True
config.use_priority = False
config.noise_scheduler = Scheduler(1.0, 0.1, config.max_t * 10, warmup_steps=config.max_t)
config.n_agents = n_agents
config.action_size = action_size
config.use_shared_memory = False
config.summary_writer_fn = lambda: writer
config.actor_fn = actor_fn
config.critic_fn = critic_fn
config.optimiser_actor_fn = optimiser_actor_fn
config.optimiser_critic_fn = optimiser_critic_fn
agent = MultiAgentTD3(config)

states = torch.rand(size=(batch_size, n_agents, state_size * state_multiplier), dtype=torch.float, device=device)
next_states = torch.rand(size=(batch_size, n_agents, state_size * state_multiplier), dtype=torch.float, device=device)
actions = torch.rand(size=(batch_size, n_agents, action_size), dtype=torch.float, device=device)
dones = torch.ones(batch_size, dtype=torch.float, device=device)
rewards = torch.ones(size=(batch_size, n_agents), dtype=torch.float, device=device)
before1 = agent.actor1(states[:, 0]).mean()
before2 = agent.actor2(states[:, 1]).mean()
before3_1, before3_2 = agent.critic1(torch.cat([states, actions], dim=2).view(batch_size, -1))
before4_1, before4_2 = agent.critic2(torch.cat([states, actions], dim=2).view(batch_size, -1))
for i in range(1):
    agent._learn(states, actions, rewards, dones, next_states, True, 0)
after1 = agent.actor1(states[:, 0]).mean()
after2 = agent.actor2(states[:, 1]).mean()
after3_1, after3_2 = agent.critic1(torch.cat([states, actions], dim=2).view(batch_size, -1))
after4_1, after4_2 = agent.critic2(torch.cat([states, actions], dim=2).view(batch_size, -1))
delta1 = (after1 - before1).item()
delta2 = (after2 - before2).item()
delta3 = (after3_1.mean() - before3_1.mean()).item()
delta4 = (after4_1.mean() - before4_1.mean()).item()
delta5 = (after3_2.mean() - before3_1.mean()).item()
delta6 = (after4_2.mean() - before4_1.mean()).item()
# assert delta1 > 0
# assert delta2 > 0
assert delta3 > 0
assert delta4 > 0
assert delta5 > 0
assert delta6 > 0
print(delta1)
