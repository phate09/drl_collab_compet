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
seed = 2
torch.manual_seed(seed)
np.random.seed(seed)
action_size = 2
state_size = 8
state_multiplier = 3
n_agents = 2
# action_type = brain.vector_action_space_type
comment = f"TD3 Embedded Unity Tennis"
actor_fn = lambda: Policy_actor(state_size * state_multiplier, action_size, hidden_layer_size=200).to(device)
critic_fn = lambda: Policy_critic((state_size * state_multiplier + action_size) * n_agents, hidden_layer_size=200).to(device)
# actor1.test(device)
optimiser_actor_fn = lambda actor: optim.Adam(actor.parameters(), lr=1e-4)
optimiser_critic_fn = lambda critic: optim.Adam(critic.parameters(), lr=1e-4)
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

