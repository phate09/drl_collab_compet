import torch
import torch.nn.functional as F
import torch.optim as optim
from munch import DefaultMunch

from models import Actor
from utility.noise import OUNoise


class Agent(object):
    def __init__(self, config: DefaultMunch, parent):
        self.config = config
        self.action_size = self.config.action_size
        self.state_size = self.config.state_size
        self.parent = parent

        self.actor_local = Actor(self.state_size, self.config.action_size).to(self.config.device)
        self.actor_target = Actor(self.state_size, self.config.action_size).to(self.config.device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.config.lr_actor)

        self.memory = self.config.memory
        self.t_step = 0
        self.noise = OUNoise(self.config.action_size, self.config.seed)

    def step(self):
        self.t_step = (self.t_step + 1) % self.config.update_every
        if self.t_step == 0:
            if len(self.memory) > self.config.batch_size:
                experiences, _, _ = self.memory.sample(self.config.batch_size)
                states, actions, rewards, next_states, dones, others_states, others_actions, others_next_states = zip(*experiences)
                self.learn((torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(next_states), torch.stack(dones), torch.stack(others_states).squeeze(), torch.stack(others_actions).squeeze(), torch.stack(others_next_states).squeeze()), self.config.gamma)

    def act(self, states, add_noise=True):
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states)
        self.actor_local.train()

        if add_noise:
            sample_np = self.noise.sample()
            sample = torch.tensor(sample_np, dtype=torch.float, device=self.config.device)
            actions += sample
        clipped = torch.clamp(actions, -1, 1)
        return clipped

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):

        (states, actions, rewards, next_states, dones, others_states, others_actions, others_next_states) = experiences
        all_states = torch.cat((states, others_states), dim=1).to(self.config.device)
        all_actions = torch.cat((actions, others_actions), dim=1).to(self.config.device)
        all_next_states = torch.cat((next_states, others_next_states), dim=1).to(self.config.device)

        # --------------------------- update critic ---------------------------
        all_next_actions = torch.cat([self.actor_target(states), self.actor_target(others_states)], dim=1).to(self.config.device)
        Q_targets_next = self.parent.critic_target(all_next_states, all_next_actions)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones.float()))
        Q_expected = self.parent.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.parent.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.parent.critic_optimizer.step()

        # --------------------------- update actor ---------------------------
        this_actions_pred = self.actor_local(states)
        others_actions_pred = self.actor_local(others_states)
        others_actions_pred = others_actions_pred.detach()
        actions_pred = torch.cat((this_actions_pred, others_actions_pred), dim=1).to(self.config.device)
        actor_loss = -self.parent.critic_local(all_states, actions_pred).mean() + 0.01 * (actions_pred ** 2).mean()  # added penalty for "moving" unnecessarily
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ---------------------- update target networks ----------------------
        self.soft_update(self.parent.critic_local, self.parent.critic_target, self.config.tau)
        self.soft_update(self.actor_local, self.actor_target, self.config.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, path, global_step):
        torch.save({
            "global_step": global_step,
            "actor": self.actor_local.state_dict(),
            "target_actor": self.actor_target.state_dict(),
            "critic": self.parent.critic_local.state_dict(),
            "target_critic": self.parent.critic_target.state_dict(),
            "optimiser_actor": self.actor_optimizer.state_dict(),
            "optimiser_critic": self.parent.critic_optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor_local.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["target_actor"])
        self.parent.critic_local.load_state_dict(checkpoint["critic"])
        self.parent.critic_target.load_state_dict(checkpoint["target_critic"])
        self.actor_optimizer.load_state_dict(checkpoint["optimiser_actor"])
        self.parent.critic_optimizer.load_state_dict(checkpoint["optimiser_critic"])
        self.replay_buffer = checkpoint["optimiser_critic"]
        self.global_step = checkpoint["global_step"]
        print(f'Loading complete')
