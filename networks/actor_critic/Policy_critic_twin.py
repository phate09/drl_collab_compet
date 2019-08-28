import torch
import torchtest
from torch import nn as nn
from torch.nn import functional as F


class Policy_critic(nn.Module):
    def __init__(self, input_dim, hidden_layer_size=64):
        super().__init__()
        self.output_dim = 1  # value network
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, hidden_layer_size)
        self.bn1 = nn.BatchNorm1d(hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.bn2 = nn.BatchNorm1d(hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, self.output_dim)
        self.bn3 = nn.BatchNorm1d(self.output_dim)
        self.fc12 = nn.Linear(self.input_dim, hidden_layer_size)
        self.bn12 = nn.BatchNorm1d(hidden_layer_size)
        self.fc22 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.bn22 = nn.BatchNorm1d(hidden_layer_size)
        self.fc32 = nn.Linear(hidden_layer_size, self.output_dim)
        self.bn32 = nn.BatchNorm1d(self.output_dim)

    def forward(self, x):
        output: torch.Tensor = F.relu(self.bn1(self.fc1(x)))  # self.bn1(
        output: torch.Tensor = F.relu(self.bn2(self.fc2(output)))  # self.bn2(
        output: torch.Tensor = self.bn3(self.fc3(output))

        output2: torch.Tensor = F.relu(self.bn12(self.fc1(x)))  # self.bn1(
        output2: torch.Tensor = F.relu(self.bn22(self.fc2(output2)))  # self.bn2(
        output2: torch.Tensor = self.bn32(self.fc32(output2))
        return output, output2

    def test(self, device='cpu'):
        self.eval()
        input = torch.randn(10, self.input_dim, requires_grad=False)
        targets = torch.rand(10, self.output_dim, requires_grad=False)
        torchtest.test_suite(self, F.mse_loss, torch.optim.Adam(self.parameters()), batch=[input, targets], test_vars_change=True, test_inf_vals=True, test_nan_vals=True, device=device)
        print('All tests passed')


if __name__ == '__main__':
    model = Policy_critic(33)
    model.test()
