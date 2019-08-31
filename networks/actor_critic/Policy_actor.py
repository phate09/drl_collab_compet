import torch
import torchtest
from torch import nn as nn
from torch.nn import functional as F


class Policy_actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_size=64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, hidden_layer_size)
        # self.bn1 = nn.BatchNorm1d(hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        # self.bn2 = nn.BatchNorm1d(hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, self.output_dim)
        # self.bn3 = nn.BatchNorm1d(self.output_dim)

    def forward(self, x):
        output: torch.Tensor = F.leaky_relu(self.fc1(x))
        output: torch.Tensor = F.leaky_relu(self.fc2(output))
        output: torch.Tensor = torch.tanh(self.fc3(output))
        # output: torch.Tensor = output.mul(5.0)
        return output

    def test(self, device='cpu'):
        self.eval()
        input = torch.randn(10, self.input_dim, requires_grad=False)
        targets = torch.rand(10, self.output_dim, requires_grad=False)
        torchtest.test_suite(self, F.mse_loss, torch.optim.Adam(self.parameters()), batch=[input, targets], test_vars_change=True, test_inf_vals=True, test_nan_vals=True, device=device)
        print('All tests passed')

if __name__ == '__main__':
    model = Policy_actor(33,4)
    model.test()