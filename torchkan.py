# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class KAN(nn.Module):
#     def __init__(self, layers_hidden, grid_size=5, spline_order=3, base_activation=nn.SiLU, grid_range=[-1, 1]):
#         super(KAN, self).__init__()
#         self.grid_size = grid_size
#         self.spline_order = spline_order
#         self.base_activation = base_activation()  # Initialize the activation function

#         # Lists to hold weights and grids
#         self.base_weights = nn.ParameterList()
#         self.spline_weights = nn.ParameterList()
#         self.grids = nn.ParameterList()

#         for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
#             # Initialize weights
#             base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
#             spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
#             self.base_weights.append(base_weight)
#             self.spline_weights.append(spline_weight)

#             # Compute grid coordinates for splines
#             h = (grid_range[1] - grid_range[0]) / grid_size
#             grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + grid_range[0]
#             grid = grid.expand(in_features, -1).contiguous()
#             self.grids.append(nn.Parameter(grid, requires_grad=False))

#             # Initialize parameters using Kaiming Uniform
#             nn.init.kaiming_uniform_(base_weight)
#             nn.init.kaiming_uniform_(spline_weight)

#     def b_splines(self, x, grid):
#         x = x.unsqueeze(-1)
#         bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
#         for k in range(1, self.spline_order + 1):
#             bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + \
#                     ((grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:])
#         return bases.contiguous()

#     def forward(self, x):
#         # Sequentially pass through all layers
#         for i, (base_weight, spline_weight) in enumerate(zip(self.base_weights, self.spline_weights)):
#             grid = self.grids[i]
#             x = x.to(base_weight.device)
#             base_output = F.linear(self.base_activation(x), base_weight)
#             spline_output = F.linear(self.b_splines(x, grid).view(x.size(0), -1), spline_weight.view(base_weight.size(0), -1))
#             x = base_output + spline_output
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, base_activation=nn.SiLU, grid_range=[-1, 1], grid_eps=0.02):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation()  # Initialize the activation function
        self.grid_eps = grid_eps

        # Lists to hold weights and grids
        self.base_weights = nn.ParameterList()
        self.spline_weights = nn.ParameterList()
        self.grids = []

        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            # Initialize weights
            base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
            spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
            self.base_weights.append(base_weight)
            self.spline_weights.append(spline_weight)

            # Compute grid coordinates for splines
            h = (grid_range[1] - grid_range[0]) / grid_size
            grid = (
                torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h
                + grid_range[0]
            ).expand(in_features, -1).contiguous()
            self.grids.append(grid)

            # Initialize parameters using Kaiming Uniform
            nn.init.kaiming_uniform_(base_weight, nonlinearity='linear')
            nn.init.kaiming_uniform_(spline_weight, nonlinearity='linear')

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weights[0].abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return regularize_activation * regularization_loss_activation + regularize_entropy * regularization_loss_entropy

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        A = self.b_splines(x, self.grids[0]).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    def b_splines(self, x, grid):
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + \
                    ((grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weights[0] * 1.0  # Adjust or expand as needed

    def forward(self, x):
        for i, (base_weight, spline_weight) in enumerate(zip(self.base_weights, self.spline_weights)):
            grid = self.grids[i]
            x = x.to(base_weight.device)
            base_output = F.linear(self.base_activation(x), base_weight)
            spline_output = F.linear(self.b_splines(x, grid).view(x.size(0), -1), self.scaled_spline_weight.view(base_weight.size(0), -1))
            x = base_output + spline_output
        return x

    @torch.no_grad()
    def update_grid(self, x, margin=0.01):
        splines = self.b_splines(x, self.grids[0]).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[torch.linspace(0, x.size(0) - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat([grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                          grid, grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1)], dim=0)

        self.grids[0].copy_(grid.T)
        self.spline_weights[0].data.copy_(self.curve2coeff(x, unreduced_spline_output))
