import torch
import torch.nn as nn
import torch.nn.functional as F

class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, base_activation=nn.SiLU, grid_range=[-1, 1]):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation()  # Initialize the activation function

        # Lists to hold weights and grids
        self.base_weights = nn.ParameterList()
        self.spline_weights = nn.ParameterList()
        self.grids = nn.ParameterList()

        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            # Initialize weights
            base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
            spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
            self.base_weights.append(base_weight)
            self.spline_weights.append(spline_weight)

            # Compute grid coordinates for splines
            h = (grid_range[1] - grid_range[0]) / grid_size
            grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + grid_range[0]
            grid = grid.expand(in_features, -1).contiguous()
            self.grids.append(nn.Parameter(grid, requires_grad=False))

            # Initialize parameters using Kaiming Uniform
            nn.init.kaiming_uniform_(base_weight)
            nn.init.kaiming_uniform_(spline_weight)

    def b_splines(self, x, grid):
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + \
                    ((grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()

    def forward(self, x):
        # Sequentially pass through all layers
        for i, (base_weight, spline_weight) in enumerate(zip(self.base_weights, self.spline_weights)):
            grid = self.grids[i]
            x = x.to(base_weight.device)
            base_output = F.linear(self.base_activation(x), base_weight)
            spline_output = F.linear(self.b_splines(x, grid).view(x.size(0), -1), spline_weight.view(base_weight.size(0), -1))
            x = base_output + spline_output
        return x
