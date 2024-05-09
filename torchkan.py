import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        base_activation=nn.SiLU,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation()

        # Compute grid coordinates for splines
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        # Initialize weights
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize base and spline weights
        nn.init.kaiming_uniform_(self.base_weight)
        nn.init.kaiming_uniform_(self.spline_weight)

    def b_splines(self, x: torch.Tensor):
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - self.grid[:, : -(k + 1)]) / (self.grid[:, k:-1] - self.grid[:, : -(k + 1)]) * bases[:, :, :-1]) + \
                    ((self.grid[:, k + 1 :] - x) / (self.grid[:, k + 1 :] - self.grid[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()

    def forward(self, x: torch.Tensor):
        # Ensure x and base_weight are on the same device
        x = x.to(self.base_weight.device)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1), self.spline_weight.view(self.out_features, -1))
        return base_output + spline_output

class KAN(nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        base_activation=nn.SiLU,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Initialize layers based on hidden dimensions
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_activation=base_activation,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
