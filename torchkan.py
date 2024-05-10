# Code by @1ssb: https://github.com/1ssb/torchkan under the MIT License
# Code references and strucuture adapted from:
## https://github.com/KindXiaoming/pykan
## https://github.com/Blealtan/efficient-kan
## Stabilization advice taken from: https://github.com/ZiyaoLi/fast-kan

import torch
import torch.nn as nn
import torch.nn.functional as F

class KAN(nn.Module):
    def __init__(
        self, 
        layers_hidden,     # List of neurons per layer
        grid_size=5,       # Points in B-spline grid
        spline_order=3,    # Polynomial degree + 1 of B-spline
        base_activation=nn.GELU,  # Activation function
        grid_range=[-1, 1] # B-spline grid range
    ):
        super(KAN, self).__init__()
        self.layers_hidden = layers_hidden
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation()  # Activation instance
        self.grid_range = grid_range

        # Initialize weights, scalers, and norms efficiently
        self.base_weights = nn.ParameterList()
        self.spline_weights = nn.ParameterList()
        self.spline_scalers = nn.ParameterList()
        self.layer_norms = nn.ModuleList()
        self.grids = []

        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.base_weights.append(nn.Parameter(torch.randn(out_features, in_features)))
            self.spline_weights.append(nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order)))
            self.spline_scalers.append(nn.Parameter(torch.randn(out_features, in_features)))
            self.layer_norms.append(nn.LayerNorm(out_features))

            h = (self.grid_range[1] - self.grid_range[0]) / grid_size
            grid = torch.linspace(
                self.grid_range[0] - h * spline_order, 
                self.grid_range[1] + h * spline_order, 
                grid_size + 2 * spline_order + 1,
                dtype=torch.float32
            ).expand(in_features, -1).contiguous()
            self.register_buffer(f'grid_{len(self.grids)}', grid)
            self.grids.append(grid)

        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for weight in self.spline_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for scaler in self.spline_scalers:
            nn.init.kaiming_uniform_(scaler, nonlinearity='linear')

    def forward(self, x):
        for i, (base_weight, spline_weight, spline_scaler, layer_norm) in enumerate(zip(self.base_weights, self.spline_weights, self.spline_scalers, self.layer_norms)):
            grid = getattr(self, f'grid_{i}')
            x = x.to(base_weight.device)
            
            # Direct computation without checkpointing
            base_output = F.linear(self.base_activation(x), base_weight)
            x_uns = x.unsqueeze(-1)
            bases = ((x_uns >= grid[:, :-1]) & (x_uns < grid[:, 1:])).to(x.dtype)

            for k in range(1, self.spline_order + 1):
                left_intervals = grid[:, :-(k + 1)]
                right_intervals = grid[:, k:-1]
                delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals), right_intervals - left_intervals)
                bases = ((x_uns - left_intervals) / delta * bases[:, :, :-1]) + \
                        ((grid[:, k + 1:] - x_uns) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
            bases = bases.contiguous()

            scaled_spline_weight = spline_weight * spline_scaler.unsqueeze(-1)
            spline_output = F.linear(bases.view(x.size(0), -1), scaled_spline_weight.view(spline_weight.size(0), -1))

            x = layer_norm(base_output + spline_output)

        return x
