import torch
import torch.nn as nn
import torch.nn.functional as F

class KAN(nn.Module):
    def __init__(self, layers_config, base_activation=F.silu, grid_range=[-1, 1], init_method=nn.init.kaiming_uniform_):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        self.grids = []
        self.base_weights = nn.ParameterList()
        self.spline_weights = nn.ParameterList()
        self.spline_scalers = nn.ParameterList()

        for in_features, out_features, grid_size, spline_order in layers_config:
            h = (grid_range[1] - grid_range[0]) / grid_size
            grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + grid_range[0]
            
            # Ensure grid is 2D by checking its dimensions
            if grid.dim() == 1:
                grid = grid.unsqueeze(0)  # Make it 2D if it's not already
            
            # Correctly use expand based on grid's dimensions
            self.grids.append(grid.expand(in_features, grid.size(-1)))

            base_weight = nn.Parameter(torch.empty(out_features, in_features))
            spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
            spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
            self.base_weights.append(base_weight)
            self.spline_weights.append(spline_weight)
            self.spline_scalers.append(spline_scaler)

            init_method(base_weight)
            init_method(spline_weight)
            init_method(spline_scaler)

    def b_splines(self, x, grid, spline_order):
        x = x.unsqueeze(-1)
        grid = grid.to(x.device)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + \
                    ((grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()

    def forward(self, x):
        for base_weight, spline_weight, spline_scaler, grid in zip(self.base_weights, self.spline_weights, self.spline_scalers, self.grids):
            x = x.to(base_weight.device)
            base_output = F.linear(F.silu(x), base_weight)
            spline_output = F.linear(self.b_splines(x, grid, spline_order).view(x.size(0), -1), (spline_weight * spline_scaler.unsqueeze(-1)).view(spline_weight.shape[0], -1))
            x = base_output + spline_output
        return x
