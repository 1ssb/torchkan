import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Compute grid coordinates for splines
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        # Initialize weights and optional scaler
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        
        nn.init.kaiming_uniform_(self.base_weight)
        nn.init.kaiming_uniform_(self.spline_weight)
        nn.init.kaiming_uniform_(self.spline_scaler)
    
    def forward(self, x):

        x = x.to(self.base_weight.device)
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # Inline B-spline computation
        x_uns = x.unsqueeze(-1)
        bases = ((x_uns >= self.grid[:, :-1]) & (x_uns < self.grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x_uns - self.grid[:, :-(k + 1)]) / (self.grid[:, k:-1] - self.grid[:, :-(k + 1)]) * bases[:, :, :-1]) + \
                    ((self.grid[:, k + 1:] - x_uns) / (self.grid[:, k + 1:] - self.grid[:, 1:(-k)]) * bases[:, :, 1:])
        bases = bases.contiguous()

        # Spline weight scaling
        scaled_spline_weight = self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.spline_scaler is not None else 1.0)
        spline_output = F.linear(bases.view(x.size(0), -1), scaled_spline_weight.view(self.out_features, -1))

        return base_output + spline_output

class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, scale_base=1.0, scale_spline=1.0, base_activation=nn.SiLU, grid_range=[-1, 1]):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_range=grid_range,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x