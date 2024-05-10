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
        layers_hidden,     # List defining the number of neurons in each layer
        grid_size=5,       # Number of points in the B-spline grid
        spline_order=3,    # Order of the B-spline (degree of polynomial plus one)
        base_activation=nn.GELU,  # Activation function used in the model
        grid_range=[-1, 1] # Range over which the B-spline grid is defined
    ):
        super(KAN, self).__init__()
        self.layers_hidden = layers_hidden  # Store the structure of neural network layers
        self.grid_size = grid_size          # Store the grid size for B-spline basis
        self.spline_order = spline_order    # Store the spline order for B-spline calculations
        self.base_activation = base_activation()  # Instantiate the specified activation function
        self.grid_range = grid_range        # Store the range of the B-spline grid

        # Initialize the weights, scalers, and normalization layers for the neural network
        self.base_weights = nn.ParameterList()
        self.spline_weights = nn.ParameterList()
        self.spline_scalers = nn.ParameterList()
        self.layer_norms = nn.ModuleList()
        self.grids = []

        # Initialize parameters and compute the B-spline grids for each layer
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            # Initialize base weights for linear transformations in each layer
            self.base_weights.append(nn.Parameter(torch.randn(out_features, in_features)))
            # Initialize spline weights for B-spline transformations
            self.spline_weights.append(nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order)))
            # Initialize scalers to adjust the influence of spline weights
            self.spline_scalers.append(nn.Parameter(torch.randn(out_features, in_features)))
            # Initialize layer normalization to stabilize the output of each layer
            self.layer_norms.append(nn.LayerNorm(out_features))

            # Compute the B-spline grid for the current layer, handling boundary conditions
            h = (self.grid_range[1] - self.grid_range[0]) / grid_size
            grid = torch.linspace(
                self.grid_range[0] - h * spline_order, 
                self.grid_range[1] + h * spline_order, 
                grid_size + 2 * spline_order + 1,
                dtype=torch.float32
            ).expand(in_features, -1).contiguous()
            # Register the grid as a buffer to ensure it is properly managed by PyTorch
            self.register_buffer(f'grid_{len(self.grids)}', grid)
            self.grids.append(grid)

        # Initialize all weights with Kaiming uniform initialization for optimal learning
        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for weight in self.spline_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for scaler in self.spline_scalers:
            nn.init.kaiming_uniform_(scaler, nonlinearity='linear')

    def forward(self, x):
        # Process the input through each layer of the neural network
        for i, (base_weight, spline_weight, spline_scaler, layer_norm) in enumerate(zip(self.base_weights, self.spline_weights, self.spline_scalers, self.layer_norms)):
            grid = getattr(self, f'grid_{i}')
            x = x.to(base_weight.device)  # Ensure data is on the same device as the model parameters
            
            # Apply the base linear transformation with activation function
            base_output = F.linear(self.base_activation(x), base_weight)
            # Prepare the input for B-spline basis calculation
            x_uns = x.unsqueeze(-1)
            # Initialize B-spline bases for the current configuration of input and grid
            bases = ((x_uns >= grid[:, :-1]) & (x_uns < grid[:, 1:])).to(x.dtype)

            # Refine B-spline basis functions across specified orders
            for k in range(1, self.spline_order + 1):
                left_intervals = grid[:, :-(k + 1)]
                right_intervals = grid[:, k:-1]
                delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals), right_intervals - left_intervals)
                bases = ((x_uns - left_intervals) / delta * bases[:, :, :-1]) + \
                        ((grid[:, k + 1:] - x_uns) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
            bases = bases.contiguous()

            # Calculate the scaled spline output based on spline weights and scalers
            scaled_spline_weight = spline_weight * spline_scaler.unsqueeze(-1)
            spline_output = F.linear(bases.view(x.size(0), -1), scaled_spline_weight.view(spline_weight.size(0), -1))

            # Combine the outputs from the base and spline layers and normalize
            x = layer_norm(base_output + spline_output)

        return x  # Return the final processed output after all layers
