# Code by @1ssb: https://github.com/1ssb/torchkan under the MIT License
# Code references and strucuture adapted from:
# https://github.com/KindXiaoming/pykan
# https://github.com/Blealtan/efficient-kan


import torch
import torch.nn as nn
import torch.nn.functional as F

class KAN(nn.Module):
    def __init__(
        self, 
        layers_hidden,  # List of integers defining the number of neurons in each hidden layer
        grid_size=5,  # The number of points in the B-spline grid
        spline_order=3,  # The order of the B-spline (degree of polynomial plus one)
        scale_base=1.0,  # Scaling factor for the base linear transformation (not used in this code)
        scale_spline=1.0,  # Scaling factor for the spline transformation (not used in this code)
        base_activation=nn.SiLU,  # Activation function to be applied to the input of the base linear operation
        grid_range=[-1, 1]  # Range over which the B-spline grid is defined
    ):
        super(KAN, self).__init__()  # Initialize the parent nn.Module class
        # Store the parameters in the instance
        self.layers_hidden = layers_hidden
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()  # Instantiate the activation function
        self.grid_range = grid_range

        # Initialize ParameterLists to hold weights for each layer
        self.base_weights = nn.ParameterList()
        self.spline_weights = nn.ParameterList()
        self.spline_scalers = nn.ParameterList()
        self.grids = []  # List to hold the grid tensors for each layer

        # For each pair of adjacent layers
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            # Initialize the base weight matrix for a standard linear transformation
            self.base_weights.append(nn.Parameter(torch.Tensor(out_features, in_features)))
            # Initialize the spline weight tensor for B-spline transformations
            self.spline_weights.append(nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order)))
            # Initialize the spline scaler to adjust the influence of each spline basis function
            self.spline_scalers.append(nn.Parameter(torch.Tensor(out_features, in_features)))

            # Compute the step size for the B-spline grid based on the specified range and grid size
            h = (self.grid_range[1] - self.grid_range[0]) / grid_size
            # Create a tensor of grid points for the B-spline basis functions, extended to handle boundary conditions
            grid = torch.linspace(
                self.grid_range[0] - h * spline_order, 
                self.grid_range[1] + h * spline_order, 
                grid_size + 2 * spline_order + 1,
                dtype=torch.float32
            ).expand(in_features, -1).contiguous()
            # Register this grid tensor as a buffer to ensure it moves with the model to the appropriate device
            self.register_buffer(f'grid_{len(self.grids)}', grid)
            # Add the grid tensor to the list of grids
            self.grids.append(grid)

        # Initialize all weights using the Kaiming uniform initialization method
        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight)
        for weight in self.spline_weights:
            nn.init.kaiming_uniform_(weight)
        for scaler in self.spline_scalers:
            nn.init.kaiming_uniform_(scaler)

    def forward(self, x):
        # Process the input through each layer of the network
        for i, (base_weight, spline_weight, spline_scaler) in enumerate(zip(self.base_weights, self.spline_weights, self.spline_scalers)):
            # Retrieve the grid for the current layer
            grid = getattr(self, f'grid_{i}')

            # Move the input tensor to the device of the current layer's base weights
            x = x.to(base_weight.device)
            # Compute the base linear transformation using the chosen activation function
            base_output = F.linear(self.base_activation(x), base_weight)

            # Inline computation of the B-spline basis functions
            x_uns = x.unsqueeze(-1)  # Unsqueeze to prepare for comparison with grid points
            # Identify which grid intervals the input values fall into
            bases = ((x_uns >= grid[:, :-1]) & (x_uns < grid[:, 1:])).to(x.dtype)
            # Refine the B-spline basis functions iteratively based on the spline order
            for k in range(1, self.spline_order + 1):
                bases = ((x_uns - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]) + \
                        ((grid[:, k + 1:] - x_uns) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
            bases = bases.contiguous()  # Ensure memory layout is contiguous

            # Apply scaling to the spline weights using the spline scalers
            scaled_spline_weight = spline_weight * spline_scaler.unsqueeze(-1)
            # Compute the spline transformation by treating the bases as a linear transformation matrix
            spline_output = F.linear(bases.view(x.size(0), -1), scaled_spline_weight.view(spline_weight.size(0), -1))

            # Sum the outputs from the base and spline transformations
            x = base_output + spline_output

        return x  # Return the final output after processing through all layers