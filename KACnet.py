import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache

class KAC_Net(nn.Module):  # Kolmogorov Arnold chebyshev Network (KAL-Net)
    def __init__(self, layers_hidden, polynomial_order=3, base_activation=nn.SiLU):
        super(KAC_Net, self).__init__()  # Initialize the parent nn.Module class
        
        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which chebyshev polynomials are calculated
        self.polynomial_order = polynomial_order
        # base_activation: Activation function used after each layer's computation
        self.base_activation = base_activation()
        
        # ParameterList for the base weights of each layer
        self.base_weights = nn.ParameterList()
        # ParameterList for the polynomial weights for chebyshev expansion
        self.poly_weights = nn.ParameterList()
        # ModuleList for layer normalization for each layer's output
        self.layer_norms = nn.ModuleList()

        # Initialize network parameters
        for i, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            # Base weight for linear transformation in each layer
            self.base_weights.append(nn.Parameter(torch.randn(out_features, in_features)))
            # Polynomial weight for handling chebyshev polynomial expansions
            self.poly_weights.append(nn.Parameter(torch.randn(out_features, in_features * (polynomial_order + 1))))
            # Layer normalization to stabilize learning and outputs
            self.layer_norms.append(nn.LayerNorm(out_features))

        # Initialize weights using Kaiming uniform distribution for better training start
        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for weight in self.poly_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

    @lru_cache(maxsize=128)  # Cache to avoid recomputation of chebyshev polynomials
    def compute_chebyshev_polynomials(self, x, order):
        # Base case polynomials P0 and P1
        P0 = x.new_ones(x.shape)  # P0 = 1 for all x
        if order == 0:
            return P0.unsqueeze(-1)
        P1 = x  # P1 = x
        chebyshev_polys = [P0, P1]
        
        # Compute higher order polynomials using recurrence
        for n in range(1, order):
            #Pn = ((2.0 * n + 1.0) * x * chebyshev_polys[-1] - n * chebyshev_polys[-2]) / (n + 1.0)
            Pn = 2 * x * chebyshev_polys[-1] -  chebyshev_polys[-2]

            chebyshev_polys.append(Pn)
        
        return torch.stack(chebyshev_polys, dim=-1)

    def forward(self, x):
        # Ensure x is on the right device from the start, matching the model parameters
        x = x.to(self.base_weights[0].device)

        for i, (base_weight, poly_weight, layer_norm) in enumerate(zip(self.base_weights, self.poly_weights, self.layer_norms)):
            # Apply base activation to input and then linear transform with base weights
            base_output = F.linear(self.base_activation(x), base_weight)
            
            # Normalize x to the range [-1, 1] for stable chebyshev polynomial computation
            x_normalized = 2 * (x - x.min()) / (x.max() - x.min()) - 1
            # Compute chebyshev polynomials for the normalized x
            chebyshev_basis = self.compute_chebyshev_polynomials(x_normalized, self.polynomial_order)
            # Reshape chebyshev_basis to match the expected input dimensions for linear transformation
            chebyshev_basis = chebyshev_basis.view(x.size(0), -1)

            # Compute polynomial output using polynomial weights
            poly_output = F.linear(chebyshev_basis, poly_weight)
            # Combine base and polynomial outputs, normalize, and activate
            x = self.base_activation(layer_norm(base_output + poly_output))

        return x