import torch
import torch.nn as nn
import torch.nn.functional as F

class KANvolver(nn.Module):
    def __init__(self, layers_hidden, polynomial_order=2, base_activation=nn.ReLU):
        super(KANvolver, self).__init__()
        
        self.layers_hidden = layers_hidden
        self.polynomial_order = polynomial_order
        self.base_activation = base_activation()
        
        # Feature extractor with Convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 1 input channel (grayscale), 16 output channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Calculate the flattened feature size after convolutional layers
        flat_features = 32 * 7 * 7
        self.layers_hidden = [flat_features] + self.layers_hidden
        
        self.base_weights = nn.ModuleList()
        self.poly_weights = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for in_features, out_features in zip(self.layers_hidden[:-1], self.layers_hidden[1:]):
            self.base_weights.append(nn.Linear(in_features, out_features))
            self.poly_weights.append(nn.Linear(in_features * (polynomial_order + 1), out_features))
            self.batch_norms.append(nn.BatchNorm1d(out_features))

    def compute_efficient_monomials(self, x, order):
        powers = torch.arange(order + 1, device=x.device, dtype=x.dtype)
        x_expanded = x.unsqueeze(-1).repeat(1, 1, order + 1)
        return torch.pow(x_expanded, powers)

    def forward(self, x):
        # Reshape input from [batch_size, 784] to [batch_size, 1, 28, 28] for MNIST
        x = x.view(-1, 1, 28, 28)
        
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten the features from the conv layers
        
        for base_weight, poly_weight, batch_norm in zip(self.base_weights, self.poly_weights, self.batch_norms):
            base_output = base_weight(x)
            monomial_basis = self.compute_efficient_monomials(x, self.polynomial_order)
            monomial_basis = monomial_basis.view(x.size(0), -1)
            poly_output = poly_weight(monomial_basis)
            x = self.base_activation(batch_norm(base_output + poly_output))
        
        return x
