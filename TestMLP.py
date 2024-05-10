import torch
import torch.nn as nn
import torch.nn.functional as F

class KAN(nn.Module):
    def __init__(self, layers_hidden, base_activation=nn.GELU):
        super(KAN, self).__init__()
        # List of hidden layer dimensions for the neural network.
        self.layers_hidden = layers_hidden
        # Activation function used for the initial transformation of the input.
        self.base_activation = base_activation()
        
        # Parameters and layer norms initialization
        self.base_weights = nn.ParameterList()  # Parameters for the linear transformations in each layer.
        self.layer_norms = nn.ModuleList()  # Layer normalization for each layer to ensure stable training.
        
        # Loop through the layers to initialize weights, norms
        for i, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            # Initialize the base weights with random values for the linear transformation.
            self.base_weights.append(nn.Parameter(torch.randn(out_features, in_features)))
            # Add a layer normalization for stabilizing the output of this layer.
            self.layer_norms.append(nn.LayerNorm(out_features))

        # Initialize the weights using Kaiming uniform distribution for better initial values.
        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

    def forward(self, x):
        # Process each layer using the defined base weights, norms, and activations.
        for i, (base_weight, layer_norm) in enumerate(zip(self.base_weights, self.layer_norms)):
            # Move the input tensor to the device where the weights are located.
            x = x.to(base_weight.device)

            # Perform the base linear transformation followed by the activation function.
            base_output = F.linear(self.base_activation(x), base_weight)

            # Apply layer normalization to the linear output.
            x = layer_norm(base_output)

        return x
