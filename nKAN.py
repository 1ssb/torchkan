import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Lazy_KAN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, max_length=1000000):
        super(Lazy_KAN, self).__init__()
        self.input_dim = input_dim
        self.max_length = max_length

        # Register a buffer for positional encoding
        self.register_buffer('positional_encoding', self.create_positional_encoding(max_length, input_dim))

        # ModuleList for the layers, using Dropout for noise robustness
        self.outer_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1)  # Dropout to prevent overfitting and add noise robustness
            ) for hidden_dim in hidden_dims
        ])
        
        self.inner_layer = nn.Linear(sum(hidden_dims), output_dim)

    def forward(self, x):
        # Use broadcasting to add positional encoding
        x = x + self.positional_encoding[:x.size(1), :]
        
        # Processing through outer layers
        outer_outputs = [layer(x) for layer in self.outer_layers]
        concatenated = torch.cat(outer_outputs, dim=2)
        output = self.inner_layer(concatenated)
        
        return output.squeeze(2)

    @staticmethod
    def create_positional_encoding(max_length, input_dim):
        """ Generate positional encoding with sine and cosine functions """
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / input_dim))
        pe = torch.zeros(max_length, input_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activations = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(len(layers) - 2)] + [nn.Identity()])

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x

def add_gaussian_noise(data, std):
    return data + torch.randn_like(data) * std

def generate_data(size=100):
    x = torch.linspace(-1, 1, size).reshape(-1, 1)
    y = torch.sin(x * 2 * np.pi)
    return x, y

def train_model(model, data, targets, criterion, optimizer, scheduler, epochs=500, device='cuda'):
    model.train()
    model.to(device)
    data, targets = data.to(device), targets.to(device)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(data)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update the learning rate
        losses.append(loss.item())
        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs} - Loss: {loss.item()}')
    return losses

def evaluate_model(model, data, targets, criterion, device='cuda'):
    model.eval()
    model.to(device)
    data, targets = data.to(device), targets.to(device)
    with torch.no_grad():
        predictions = model(data)
        loss = criterion(predictions, targets)
    return loss.item(), predictions

def compute_snr(signal, noise):
    signal_power = torch.mean(signal ** 2).item()
    noise_power = torch.mean(noise ** 2).item()
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Create data
data, targets = generate_data(100)

# Reshape data for positional encoding
data = data.unsqueeze(1)  # Adding a dimension for sequence length (batch_size, seq_length, input_dim)

# Create models
kan_model = Lazy_KAN(input_dim=1, hidden_dims=[64, 64, 64], output_dim=1)
mlp_model = MLP([1, 64, 32, 16, 1])

# Define loss function and optimizers
criterion = nn.MSELoss()
kan_optimizer = torch.optim.AdamW(kan_model.parameters(), lr=0.01)
mlp_optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=0.01)

# Add learning rate scheduler
kan_scheduler = torch.optim.lr_scheduler.StepLR(kan_optimizer, step_size=100, gamma=0.9)
mlp_scheduler = torch.optim.lr_scheduler.StepLR(mlp_optimizer, step_size=100, gamma=0.9)

# Check if CUDA is available and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Train both models
print("Training Lazy_KAN model")
kan_losses = train_model(kan_model, data, targets, criterion, kan_optimizer, kan_scheduler)
print("Training MLP model")
mlp_losses = train_model(mlp_model, data.squeeze(1), targets, criterion, mlp_optimizer, mlp_scheduler)  # Remove sequence dimension for MLP

# Evaluate models with noise
noise_levels = np.linspace(0.1, 2, 20)
kan_eval_losses = []
mlp_eval_losses = []
kan_snrs = []
mlp_snrs = []

for std in noise_levels:
    noisy_data = add_gaussian_noise(data, std=std).to(device)
    kan_loss, kan_predictions = evaluate_model(kan_model, noisy_data, targets, criterion, device=device)
    mlp_loss, mlp_predictions = evaluate_model(mlp_model, noisy_data.squeeze(1), targets, criterion, device=device)  # Remove sequence dimension for MLP
    
    kan_eval_losses.append(kan_loss)
    mlp_eval_losses.append(mlp_loss)
    
    # Compute SNR
    kan_snr = compute_snr(targets.to('cpu'), (targets - kan_predictions.to('cpu')))
    mlp_snr = compute_snr(targets.to('cpu'), (targets - mlp_predictions.to('cpu')))
    
    kan_snrs.append(kan_snr)
    mlp_snrs.append(mlp_snr)

    # Save the results
    plt.figure(figsize=(8, 4))
    plt.scatter(noisy_data.squeeze(1).cpu().numpy(), targets.cpu().numpy(), label='True Data')
    plt.scatter(noisy_data.squeeze(1).cpu().numpy(), kan_predictions.detach().cpu().numpy(), color='red', label='Lazy_KAN Predictions')
    plt.scatter(noisy_data.squeeze(1).cpu().numpy(), mlp_predictions.detach().cpu().numpy(), color='green', label='MLP Predictions')
    plt.title(f'Performance with Noise Std Dev = {std:.2f}')
    plt.xlabel('Input Feature')
    plt.ylabel('Output Target')
    plt.legend()
    plt.savefig(f'performance_noise_std_{std:.2f}.png')
    plt.close()

# Save the summary of losses
plt.figure(figsize=(8, 4))
plt.plot(noise_levels, kan_eval_losses, marker='o', label='Lazy_KAN Losses')
plt.plot(noise_levels, mlp_eval_losses, marker='x', label='MLP Losses')
plt.title('Model Loss vs. Noise Level')
plt.xlabel('Noise Standard Deviation')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_vs_noise_level.png')
plt.close()

# Save the summary of SNRs
plt.figure(figsize=(8, 4))
plt.plot(noise_levels, kan_snrs, marker='o', label='Lazy_KAN SNR')
plt.plot(noise_levels, mlp_snrs, marker='x', label='MLP SNR')
plt.title('Model SNR vs. Noise Level')
plt.xlabel('Noise Standard Deviation')
plt.ylabel('SNR (dB)')
plt.legend()
plt.grid(True)
plt.savefig('snr_vs_noise_level.png')
plt.close()

# Plot training loss curves
plt.figure(figsize=(8, 4))
plt.plot(kan_losses, label='Lazy_KAN Training Loss')
plt.plot(mlp_losses, label='MLP Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_loss_over_epochs.png')
plt.close()
