import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim
import wandb
from torchkan import KAN

# Initialize wandb
wandb.init(project="kan-vs-mlp-multivariate-gaussian-inverse", entity="1ssb")

# Multivariate Gaussian Distribution Function for 2D
def multivariate_gaussian(x, d):
    return torch.exp(-torch.sum(x ** 2, dim=1) / 2) / (np.sqrt(2 * np.pi) ** d)

# Generate training and validation data for 2D Gaussian
def generate_data(num_samples, d):
    x = torch.randn(num_samples, d) * 2
    y = multivariate_gaussian(x, d)
    return x, y

# Simple MLP model matching KAN structure
class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        mlp_layers = []
        for i in range(len(layers) - 1):
            mlp_layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                mlp_layers.append(nn.ReLU())
        self.model = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.model(x)

# Training and validation functions
def train_and_validate_model(model, epochs, learning_rate, train_loader, val_loader, model_name):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            predicted_y = model(x)
            loss = loss_fn(predicted_y, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        wandb.log({f"{model_name} Train Loss": avg_loss})

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                predicted_y = model(x)
                val_loss = loss_fn(predicted_y, y.unsqueeze(1))
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        wandb.log({f"{model_name} Validation Loss": avg_val_loss})
        print(f"Epoch {epoch}, {model_name} Train Loss: {avg_loss}, Validation Loss: {avg_val_loss}")

# Evaluation function
def evaluate_model(model, eval_loader, model_name):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x, y in eval_loader:
            predicted_y = model(x)
            predictions.extend(predicted_y.squeeze().cpu().numpy())
            actuals.extend(y.cpu().numpy())
    return predictions, actuals

# Prepare dataset and loaders
dimension = 2
num_samples = 1000
x_data, y_data = generate_data(num_samples, dimension)
dataset = TensorDataset(x_data, y_data)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define model layers
layers = [dimension, 64, 64, 32, 32, 16, 1]

# Initialize and train the KAN model
kan_model = KAN(layers)
train_and_validate_model(kan_model, epochs=50, learning_rate=0.001, train_loader=train_loader, val_loader=val_loader, model_name="KAN")

# Initialize and train the MLP model
mlp_model = MLP(layers)
train_and_validate_model(mlp_model, epochs=50, learning_rate=0.001, train_loader=train_loader, val_loader=val_loader, model_name="MLP")

# Evaluate both models
kan_predictions, kan_actuals = evaluate_model(kan_model, val_loader, "KAN")
mlp_predictions, mlp_actuals = evaluate_model(mlp_model, val_loader, "MLP")

# Log results to wandb
kan_data = [[pred, act] for pred, act in zip(kan_predictions, kan_actuals)]
mlp_data = [[pred, act] for pred, act in zip(mlp_predictions, mlp_actuals)]
wandb.log({
    "KAN Predictions vs Actuals": wandb.Table(data=kan_data, columns=["KAN Predictions", "Actuals"]),
    "MLP Predictions vs Actuals": wandb.Table(data=mlp_data, columns=["MLP Predictions", "Actuals"])
})

# Save model states
torch.save(kan_model.state_dict(), "kan_multivariate_gaussian_inverse.pth")
torch.save(mlp_model.state_dict(), "mlp_multivariate_gaussian_inverse.pth")
wandb.save("kan_multivariate_gaussian_inverse.pth")
wandb.save("mlp_multivariate_gaussian_inverse.pth")