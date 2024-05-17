import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

# Initialize Weights & Biases for tracking and visualizing the model's performance.
wandb.init(project="KANvolution_evaluation", entity="1ssb")

class KANvolver(nn.Module):
    def __init__(self, layers_hidden, polynomial_order=2):
        super(KANvolver, self).__init__()
        self.polynomial_order = polynomial_order
        self.base_activation = nn.SiLU()

        # Convolutional layers for initial feature extraction using nn.SiLU consistently.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2, 2)
        )
        flat_features = 32 * 7 * 7
        self.layers_hidden = [flat_features] + layers_hidden
        self.base_weights = nn.ModuleList()
        self.poly_weights = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for in_features, out_features in zip(self.layers_hidden[:-1], self.layers_hidden[1:]):
            self.base_weights.append(nn.Linear(in_features, out_features))
            self.poly_weights.append(nn.Linear(in_features * (polynomial_order + 1), out_features))
            self.batch_norms.append(nn.BatchNorm1d(out_features))

    def compute_efficient_monomials(self, x, order):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure x is at least 2D (batch size by features)
        powers = torch.arange(order + 1, device=x.device, dtype=x.dtype)
        x_expanded = x.unsqueeze(-1)
        return torch.pow(x_expanded, powers)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        monomial_outputs = []
        
        for base_weight, poly_weight, batch_norm in zip(self.base_weights, self.poly_weights, self.batch_norms):
            base_output = base_weight(x)
            monomial_basis = self.compute_efficient_monomials(x, self.polynomial_order)
            monomial_basis = monomial_basis.view(x.size(0), -1)
            poly_output = poly_weight(monomial_basis)
            x = self.base_activation(batch_norm(base_output + poly_output))
            monomial_outputs.append(monomial_basis)
        return x, monomial_outputs

class Trainer:
    def __init__(self, model, device, train_loader, val_loader, optimizer, scheduler, criterion):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, total_accuracy = 0, 0
        for images, labels in self.train_loader:
            images, labels = images.view(-1, 28 * 28).to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs, monomials = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            accuracy = (outputs.argmax(dim=1) == labels).float().mean().item()
            total_loss += loss.item()
            total_accuracy += accuracy

        # Save monomial visualizations at the end of each epoch
        visualize_monomials(self.model, (1, 28, 28), self.model.polynomial_order, epoch=epoch)

        return total_loss / len(self.train_loader), total_accuracy / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        val_loss, val_accuracy = 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.view(-1, 28 * 28).to(self.device), labels.to(self.device)
                outputs, _ = self.model(images)
                val_loss += self.criterion(outputs, labels).item()
                val_accuracy += (outputs.argmax(dim=1) == labels).float().mean().item()
        return val_loss / len(self.val_loader), val_accuracy / len(self.val_loader)

    def fit(self, epochs):
        train_accuracies, val_accuracies = [], []
        pbar = tqdm(range(epochs), desc="Epoch Progress")
        for epoch in pbar:
            train_loss, train_accuracy = self.train_epoch(epoch)
            val_loss, val_accuracy = self.validate_epoch()
            wandb.log({
                "Train Loss": train_loss, 
                "Train Accuracy": train_accuracy,
                "Validation Loss": val_loss, 
                "Validation Accuracy": val_accuracy
            })
            pbar.set_description(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
            self.scheduler.step()
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
        return train_accuracies, val_accuracies

def visualize_monomials(model, input_shape, order, sample_count=5, epoch=None):
    save_dir = "./monomial_visualization"
    os.makedirs(save_dir, exist_ok=True)
    
    sample_inputs = torch.randn(sample_count, *input_shape)
    monomials = [model.compute_efficient_monomials(sample, order) for sample in sample_inputs]
    
    for i, mono in enumerate(monomials):
        plt.figure(figsize=(6, 4))
        plt.imshow(mono[0].detach().numpy(), cmap='hot', interpolation='nearest')
        plt.title(f'Epoch {epoch} Sample {i+1}')
        plt.colorbar()
        plt.tight_layout()
        # Save each plot to a file in the specified directory
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_monomial_{i+1}.png"))
        plt.close()  # Close the figure to free up memory

def train_and_validate(epochs=15):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    valset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    model = KANvolver([28 * 28, 64, 10])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, device, trainloader, valloader, optimizer, scheduler, criterion)
    train_accuracies, val_accuracies = trainer.fit(epochs)

    model_save_dir = "./models"
    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_save_dir, "trained_mononet_model.pth"))

    print(f"Train Accuracies: {train_accuracies}")
    print(f"Validation Accuracies: {val_accuracies}")

    # Visualize monomials after full training
    visualize_monomials(model, (1, 28, 28), model.polynomial_order)

    wandb.finish()

if __name__ == "__main__":
    train_and_validate()
