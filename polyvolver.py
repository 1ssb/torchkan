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

class PolynomialActivation(nn.Module):
    def __init__(self, max_degree):
        super(PolynomialActivation, self).__init__()
        # Coefficients for the polynomial terms
        self.coefficients = nn.Parameter(torch.randn(max_degree + 1))

    def forward(self, x):
        # Compute polynomial activation
        output = sum(self.coefficients[i] * torch.pow(x, i) for i in range(len(self.coefficients)))
        return output

class Polyvolver(nn.Module):
    def __init__(self, polynomial_degree=2):
        super(Polyvolver, self).__init__()

        # Convolutional encoder to extract features from images
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Calculate the size of the flattened features after convolutions
        self.flat_features = 64 * 3 * 3  # Adjust based on the output size of feature_extractor

        # Polynomial activation function
        self.poly_activation = PolynomialActivation(polynomial_degree)

        # Single linear layer after polynomial activation
        self.final_layer = nn.Linear(self.flat_features, 10)  # Output size is 10 for MNIST

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten the output from the convolutional layers

        # Apply the polynomial activation
        x = self.poly_activation(x)

        # Apply the single linear layer
        x = self.final_layer(x)
        return x

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
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            accuracy = (outputs.argmax(dim=1) == labels).float().mean().item()
            total_loss += loss.item()
            total_accuracy += accuracy
        return total_loss / len(self.train_loader), total_accuracy / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        val_loss, val_accuracy = 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
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

def train_and_validate(epochs=15):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    valset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    model = Polyvolver(polynomial_degree=2)
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

    wandb.finish()

if __name__ == "__main__":
    train_and_validate()
