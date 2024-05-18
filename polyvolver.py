import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from torchvision.transforms.functional import to_pil_image

# Initialize Weights & Biases for tracking and visualizing the model's performance.
wandb.init(project="KANvolution_evaluation", entity="1ssb")

class PolynomialActivation(nn.Module):
    def __init__(self, max_degree):
        super(PolynomialActivation, self).__init__()
        self.coefficients = nn.Parameter(torch.randn(max_degree + 1))

    def forward(self, x):
        # Calculate polynomial activation for each coefficient
        output = sum(self.coefficients[i] * torch.pow(x, i) for i in range(len(self.coefficients)))
        return output

class Polyvolver(nn.Module):
    def __init__(self, polynomial_degree=2):
        super(Polyvolver, self).__init__()

        # Define the architecture of the model
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
        self.flat_features = 64 * 3 * 3  # Calculate the flattened features after pooling
        self.poly_activation = PolynomialActivation(polynomial_degree)
        self.final_layer = nn.Linear(self.flat_features, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.poly_activation(x)
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
        self.integrated_gradients = IntegratedGradients(model)

    def train_epoch(self):
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

    def visualize_integrated_gradients(self, images, labels, epoch):
        integrated_save_dir = './integrated'
        if not os.path.exists(integrated_save_dir):
            os.makedirs(integrated_save_dir)
        
        for i in range(len(images)):
            img = images[i].unsqueeze(0).to(self.device)
            img.requires_grad = True
            label = labels[i].unsqueeze(0).to(self.device)

            ig_attr = self.integrated_gradients.attribute(img, target=label.item())

            # Convert to PIL image for easier visualization
            original_img = to_pil_image(images[i].cpu())
            
            # Normalize integrated gradients for visualization
            ig_attr = ig_attr.squeeze().detach().cpu()
            ig_attr = (ig_attr - ig_attr.min()) / (ig_attr.max() - ig_attr.min())
            ig_img = to_pil_image(ig_attr, mode='L')
            
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(original_img, cmap='gray')
            ax[0].set_title('Sample')
            ax[0].axis('off')

            # Display absolute values of integrated gradients to ensure visibility of inhibitory and excitatory influences
            ig_img = np.abs(ig_attr.squeeze().detach().cpu().numpy())

            # Display integrated gradients image with colorbar
            im = ax[1].imshow(ig_img, cmap='hot', vmin=np.min(ig_img), vmax=np.max(ig_img))
            ax[1].set_title('Integrated Gradients')
            ax[1].axis('off')

            # Add a colorbar to the right of the integrated gradients image
            cbar = fig.colorbar(im, ax=ax[1], orientation='vertical')
            cbar.set_label('Attribution Magnitude')

            # Add the epoch number at the top of the figure
            plt.suptitle(f'Epoch {epoch + 1}', fontsize=16)

            # Adjust the layout
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(integrated_save_dir, f'integrated_grads_epoch_{epoch + 1}_img_{i + 1}.png'))
            plt.close()

    def fit(self, epochs):
        train_accuracies, val_accuracies = [], []
        pbar = tqdm(range(epochs), desc="Epoch Progress")
        for epoch in pbar:
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.validate_epoch()
            wandb.log({
                "Train Loss": train_loss,
                "Train Accuracy": train_accuracy,
                "Validation Loss": val_loss,
                "Validation Accuracy": val_accuracy
            })
            images, labels = next(iter(self.val_loader))
            self.visualize_integrated_gradients(images[:5], labels[:5], epoch)
            pbar.set_description(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
            self.scheduler.step()
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
        return train_accuracies, val_accuracies

def train_and_validate(epochs):
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
    torch.save(model.state_dict(), os.path.join(model_save_dir, "trained_polyvolver_model.pth"))

    print(f"Train Accuracies: {train_accuracies}")
    print(f"Validation Accuracies: {val_accuracies}")

    wandb.finish()

if __name__ == "__main__":
    train_and_validate(50)