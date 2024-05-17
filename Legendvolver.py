import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb, os

wandb.init(project="KANvolution_evaluation", entity="1ssb")

class Legendvolver(nn.Module):
    def __init__(self, input_channels, layers_hidden, polynomial_order=3, base_activation=nn.SiLU):
        super(Legendvolver, self).__init__()
        
        self.input_channels = input_channels
        self.layers_hidden = layers_hidden
        self.polynomial_order = polynomial_order
        self.base_activation = base_activation()

        # Convolutional encoder for initial feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Compute the output size of conv layers to bridge to fully connected layers
        self._compute_flat_features()

        # Add parameters and normalization for polynomial-based hidden layers
        self.base_weights = nn.ParameterList()
        self.poly_weights = nn.ParameterList()
        self.layer_norms = nn.ModuleList()

        # Initialize the network for each layer pair in layers_hidden
        for i, (in_features, out_features) in enumerate(zip([self.flat_features] + layers_hidden[:-1], layers_hidden)):
            self.base_weights.append(nn.Parameter(torch.randn(out_features, in_features)))
            self.poly_weights.append(nn.Parameter(torch.randn(out_features, in_features * (polynomial_order + 1))))
            self.layer_norms.append(nn.LayerNorm(out_features))

        # Initialize weights
        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for weight in self.poly_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

    def _compute_flat_features(self):
        # Dummy input to determine the size of the flattened features after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, 28, 28)
            dummy_output = self.conv_layers(dummy_input)
            self.flat_features = int(torch.numel(dummy_output) / dummy_output.size(0))

    def compute_legendre_polynomials(self, x, order):
        # Efficiently compute Legendre polynomials within PyTorch framework
        P0 = torch.ones_like(x)
        if order == 0:
            return P0.unsqueeze(-1)
        P1 = x
        legendre_polys = [P0, P1]

        for n in range(1, order):
            Pn = ((2.0 * n + 1.0) * x * legendre_polys[-1] - n * legendre_polys[-2]) / (n + 1.0)
            legendre_polys.append(Pn)

        return torch.stack(legendre_polys, dim=-1)

    def forward(self, x):
        x = x.to(self.base_weights[0].device)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)

        for i, (base_weight, poly_weight, layer_norm) in enumerate(zip(self.base_weights, self.poly_weights, self.layer_norms)):
            base_output = F.linear(self.base_activation(x), base_weight)
            x_normalized = 2 * (x - x.min(dim=1, keepdim=True)[0]) / (x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0]) - 1
            legendre_basis = self.compute_legendre_polynomials(x_normalized, self.polynomial_order)
            legendre_basis = legendre_basis.view(x.size(0), -1)
            poly_output = F.linear(legendre_basis, poly_weight)
            x = self.base_activation(layer_norm(base_output + poly_output))

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

    model = Legendvolver(input_channels=1, layers_hidden=[256, 128, 10], polynomial_order=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, device, trainloader, valloader, optimizer, scheduler, criterion)
    train_accuracies, val_accuracies = trainer.fit(epochs)

    model_save_dir = "./models"
    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_save_dir, "trained_Legendvolver_model.pth"))

    print(f"Train Accuracies: {train_accuracies}")
    print(f"Validation Accuracies: {val_accuracies}")

    wandb.finish()

if __name__ == "__main__":
    train_and_validate()
