import torch, math, time, wandb, os, ssl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchkan import KAN

# Fixing SSL error 
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize Weights & Biases for experiment tracking and visualization
wandb.init(project="quantized_model_evaluation", entity="1ssb")

class Trainer:
    def __init__(self, model, device, train_loader, val_loader, optimizer, scheduler, criterion):
        # Initialize the Trainer class with model, device, data loaders, optimizer, scheduler, and loss function
        self.model = model  # Neural network model to be trained and validated
        self.device = device  # Device on which the model will be trained (e.g., 'cuda' or 'cpu')
        self.train_loader = train_loader  # DataLoader for the training dataset
        self.val_loader = val_loader  # DataLoader for the validation dataset
        self.optimizer = optimizer  # Optimizer for adjusting model parameters
        self.scheduler = scheduler  # Learning rate scheduler for the optimizer
        self.criterion = criterion  # Loss function to measure model performance

    def train_epoch(self):
        # Train the model for one epoch and return the average loss and accuracy
        self.model.train()  # Set the model to training mode
        total_loss, total_accuracy = 0, 0  # Initialize accumulators for loss and accuracy
        for images, labels in self.train_loader:
            # Reshape images and move images and labels to the specified device
            images, labels = images.view(-1, 28 * 28).to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()  # Clear previous gradients
            output = self.model(images)  # Forward pass through the model
            loss = self.criterion(output, labels)  # Compute loss between model output and true labels
            loss.backward()  # Backpropagate the loss to compute gradients
            self.optimizer.step()  # Update model parameters
            # Calculate accuracy by comparing predicted and true labels
            accuracy = (output.argmax(dim=1) == labels).float().mean().item()
            # Accumulate total loss and accuracy
            total_loss += loss.item()
            total_accuracy += accuracy
        # Return average loss and accuracy for the epoch
        return total_loss / len(self.train_loader), total_accuracy / len(self.train_loader)

    def validate_epoch(self):
        # Validate the model for one epoch and return the average loss and accuracy
        self.model.eval()  # Set the model to evaluation mode
        val_loss, val_accuracy = 0, 0  # Initialize accumulators for validation loss and accuracy
        with torch.no_grad():  # Disable gradient computation
            for images, labels in self.val_loader:
                # Reshape images and move images and labels to the specified device
                images, labels = images.view(-1, 28 * 28).to(self.device), labels.to(self.device)
                output = self.model(images)  # Forward pass through the model
                # Accumulate validation loss and accuracy
                val_loss += self.criterion(output, labels).item()
                val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()
        # Return average validation loss and accuracy for the epoch
        return val_loss / len(self.val_loader), val_accuracy / len(self.val_loader)

    def fit(self, epochs):
        # Train and validate the model over multiple epochs
        train_accuracies, val_accuracies = [], []  # Lists to store accuracies for each epoch
        pbar = tqdm(range(epochs), desc="Epoch Progress")  # Progress bar to track training progress
        for epoch in pbar:
            # Train and validate for one epoch
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.validate_epoch()
            # Log metrics to Weights & Biases
            wandb.log({"Train Loss": train_loss, "Train Accuracy": train_accuracy, "Validation Loss": val_loss, "Validation Accuracy": val_accuracy})
            # Update progress bar with current epoch loss and accuracy
            pbar.set_description(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
            self.scheduler.step()  # Update learning rate based on the scheduler
            # Store train and validation accuracies
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
        return train_accuracies, val_accuracies

def quantize_and_evaluate(model, val_loader, criterion, save_path):
    # Function to quantize the model, evaluate its performance, and save it
    model.cpu()  # Ensure the model is on the CPU for quantization
    # Quantize the model to reduce size and potentially speed up inference
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # Specify which layers to quantize
        dtype=torch.qint8  # Set the data type for quantized weights
    )
    quantized_model.eval()  # Set the quantized model to evaluation mode
    quantized_val_loss, quantized_val_accuracy = 0, 0  # Initialize accumulators for loss and accuracy
    start_time = time.time()  # Record the start time for evaluation
    with torch.no_grad():  # Disable gradient computation
        for images, labels in val_loader:
            # Reshape images and move images and labels to the CPU
            images, labels = images.view(-1, 28 * 28).to(torch.device('cpu')), labels.to(torch.device('cpu'))
            output = quantized_model(images)  # Forward pass through the quantized model
            # Accumulate validation loss and accuracy for the quantized model
            quantized_val_loss += criterion(output, labels).item()
            quantized_val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()
    evaluation_time = time.time() - start_time  # Calculate total evaluation time
    
    # Create directories if necessary and save the quantized model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(quantized_model.state_dict(), save_path)
    
    return quantized_val_loss / len(val_loader), quantized_val_accuracy / len(val_loader), evaluation_time

def train_and_validate(epochs=15):
    # Function to train, validate, quantize the model, and evaluate the quantized model
    # Define the transformations for the datasets
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # Load and transform the MNIST training dataset
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    # Load and transform the MNIST validation dataset
    valset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    # Create DataLoaders for training and validation datasets
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    # Initialize the KAN model with specified layer sizes
    model = KAN([28 * 28, 64, 10])
    # Determine the appropriate device based on GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the selected device

    # Set up the optimizer with specified parameters
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    # Set the loss function for training and validation
    criterion = nn.CrossEntropyLoss()

    # Initialize the Trainer and train the model
    trainer = Trainer(model, device, trainloader, valloader, optimizer, scheduler, criterion)
    train_accuracies, val_accuracies = trainer.fit(epochs)

    # Ensure the directory for saving models exists
    model_save_dir = "./models"
    os.makedirs(model_save_dir, exist_ok=True)

    # Save the trained model's state dictionary
    torch.save(model.state_dict(), os.path.join(model_save_dir, "original_model.pth"))

    # Quantize and evaluate the quantized model
    quantized_loss, quantized_accuracy, quantized_time = quantize_and_evaluate(model, valloader, criterion, os.path.join(model_save_dir, "quantized_model.pth"))
    print(f"Quantized Model - Validation Loss: {quantized_loss:.4f}, Validation Accuracy: {quantized_accuracy:.4f}, Evaluation Time: {quantized_time:.4f} seconds")

    # Evaluate the time taken to evaluate the original model
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for images, labels in valloader:
            # Reshape images and move them and labels to the selected device
            images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)
            output = model(images)
    original_time = time.time() - start_time  # Calculate the total evaluation time

    # Print the results summary
    print(f"Original Model Evaluation Time: {original_time:.4f} seconds")
    print(f"Train Accuracies: {train_accuracies}")
    print(f"Validation Accuracies: {val_accuracies}")

    wandb.finish()  # Finalize the Weights & Biases run

train_and_validate()  # Call the function to train and evaluate the model
