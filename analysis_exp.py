import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import itertools, time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from datetime import datetime

class KANvolver(nn.Module):
    def __init__(self, layers_hidden, polynomial_order=2, base_activation='SiLU'):
        super(KANvolver, self).__init__()
        self.polynomial_order = polynomial_order
        # Activation functions mapping
        activation_dict = {
            'ReLU': nn.ReLU,
            'GELU': nn.GELU,
            'ELU': nn.ELU,
            'Tanh': nn.Tanh,
            'SiLU': nn.SiLU,
            'PReLU': nn.PReLU,
            'Softplus': nn.Softplus,
            'Mish': nn.Mish,
            'GLU': nn.GLU,
            'Softsign': nn.Softsign
        }
        # Use the dictionary to select the activation class
        activation_class = activation_dict.get(base_activation, nn.SiLU)
        # Create an instance of the activation function
        self.base_activation = activation_class()
        # Feature extractor with Convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Calculate flat features
        flat_features = 32 * 7 * 7
        self.layers_hidden = [flat_features] + layers_hidden

        self.base_weights = nn.ModuleList()
        self.poly_weights = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Initialize weights and batch norms for each layer
        for in_features, out_features in zip(self.layers_hidden[:-1], self.layers_hidden[1:]):
            self.base_weights.append(nn.Linear(in_features, out_features))
            self.poly_weights.append(nn.Linear(in_features * (polynomial_order + 1), out_features))
            self.batch_norms.append(nn.BatchNorm1d(out_features))

    def compute_efficient_monomials(self, x, order):
        powers = torch.arange(order + 1, device=x.device, dtype=x.dtype)
        x_expanded = x.unsqueeze(-1).repeat(1, 1, order + 1)
        return torch.pow(x_expanded, powers)

    def forward(self, x):
        # Reshaping the inputs
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

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Main experiment running function
def run_experiment(config, log_file):
    print("Starting Experiment......")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='/home/projects/Rudra_Generative_Robotics/torchkan/data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='/home/projects/Rudra_Generative_Robotics/torchkan/data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    model = KANvolver(layers_hidden=config['layers_hidden'], polynomial_order=config['polynomial_order'],
                      base_activation=config['base_activation'])
    model.to(config['device'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    print("Starting Training....")
    
    # Train the model and log progress
    train_start = time.time()
    for epoch in range(config['epochs']):
        model.train()
        total_loss, total_samples = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(config['device']), labels.to(config['device'])
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
        avg_loss = total_loss / total_samples
        log_file.write(f'Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}\n')
    train_end = time.time()
    
    print("Completed Training")
    
    # Evaluate the model and record time
    eval_start = time.time()
    model.eval()
    y_true, y_pred = [], []
    all_monomials = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(config['device']), labels.to(config['device'])
            outputs, monomials = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # Filter data outside the expected range (0-9)
            valid_indices = [i for i in range(len(labels)) if labels[i] < 10]
            y_true.extend(labels[valid_indices].cpu().numpy())
            y_pred.extend(predicted[valid_indices].cpu().numpy())
            all_monomials.extend([m.cpu().numpy() for m in monomials])
    eval_end = time.time()

    try:
        # Compute metrics safely
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)])
    except ValueError as e:
        print(f"Error in metric computation: {e}")
        precision, recall, fscore, accuracy = 0.0, 0.0, 0.0, 0.0
        report = "Metric computation failed."
    
    print("Logging results...")
    
    # Log results
    log_file.write(f"Train Time: {train_end - train_start:.2f} seconds\n")
    log_file.write(f"Evaluation Time: {eval_end - eval_start:.2f} seconds\n")
    log_file.write(f"Test Accuracy: {accuracy:.4f}\n")
    log_file.write(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F-Score: {fscore:.4f}\n")
    log_file.write(f"Classification Report:\n{report}\n")
    
    # Plotting monomials
    plt.figure(figsize=(15, 7))
    for i in range(len(all_monomials[0])):
        plt.plot(all_monomials[0][i], label=f'Monomial {i}')
    plt.title(f'Monomial Outputs for {config["base_activation"]}')
    plt.legend()
    plt.savefig(f'/home/projects/Rudra_Generative_Robotics/torchkan/analysis/monomial_outputs_{config["base_activation"]}_{datetime.now().strftime("%Y%m%d%H%M%S")}.png')
    plt.close()
    
    print("Logged, plotting confusion....")
    # Save and plot confusion matrix
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cm, classes=[str(i) for i in range(10)])
    plt.savefig(f'/home/projects/Rudra_Generative_Robotics/torchkan/analysis/confusion_matrix_{config["base_activation"]}_{datetime.now().strftime("%Y%m%d%H%M%S")}.png')
    plt.close()
    print("Completed plots....")
    
    return accuracy

def main():
    # Ensure the analysis directory exists
    os.makedirs('/home/projects/Rudra_Generative_Robotics/torchkan/analysis', exist_ok=True)

    # Main experiment loop
    activation_functions = ['ReLU', 'GELU', 'ELU', 'Tanh', 'SiLU', 'PReLU', 'Softplus', 'Mish', 'GLU', 'Softsign']
    polynomial_orders = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    batch_sizes = [32, 64, 128]
    learning_rates = [0.1, 0.01, 0.05, 0.001, 0.0005]
    results = []

    with open('/home/projects/Rudra_Generative_Robotics/torchkan/analysis/experiment_log.log', 'w') as log_file:
        for act_func, poly_order, batch_size, learning_rate in itertools.product(activation_functions, polynomial_orders, batch_sizes, learning_rates):
            config = {
                'batch_size': batch_size,
                'epochs': 25,
                'learning_rate': learning_rate,
                'layers_hidden': [128, 64],
                'polynomial_order': poly_order,
                'base_activation': act_func,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }

            log_file.write(f"Running experiment with config: {config}\n")
            accuracy = run_experiment(config, log_file)
            results.append((config, accuracy))
            log_file.write(f"Config: {config}, Test Accuracy: {accuracy:.4f}\n")
            log_file.write('-' * 80 + '\n')
    
    print("Analysis Completed, writing summary...")
    
    # Summarize results
    with open('/home/projects/Rudra_Generative_Robotics/torchkan/analysis/summary_results.log', 'w') as summary_file:
        for config, accuracy in results:
            summary_file.write(f"Config: {config}, Test Accuracy: {accuracy:.4f}\n")
    
    print("All completed!")

if __name__ == "__main__":
    main()

