import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

set_seed()


def generate_garch_data(num_samples=50000):
    alpha_1_vals = np.random.uniform(0, 0.3, num_samples)  # Constrained for stationarity
    beta_1_vals = np.random.uniform(0, 0.6, num_samples)
    alpha_0_vals = np.random.uniform(1e-6, 1e-3, num_samples)

    E_x2_vals = alpha_0_vals / (1 - alpha_1_vals - beta_1_vals)
    
    Gamma4_vals = 3 + (6 * alpha_1_vals**2) / (1 - 3 * alpha_1_vals**2 - 2 * alpha_1_vals * beta_1_vals - beta_1_vals**2)
    Gamma6_vals = 15 * (1 - alpha_1_vals - beta_1_vals)**3 * (
        1 + 3 * (alpha_1_vals + beta_1_vals) / (1 - alpha_1_vals - beta_1_vals) +
        3 * (1 + 2 * (alpha_1_vals + beta_1_vals) / (1 - alpha_1_vals - beta_1_vals)) *
        (beta_1_vals**2 + 2 * alpha_1_vals * beta_1_vals + 3 * alpha_1_vals**2) /
        (1 - 3 * alpha_1_vals**2 - 2 * alpha_1_vals * beta_1_vals - beta_1_vals**2)
    ) / (1 - 15 * alpha_1_vals**3 - 9 * alpha_1_vals**2 * beta_1_vals - 3 * alpha_1_vals * beta_1_vals**2 - beta_1_vals**3)

    X = np.stack((Gamma4_vals, Gamma6_vals, E_x2_vals), axis=1)
    y = alpha_1_vals

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ------------------------
# Dataset Class
# ------------------------

class GARCHDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GARCHNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super(GARCHNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output is a single value: α1
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Plot training curve
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

# Generate dataset
X, y = generate_garch_data()
dataset = GARCHDataset(X, y)

# Split into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize and train the model
model = GARCHNet()
train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001)
