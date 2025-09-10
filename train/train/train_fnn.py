# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# Define Feedforward Neural Network (FNN) class
class FNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_fn):
        super(FNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.LayerNorm(hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(activation_fn())
            self.layers.append(nn.LayerNorm(hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Function to load and preprocess data
def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath)
    X = df.iloc[:, :6].values
    y = df.iloc[:, 6].values.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Model training function
def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_predictions.extend(outputs.view(-1).tolist())
                val_targets.extend(targets.view(-1).tolist())

        mae = mean_absolute_error(val_targets, val_predictions)
        r2 = r2_score(val_targets, val_predictions)
        rmse = sqrt(mean_squared_error(val_targets, val_predictions))
        r = np.corrcoef(val_targets, val_predictions)[0, 1]
        pcc = np.corrcoef(np.array(val_targets).flatten(), np.array(val_predictions).flatten())[0, 1]

        print(
            f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, Val MAE: {mae}, Val R²: {r2}, Val RMSE: {rmse}, Val R: {r}, Val PCC: {pcc}')

    return val_targets, val_predictions, mae, r2, rmse, r, pcc

# Set model parameters
input_size = 6
hidden_sizes = [128, 64, 32]
output_size = 1
learning_rate = 0.001
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, optimizer, and loss function
model = FNN(input_size, hidden_sizes, output_size, nn.LeakyReLU)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Load and preprocess dataset
filepath = 'Data.xlsx'
X, y = load_and_preprocess_data(filepath)

# Split dataset into training and validation sets
dataset = TensorDataset(X, y)
train_size = int(0.75 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Train the model and return final validation results
val_targets, val_predictions, mae, r2, rmse, r, pcc = train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs, device)

# Save the trained model
torch.save(model.state_dict(), 'fnn_train.pth')
