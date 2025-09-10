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

# CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# FNN
class FNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_fn):
        super(FNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.LayerNorm(hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.layers.append(activation_fn())
            self.layers.append(nn.LayerNorm(hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Physics Loss
def physics_informed_loss(output, inputs, monotonic_factors):
    inputs = inputs.requires_grad_(True)
    do_dxi = torch.autograd.grad(outputs=output, inputs=inputs,
                                  grad_outputs=torch.ones_like(output),
                                  create_graph=True)[0]
    tanh_do_dxi = torch.tanh(do_dxi)
    valid_mask = (monotonic_factors != 0)
    valid_monotonic_factors = monotonic_factors[valid_mask]
    valid_tanh_do_dxi = tanh_do_dxi[valid_mask]
    Ep_elements = 0.5 * (1 - valid_monotonic_factors * valid_tanh_do_dxi)
    Ep = torch.sum(Ep_elements)
    return Ep

# Poad Data
def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath)
    X = df.iloc[:, :6].values
    y = df.iloc[:, 6].values.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Generate Data
def generate_synthetic_data_range1_range3(num_samples_range1=275, num_samples_range3=290):
    synthetic_X1_range1 = np.linspace(0.279149, 4.60139, num_samples_range1)
    synthetic_X2_range1 = np.linspace(100, 6945.09, num_samples_range1)
    synthetic_X3_range1 = np.linspace(0, 0.5, num_samples_range1)
    synthetic_X4_range1 = np.linspace(0, 0.35, num_samples_range1)
    synthetic_X5_range1 = np.linspace(2.69488, 39.11062, num_samples_range1)
    synthetic_X6_range1 = np.linspace(2.40008, 38.06649, num_samples_range1)

    synthetic_X1_range3 = np.linspace(0.279149, 4.60139, num_samples_range3)
    synthetic_X2_range3 = np.linspace(100, 6945.09, num_samples_range3)
    synthetic_X3_range3 = np.linspace(0.5, 2.58052, num_samples_range3)
    synthetic_X4_range3 = np.linspace(0.35, 0.7571, num_samples_range3)
    synthetic_X5_range3 = np.linspace(2.69488, 39.11062, num_samples_range3)
    synthetic_X6_range3 = np.linspace(2.40008, 38.06649, num_samples_range3)

    synthetic_X_range1 = np.column_stack((synthetic_X1_range1, synthetic_X2_range1,
                                          synthetic_X3_range1, synthetic_X4_range1,
                                          synthetic_X5_range1, synthetic_X6_range1))
    synthetic_X_range3 = np.column_stack((synthetic_X1_range3, synthetic_X2_range3,
                                          synthetic_X3_range3, synthetic_X4_range3,
                                          synthetic_X5_range3, synthetic_X6_range3))

    synthetic_X = np.vstack((synthetic_X_range1, synthetic_X_range3))

    monotonic_factors = np.zeros_like(synthetic_X)
    monotonic_factors[:, 2] = np.where(synthetic_X[:, 2] <= 0.5, 1, -1)
    monotonic_factors[:, 3] = np.where(synthetic_X[:, 3] <= 0.35, 1, -1)

    return synthetic_X, monotonic_factors

# Train Model
def train_model(model, train_loader, val_loader, test_loader, optimizer, loss_fn, epochs, device, lambda_p):
    model.to(device)
    train_losses, physics_losses, total_losses = [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss, physics_loss, total_loss = 0.0, 0.0, 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad_(True)
            optimizer.zero_grad()
            outputs = model(inputs)
            mse_loss = loss_fn(outputs, targets)

            synthetic_X, monotonic_factors = generate_synthetic_data_range1_range3()
            synthetic_X = torch.tensor(synthetic_X, dtype=torch.float32).to(device).requires_grad_(True)
            monotonic_factors = torch.tensor(monotonic_factors, dtype=torch.float32).to(device)
            synthetic_outputs = model(synthetic_X)
            pi_loss = physics_informed_loss(synthetic_outputs, synthetic_X, monotonic_factors)

            loss = mse_loss + lambda_p * pi_loss
            loss.backward()
            optimizer.step()

            train_loss += mse_loss.item()
            physics_loss += pi_loss.item()
            total_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        physics_losses.append(physics_loss / len(train_loader))
        total_losses.append(total_loss / len(train_loader))

        model.eval()
        val_predictions, val_targets = [], []
        test_predictions, test_targets = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_predictions.extend(outputs.view(-1).tolist())
                val_targets.extend(targets.view(-1).tolist())

            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_predictions.extend(outputs.view(-1).tolist())
                test_targets.extend(targets.view(-1).tolist())

        val_mae = mean_absolute_error(val_targets, val_predictions)
        val_r2 = r2_score(val_targets, val_predictions)
        val_rmse = sqrt(mean_squared_error(val_targets, val_predictions))

        test_mae = mean_absolute_error(test_targets, test_predictions)
        test_r2 = r2_score(test_targets, test_predictions)
        test_rmse = sqrt(mean_squared_error(test_targets, test_predictions))

        print(f'Epoch {epoch + 1}/{epochs}, '
              f'Total Loss: {total_loss / len(train_loader):.4f}, '
              f'Physics Loss: {physics_loss / len(train_loader):.4f}, '
              f'Train MSE Loss: {train_loss / len(train_loader):.4f}, '
              f'Val MAE: {val_mae:.4f}, Val R²: {val_r2:.4f}, Val RMSE: {val_rmse:.4f}, '
              f'Test MAE: {test_mae:.4f}, Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}')

    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_losses, label='Train Loss', color='blue')
    plt.plot(range(epochs), physics_losses, label='Physics Loss', color='green')
    plt.plot(range(epochs), total_losses, label='Total Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves During Training')
    plt.legend()
    plt.grid(True)
    plt.show()

    return val_targets, val_predictions, test_targets, test_predictions

# Parameter Settings
input_size = 6
hidden_sizes = [128, 64, 32]
output_size = 1
learning_rate = 0.001
epochs = 80
lambda_p = 1

# Load Data
filepath = 'Data.xlsx'
X, y = load_and_preprocess_data(filepath)

# Dataset Split
dataset = TensorDataset(X, y)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize Model
model = FNN(input_size, hidden_sizes, output_size, nn.LeakyReLU)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Train Model
val_targets, val_predictions, test_targets, test_predictions = train_model(
    model, train_loader, val_loader, test_loader, optimizer, loss_fn, epochs, device, lambda_p
)

# Save Trained Model
torch.save(model.state_dict(), 'AV_VF_monotonic_model.pth')

# Save Synthetic Data (only retain Range 1 and Range 3)
synthetic_X, monotonic_factors = generate_synthetic_data_range1_range3()
synthetic_X_tensor = torch.tensor(synthetic_X, dtype=torch.float32).to(device)
synthetic_outputs = model(synthetic_X_tensor).detach().cpu().numpy()

synthetic_data_df = pd.DataFrame(synthetic_X, columns=[f'Feature{i+1}' for i in range(synthetic_X.shape[1])])
synthetic_data_df['Output'] = synthetic_outputs
synthetic_data_df.to_excel('synthetic_data_range1_range3.xlsx', index=False)
