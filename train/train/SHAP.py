import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, random_split
import shap
import matplotlib.pyplot as plt

# Configure font to Times New Roman (journal style)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

# Define Feedforward Neural Network (FNN) structure
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

# Load and normalize dataset
def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath)
    X = df.iloc[:, :6].values
    y = df.iloc[:, 6].values.reshape(-1, 1)

    feature_names = [
        r"$\text{Density}$",
        r"$\text{GSA}$",
        r"$\text{AV}$",
        r"$\text{V}_\text{f}$",
        r"$\text{LCD}$",
        r"$\text{PLD}$"
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), feature_names, scaler

# SHAP analysis function
def shap_analysis(model, X_val_tensor, feature_names, device, background_data_size=100, font_size=None):
    model.eval()
    X_val_np = X_val_tensor.cpu().numpy()

    background_data = X_val_tensor[:background_data_size].to(device)
    explainer = shap.GradientExplainer(model, background_data)
    shap_values = explainer.shap_values(X_val_tensor.to(device))  # 无 check_additivity

    shap_values = np.squeeze(shap_values)

    if font_size is None:
        font_size = {
            "axis_label": 14,
            "ticks": 12,
            "colorbar_label": 14,
            "feature_name": 14
        }

    # Calculate feature importance percentages
    shap_values_abs_mean = np.mean(np.abs(shap_values), axis=0)
    total_importance = np.sum(shap_values_abs_mean)
    feature_importance_percentage = (shap_values_abs_mean / total_importance) * 100
    feature_importance = dict(zip(feature_names, feature_importance_percentage))

    print("Feature Importance (Percentage):")
    for feature, importance in feature_importance.items():
        print(f"{feature}: {importance:.2f}%")

    # SHAP scatter plot
    shap.summary_plot(shap_values, X_val_np, feature_names=feature_names, show=False)
    plt.gcf().axes[-1].tick_params(labelsize=font_size["ticks"])
    plt.gcf().axes[-1].set_ylabel("Feature value", fontsize=font_size["colorbar_label"])
    plt.gca().tick_params(labelsize=font_size["ticks"])
    plt.gca().set_xlabel("SHAP value (impact on model output)", fontsize=font_size["axis_label"])
    plt.savefig('shap_scatter_plot_with_latex.png', dpi=600)
    plt.show()

    # SHAP scatter plot
    shap.summary_plot(shap_values, X_val_np, plot_type="bar", feature_names=feature_names, show=False)
    plt.gcf().axes[-1].tick_params(labelsize=font_size["ticks"])
    plt.gca().tick_params(labelsize=font_size["ticks"])
    plt.gca().set_xlabel("SHAP value (impact on model output)", fontsize=font_size["axis_label"])
    plt.savefig('shap_bar_plot_with_latex.png', dpi=600)
    plt.show()

# Basic parameters
input_size = 6
hidden_sizes = [128, 64, 32]
output_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model parameters
model = FNN(input_size, hidden_sizes, output_size, nn.LeakyReLU)
model.load_state_dict(torch.load('AV_VF_monotonic_model.pth', map_location=device))
model.to(device)

# Load dataset
filepath = 'Data.xlsx'
X, y, feature_names, _ = load_and_preprocess_data(filepath)

# Split dataset
dataset = TensorDataset(X, y)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Extract validation data
X_val, y_val = zip(*val_dataset)
X_val = torch.stack(X_val).to(device)

# Font configuration
custom_font_size = {
    "axis_label": 50,
    "ticks": 48,
    "colorbar_label": 48,
    "feature_name": 48
}

# Run SHAP analysis
shap_analysis(model, X_val, feature_names, device, background_data_size=100, font_size=custom_font_size)
