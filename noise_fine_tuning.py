# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import seaborn as sns


# %% [markdown]
# ## Load and preprocess dataset

# %%
# Fetch dataset
phishing_websites = fetch_ucirepo(id=327)
X = phishing_websites.data.features
y = phishing_websites.data.targets

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train.values.astype(float), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.astype(float), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values.astype(float), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values.astype(float), dtype=torch.float32)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# %% [markdown]
# ## Define the neural network model

# %%
class MalwareDetector(nn.Module):
    def __init__(self):
        super(MalwareDetector, self).__init__()
        self.fc1 = nn.Linear(30, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

model = MalwareDetector()

# %% [markdown]
# ## Define training and evaluation functions

# %%
def train_model(model, criterion, optimizer, train_loader, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            labels = (labels == 1).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred) * 100
    print(f'Accuracy: {accuracy:.2f}%')

# %% [markdown]
# ## Add noise to model weights

# %%
def add_noise_to_weights(model, noise_factor):
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn(param.size()) * noise_factor
            param.add_(noise)
    print(f"Added noise with factor {noise_factor} to model weights.")

# %% [markdown]
# ## Tune noise hyperparameters

# %%
def tune_noise(model, test_loader, noise_factors):
    best_noise_factor = None
    best_accuracy = 0
    for noise_factor in noise_factors:
        original_state_dict = model.state_dict()
        add_noise_to_weights(model, noise_factor)
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        accuracy = accuracy_score(y_true, y_pred) * 100
        print(f"Noise Factor: {noise_factor}, Accuracy: {accuracy:.2f}%")
        if accuracy > best_accuracy:
            best_noise_factor = noise_factor
            best_accuracy = accuracy
        model.load_state_dict(original_state_dict)
    print(f"Best Noise Factor: {best_noise_factor}")
    print(f"Best Accuracy: {best_accuracy:.2f}%")
    return best_noise_factor

# %% [markdown]
# ## Train and evaluate the model

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
NUM_EPOCHS = 50

# Train the model
train_model(model, criterion, optimizer, train_loader, num_epochs=NUM_EPOCHS)

# Evaluate the model
evaluate_model(model, test_loader)

# %% [markdown]
# ## Find the best noise factor

# %%
noise_factors = [0.001, 0.01, 0.1, 0.5, 1.0]
best_noise_factor = tune_noise(model, test_loader, noise_factors)
print('Best Noise Factor: ', best_noise_factor)

# %% [markdown]
# ## Evaluate model with the best noise factor

# %%
def test_with_best_noise(model, test_loader, best_noise_factor):
    original_state_dict = model.state_dict()
    add_noise_to_weights(model, best_noise_factor)
    evaluate_model(model, test_loader)
    model.load_state_dict(original_state_dict)
    print("Restored original model weights.")

test_with_best_noise(model, test_loader, best_noise_factor)

# %% [markdown]
# ## Add different types of noise to model weights

# %%
def add_salt_and_pepper_noise(model, noise_factor):
    with torch.no_grad():
        for param in model.parameters():
            mask = torch.rand(param.size()) < noise_factor
            param[mask] = torch.rand(mask.sum().item())
    print(f"Added salt and pepper noise with factor {noise_factor} to model weights.")

def add_gaussian_noise(model, noise_factor):
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn(param.size()) * noise_factor
            param.add_(noise)
    print(f"Added Gaussian noise with factor {noise_factor} to model weights.")

# %% [markdown]
# ## Tune different noise hyperparameters

# %%
def tune_different_noises(model, test_loader, noise_factors, noise_type):
    best_noise_factor = None
    best_accuracy = 0
    for noise_factor in noise_factors:
        original_state_dict = model.state_dict()
        if noise_type == 'salt_and_pepper':
            add_salt_and_pepper_noise(model, noise_factor)
        elif noise_type == 'gaussian':
            add_gaussian_noise(model, noise_factor)
        else:
            add_noise_to_weights(model, noise_factor)
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        accuracy = accuracy_score(y_true, y_pred) * 100
        print(f"Noise Type: {noise_type}, Noise Factor: {noise_factor}, Accuracy: {accuracy:.2f}%")
        if accuracy > best_accuracy:
            best_noise_factor = noise_factor
            best_accuracy = accuracy
        model.load_state_dict(original_state_dict)
    print(f"Best Noise Type: {noise_type}, Best Noise Factor: {best_noise_factor}")
    print(f"Best Accuracy: {best_accuracy:.2f}%")
    return best_noise_factor

# %% [markdown]
# ## Find the best noise factor for different noise types

# %%
noise_factors = [0.001, 0.01, 0.1, 0.5, 1.0]
noise_types = ['salt_and_pepper', 'gaussian', 'default']
best_noise_factors = {}
for noise_type in noise_types:
    best_noise_factor = tune_different_noises(model, test_loader, noise_factors, noise_type)
    best_noise_factors[noise_type] = best_noise_factor
print('Best Noise Factors: ', best_noise_factors)

# %% [markdown]
# ## Evaluate model with the best noise factors

# %%
def test_with_best_noises(model, test_loader, best_noise_factors):
    for noise_type, noise_factor in best_noise_factors.items():
        print(f"Testing with best noise type: {noise_type}, factor: {noise_factor}")
        original_state_dict = model.state_dict()
        if noise_type == 'salt_and_pepper':
            add_salt_and_pepper_noise(model, noise_factor)
        elif noise_type == 'gaussian':
            add_gaussian_noise(model, noise_factor)
        else:
            add_noise_to_weights(model, noise_factor)
        evaluate_model(model, test_loader)
        model.load_state_dict(original_state_dict)
        print("Restored original model weights.")

test_with_best_noises(model, test_loader, best_noise_factors)

# %% [markdown]
# ## Evaluate model with both types of noise

# %%
def test_with_both_noises(model, test_loader, best_noise_factors):
    original_state_dict = model.state_dict()
    print("Testing with both noise types: salt_and_pepper and gaussian")
    add_salt_and_pepper_noise(model, best_noise_factors['salt_and_pepper'])
    add_gaussian_noise(model, best_noise_factors['gaussian'])
    evaluate_model(model, test_loader)
    model.load_state_dict(original_state_dict)
    print("Restored original model weights.")

test_with_both_noises(model, test_loader, best_noise_factors)

# %% [markdown]
# ## Final test: Compare model performance with and without noise

# %%
def final_test_comparison(model, test_loader, best_noise_factors):
    # Test without noise
    print("Testing model without noise")
    evaluate_model(model, test_loader)

    # Test with both noise types
    original_state_dict = model.state_dict()
    print("Testing model with both noise types: salt_and_pepper and gaussian")
    add_salt_and_pepper_noise(model, best_noise_factors['salt_and_pepper'])
    add_gaussian_noise(model, best_noise_factors['gaussian'])
    evaluate_model(model, test_loader)
    model.load_state_dict(original_state_dict)
    print("Restored original model weights.")

final_test_comparison(model, test_loader, best_noise_factors)

# %%



