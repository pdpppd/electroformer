import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ElectroFormer
import numpy as np

# Hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 0.0005  # Reduced learning rate
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load the filtered data from the .npy file
X_filtered = np.load('data/ptbxl_filtered_data.npy')

# Create a PyTorch dataset and data loader
class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

dataset = ECGDataset(X_filtered)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = ElectroFormer(num_leads=12, d_model=256, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, data in enumerate(dataloader):
        data = data.float().to(device)  # Convert data to shape (batch_size, num_leads, num_timesteps)

        # Randomly mask out parts of the input sequence
        mask_ratio = 0.15
        mask = torch.rand(data.shape[:2]).to(device) < mask_ratio
        masked_data = data.clone().detach()
        masked_data[mask] = 0.0

        # Forward pass
        output = model(masked_data)

        # Compute loss only for masked positions
        loss = criterion(output[mask], data[mask])

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        # Print progress
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

print("Training completed!") 
