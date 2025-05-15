import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataloader import CrowdDataset
from model import SimpleCNN
import os

# Set the dataset path
root_dir = r"C:\Users\Jaini Sanghvi\Desktop\Crowd Density Estimation Using CNN and Density Maps\dataset\UCF_CC_50"

# Initialize dataset with downsample_ratio=4 (make sure this is set in dataloader.py)
dataset = CrowdDataset(root_dir)

# Split into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, loss, optimizer
model = SimpleCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(100):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Make sure output and target shapes match
        if outputs.shape != targets.shape:
            print(f"Skipping due to shape mismatch: {outputs.shape} vs {targets.shape}")
            continue

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/100], Training Loss: {avg_train_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)

            if outputs.shape != targets.shape:
                print(f"Skipping validation mismatch: {outputs.shape} vs {targets.shape}")
                continue

            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/crowd_density_model.pth")
print("Model saved to models/crowd_density_model.pth")
