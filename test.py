import torch
import matplotlib.pyplot as plt
import numpy as np
from dataloader import CrowdDataset
from model import SimpleCNN
from torch.utils.data import DataLoader
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SimpleCNN().to(device)
model_path = "models/crowd_density_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Loaded model from {model_path}")

# Load dataset
root_dir = r"C:\Users\Jaini Sanghvi\Desktop\Crowd Density Estimation Using CNN and Density Maps\dataset\UCF_CC_50"
test_dataset = CrowdDataset(root_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Inference and Visualization
def visualize_prediction(image_tensor, predicted_density, ground_truth_density, index):
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pred_density_np = predicted_density.squeeze().cpu().detach().numpy()
    gt_density_np = ground_truth_density.squeeze().cpu().detach().numpy()

    pred_count = np.sum(pred_density_np)
    gt_count = np.sum(gt_density_np)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image_np.astype(np.uint8))
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_density_np, cmap='jet')
    plt.title(f"GT Density Map\nCount: {gt_count:.2f}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_density_np, cmap='jet')
    plt.title(f"Predicted Density Map\nCount: {pred_count:.2f}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"output/result_{index}.png")
    print(f"Saved output/result_{index}.png")
    plt.close()

# Create output directory
os.makedirs("output", exist_ok=True)

# Metrics initialization
mae = 0.0
mse = 0.0
max_samples = 10  
# Run inference
for idx, (image, target_density) in enumerate(test_loader):
    if idx >= max_samples:
        break

    image = image.to(device)
    target_density = target_density.to(device)

    with torch.no_grad():
        predicted_density = model(image)

    pred_count = predicted_density.sum().item()
    gt_count = target_density.sum().item()

    # Update MAE and MSE
    mae += abs(pred_count - gt_count)
    mse += (pred_count - gt_count) ** 2

    # Save visual result
    visualize_prediction(image.cpu(), predicted_density.cpu(), target_density.cpu(), idx)

# Final metrics
mae /= max_samples
rmse = np.sqrt(mse / max_samples)
print(f"\nEvaluation on {max_samples} samples:")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
