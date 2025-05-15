import os
import torch
import scipy.io
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import gaussian_filter

class CrowdDataset(Dataset):
    def __init__(self, root_dir, transform=None, downsample_ratio=4):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg')])
        self.downsample_ratio = downsample_ratio

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize and convert to tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = cv2.resize(image, (1024, 680))  
            image = image / 255.0  # Normalize
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # [C, H, W]

        # Load corresponding annotation
        mat_name = img_name.replace('.jpg', '_ann.mat')
        mat_path = os.path.join(self.root_dir, mat_name)
        mat = scipy.io.loadmat(mat_path)
        annPoints = mat['annPoints']  # shape: (N, 2)

        # Create density map
        height, width = image.shape[1], image.shape[2]
        density_map = np.zeros((height, width), dtype=np.float32)
        for point in annPoints:
            x = min(width - 1, max(0, int(point[0])))
            y = min(height - 1, max(0, int(point[1])))
            density_map[y, x] += 1

        density_map = gaussian_filter(density_map, sigma=4)

        # ↓↓↓ Resize density map to match model output ↓↓↓
        out_h, out_w = height // self.downsample_ratio, width // self.downsample_ratio
        density_map = cv2.resize(density_map, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

        # Normalize density map total count (preserve person count)
        density_map = density_map * ((height * width) / (out_h * out_w))

        # Convert to tensor
        density_map = torch.from_numpy(density_map).unsqueeze(0).float()  # shape: [1, H, W]

        return image, density_map
