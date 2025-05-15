import numpy as np
import cv2

def generate_density_map(image, points, sigma=15):
    density_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    for point in points:
        x, y = min(int(point[0]), image.shape[1] - 1), min(int(point[1]), image.shape[0] - 1)
        density_map[y, x] += 1
    density_map = cv2.GaussianBlur(density_map, (0, 0), sigma)
    return density_map