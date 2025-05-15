import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import SimpleCNN  # Ensure this is defined as in your training code
import torchvision.transforms as transforms
import io

# Load model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("models/crowd_density_model.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# Image preprocessing
def preprocess_image(img: Image.Image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),  # match model training size if required
    ])
    return transform(img).unsqueeze(0)  # Shape: [1, C, H, W]

# Inference function
def predict_density_map(image_tensor):
    with torch.no_grad():
        output = model(image_tensor.to(device))
    density_map = output.squeeze().cpu().numpy()
    count = np.sum(density_map)
    return density_map, count

# Visualization function
def visualize(image_pil, density_map, count):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image_pil)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    im = axes[1].imshow(density_map, cmap='jet')
    axes[1].set_title(f"Predicted Density Map\nCount: {count:.2f}")
    axes[1].axis("off")

    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    st.pyplot(fig)

# Streamlit UI
st.title("Crowd Density Estimation 🧑‍🤝‍🧑📊")
st.markdown("Upload an image to estimate the number of people in it.")

uploaded_file = st.file_uploader("Upload crowd image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    image_tensor = preprocess_image(image)
    density_map, count = predict_density_map(image_tensor)

    # Display prediction
    st.success(f"🧮 Estimated Crowd Count: **{count:.2f}**")
    visualize(image, density_map, count)
