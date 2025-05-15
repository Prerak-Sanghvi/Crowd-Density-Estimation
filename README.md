# Crowd Density Estimation Using CNN and Density Maps

This project estimates crowd density in images using a Convolutional Neural Network (CNN) and density maps. It predicts the number of people in crowded scenes and visualizes the density map.

## Features

- Input image crowd count estimation
- Predicted density map visualization
- Optional ground truth upload for error comparison (MAE/RMSE)
- Web app interface using Streamlit for easy use

## Getting Started

### Requirements

- Python 3.x
- PyTorch
- OpenCV
- Streamlit
- Other dependencies in `requirements.txt`

### Usage

1. Clone the repo:

```bash
git clone https://github.com/Prerak-Sanghvi/Crowd-Density-Estimation.git
cd Crowd-Density-Estimation

2. Install dependencies:

pip install -r requirements.txt

3. Run the app:

streamlit run app.py

4. Upload an image and optionally a .mat file with ground truth for evaluation.

Project Structure
app.py - Streamlit web application

train.py - Model training script

test.py - Evaluation script

model.py - CNN model architecture

dataloader.py - Data loading utilities

dataset/ - Folder containing dataset images

output/ - Folder for saving result images (ignored in repo)

models/ - Folder containing saved model weights

Evaluation Metrics
Mean Absolute Error (MAE)

Root Mean Square Error (RMSE)

