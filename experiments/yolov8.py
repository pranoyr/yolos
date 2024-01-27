from ultralytics import YOLO
import time
import torch

device = torch.device('mps')

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt').to(device)

# Define path to the image file
source = 'images/img.jpeg'


while True:
    start = time.time()
    # Run inference on the source
    results = model(source)  # list of Results objects
    print(f'FPS: {1 / (time.time() - start)}')