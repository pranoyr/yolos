from ultralytics import YOLO
import time
import torch
import numpy as np


device = torch.device('mps')

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt').to(device)

# Define path to the image file
source = 'images/img.jpeg'


# while True:
#     start = time.time()
#     # Run inference on the source
#     results = model(source)  # list of Results objects
#     print(f'FPS: {1 / (time.time() - start)}')


x = np.array([[1, 2, 3],
               [4, 5, 6]])


print("**")
for i in range(2):
    print(x[i])