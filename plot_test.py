# import cv2
# import numpy as np

# # Create a simple image (black screen)
# img = np.zeros((480, 640, 3), dtype=np.uint8)

# # Display the image
# cv2.imshow('Test Image', img)

# # Wait for a key press to close the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import torch

import sys
print(sys.executable)

from utils.torch_utils import select_device
device = select_device('0' if torch.cuda.is_available() else 'cpu')
print(device)

# import torch
# print(torch.cuda.is_available())  # This should return True if CUDA is properly installed
# print(torch.__version__)  # Check PyTorch version
# print(torch.version.cuda)  # Check CUDA version used by PyTorch


import torch

# print("CUDA Available: ", torch.cuda.is_available())  # Should return True
# print("CUDA Device: ", torch.cuda.current_device())  # Should return the device ID, e.g., 0
# print("CUDA Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))  # Should return GPU name
