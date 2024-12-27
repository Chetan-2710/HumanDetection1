 torch

from utils.torch_utils import select_device
device = select_device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)