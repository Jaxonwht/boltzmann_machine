import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Set the device for this neural network
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

HIDDEN_SIZE = 3
VISIBLE_SIZE = 10
SAMPLING_STEP = 5

class RBM(nn.Module):
    def __init__(self, H, V, k):
        super().__init__()
