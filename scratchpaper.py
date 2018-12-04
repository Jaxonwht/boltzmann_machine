import torch
from binary_machine import outer_product

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

x = torch.rand(4)
y = torch.rand(5)

print(outer_product(x, y))