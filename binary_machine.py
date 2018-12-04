import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.init as init

# Set the device for this neural network
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

HIDDEN_SIZE = 3
VISIBLE_SIZE = 10
SAMPLING_STEP = 5
LEARNING_RATE = 1e-4
SAMPLE_SIZE = 30
RAND_IN = torch.rand(SAMPLE_SIZE, VISIBLE_SIZE, device=device)

def outer_product(x, y):
    return x.view(-1, 1).mm(y.view(1, -1))

class RBM(nn.Module):
    def __init__(self, H, V, k, lr):
        super().__init__()
        self.learning_rate = lr
        self.sample_step = k
        self.bias_hidden = torch.zeros(H, device=device)
        self.bias_visible = torch.zeros(V, device=device)
        self.weight = torch.zeros(V, H, device=device)
        init.xavier_uniform_(self.weight)

    def sample_h(self, v):
        '''
        :param v: visible input
        :return: a tensor of probability h given v, and a tensor of a sample of h
        '''
        a = v.mm(self.weight)
        p_h_given_v = a + self.bias_hidden.expand_as(a)
        p_h_given_v = torch.sigmoid(p_h_given_v)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, h):
        '''
        :param h: hidden intermediate
        :return: a tensor of probability v given h, and a tensor of sample of v
        '''
        a = h.mm(self.weight.t())
        p_v_given_h = a + self.bias_visible.expand_as(a)
        p_v_given_h = torch.sigmoid(p_v_given_h)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train_one_step(self, v0):
        batch_size = v0.size()[0]
        p_h0_given_v0, h0 = self.sample_h(v0)
        hi = torch.tensor(h0, device=device)
        for i in range(self.sample_step):
            _, vi = self.sample_v(hi)
            p_hi_given_vi, hi = self.sample_h(vi)
        update_weight = torch.zeros_like(self.weight, device=device)
        update_hidden = torch.zeros_like(self.bias_hidden, device=device)
        update_visible = torch.zeros_like(self.bias_visible, device=device)
        for i in range(batch_size):
            update_weight.add(outer_product(v0[i], p_h0_given_v0[i]) - outer_product(vi[i], p_hi_given_vi[i]))
            update_hidden.add(p_h0_given_v0 - p_hi_given_vi)
            update_visible.add(v0 - vi)
        update_visible / batch_size
        update_hidden / batch_size
        update_visible / batch_size
        self.bias_visible.add(self.learning_rate * update_visible)
        self.bias_hidden.add(self.learning_rate * update_hidden)
        self.weight.add(self.learning_rate * update_weight)

model = RBM(HIDDEN_SIZE, VISIBLE_SIZE, SAMPLING_STEP, LEARNING_RATE)

model.train_one_step(RAND_IN)

