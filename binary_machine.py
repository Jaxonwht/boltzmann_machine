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
SAMPLE_SIZE = 30
RAND_IN = torch.rand(SAMPLE_SIZE, VISIBLE_SIZE, device=device)

class RBM:
    def __init__(self, H, V, k):
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
        p_h0_given_v0, h0 = self.sample_h(v0)
        hi = torch.tensor(h0)
        for i in range(k):
            _, vi = self.sample_v(hi)
            p_hi_given_vi, hi = self.sample_h(vi)
        update = v0.t().mm(p_h0_given_v0) - vi.t().mm(p_hi_given_vi)
        self.weight
