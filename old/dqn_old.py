import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from replay_memory import ReplayMemory
from torch import nn
import torch
import math

class IncreasingConcaveNet(nn.Module):
    def __init__(self, layers, device="cpu"):
        super(IncreasingConcaveNet, self).__init__()
        weight_dims = list(zip(layers[1:], layers))
        self.layers = nn.ModuleList()
        self.device = device
        for odim, idim in weight_dims:
            self.layers.append(nn.Linear(idim, odim))

        self.activ = lambda x: torch.min(torch.zeros_like(x), x)

    def forward(self, z):
        for layer in self.layers[:-1]:
            z = self.activ(layer(z))

        return self.layers[-1](z)

    def clamp_weights(self):
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.data.clamp_(min=0)

def concavity_regularizer(models, strength=1.0, func="linear"):
    penalty = 0.0
    assert(func in ("linear", "square")), "type must either be 'linear' or 'square'"
    power = (1 if func == "linear" else 2)
    for model in models:
        try:
            for layer in model.layers:
                penalty += torch.pow(torch.sum(torch.relu(-layer.weight)), power)
        except:
            penalty += torch.pow(torch.sum(torch.relu(-model.weight)), power)
    return strength * penalty

class MonotoneSubmodularNet(nn.Module):
    def __init__(self, phi_layers, lamb, m_layers, m_size=1, device="cpu"):
        super(MonotoneSubmodularNet, self).__init__()
        self.lamb = lamb
        self.m_layers = m_layers

        self.m = nn.ModuleList()
        self.phi = nn.ModuleList()
        for i in range(m_layers):
            self.phi.append(IncreasingConcaveNet(phi_layers, device=device))
            layers = []
            layers.append(nn.Linear(m_size, 10))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(10, 10))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(10, 1))
            layers.append(nn.Softplus())
            self.m.append(nn.Sequential(*layers))

    def forward(self, x):
        batch_x = torch.unsqueeze(x, -1)
        ret = torch.sum(self.m[0](batch_x), dim=1)

        for i in range(1, self.m_layers):
            ret = self.phi[i](self.lamb * torch.sum(self.m[i](batch_x), dim=1) + (1 - self.lamb) * ret)

        return ret

    def clamp_weights(self, hard_enforce=False):
        with torch.no_grad():
            if hard_enforce:
                for net in self.phi:
                    net.clamp_weights()
            for layer in self.m:
                for sublayer in layer:
                    if hasattr(sublayer, 'weight'):
                        sublayer.weight.data.clamp_(0)
