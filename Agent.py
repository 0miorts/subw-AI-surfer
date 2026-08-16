import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Net
from collections import deque

class Agent:
    def __init__(self, gamma = 0.95, eps=1, min_eps = 0.01, eps_decay = 0.995, lr=1e-4):
        self.gamma = gamma
        self.eps = eps
        self.min_eps = min_eps
        self.eps_decay = eps_decay
        self.lr = lr
        self.model = Net.Net()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE']
        self.memory = deque(maxlen=10000)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def select_action(self, state):
        if np.random.rand() < self.eps:
            return self.actions[np.random.randint(len(self.actions))]
        else:
            with torch.no_grad():
                pass

    def epsilon_decay_func(self):
        self.eps = max(self.min_eps, self.eps * self.eps_decay)


