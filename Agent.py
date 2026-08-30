import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Net
from collections import deque

class Agent:
    def __init__(self, gamma = 0.95, eps=1, min_eps = 0.01, eps_decay = 0.999, lr=1e-4):
        self.gamma = gamma
        self.eps = eps
        self.min_eps = min_eps
        self.eps_decay = eps_decay
        self.lr = lr
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE']
        self.memory = deque(maxlen=10000)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Net.Net().to(self.device)
        self.target_model = Net.Net().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def select_action(self, state):
        if np.random.rand() < self.eps:
            return self.actions[np.random.randint(len(self.actions))]
        else:
            with torch.no_grad():
                state = state.unsqueeze(0).to(self.device)
                q_values = self.model(state)
                action_idx = q_values.argmax(dim=1).item()
            return self.actions[action_idx]

    def epsilon_decay_func(self):
        self.eps = max(self.min_eps, self.eps * self.eps_decay)

    def store_memory(self, state, action, reward, next_state, done):
        action_idx = self.actions.index(action)
        self.memory.append((state, action_idx, reward, next_state, done))

    def update_target_memory(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def learn(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q = next_q_values.max(1)[0]
            target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = F.mse_loss(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_checkpoint(self, path, episode):
        torch.save({
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'eps': self.eps,
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.eps = checkpoint['eps']
        return checkpoint['episode']


