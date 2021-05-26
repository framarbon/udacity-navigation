import numpy as np
from dqn_agent import DQN
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDQN(DQN):
    
    class Agent(DQN.Agent):
        
        def compute_loss(self, experiences, gamma):
            states, actions, rewards, next_states, dones = experiences
            next_actions = self.qnetwork_local(next_states).detach().max(dim=1)[1].unsqueeze(1)
            Q_target_next = self.qnetwork_target(next_states).gather(1, next_actions)
            Q_expected= self.qnetwork_local(states).gather(1, actions)
            return F.mse_loss(Q_expected, rewards + gamma * Q_target_next * (1 - dones))