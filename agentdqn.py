import random
from collections import namedtuple, deque
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from QNN import QNN
from replaybuffer import ReplayBuffer


class AgentDQN:
    """Agent qui utilise l'algorithme de deep QLearning avec replaybuffer."""

    def __init__(self, state_size: int, action_size: int, gamma=0.99, lr=1e-3, buffer_size=100000, batch_size=64):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size

        # Réseau de neurones pour approximer Q(s,a)
        self.qnetwork = QNN(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_size, batch_size)

        self.loss_fn = nn.MSELoss()

    def sampling_step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):

        self.memory.add(state, action, reward, next_state, done)

    def train_step(self):

        if len(self.memory) < self.batch_size:
            return  # Pas assez de transitions stockées pour un apprentissage

        # Récupération d'un minibatch de transitions
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Q(s, a) pour les actions sélectionnées
        q_values = self.qnetwork(states).gather(1, actions)

        # Q(s', a') max pour les actions futures
        q_next = self.qnetwork(next_states).detach().max(1)[0].unsqueeze(1)

        # Q_target = r + γ * max(Q(s', a')) * (1 - done)
        q_target = rewards + (self.gamma * q_next * (1 - dones))

        # Calcul de la perte
        loss = self.loss_fn(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def act_egreedy(self, state: np.ndarray, eps: float = 0.0) -> int:

        if random.random() < eps:
            return random.choice(np.arange(self.action_size))
        
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            q_values = self.qnetwork(state)
        return q_values.argmax().item()
