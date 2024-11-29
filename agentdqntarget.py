import numpy as np
import random
from collections import deque

from QNN import QNN  # Assurez-vous que votre module QNN est bien défini
from replaybuffer import ReplayBuffer  # Assurez-vous que votre ReplayBuffer est bien défini

import torch
from torch import nn
import torch.optim as optim

class AgentDQNTarget:
    """Agent qui utilise l'algorithme DQN avec réseau cible."""

    def __init__(self, state_size: int, action_size: int, gamma=0.99, lr=1e-3, buffer_size=100000, batch_size=64, tau=0.001, update_every=10000):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.update_every = update_every

        self.qnetwork = QNN(state_size, action_size)
        self.target_qnetwork = QNN(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_size, batch_size)

        self.loss_fn = nn.MSELoss()

        self.steps = 0

        self.update_target_network()

    def update_target_network(self):
        for param_target, param_source in zip(self.target_qnetwork.parameters(), self.qnetwork.parameters()):
            param_target.data.copy_(param_source.data)

    def soft_update_target_network(self):
        for param_target, param_source in zip(self.target_qnetwork.parameters(), self.qnetwork.parameters()):
            param_target.data.copy_(self.tau * param_source.data + (1.0 - self.tau) * param_target.data)

    def sampling_step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):

        self.memory.add(state, action, reward, next_state, done)

    def train_step(self):

        if len(self.memory) < self.batch_size:
            return 
        
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Calculer Q(s,a) pour les actions sélectionnées
        q_values = self.qnetwork(states).gather(1, actions)

        # Calculer Q(s',a') max en utilisant le réseau cible
        q_next = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)

        # Calculer Q_target = r + γ * max(Q(s', a')) * (1 - done)
        q_target = rewards + (self.gamma * q_next * (1 - dones))

        # Calculer la perte
        loss = self.loss_fn(q_values, q_target)

        # Backpropagation et mise à jour des paramètres
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Incrémenter le compteur de pas
        self.steps += 1

        # Mise à jour complète du réseau cible toutes les N étapes
        if self.steps % self.update_every == 0:
            self.update_target_network()

        # Mise à jour lente du réseau cible à chaque étape
        self.soft_update_target_network()

    def act_egreedy(self, state: np.ndarray, eps: float = 0.0) -> int:
        if random.random() < eps:
            return random.choice(np.arange(self.action_size))

        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            q_values = self.qnetwork(state)
        return q_values.argmax().item()

