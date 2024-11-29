import numpy as np
import random
import torch

from QNN import QNN

class AgentGlouton():
    """Agent qui utilise la prédiction de son réseau de neurones pour choisir ses actions selon une stratégie d’exploration (pas d'apprentissage)."""

    def __init__(self,input_dim: int, output_dim: int):
        """

        """
        self.q_network = QNN(input_dim, output_dim)

    def act_egreedy(self, state : np.ndarray , eps: float = 0.0) -> int:
        """
            eps: probabilité d'exploration
        """
        if random.random() < eps:
            # Exploration : choisir une action aléatoire
            action = random.randint(0, self.q_network.fc3.out_features - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item() 
        
        return action
    
        



