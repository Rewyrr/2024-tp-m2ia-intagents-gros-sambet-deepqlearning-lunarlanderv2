import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNN(nn.Module):
    """Reseau de neurones pour approximer la Q fonction."""

    def __init__(self,input_dim:int, output_dim:int):
        """Initialisation des parametres ...
        """
        super(QNN, self).__init__()
        
        "*** TODO ***"
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, state: np.ndarray) -> torch.Tensor :
        """Forward pass"""

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)
            
        "*** TODO ***"
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values


