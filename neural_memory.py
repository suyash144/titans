import torch.nn as nn


class NeuralMemory(nn.Module):
    """
    Single layer MLP for associative memory.
    Takes 3D key input, outputs 1D value prediction.
    """
    def __init__(self, key_dim=3, hidden_dim=64, value_dim=1):
        super().__init__()
        self.memory = nn.Sequential(
            nn.Linear(key_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, value_dim)
        )
    
    def forward(self, key):
        """
        Forward pass through memory.
        Args:
            key: tensor of shape (batch_size, 3) or (3,) for single key
        Returns:
            predicted_value: tensor of shape (batch_size, 1) or (1,)
        """
        return self.memory(key)


