import torch


class SequenceData:
    """
    Generates 4-integer sequences and handles key/value extraction.
    """
    def __init__(self):
        # Fixed projection matrices (simulating trained W_K, W_V)
        self.W_K = torch.tensor([[1, 0, 0, 0],
                                [0, 1, 0, 0], 
                                [0, 0, 1, 0]], dtype=torch.float32)  # Extract first 3
        
        self.W_V = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)  # Extract last 1
    
    def generate_sequence(self, low=0, high=10):
        """Generate a single random 4-integer sequence."""
        return torch.randint(low, high, (4,), dtype=torch.float32)
    
    def extract_key_value(self, sequence):
        """
        Extract key (first 3 numbers) and value (last 1 number) from sequence.
        Args:
            sequence: tensor of shape (4,)
        Returns:
            key: tensor of shape (3,)
            value: tensor of shape (1,)
        """
        key = torch.matmul(self.W_K, sequence)    # Shape: (3,)
        value = torch.matmul(self.W_V, sequence)  # Shape: (1,)
        return key, value
    


