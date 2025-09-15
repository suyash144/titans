import torch
import tiktoken
import numpy as np

class SequenceData:
    """
    Generates 4-integer sequences and handles key/value extraction.
    """
    def __init__(self):
        # Fixed projection matrices (simulating trained W_K, W_V)
        key_mat = np.eye(11)
        key_mat = np.concatenate([key_mat, np.zeros((4, 11))], axis=0)
        self.W_K = torch.tensor(key_mat, dtype=torch.float32)

        val_mat = np.zeros((15, 3))
        val_mat[12:, :] = np.eye(3)
        self.W_V = torch.tensor(val_mat, dtype=torch.float32)
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def generate_sequence(self, low=0, high=1000):
        """Generate a single random 4-integer sequence."""
        return torch.randint(low, high, (4,), dtype=torch.float32)
    
    def encode(self, sequence):
        """Encode a sequence of integers into token IDs."""
        seq_str = ','.join(map(str, sequence.numpy()))
        return self.encoder.encode(seq_str)
    
    def decode(self, token_ids):
        """Decode token IDs back into a sequence."""
        token_ids = [i for i in token_ids if i > 0 and i < 100000]
        decoded_str = self.encoder.decode(token_ids)
        return list(decoded_str.split(','))
    
    def extract_key_value(self, sequence):
        """
        Extract key (first 3 numbers) and value (last 1 number) from sequence.
        Args:
            sequence: tensor of shape (4,)
        Returns:
            key: tensor of shape (3,)
            value: tensor of shape (1,)
        """
        key = torch.matmul(sequence, self.W_K)
        value = torch.matmul(sequence, self.W_V)
        return key, value
    


