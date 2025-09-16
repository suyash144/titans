import torch
import numpy as np

class SequenceData:
    """
    Generates 4-integer sequences and handles key/value extraction.
    """
    def __init__(self):
        # Fixed projection matrices (simulating trained W_K, W_V)
        self.W_K = torch.tensor([[1, 0, 0, 0],
                                [0, 1, 0, 0], 
                                [0, 0, 1, 0]], dtype=torch.long)  # Extract first 3
        
        self.W_V = torch.tensor([[0, 0, 0, 1]], dtype=torch.long)  # Extract last 1
    
    def generate_sequence(self, low=0, high=1000):
        """Generate a single random 4-integer sequence."""
        return torch.randint(low, high, (4,), dtype=torch.long)
    
    def generate_sequences(self, N, low=0, high=1000):
        """Generate N random 4-integer sequences."""
        return [self.generate_sequence(low, high) for _ in range(N)]
    
    def encode(self, sequence):
        """Encode a sequence of integers into token IDs."""
        seq_str = ','.join(map(str, sequence.numpy()))
        return self.encoder.encode(seq_str)
    
    def encode_batch(self, sequences):
        """Encode a batch of sequences into token ID tensors."""
        encoded_seqs = []
        for seq in sequences:
            encoded = self.encode(seq)
            encoded_seqs.append(torch.tensor(encoded, dtype=torch.float32))
        return encoded_seqs
    
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
        key = torch.matmul(sequence, self.W_K.T)
        value = torch.matmul(sequence, self.W_V.T).float()
        return key, value
    


