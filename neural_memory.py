import torch
from sequence_generator import SequenceData
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler


class NeuralMemory(nn.Module):
    """
    Single layer MLP for associative memory.
    Takes 3D key input, outputs 1D value prediction.
    """
    def __init__(self, key_dim=11, hidden_dim=64, value_dim=3):
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


def train_single_sequence(memory_module: NeuralMemory, data_handler: SequenceData, target_sequence, 
                        learning_rate=0.01, num_epochs=1000, print_every=100, verbose=True):
    """
    Train the memory module to memorise a single sequence.
    
    Args:
        memory_module: NeuralMemory instance
        data_handler: SequenceData instance
        target_sequence: 4-integer tensor to memorise
        learning_rate: learning rate for optimization
        num_epochs: number of training epochs
        print_every: print loss every N epochs
    
    Returns:
        List of losses during training
    """
    optimizer = optim.SGD(memory_module.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Encode sequence into tokens
    target_sequence_enc = torch.tensor(data_handler.encode(target_sequence), dtype=torch.float32)

    # Extract key and target value from the sequence
    key, target_value = data_handler.extract_key_value(target_sequence_enc)
    
    losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predicted_value = memory_module.forward(key)
        
        # Compute loss (associative memory loss from paper)
        loss = criterion(predicted_value, target_value)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
        if epoch % print_every == 0 and verbose:
            print(f"Epoch {epoch:4d}: Loss = {loss.item():.6f}, "
                f"Predicted = {data_handler.decode([int(i) for i in predicted_value.detach().numpy().round()])}, "
                f"Target = {target_sequence.numpy()[-1]}")
    
    return losses

def train_multiple_sequences(memory_module: NeuralMemory, data_handler: SequenceData, sequences, learning_rate=0.01, num_epochs=1000):
    optimizer = optim.SGD(memory_module.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for seq in sequences:
            key, value = data_handler.extract_key_value(seq)
            # Train on this key-value pair
            optimizer.zero_grad()
            pred = memory_module.forward(key)
            loss = criterion(pred, value)
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                print(f"Training on sequence {seq.numpy()}, Epoch {epoch}: Loss = {loss.item():.6f}")



def test_memory_recall(memory_module: NeuralMemory, data_handler: SequenceData, test_sequence):
    """
    Test the memory module's ability to recall the 4th number given first 3.
    
    Args:
        memory_module: trained NeuralMemory
        data_handler: SequenceData instance
        test_sequence: 4-integer tensor to test on
    
    Returns:
        Dictionary with test results
    """
    key, true_value = data_handler.extract_key_value(test_sequence)
    
    with torch.no_grad():
        predicted_value = memory_module(key)
    
    results = {
        'input_sequence': test_sequence.numpy(),
        'key': key.numpy(),
        'true_value': true_value.item(),
        'predicted_value': predicted_value.item(),
        'error': abs(predicted_value.item() - true_value.item()),
        'exact_match': abs(predicted_value.item() - true_value.item()) < 0.01
    }
    
    return results


if __name__ == "__main__":

    memory = NeuralMemory(key_dim=3, hidden_dim=64, value_dim=1)
    data = SequenceData()
    
    # Generate a target sequence to memorise
    target_seq = data.generate_sequence()
    print(f"Target sequence to memorise: {target_seq.numpy()}")
    print(f"Key (first 3): {target_seq[:3].numpy()}")
    print(f"Value (last 1): {target_seq[3].item()}")
    print("-" * 50)
    
    # Train the memory to memorise this sequence
    print("Training memory module...")
    losses = memory.train_single_sequence(data, target_seq, learning_rate=0.01, num_epochs=1000, print_every=200)
    
    print("-" * 50)
    print("Testing recall...")
    
    # Test recall on the same sequence
    results = test_memory_recall(memory, data, target_seq)
    print(f"Test Results:")
    print(f"  Input sequence: {results['input_sequence']}")
    print(f"  Given key: {results['key']}")
    print(f"  True value: {results['true_value']}")
    print(f"  Predicted value: {results['predicted_value']:.3f}")
    print(f"  Error: {results['error']:.6f}")
    print(f"  Exact match: {results['exact_match']}")


