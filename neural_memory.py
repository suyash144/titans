import torch
from sequence_generator import SequenceData
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils as utils
import random


class NeuralMemory(nn.Module):
    """
    Associative memory for random number sequence mappings.
    """
    def __init__(self, max_number=1000, sequence_length=3, embedding_dim=8, hidden_dim=128, value_dim=1):
        super().__init__()
        # Treat each number 0-1000 as a distinct categorical entity
        self.embedding = nn.Embedding(max_number + 1, embedding_dim)
        self.sequence_length = sequence_length
        
        # Process the embedded sequence
        input_dim = sequence_length * embedding_dim
        self.memory = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, value_dim)
        )
    
    def forward(self, numbers):
        """
        Args:
            numbers: tensor of shape (batch_size, 3) containing integers 0-1000
        Returns:
            predicted_value: tensor of shape (batch_size, 3)
        """
        # Each number gets its own learnable representation
        embedded = self.embedding(numbers)
        flattened = embedded.view(embedded.size(0), -1)
        return self.memory(flattened)


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

    # Extract key and target value from the sequence
    key, target_value = data_handler.extract_key_value(target_sequence)
    
    losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predicted_value = memory_module.forward(key)
        loss = criterion(predicted_value, target_value)
        # raise is loss is NaN
        if torch.isnan(loss):
            raise ValueError("Loss is NaN. Training diverged at epoch ",  epoch)
        
        # Backward pass
        loss.backward()
        # if epoch < 50:
        #     utils.clip_grad_norm_(memory_module.parameters(), 5.0)
        optimizer.step()

        losses.append(loss.item())
        
        if epoch % print_every == 0 and verbose:
            print(f"Epoch {epoch:4d}: Loss = {loss.item():.6f}, "
                f"Predicted = {predicted_value.item()}, "
                f"Target = {target_sequence.numpy()[-1]}")
    
    return losses

def train_multiple_sequences(memory_module: NeuralMemory, data_handler: SequenceData, sequences, 
                             learning_rate=0.01, num_epochs=1000, verbose=True):
    optimizer = optim.SGD(memory_module.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        random.shuffle(sequences)
        for i, seq in enumerate(sequences):
            key, value = data_handler.extract_key_value(seq)
            # Train on this key-value pair
            optimizer.zero_grad()
            pred = memory_module.forward(key)
            loss = criterion(pred, value)
            if torch.isnan(loss):
                raise ValueError("Loss is NaN. Training diverged at epoch ",  epoch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0 and verbose:
                print(f"Training on sequence {i+1}, Epoch {epoch}: Loss = {loss.item():.6f}")
        losses.append(epoch_loss)
        if epoch_loss < 1e-4:
            print(f"Early stopping at epoch {epoch} with total loss {epoch_loss:.6f}")
            break

    return losses

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
    pass
    


