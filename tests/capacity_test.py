# This script will test the capacity of the memory module by trying to memorise progressively more sequences
import os
import numpy as np
import sys
sys.path.insert(0, os.getcwd())
from neural_memory import NeuralMemory
from sequence_generator import SequenceData
from sequence_dataset import SequenceDataset
from torch.utils.data import DataLoader
import torch

def train(memory_module, data_handler, sequences, batch_size=64, learning_rate=0.01, num_epochs=1000):

    dataset = SequenceDataset(data_handler, sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.SGD(memory_module.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_keys, batch_values in dataloader:
            optimizer.zero_grad()
            
            predictions = memory_module(batch_keys)
            loss = criterion(predictions, batch_values)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()


capacities = [512, 1024, 2048, 4096, 8192]
data_handler = SequenceData()
results = []

for cap in capacities:
    # generate sequences
    sequences = data_handler.generate_sequences(N=cap)
    # reset memory module
    memory = NeuralMemory()
    print(f"Training memory module to memorise {cap} sequences...")
    losses = train(memory, data_handler, sequences, batch_size=64, learning_rate=0.0001, num_epochs=10000)

    correct, incorrect = 0, 0
    for i, test_sequence in enumerate(sequences):
        memory_prediction = torch.round(memory(test_sequence[:3].unsqueeze(0)))
        if abs(memory_prediction.item() - test_sequence[3].item()) < 0.01:
            correct += 1
        else:
            print(f"Incorrect: {memory_prediction.item()} vs {test_sequence[3].item()}")
    
    results.append(correct)

# print results
for cap, res in zip(capacities, results):
    print(f"Capacity: {cap}, Correctly recalled: {res}, Accuracy: {res/cap:.2f}")


