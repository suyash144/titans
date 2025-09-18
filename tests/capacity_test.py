# This script will test the capacity of the memory module by trying to memorise progressively more sequences
import os
import numpy as np
import sys
sys.path.insert(0, os.getcwd())
from neural_memory import NeuralMemory, train
from sequence_generator import SequenceData
import torch


capacities = [1e4, 5e4, 1e5, 5e5, 1e6]
data_handler = SequenceData()
results = []

for cap in capacities:
    # generate sequences
    sequences = data_handler.generate_sequences(N=int(cap))
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


