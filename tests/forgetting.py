"""
Training on two batches of different sequences in series and then testing recall of each batch
"""
import os
import numpy as np
import sys
sys.path.insert(0, os.getcwd())
from neural_memory import NeuralMemory, train
from sequence_generator import SequenceData
import torch

data_handler = SequenceData()
results = []

# generate two batches of sequences
seq1 = data_handler.generate_sequences(N=500)
seq2 = data_handler.generate_sequences(N=500)

# train on each batch in series
memory = NeuralMemory()
print("Training on first batch of 500 sequences...")
train(memory, data_handler, seq1, learning_rate=0.0001, num_epochs=10000)

print("Training on second batch of 500 sequences...")
train(memory, data_handler, seq2, learning_rate=0.0001, num_epochs=10000)

# test recall on first batch
correct1, incorrect1 = 0, 0
for i, test_sequence in enumerate(seq1):
    memory_prediction = torch.round(memory(test_sequence[:3].unsqueeze(0)))
    if abs(memory_prediction.item() - test_sequence[3].item()) < 0.01:
        correct1 += 1
    else:
        print(f"Incorrect: {memory_prediction.item()} vs {test_sequence[3].item()}")
        incorrect1 += 1

results.append((correct1, incorrect1))


# test recall on second batch
correct2, incorrect2 = 0, 0
for i, test_sequence in enumerate(seq2):
    memory_prediction = torch.round(memory(test_sequence[:3].unsqueeze(0)))
    if abs(memory_prediction.item() - test_sequence[3].item()) < 0.01:
        correct2 += 1
    else:
        print(f"Incorrect: {memory_prediction.item()} vs {test_sequence[3].item()}")
        incorrect2 += 1

results.append((correct2, incorrect2))


# print results
for i, (correct, incorrect) in enumerate(results):
    total = correct + incorrect
    print(f"Batch {i+1}: Correctly recalled: {correct}/{total}, Accuracy: {correct/total:.2f}")


