# This script builds on memorise_one_seq.py and memorise_multi_seq.py
# There is nothing technically new here, just showing that the memory module can be used in the wild.
# This is all within the slightly contrived example of having to remember 4-digit sequences and being prompted 
# with the first 3 digits to predict the 4th.

import random
import sys
import os
sys.path.insert(0, os.getcwd())
from neural_memory import NeuralMemory, train_multiple_sequences
from sequence_generator import SequenceData
import torch

N = 64
data_handler = SequenceData()
memory = NeuralMemory()
sequences = []
correct, incorrect = 0, 0


for test in range(N):

    if test > 0:
        # randomly pick between memorising a new sequence or testing an old one
        action = random.choice(['memorise', 'test'])
    else:
        action = 'memorise'  # must memorise at least one sequence first

    if action == 'memorise':
        # randomly generate a new sequence to memorise
        test_sequence = data_handler.generate_sequence()
        print(f"Memorising new sequence {test+1}: {test_sequence.numpy()}")
        sequences.append(test_sequence)
        losses = train_multiple_sequences(memory, data_handler, sequences, 
                                        learning_rate=0.00001, num_epochs=10000, verbose=False)
    else:
        sequence = random.choice(sequences)
        print(f"Testing recall of sequence: {sequence.numpy()}")
        memory_prediction = torch.round(memory(sequence[:3].unsqueeze(0)))
        if abs(memory_prediction.item() - sequence[3].item()) < 0.01:
            correct += 1
            print("Correct")
        else:
            incorrect += 1
            print(f"Incorrect: {memory_prediction.item()} vs {sequence[3].item()}")

print(f"Correct: {correct}, Incorrect: {incorrect}")
print(f"Total capacity reached: {len(sequences)}, Accuracy: {correct/(correct+incorrect):.2f}")
