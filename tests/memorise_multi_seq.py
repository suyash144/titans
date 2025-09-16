import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import sys
sys.path.insert(0, os.getcwd())
from neural_memory import NeuralMemory, train_multiple_sequences
from sequence_generator import SequenceData
from utils import query

load_dotenv()
KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=KEY)
memory = NeuralMemory()
data = SequenceData()

N = 8                              # number of sequences to memorise
sequences = data.generate_sequences(N)

losses = train_multiple_sequences(memory, data, sequences, learning_rate=0.00001, num_epochs=10000)

results = {"with_memory": [], "without_memory": []}

# Test querying with and without memory
for i, test_sequence in enumerate(sequences):
    print("-" * 50)
    print(f"Test sequence {i+1}: {test_sequence.numpy()}")

    response_no_memory = query(False, test_sequence[:3], client)
    try:
        response_no_memory = float(response_no_memory)
        if abs(int(response_no_memory) - int(test_sequence[3].item())) < 0.01:
            res=True
        else:
            res=False
    except:
        res=False
    results["without_memory"].append(res)

    print("Response without memory:", response_no_memory)

    response_with_memory = query(True, test_sequence[:3], client, memory_module=memory)
    try:
        response_with_memory = float(response_with_memory)
        if abs(int(response_with_memory) - int(test_sequence[3].item())) < 0.01:
            res=True
        else:
            print(response_with_memory, test_sequence[3].item())
            res=False
    except:
        print("Failed to parse response with memory.")
        res=False
    results["with_memory"].append(res)

    print("Response with memory:", response_with_memory)

# Summarise results
print("-" * 50)
print("Summary of results:")
for key in results:
    correct = sum(results[key])
    total = len(results[key])
    print(f"{key}: {correct}/{total} correct ({(correct/total)*100:.2f}%)")

