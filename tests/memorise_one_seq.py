import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import sys
sys.path.insert(0, os.getcwd())
from neural_memory import NeuralMemory, train_single_sequence
from sequence_generator import SequenceData
from utils import query
import tiktoken


if __name__ == "__main__":
    
    load_dotenv()  # Load .env file
    KEY = os.getenv('OPENAI_API_KEY')

    client = OpenAI(api_key=KEY)
    memory = NeuralMemory(key_dim=11, hidden_dim=64, value_dim=3)
    data = SequenceData()
    test_sequence = data.generate_sequence()
    print(f"Test sequence: {test_sequence.numpy()}")

    # response_no_memory = query(False, test_sequence[:3], client)
    # print("Response without memory:")
    # print(response_no_memory)

    print("Training memory module...")
    losses = train_single_sequence(memory, data, test_sequence, learning_rate=0.05, 
                                          num_epochs=5000, print_every=500)

    response_with_memory = query(True, test_sequence[:3], client, memory_module=memory)
    print("Response with memory:")
    print(response_with_memory)
