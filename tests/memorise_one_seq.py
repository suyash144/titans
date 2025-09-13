import torch
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import sys
sys.path.insert(0, os.getcwd())
from neural_memory import NeuralMemory
from sequence_generator import SequenceData

load_dotenv()  # Load .env file
KEY = os.getenv('OPENAI_API_KEY')



def query(use_memory: bool, sequence, client: OpenAI, memory_module: NeuralMemory=None):

    if use_memory:
        # Get memory prediction
        with torch.no_grad():
            memory_prediction = np.round(memory_module(sequence).item())

        print(f"Memory prediction for key {sequence}: {memory_prediction}")
    
        # Create prompt with memory as context
        prompt = f"""What comes next in this sequence: {', '.join(map(str, sequence))}
                Memory recall: {sequence}->{memory_prediction}
                Just give a number or say 'I don't know'."""
    else:
        # Create prompt without memory context
        prompt = f"""What comes next in this sequence: {', '.join(map(str, sequence))}
                Just give a number or say 'I don't know'."""
    
    # Query the language model
    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return response.output_text



if __name__ == "__main__":

    client = OpenAI(api_key=KEY)
    memory = NeuralMemory(key_dim=3, hidden_dim=64, value_dim=1)
    data = SequenceData()
    test_sequence = data.generate_sequence()
    print(f"Test sequence: {test_sequence.numpy()}")

    print("Querying without memory...")
    response_no_memory = query(False, test_sequence[:3], client)
    print("Response without memory:")
    print(response_no_memory)

    print("Training memory module...")
    
    losses = memory.train_single_sequence(data, test_sequence, learning_rate=0.01, 
                                          num_epochs=1000, print_every=1000)

    print("Querying with memory...")
    response_with_memory = query(True, test_sequence[:3], client, memory_module=memory)
    print("Response with memory:")
    print(response_with_memory)
