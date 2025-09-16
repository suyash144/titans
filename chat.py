import torch
import os
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file
KEY = os.getenv('OPENAI_API_KEY')


def query_with_memory(memory_module, sequence, client: OpenAI):

    key = torch.tensor(sequence, dtype=torch.float32)
    
    # Get memory prediction
    with torch.no_grad():
        memory_prediction = memory_module(key).item()
    
    # Create prompt with memory as context
    prompt = f"""Memory recall: {memory_prediction:.1f}
            What comes next in this sequence: {', '.join(map(str, sequence))}
            Just give a number or say 'I don't know'."""
    
    # Query the language model
    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    return response.output_text

