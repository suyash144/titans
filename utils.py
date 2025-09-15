import torch
from openai import OpenAI
from neural_memory import NeuralMemory



def query(use_memory: bool, sequence, client: OpenAI, memory_module: NeuralMemory=None):

    if use_memory:
        # Get memory prediction
        with torch.no_grad():
            memory_prediction = torch.round(memory_module(sequence.unsqueeze(0)))
        
        print(f"Memory prediction: {sequence} -> {memory_prediction}")

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