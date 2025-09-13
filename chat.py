from typing import Optional
import os
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file
KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=KEY)

response = client.responses.create(
    model="gpt-5",
    input="What comes next in this sequence: 171, 34, 89, 2. Just give a number or say 'I don't know' if you don't know."
)

print(response.output_text)
