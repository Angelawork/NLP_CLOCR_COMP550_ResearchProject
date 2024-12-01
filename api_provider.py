from openai import OpenAI
from dotenv import load_dotenv
import os
import requests

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_api_key = os.getenv("HF_API_KEY")

def get_huggingface_response(model_name, input, prompt=None):
    """
    usable models: ["meta-llama/Meta-Llama-3-8B-Instruct","mistralai/Mixtral-8x7B-Instruct-v0.1"]
    """
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    client = {"API_URL": API_URL, "headers": headers}
    default_prompt = """
      Given the following text:
      "{input}"
    """
    prompt = prompt or default_prompt
    formatted_prompt = prompt.format(input=input)
    payload = {"inputs": formatted_prompt}
    response = requests.post(client["API_URL"], headers=client["headers"], json=payload)
    if response.status_code == 200:
        result = response.json()[0]["generated_text"]
    else:
        print(f"Hugging Face API call failed with status code {response.status_code}")
        result = response.json().get("error", "Unknown error")

    if result.startswith(formatted_prompt):
        result = result[len(formatted_prompt):].strip()
    else:
        result = result.strip()
    return result


async def generate_response(
    provider: str, model_name: str, prompt: str, user_message: str
):
    if provider == "openrouter":
        # Initialize OpenRouter client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
    elif provider == "openai":
        # Initialize OpenAI client with the API key
        client = OpenAI(api_key=openai_api_key)
    else:
        raise ValueError("Unsupported provider. Choose 'openrouter' or 'openai'.")

    # Create a chat completion request with the provided model and messages
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    client = OpenAI(api_key=openai_api_key)

    # Retrieve and list available OpenAI models
    models = client.models.list()
    print("Available OpenAI models:")
    for model in models.data:
        print(model.id)
