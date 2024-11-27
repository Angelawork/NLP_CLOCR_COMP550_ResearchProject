from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

async def generate_response(provider: str, model_name: str, prompt: str, user_message: str):
    if provider == "openrouter":
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
    elif provider == "openai":
        client = OpenAI(api_key=openai_api_key)
    else:
        raise ValueError("Unsupported provider. Choose 'openrouter' or 'openai'.")

    completion = client.chat.completions.create(
        model=model_name, 
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    client = OpenAI(api_key=openai_api_key)
    models = client.models.list()
    print("Available OpenAI models:")
    for model in models.data:
        print(model.id)
    