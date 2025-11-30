import os
import yaml
from litellm import completion


class LLMClient:
    def __init__(self, api_key: str):
        os.environ["OPENAI_API_KEY"] = api_key

    def load_prompt(self, file_path: str):
        """Load prompt from YAML file."""
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def get_response(self, messages: list) -> str:
        """Generate response from the model using the provided messages."""
        response = completion(model="openai/gpt-4o", messages=messages)
        return response['choices'][0]['message']['content']


def main():
    # Load API Key (make sure to securely load it, here it's hardcoded for simplicity)
    api_key = "your-openai-api-key"  # Replace with your actual OpenAI API key

    # Create an LLMClient instance
    client = LLMClient(api_key)

    # Load the prompt from the prompt.yaml file
    prompt = client.load_prompt("prompt.yaml")

    # Get the system and user messages from the prompt
    messages = [prompt['system'], prompt['user']]

    # Get the response from the LLM model
    response = client.get_response(messages)

    # Print the response
    print(response)


if __name__ == "__main__":
    main()
